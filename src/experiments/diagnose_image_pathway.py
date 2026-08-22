"""Diagnose the Gemma image pathway: TransformerBridge vs raw HuggingFace, same image, same prompt.

WHY THIS EXISTS. Gemma-3-4B is the only model whose published Stage F numbers came from
TransformerBridge; every other model went through raw HF. It is also the only run whose no-context
image separation is degraded:

    Gemma published (bridge)   discriminability gap +0.374   AUC 0.788
    Gemma re-run   (raw HF)    gap +1.674   AUC 0.982
    Qwen published (raw HF)    gap +1.558   AUC 0.987
    LLaVA published(raw HF)    gap +1.471   AUC 0.989

A weak image signal inflates the override rate directly — text wins by default when the image barely
registers — which is consistent with the published +65% override gap collapsing to +12% on raw HF.
This script tests whether the two stacks actually feed the model different images, and whether they
produce different answers for the same input.

It compares, for ONE image and ONE context, in this order (cheapest first):
  1. `input_ids`      — identical token sequences? (rules out a prompt/splicing difference)
  2. `pixel_values`   — same shape, range and values? (the preprocessing hypothesis)
  3. emotion logprobs — do the two stacks answer differently? (the behavioral consequence)

Run on Colab, where both stacks and HF_TOKEN are available:

    python -m src.experiments.diagnose_image_pathway
    python -m src.experiments.diagnose_image_pathway --n-images 5   # a few images, logprobs only
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from ..data.conflict_contexts import NEGATIVE_CONTEXTS, context_prompt
from ..data.labels import EMOTION_LABELS
from .stage_f_qwen import emotion_logprobs, emotion_token_ids, select_extreme_images


def _versions() -> None:
    import transformers
    print("  transformers      ", transformers.__version__)
    try:
        import transformer_lens
        print("  transformer_lens  ", transformer_lens.__version__,
              "  (>=3.2.1 required for the Gemma3 multimodal hotfix)")
    except Exception as e:
        print("  transformer_lens   unavailable:", e)


def _stats(name: str, t) -> dict:
    a = t.detach().float().cpu().numpy()
    d = {"shape": list(a.shape), "mean": float(a.mean()), "std": float(a.std()),
         "min": float(a.min()), "max": float(a.max())}
    print(f"    {name:16s} shape={d['shape']}  mean={d['mean']:+.4f}  std={d['std']:.4f}  "
          f"range=[{d['min']:+.3f}, {d['max']:+.3f}]")
    return d


def compare(model_name: str, n_images: int, device: str = "cuda") -> dict:
    print("=== versions ===")
    _versions()

    sentence = NEGATIVE_CONTEXTS[0]
    prompt = context_prompt(sentence)          # the exact scaffold the published run used
    sel = select_extreme_images(max(2, n_images * 2))
    paths = sel["image_path"].tolist()[:n_images] if n_images > 1 else [sel["image_path"].iloc[0]]

    # ---------------- raw HF
    print("\n=== raw HuggingFace ===")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    hf = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto").eval()
    tok_hf = emotion_token_ids(processor)

    # ---------------- bridge
    print("\n=== TransformerBridge ===")
    from ..bridge.boot import boot_gemma
    from ..bridge.multimodal import build_image_inputs
    bridge = boot_gemma(model_name, device=device)
    tok_br = {w: bridge.tokenizer.encode(" " + w, add_special_tokens=False)[0]
              for w in EMOTION_LABELS}
    print(f"  emotion token ids identical across stacks: {tok_hf == tok_br}")
    if tok_hf != tok_br:
        diff = {w: (tok_hf[w], tok_br[w]) for w in EMOTION_LABELS if tok_hf[w] != tok_br[w]}
        print(f"  [!] DIFFERING IDS (hf, bridge): {diff}")

    out: dict = {"model": model_name, "token_ids_match": tok_hf == tok_br, "images": []}
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        print(f"\n--- image {i+1}/{len(paths)}: {path.split('/')[-1]}  (size {img.size}) ---")

        br_in = build_image_inputs(bridge, img, prompt=prompt)
        hf_in = processor(text=prompt, images=[img], return_tensors="pt")

        ids_b, ids_h = br_in["input_ids"], hf_in["input_ids"]
        same_ids = ids_b.shape == ids_h.shape and bool((ids_b.cpu() == ids_h.cpu()).all())
        print(f"  input_ids: bridge {list(ids_b.shape)}  hf {list(ids_h.shape)}  identical={same_ids}")

        rec = {"path": path, "input_ids_identical": same_ids,
               "n_tokens": [int(ids_b.shape[-1]), int(ids_h.shape[-1])]}

        pv_b, pv_h = br_in.get("pixel_values"), hf_in.get("pixel_values")
        if pv_b is not None and pv_h is not None:
            print("  pixel_values:")
            rec["pixel_bridge"] = _stats("bridge", pv_b)
            rec["pixel_hf"] = _stats("hf", pv_h)
            if pv_b.shape == pv_h.shape:
                d = (pv_b.detach().float().cpu() - pv_h.detach().float().cpu()).abs()
                rec["pixel_max_abs_diff"] = float(d.max())
                rec["pixel_mean_abs_diff"] = float(d.mean())
                print(f"    max|Δ|={d.max():.6f}   mean|Δ|={d.mean():.6f}"
                      f"   {'IDENTICAL' if d.max() < 1e-6 else '<-- DIFFERENT PIXELS'}")
            else:
                rec["pixel_max_abs_diff"] = None
                print(f"    [!] SHAPE MISMATCH {list(pv_b.shape)} vs {list(pv_h.shape)}")

        # behavioral consequence: same question, two stacks
        with torch.no_grad():
            lg_b = bridge.run_with_hooks(ids_b, pixel_values=pv_b, fwd_hooks=[])[0, -1]
            lg_h = hf(**{k: v.to(hf.device) for k, v in hf_in.items()}).logits[0, -1]
        lp_b, lp_h = emotion_logprobs(lg_b, tok_br), emotion_logprobs(lg_h, tok_hf)
        top_b, top_h = max(lp_b, key=lp_b.get), max(lp_h, key=lp_h.get)
        md = max(abs(lp_b[w] - lp_h[w]) for w in EMOTION_LABELS)
        rec.update({"argmax_bridge": top_b, "argmax_hf": top_h, "max_logprob_diff": float(md)})
        print(f"  answer:   bridge -> {top_b:9s}   hf -> {top_h:9s}   max|Δ logprob| = {md:.3f}"
              f"   {'AGREE' if top_b == top_h else '<-- DISAGREE'}")
        out["images"].append(rec)

    agree = sum(r["argmax_bridge"] == r["argmax_hf"] for r in out["images"])
    pix = [r.get("pixel_max_abs_diff") for r in out["images"] if r.get("pixel_max_abs_diff") is not None]
    print(f"\n=== summary over {len(out['images'])} image(s) ===")
    print(f"  stacks agree on the answer: {agree}/{len(out['images'])}")
    if pix:
        print(f"  pixel_values max|Δ| across images: {max(pix):.6f}")
        print("  -> non-zero means the two stacks feed the model DIFFERENT images, which is the "
              "leading explanation for the published Gemma run's degraded image separation "
              "(AUC 0.788 vs 0.982).")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--n-images", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    compare(a.model, a.n_images, a.device)


if __name__ == "__main__":
    main()
