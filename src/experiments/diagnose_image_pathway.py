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
from .stage_f_qwen import (emotion_logprobs, emotion_token_ids, select_extreme_images,
                           valence_score)


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
    # EMOTIC annotates per PERSON, so image_path recurs across rows — taking a head slice returns the
    # same photo several times (and only from one valence group). Dedupe, then interleave the two
    # groups so the sample spans both positive and negative images.
    sel = select_extreme_images(max(4, n_images * 4))
    pos = sel[sel["image_group"] == "positive"]["image_path"].drop_duplicates().tolist()
    neg = sel[sel["image_group"] == "negative"]["image_path"].drop_duplicates().tolist()
    paths = [p for pair in zip(pos, neg) for p in pair][:n_images] or pos[:n_images]
    print(f"\n  sampling {len(paths)} distinct image(s): "
          f"{sum(p in pos for p in paths)} positive / {sum(p in neg for p in paths)} negative")

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

    # ---------------- (A) TEXT-ONLY PARITY — the scoping test.
    # Inputs are already known to be byte-identical while the answers differ, so the bridge's FORWARD
    # differs from raw HF. This asks whether that is image-specific. If the two stacks agree with no
    # image, the language path is sound and only image-conditioned results are affected (Stages C, D,
    # E and the Gemma half of F). If they disagree here too, the bridge is wrong for text as well and
    # the text-side Stage A probes/steering are implicated on top.
    print("\n=== (A) text-only parity (no image at all) ===")
    ctx = f"Context: {sentence} "
    text_prompt = (f"<start_of_turn>user\n{ctx}What single emotion is this person feeling?"
                   f"<end_of_turn>\n<start_of_turn>model\n")
    ids_t = processor.tokenizer(text_prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lt_b = bridge.run_with_hooks(ids_t.to(device), fwd_hooks=[])[0, -1]
        lt_h = hf(input_ids=ids_t.to(hf.device)).logits[0, -1]
    lpt_b, lpt_h = emotion_logprobs(lt_b, tok_br), emotion_logprobs(lt_h, tok_hf)
    tb, th = max(lpt_b, key=lpt_b.get), max(lpt_h, key=lpt_h.get)
    mdt = max(abs(lpt_b[w] - lpt_h[w]) for w in EMOTION_LABELS)
    # Grade on the quantity the paper actually reports, not on a raw logprob threshold. `max|Δ
    # logprob|` over 13 labels is dominated by the LEAST likely label, where a tiny probability change
    # is a large log change, so it overstates disagreement. Behavioural valence (P(pos) − P(neg)) is
    # what every Stage A/C/D/F number is built from, and the argmax is what the override rate uses.
    vb, vh = valence_score(lt_b, tok_br), valence_score(lt_h, tok_hf)
    dv = abs(vb - vh)
    text_ok = tb == th and dv < 0.05
    print(f"  bridge -> {tb:9s}   hf -> {th:9s}   max|Δ logprob| = {mdt:.4f}")
    print(f"  behavioural valence: bridge {vb:+.4f}   hf {vh:+.4f}   |Δ| = {dv:.4f}")
    print(f"  VERDICT: {'text path AGREES on argmax AND valence — bug is image-specific' if text_ok else ('argmax agrees but valence differs by %.3f — text path is approximate, not exact' % dv if tb == th else 'text path DISAGREES on the argmax — bridge is wrong for text too')}")

    out: dict = {"model": model_name, "token_ids_match": tok_hf == tok_br,
                 "text_only": {"argmax_bridge": tb, "argmax_hf": th,
                               "max_logprob_diff": float(mdt), "valence_bridge": float(vb),
                               "valence_hf": float(vh), "valence_diff": float(dv),
                               "argmax_agree": bool(tb == th), "agree": bool(text_ok)},
                 "images": []}
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
        ivb, ivh = valence_score(lg_b, tok_br), valence_score(lg_h, tok_hf)
        rec.update({"argmax_bridge": top_b, "argmax_hf": top_h, "max_logprob_diff": float(md),
                    "valence_bridge": float(ivb), "valence_hf": float(ivh),
                    "valence_diff": float(abs(ivb - ivh))})
        print(f"  answer:   bridge -> {top_b:9s}   hf -> {top_h:9s}   max|Δ logprob| = {md:.3f}"
              f"   {'AGREE' if top_b == top_h else '<-- DISAGREE'}")
        print(f"  behavioural valence: bridge {ivb:+.4f}   hf {ivh:+.4f}   |Δ| = {abs(ivb-ivh):.4f}")

        # (C) HOW MUCH DOES THE IMAGE MOVE EACH STACK? Same text, image present vs absent.
        # The published Gemma image separation was AUC 0.788 vs 0.982 on raw HF, so the bridge
        # appears to under-weight the image rather than ignore it. This quantifies that directly:
        # a small bridge shift next to a large HF shift means the image is being partially applied.
        with torch.no_grad():
            nb = bridge.run_with_hooks(ids_t.to(device), fwd_hooks=[])[0, -1]
            nh = hf(input_ids=ids_t.to(hf.device)).logits[0, -1]
        lpn_b, lpn_h = emotion_logprobs(nb, tok_br), emotion_logprobs(nh, tok_hf)
        shift_b = max(abs(lp_b[w] - lpn_b[w]) for w in EMOTION_LABELS)
        shift_h = max(abs(lp_h[w] - lpn_h[w]) for w in EMOTION_LABELS)
        rec.update({"image_shift_bridge": float(shift_b), "image_shift_hf": float(shift_h)})
        print(f"  image influence (max|Δ logprob| vs no-image, same text): "
              f"bridge {shift_b:.3f}   hf {shift_h:.3f}"
              + ("   <-- bridge under-weights the image" if shift_b < 0.5 * shift_h else ""))
        out["images"].append(rec)

    agree = sum(r["argmax_bridge"] == r["argmax_hf"] for r in out["images"])
    pix = [r.get("pixel_max_abs_diff") for r in out["images"] if r.get("pixel_max_abs_diff") is not None]
    ids_ok = all(r["input_ids_identical"] for r in out["images"])
    sb = np.mean([r["image_shift_bridge"] for r in out["images"]])
    sh = np.mean([r["image_shift_hf"] for r in out["images"]])
    print(f"\n=== summary over {len(out['images'])} image(s) ===")
    print(f"  input_ids identical:        {ids_ok}")
    print(f"  pixel_values max|Δ|:        {max(pix) if pix else float('nan'):.6f}")
    print(f"  text-only stacks agree:     {out['text_only']['agree']} "
          f"(max|Δ| {out['text_only']['max_logprob_diff']:.4f})")
    print(f"  image-conditioned agree:    {agree}/{len(out['images'])}")
    print(f"  mean image influence:       bridge {sb:.3f}  vs  hf {sh:.3f}")

    print("\n  READING:")
    if ids_ok and (not pix or max(pix) < 1e-6):
        print("   * inputs are byte-identical, so preprocessing is NOT the cause.")
    if out["text_only"]["agree"] and agree < len(out["images"]):
        print("   * the stacks agree with NO image and diverge WITH one: the defect is specific to")
        print("     the bridge's multimodal path. Text-side results (Stage A) are unaffected;")
        print("     every image-conditioned bridge result is (Stages C, D, E, and Gemma in F).")
    elif out["text_only"]["argmax_agree"]:
        dvt = out["text_only"]["valence_diff"]
        dvi = np.mean([r["valence_diff"] for r in out["images"]])
        print(f"   * TEXT: same argmax, valence differs by {dvt:.3f}. IMAGE: valence differs by "
              f"{dvi:.3f} ({dvi/max(dvt,1e-9):.0f}x larger).")
        print("     The text path is approximate but broadly consistent; the multimodal path is not.")
        print("     Stage A is likely recoverable, image-conditioned bridge results are not.")
    else:
        print("   * the stacks disagree on the ARGMAX even with no image: the bridge forward is wrong")
        print("     for text too, implicating the Stage A probes/steering as well.")
    if sb < 0.5 * sh:
        print("   * the bridge under-weights the image relative to raw HF, which is exactly what")
        print("     would depress image separation (AUC 0.788 vs 0.982) and inflate the override rate.")
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
