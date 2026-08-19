"""Stage F — third-architecture robustness: cross-modal negativity dominance on LLaVA (T2.1).

Gemma-3 (SigLIP pooled to 256 soft tokens) and Qwen-VL (native dynamic-resolution ViT) are two VLM
"unification" designs. LLaVA-1.5 is the third major family: a FROZEN CLIP ViT whose patch features are
mapped through a linear/MLP projector into UNPOOLED patch tokens. Replicating the behavioral effect on
all three turns "two architectures" into "the dominant VLM unification designs" — a categorically
stronger generality claim than another checkpoint from a family already covered.

Raw HuggingFace, NO TransformerBridge and NO probe (the Gemma probe lives in Gemma's activation space).
Read-out = BEHAVIORAL VALENCE only, the closed-vocab P(positive) − P(negative) at the first answer
token — identical to the Qwen port, so the shared CPU analyzer (`analyze_stage_f`) consumes the output
unchanged and the numbers are directly comparable to Gemma + Qwen.

Reuses the model-agnostic scoring/selection from `stage_f_qwen` (valence_score, emotion_logprobs,
emotion_token_ids, readout, select_extreme_images, _user_text); only the model load and the chat
template are LLaVA-specific. Run in the `requirements-qwen.txt` env (recent transformers with
`AutoModelForImageTextToText` + llava-hf support). Needs the EMOTIC test parquet staged.

  base        — 75 high + 75 low EMOTIC-valence images × full context bank → override rate + asymmetry.
  --text-only — each context sentence with NO image → the stimulus-confound control (|neg|/|pos|).
  --reanalyze — recompute metrics from the saved parquet (CPU, no model).
`--model <hub id>` (default LLaVA-1.5-7B; LLaVA-NeXT / OneVision also load via AutoModelForImageTextToText).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE)
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_f_qwen import (_user_text, emotion_logprobs, emotion_token_ids, readout,
                           select_extreme_images, valence_score)

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"


def _conditions():
    return ([("none", "none", None)]
            + [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
            + [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
            + [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)])


# --------------------------------------------------------------------------- model
def load_llava(model_name: str = DEFAULT_MODEL):
    """Load a LLaVA-family model + processor via raw transformers (auto-selects the right class)."""
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as Cls   # covers 1.5 / NeXT / OneVision
    except ImportError:  # older transformers
        from transformers import LlavaForConditionalGeneration as Cls
    model = Cls.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def build_inputs(processor, image, context_sentence):
    """LLaVA chat inputs for one image + context; image=None gives the text-only (image-ablated) form.

    LLaVA-hf's chat template places the `<image>` placeholder (expanded to the unpooled patch tokens by
    the processor) at the head of the user turn: `USER: <image>\\n{context} {question} ASSISTANT:`.
    """
    content = ([{"type": "image"}] if image is not None else []) + \
              [{"type": "text", "text": _user_text(context_sentence)}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs = [image] if image is not None else None
    return processor(images=imgs, text=text, return_tensors="pt")


# --------------------------------------------------------------------------- analysis (shared metrics)
def _analyze_and_print(df, model_name, multi=None, n_skipped=0) -> dict:
    """Compute + save + print the override rate (primary) and graded asymmetry (secondary).

    Uses the SHARED `analyze_stage_f` metrics so LLaVA lands in the same table as Gemma + Qwen. Also
    the CPU `--reanalyze` entry point.
    """
    from .analyze_stage_f import _asymmetry_vs_floor, _flip_override
    asym = _asymmetry_vs_floor(df) if len(df) else {}
    flip = _flip_override(df) if len(df) else {}
    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name,
               "read_out": "behavioral_valence", "n_images": int(df["image_path"].nunique()) if len(df) else 0,
               "n_skipped": n_skipped, "n_rows": int(len(df)),
               "asymmetry_vs_floor": asym, "flip_override": flip, "tokenization_multi_token": multi or {}}
    save_json(metrics, STAGE_F_DIR / "conflict_llava_metrics.json")

    print(f"\nStage F [LLaVA: {model_name}] — {metrics['n_images']} images, {len(df)} rows "
          f"({n_skipped} skipped). Read-out: behavioral valence.")
    if multi:
        print(f"  multi-token labels (first sub-token scored): {list(multi)}")
    print("  RAW mean valence per cell (does the image move it?):")
    for grp in ("positive", "negative"):
        g = df[df["image_group"] == grp]
        cells = "  ".join(f"{c}={g[g['condition'] == c]['valence'].mean():+.2f}"
                          for c in ("none", "neutral", "positive", "negative"))
        print(f"    {grp:8s} img: {cells}")
    if flip:
        print(f"  NEGATIVITY DOMINANCE (override rate, argmax-emotion, cross-model): neg-ctx overrides "
              f"positive image {flip['neg_ctx_overrides_pos_img']:.0%}  vs  pos-ctx overrides negative "
              f"image {flip['pos_ctx_overrides_neg_img']:.0%}  (gap {flip['dominance_gap']:+.0%}, CI "
              f"[{flip['dominance_gap_ci95'][0]:+.0%},{flip['dominance_gap_ci95'][1]:+.0%}])")
    if "drop_pos_img_neg_ctx" in asym:
        print(f"  [graded valence] |drop|-|rise| {asym['asymmetry_index']:+.3f} "
              f"CI [{asym['asymmetry_ci95'][0]:+.3f},{asym['asymmetry_ci95'][1]:+.3f}]  head-room pull "
              f"drop {asym['headroom_norm_pull_drop']:.2f} vs rise {asym['headroom_norm_pull_rise']:.2f}")
    return metrics


# --------------------------------------------------------------------------- base pass
def run_base(config_path: str, model_name: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    n_images = limit_override or int(cfg.get("n_images", 150))
    sel = select_extreme_images(n_images)
    model, processor = load_llava(model_name)
    tok_ids = emotion_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    rows, n_skip = [], 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f llava base"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for cond, cid, sentence in _conditions():
            val, lp = readout(model, build_inputs(processor, img, sentence), tok_ids)
            rows.append({"image_path": r["image_path"], "image_valence": float(r["valence"]),
                         "image_group": r["image_group"], "condition": cond, "context_id": cid,
                         "context": sentence or "", "text_code": TEXT_CODE[cond],
                         "probe_readout": float("nan"),  # no probe on LLaVA; column kept for schema
                         "valence": val, **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    out_pq = STAGE_F_DIR / "conflict_llava.parquet"
    df.to_parquet(out_pq)
    metrics = _analyze_and_print(df, model_name, multi=multi, n_skipped=n_skip)
    print(f"  data -> {out_pq}")
    print(f"  NEXT: python -m src.experiments.stage_f_llava --text-only --model {model_name}")
    return metrics


def reanalyze(config_path: str) -> dict:
    ensure_dirs()
    pq = STAGE_F_DIR / "conflict_llava.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run the LLaVA base pass first.")
    mpath = STAGE_F_DIR / "conflict_llava_metrics.json"
    model_name = load_config(mpath).get("model", "unknown") if mpath.exists() else "unknown"
    return _analyze_and_print(pd.read_parquet(pq), model_name)


# --------------------------------------------------------------------------- text-only control
def run_text_only(config_path: str, model_name: str) -> dict:
    load_config(config_path)
    ensure_dirs()
    model, processor = load_llava(model_name)
    tok_ids = emotion_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    rows = []
    for cond, cid, sentence in _conditions():
        val, lp = readout(model, build_inputs(processor, None, sentence), tok_ids)  # image=None
        rows.append({"condition": cond, "context_id": cid, "context": sentence or "",
                     "text_code": TEXT_CODE[cond], "valence": val, "argmax_emotion": max(lp, key=lp.get),
                     **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})
    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "text_only_llava.parquet")

    neu = float(df[df["condition"] == "neutral"]["valence"].mean())
    pe = float((df[df["condition"] == "positive"]["valence"] - neu).mean())
    ne = float((df[df["condition"] == "negative"]["valence"] - neu).mean())
    pr = float(df[df["condition"] == "positive"]["valence"].mean())
    nr = float(df[df["condition"] == "negative"]["valence"].mean())
    raw_ratio = abs(nr) / abs(pr) if pr else float("nan")
    text_ratio = abs(ne) / abs(pe) if pe else float("nan")

    img_ratio = None
    bpath = STAGE_F_DIR / "conflict_llava_metrics.json"
    if bpath.exists():
        a = load_config(bpath).get("asymmetry_vs_floor", {})
        dn, dp = a.get("drop_pos_img_neg_ctx"), a.get("congruent_pos_img_pos_ctx")
        if dn is not None and dp:
            img_ratio = abs(dn) / abs(dp)

    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name, "neutral_baseline": neu,
               "pos_effect": pe, "neg_effect": ne, "pos_raw": pr, "neg_raw": nr,
               "text_only_ratio_vs_neutral": text_ratio, "text_only_ratio_raw": raw_ratio,
               "image_conditioned_ratio": img_ratio, "tokenization_multi_token": multi}
    save_json(metrics, STAGE_F_DIR / "text_only_llava_metrics.json")

    print(f"\nStage F [LLaVA: {model_name}] text-only — {len(rows)} forwards (no images).")
    if multi:
        print(f"  multi-token labels (first sub-token scored): {list(multi)}")
    print("  per-context (no image): valence | argmax emotion")
    for _, r in df.iterrows():
        print(f"    {r['context_id']:5s} {r['valence']:+6.3f}  {r['argmax_emotion']:9s}  \"{r['context'][:42]}\"")
    print(f"  vs-neutral: pos {pe:+.3f}  neg {ne:+.3f}  |neg|/|pos| = {text_ratio:.2f}")
    print(f"  RAW (vs 0): pos {pr:+.3f}  neg {nr:+.3f}  |neg|/|pos| = {raw_ratio:.2f}")
    ref = raw_ratio if np.isfinite(raw_ratio) else text_ratio
    if img_ratio is not None and np.isfinite(ref):
        verdict = ("CROSS-MODAL amplification (image inflates the ratio)" if img_ratio > 1.25 * ref
                   else "STIMULUS confound (ratios match)" if abs(img_ratio - ref) <= 0.25 * ref
                   else "image dampens (reversed)")
        print(f"  image-conditioned |neg|/|pos| = {img_ratio:.2f}  vs raw text-only {ref:.2f}  →  {verdict}")
    else:
        print("  (run the base pass first to auto-compare against the image-conditioned ratio)")
    print(f"  data -> {STAGE_F_DIR/'text_only_llava.parquet'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F third architecture — LLaVA (behavioral valence)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="LLaVA hub id (default llava-1.5-7b-hf)")
    ap.add_argument("--text-only", action="store_true", help="run the image-ablated context control")
    ap.add_argument("--reanalyze", action="store_true", help="recompute metrics from the saved parquet (CPU)")
    ap.add_argument("--limit", type=int, default=None, help="EMOTIC image count (base pass)")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze(args.config)
    elif args.text_only:
        run_text_only(args.config, args.model)
    else:
        run_base(args.config, args.model, limit_override=args.limit)


if __name__ == "__main__":
    main()
