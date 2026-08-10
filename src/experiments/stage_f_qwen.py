"""Stage F — multi-model robustness: cross-modal negativity dominance on Qwen-VL.

Replicates the CORE Stage F claims on a different-architecture VLM (Qwen3-VL / Qwen2.5-VL) via raw
HuggingFace, NO TransformerBridge and NO probe. The Gemma pleasantness probe lives in Gemma's
activation space and does not transfer, so the read-out here is BEHAVIORAL VALENCE only — the
closed-vocab P(positive emotions) − P(negative emotions) at the first answer token — which needs no
probe and carries the asymmetry, the text-only control, and (later) the patching recovery.

Two passes, both writing the SAME column schema as the Gemma base run so the shared CPU analyzer
`analyze_stage_f._asymmetry_vs_floor` consumes them unchanged:
  base       — EMOTIC valence extremes (75 high + 75 low) × full context bank → asymmetry vs floor.
  --text-only — each context sentence with NO image → the stimulus-confound control (|neg|/|pos|),
               compared to the image-conditioned ratio.

Run in the SEPARATE `requirements-qwen.txt` env (transformers>=4.57 for Qwen3-VL). Needs the EMOTIC
test parquet staged and the images at the parquet's paths. `--limit N`, `--model <hub id>`.
Qwen-VL uses VARIABLE image-token counts — we only read the LAST token's logits, so no position
bookkeeping is needed here (that comes in the patching port).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE)
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import PROCESSED_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QUESTION = "What single emotion is this person feeling?"

# Valence split — identical to stage_a_steering (copied to keep this module import-light: it must run
# in the Qwen venv, which does not have the TransformerBridge / crowd-enVENT stack).
POSITIVE = ("joy", "pride", "relief", "trust")
NEGATIVE = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")


def valence_score(logits_last, tok_ids) -> float:
    idx = torch.tensor([tok_ids[w] for w in EMOTION_LABELS], device=logits_last.device)
    probs = torch.softmax(logits_last[idx].float(), dim=-1)
    p = {w: probs[i].item() for i, w in enumerate(EMOTION_LABELS)}
    return sum(p[w] for w in POSITIVE) - sum(p[w] for w in NEGATIVE)


def emotion_logprobs(logits_last, tok_ids) -> dict[str, float]:
    idx = torch.tensor([tok_ids[w] for w in EMOTION_LABELS], device=logits_last.device)
    logp = torch.log_softmax(logits_last[idx].float(), dim=-1)
    return {w: float(logp[i]) for i, w in enumerate(EMOTION_LABELS)}


# --------------------------------------------------------------------------- model
def load_qwen(model_name: str = DEFAULT_MODEL):
    """Load a Qwen-VL model + processor (Qwen3-VL or Qwen2.5-VL) via raw transformers."""
    from transformers import AutoProcessor
    lname = model_name.lower()
    if "qwen3" in lname:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    model = Cls.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def emotion_token_ids(processor) -> dict[str, int]:
    """First-subtoken id per emotion label on the Qwen tokenizer (leading space kept)."""
    tok = processor.tokenizer
    return {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in EMOTION_LABELS}


def _user_text(context_sentence: str | None) -> str:
    ctx = "" if not context_sentence else f"Context: {context_sentence} "
    return f"{ctx}{QUESTION}"


def build_inputs(processor, image, context_sentence):
    """Qwen chat inputs for one image + context; image=None gives the text-only (image-ablated) form."""
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": _user_text(context_sentence)})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs = [image] if image is not None else None
    return processor(text=[text], images=imgs, padding=True, return_tensors="pt")


def readout(model, inputs, tok_ids):
    """(behavioral_valence, {emotion: logprob}) at the last prompt token."""
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    last = out.logits[0, -1].float()
    return valence_score(last, tok_ids), emotion_logprobs(last, tok_ids)


# --------------------------------------------------------------------------- image selection
def select_extreme_images(n: int) -> pd.DataFrame:
    """n/2 highest- and n/2 lowest-EMOTIC-valence test rows (read the parquet directly; no Gemma stack)."""
    df = pd.read_parquet(PROCESSED_DIR / "emotic_test.parquet").reset_index(drop=True)
    df = df[np.isfinite(df["valence"].to_numpy(dtype=float))].sort_values("valence")
    k = n // 2
    low = df.head(k).assign(image_group="negative")
    high = df.tail(k).assign(image_group="positive")
    return pd.concat([high, low]).reset_index(drop=True)


# --------------------------------------------------------------------------- base pass
def run_base(config_path: str, model_name: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    n_images = limit_override or int(cfg.get("n_images", 150))

    conditions = ([("none", "none", None)]
                  + [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
                  + [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
                  + [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)])

    sel = select_extreme_images(n_images)
    model, processor = load_qwen(model_name)
    tok_ids = emotion_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    rows, n_skip = [], 0
    for _, r in tqdm(list(sel.iterrows()), desc=f"stage-f qwen base"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for cond, cid, sentence in conditions:
            val, lp = readout(model, build_inputs(processor, img, sentence), tok_ids)
            rows.append({"image_path": r["image_path"], "image_valence": float(r["valence"]),
                         "image_group": r["image_group"], "condition": cond, "context_id": cid,
                         "context": sentence or "", "text_code": TEXT_CODE[cond],
                         "probe_readout": float("nan"),  # no probe on Qwen; column kept for schema
                         "valence": val, **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    out_pq = STAGE_F_DIR / "conflict_qwen.parquet"
    df.to_parquet(out_pq)

    from .analyze_stage_f import _asymmetry_vs_floor
    asym = _asymmetry_vs_floor(df) if len(df) else {}
    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name, "read_out": "behavioral_valence",
               "n_images": int(sel.shape[0] - n_skip), "n_skipped": n_skip, "n_rows": int(len(rows)),
               "asymmetry_vs_floor": asym, "tokenization_multi_token": multi}
    save_json(metrics, STAGE_F_DIR / "conflict_qwen_metrics.json")

    print(f"\nStage F [Qwen: {model_name}] base — {metrics['n_images']} images × {len(conditions)} "
          f"conditions = {len(rows)} forwards ({n_skip} skipped). Read-out: behavioral valence.")
    if multi:
        print(f"  multi-token labels (first sub-token scored): {list(multi)}")
    # RAW per-(image group × condition) means — the diagnostic for whether the IMAGE moves the output
    # at all under Qwen's saturation (e.g. is a positive face under a neutral context read as positive,
    # or floored to sadness = image ignored?).
    print("  RAW mean valence per cell (does the image move it?):")
    for grp in ("positive", "negative"):
        g = df[df["image_group"] == grp]
        cells = "  ".join(f"{c}={g[g['condition'] == c]['valence'].mean():+.2f}"
                          for c in ("none", "neutral", "positive", "negative"))
        print(f"    {grp:8s} img: {cells}")
    if "drop_pos_img_neg_ctx" in asym:
        print(f"  ASYMMETRY vs FLOOR: drop {asym['drop_pos_img_neg_ctx']:+.3f}  rise "
              f"{asym['rise_neg_img_pos_ctx']:+.3f}  |drop|-|rise| {asym['asymmetry_index']:+.3f} "
              f"CI [{asym['asymmetry_ci95'][0]:+.3f},{asym['asymmetry_ci95'][1]:+.3f}]  "
              f"MW p={asym['mannwhitney_p_greater']:.3f}")
        print(f"  → {asym['interpretation']}")
    print(f"  data -> {out_pq}")
    print("  NEXT: python -m src.experiments.stage_f_qwen --text-only --model " + model_name)
    return metrics


# --------------------------------------------------------------------------- text-only control
def run_text_only(config_path: str, model_name: str) -> dict:
    load_config(config_path)
    ensure_dirs()
    conditions = ([("none", "none", None)]
                  + [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
                  + [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
                  + [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)])
    model, processor = load_qwen(model_name)
    tok_ids = emotion_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    rows = []
    for cond, cid, sentence in conditions:
        val, lp = readout(model, build_inputs(processor, None, sentence), tok_ids)  # image=None
        top = max(lp, key=lp.get)
        rows.append({"condition": cond, "context_id": cid, "context": sentence or "",
                     "text_code": TEXT_CODE[cond], "valence": val, "argmax_emotion": top,
                     **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})
    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "text_only_qwen.parquet")

    neu = float(df[df["condition"] == "neutral"]["valence"].mean())
    none_v = float(df[df["condition"] == "none"]["valence"].mean())
    pe = float((df[df["condition"] == "positive"]["valence"] - neu).mean())
    ne = float((df[df["condition"] == "negative"]["valence"] - neu).mean())
    # raw ratio (vs 0, not vs the possibly-contaminated neutral) — robust when neutral is floored.
    pr = float(df[df["condition"] == "positive"]["valence"].mean())
    nr = float(df[df["condition"] == "negative"]["valence"].mean())
    raw_ratio = abs(nr) / abs(pr) if pr else float("nan")
    text_ratio = abs(ne) / abs(pe) if pe else float("nan")

    # compare to the image-conditioned ratio from the base run, if present
    img_ratio = None
    bpath = STAGE_F_DIR / "conflict_qwen_metrics.json"
    if bpath.exists():
        a = load_config(bpath).get("asymmetry_vs_floor", {})
        dn, dp = a.get("drop_pos_img_neg_ctx"), a.get("congruent_pos_img_pos_ctx")
        if dn is not None and dp:
            img_ratio = abs(dn) / abs(dp)

    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name,
               "neutral_baseline": neu, "none_baseline": none_v, "pos_effect": pe, "neg_effect": ne,
               "pos_raw": pr, "neg_raw": nr, "text_only_ratio_vs_neutral": text_ratio,
               "text_only_ratio_raw": raw_ratio, "image_conditioned_ratio": img_ratio,
               "tokenization_multi_token": multi}
    save_json(metrics, STAGE_F_DIR / "text_only_qwen_metrics.json")

    print(f"\nStage F [Qwen: {model_name}] text-only — {len(rows)} forwards (no images).")
    if multi:
        print(f"  ⚠ multi-token labels (first sub-token scored): {list(multi)}")
    print(f"  per-context (no image): valence | argmax emotion")
    for _, r in df.iterrows():
        print(f"    {r['context_id']:5s} {r['valence']:+6.3f}  {r['argmax_emotion']:9s}  "
              f"\"{r['context'][:42]}\"")
    print(f"  baselines: neutral {neu:+.3f}  none {none_v:+.3f}")
    print(f"  vs-neutral: pos {pe:+.3f}  neg {ne:+.3f}  |neg|/|pos| = {text_ratio:.2f}")
    print(f"  RAW (vs 0): pos {pr:+.3f}  neg {nr:+.3f}  |neg|/|pos| = {raw_ratio:.2f}  "
          f"(robust if neutral is floored)")
    # Compare the image-conditioned ratio to the RAW text-only ratio (vs-neutral is unreliable when
    # the model floors its no-information baseline, as Qwen does).
    ref = raw_ratio if np.isfinite(raw_ratio) else text_ratio
    if img_ratio is not None and np.isfinite(ref):
        verdict = ("CROSS-MODAL amplification (image inflates the ratio)" if img_ratio > 1.25 * ref
                   else "STIMULUS confound (ratios match)" if abs(img_ratio - ref) <= 0.25 * ref
                   else "image dampens (reversed)")
        print(f"  image-conditioned |neg|/|pos| = {img_ratio:.2f}  vs raw text-only {ref:.2f}  →  {verdict}")
    else:
        print("  (run the base pass first to auto-compare against the image-conditioned ratio)")
    print(f"  data -> {STAGE_F_DIR/'text_only_qwen.parquet'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F multi-model — Qwen-VL (behavioral valence)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Qwen-VL hub id (Qwen3-VL or Qwen2.5-VL)")
    ap.add_argument("--text-only", action="store_true", help="run the image-ablated context control")
    ap.add_argument("--limit", type=int, default=None, help="EMOTIC image count (base pass)")
    args = ap.parse_args()
    if args.text_only:
        run_text_only(args.config, args.model)
    else:
        run_base(args.config, args.model, limit_override=args.limit)


if __name__ == "__main__":
    main()
