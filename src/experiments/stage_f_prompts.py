"""Stage F — prompt-robustness sweep (Gemma-3-4B).

Answers the top reviewer objection on a behavioral VLM result: *did you cherry-pick the prompt?*
We hold the effect's SCORING fixed (the calibration-free argmax-emotion override rate, shared with
the Qwen port via `analyze_stage_f._flip_override`) and vary only the PROMPT TEXT — question wording,
context placement, and the `Context:` label — across a small bank of natural rephrasings. A robust
result is one where the negativity-dominance override gap (neg-ctx overrides a positive image MINUS
pos-ctx overrides a negative image) stays positive and clears 0 for EVERY variant, not just the one
we happened to write first.

Design notes (kept comparable per analysis-rules):
- The turn scaffold (`<start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n`) is IDENTICAL across
  variants — that is the read-out contract and the mechanism carrier; only the user-turn CONTENT
  changes. So every variant is scored at the same last-token position with the same 13-emotion vocab.
- We keep the SAME image bank and the SAME context bank across variants, so the only thing that moves
  between variants is the prompt phrasing. Units are otherwise identical → variants share one table.
- Primary metric per variant = `_flip_override` (calibration-free, clustered bootstrap CI). We do NOT
  re-derive the text-only confound control per variant — that was established once in the base run
  (banks symmetric at |neg|/|pos| ~ 1.0); here we test phrasing sensitivity of the override, holding
  the confound story fixed.

GPU base pass runs on Colab (HF_TOKEN + EMOTIC). `--reanalyze` recomputes per-variant override from
the saved parquet on CPU (no model), mirroring `stage_f_qwen.py`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE)
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_a_steering import emotion_logprobs, emotion_token_ids, valence_score
from .stage_f_conflict import STAGE_A_DIR_probes, _probe_and_logits, select_extreme_images


# --------------------------------------------------------------------------- prompt variants
# Each builder takes the context sentence (or None) and returns ONLY the user-turn CONTENT that goes
# between `<start_of_image>` and `<end_of_turn>`. The scaffold is added once in `_wrap` so it is
# byte-identical across variants (mechanism carrier held fixed; only phrasing/placement varies).
def _labelled(ctx: str | None, question: str) -> str:
    return (f"Context: {ctx} {question}" if ctx else question)


def _ctx_after(ctx: str | None, question: str) -> str:
    return (f"{question} Context: {ctx}" if ctx else question)


def _unlabelled(ctx: str | None, question: str) -> str:
    return (f"{ctx} {question}" if ctx else question)


# variant_name -> (context-framing fn, question string). v0 reproduces the base run exactly.
PROMPT_VARIANTS: dict[str, tuple] = {
    "v0_original":   (_labelled,   "What single emotion is this person feeling?"),
    "v1_howfeel":    (_labelled,   "How is this person feeling?"),
    "v2_oneword":    (_labelled,   "In one word, what emotion is this person experiencing?"),
    "v3_mostlikely": (_labelled,   "What is the most likely emotion of the person in this photo?"),
    "v4_ctx_after":  (_ctx_after,  "What single emotion is this person feeling?"),
    "v5_no_label":   (_unlabelled, "What single emotion is this person feeling?"),
}


def _wrap(content: str) -> str:
    """Wrap user-turn content in the fixed Gemma image+turn scaffold (identical across variants)."""
    return (f"<start_of_turn>user\n<start_of_image>{content}<end_of_turn>\n"
            f"<start_of_turn>model\n")


def variant_prompt(variant: str, ctx: str | None) -> str:
    frame, question = PROMPT_VARIANTS[variant]
    return _wrap(frame(ctx, question))


# --------------------------------------------------------------------------- base pass
def run_base(config_path: str, limit_override: int | None = None,
             variants: list[str] | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("critical_layer", 18))
    tap = cfg.get("tap", "hook_attn_out")
    seed = int(cfg.get("seed", 0))
    n_images = limit_override or int(cfg.get("n_images", 40))
    variants = variants or list(PROMPT_VARIANTS)

    probes = load_probes(STAGE_A_DIR_probes())
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    # Full context bank on both polarities + neutral — the override metric needs (positive image ×
    # negative ctx) and (negative image × positive ctx); we run the whole bank so no single sentence
    # can drive a variant's result (guards the single-context pitfall from the pilot).
    conditions = [("none", "none", None)]
    conditions += [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
    conditions += [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
    conditions += [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]

    sel = select_extreme_images(cfg.get("split", "test"), n_images)
    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    multi = {w: r for w, r in verify_label_tokenization(bridge.tokenizer).items() if not r["single_token"]}
    name = f"blocks.{layer}.{tap}"

    rows, n_skip = [], 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f prompts"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for variant in variants:
            for cond, ctx_id, sentence in conditions:
                inputs = build_image_inputs(bridge, img, prompt=variant_prompt(variant, sentence))
                probe, val, lp = _probe_and_logits(
                    bridge, inputs["input_ids"], inputs["pixel_values"], name, coef, inter, tok_ids)
                rows.append({"prompt_variant": variant, "image_path": r["image_path"],
                             "image_valence": float(r["valence"]), "image_group": r["image_group"],
                             "condition": cond, "context_id": ctx_id, "context": sentence or "",
                             "text_code": TEXT_CODE[cond], "probe_readout": probe, "valence": val,
                             **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "conflict_prompts.parquet")
    metrics = {
        "run": run_stamp(), "git": git_hash(), "layer": layer, "tap": tap, "seed": seed,
        "n_images": int(sel.shape[0] - n_skip), "n_skipped": n_skip, "n_forwards": int(len(rows)),
        "variants": {v: {"frame": PROMPT_VARIANTS[v][0].__name__, "question": PROMPT_VARIANTS[v][1]}
                     for v in variants},
        "n_conditions": len(conditions), "text_code": TEXT_CODE, "tokenization_multi_token": multi,
    }
    save_json(metrics, STAGE_F_DIR / "conflict_prompts_metrics.json")
    print(f"\nStage F prompt-robustness — {metrics['n_images']} images x {len(variants)} variants x "
          f"{len(conditions)} conditions = {len(rows)} forwards ({n_skip} skipped).")
    if multi:
        print(f"  WARNING multi-token labels (first sub-token scored): {list(multi)}")
    print(f"  data -> {STAGE_F_DIR/'conflict_prompts.parquet'}")
    print("  NEXT: python -m src.experiments.stage_f_prompts --reanalyze")
    return metrics


# --------------------------------------------------------------------------- analysis (CPU)
def analyze(df: pd.DataFrame) -> dict:
    """Per-variant calibration-free override gap + clustered CI, plus a stability summary.

    Reuses the SHARED `_flip_override` so these numbers are directly comparable to the base Gemma run
    and the Qwen port. Robustness verdict = every variant's dominance gap CI clears 0.
    """
    from .analyze_stage_f import _flip_override
    per_variant = {}
    for variant, g in df.groupby("prompt_variant"):
        per_variant[str(variant)] = _flip_override(g)

    gaps = {v: m["dominance_gap"] for v, m in per_variant.items() if m}
    lo = {v: m["dominance_gap_ci95"][0] for v, m in per_variant.items() if m}
    all_clear = bool(gaps) and all(lo[v] > 0 for v in gaps)
    summary = {
        "n_variants": len(gaps),
        "gap_min": float(min(gaps.values())) if gaps else None,
        "gap_max": float(max(gaps.values())) if gaps else None,
        "gap_mean": float(np.mean(list(gaps.values()))) if gaps else None,
        "all_variants_clear_zero": all_clear,
        "min_gap_variant": min(gaps, key=gaps.get) if gaps else None,
    }
    return {"per_variant": per_variant, "summary": summary}


def reanalyze(config_path: str) -> dict:
    load_config(config_path)
    pq = STAGE_F_DIR / "conflict_prompts.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run the base pass on the A100 first.")
    df = pd.read_parquet(pq)
    out = analyze(df)
    save_json(out, STAGE_F_DIR / "conflict_prompts_analysis.json")

    print("\nStage F prompt-robustness — per-variant override (argmax-emotion category):")
    print(f"  {'variant':<14} {'neg>pos_img':>11} {'pos>neg_img':>11} {'gap':>7}  {'95% CI':>16}")
    for v, m in sorted(out["per_variant"].items()):
        if not m:
            print(f"  {v:<14} (no override cells)")
            continue
        ci = m["dominance_gap_ci95"]
        print(f"  {v:<14} {m['neg_ctx_overrides_pos_img']:>10.0%} {m['pos_ctx_overrides_neg_img']:>11.0%} "
              f"{m['dominance_gap']:>+7.0%}  [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    s = out["summary"]
    verdict = "ROBUST — every variant clears 0" if s["all_variants_clear_zero"] else \
              "NOT robust — some variant's CI includes 0"
    print(f"\n  dominance gap across {s['n_variants']} variants: "
          f"min {s['gap_min']:+.0%} (worst: {s['min_gap_variant']}), "
          f"mean {s['gap_mean']:+.0%}, max {s['gap_max']:+.0%}")
    print(f"  VERDICT: {verdict}")
    print(f"  analysis -> {STAGE_F_DIR/'conflict_prompts_analysis.json'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — prompt-robustness sweep (Gemma)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute per-variant override from the saved parquet (CPU, no model)")
    ap.add_argument("--limit", type=int, default=None, help="EMOTIC image count")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of prompt variant names (default: all)")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze(args.config)
    else:
        run_base(args.config, limit_override=args.limit, variants=args.variants)


if __name__ == "__main__":
    main()
