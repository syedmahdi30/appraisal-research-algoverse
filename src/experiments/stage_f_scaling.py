"""Stage F — model-scale sweep of cross-modal negativity dominance (Gemma 3, 4B → 12B).

Answers the meeting's *scaling analysis* ask: does the negativity-dominance override gap hold — or
grow/shrink — as the Gemma-3 VLM scales up? We keep the stimuli, prompt, context bank, and metric
byte-identical across sizes and vary ONLY the model, so the sizes share one table (analysis-rules:
comparable units only).

Design:
- **Behavioral-only, no probe.** The frozen Stage A pleasantness probe is 4B-specific (its `d_model`
  and critical layer don't transfer to 12B), so — exactly like the Qwen port — we score with the
  calibration-free argmax-emotion OVERRIDE rate (`analyze_stage_f._flip_override`), which needs only
  the 13-emotion logits. This is the SAME metric the base Gemma-4B and Qwen numbers use, so the
  scaling curve is directly comparable to them.
- **Canonical prompt only** (the original `context_prompt`, i.e. `v0_original` from the prompt sweep) —
  scaling is a single-axis experiment; prompt variation is a separate control (`stage_f_prompts.py`).
- **One model per invocation** (`--model`) to avoid holding two VLMs in memory. Each run writes
  `conflict_scaling_<tag>.parquet`; `--reanalyze` globs them into the override-gap-vs-size curve.

GPU passes run on Colab (HF_TOKEN + EMOTIC; 12B in bf16 ~24 GB, fits an A100). `--reanalyze` is CPU.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE, context_prompt)
from ..data.emotic import load_split as load_emotic_split
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import FIGURES_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_a_steering import emotion_logprobs, emotion_token_ids, valence_score
from .shared.sampling import select_extreme_rows

# Known Gemma-3 multimodal sizes → billions of parameters (x-axis). Unknown ids fall back to a regex.
GEMMA_PARAMS_B = {
    "google/gemma-3-4b-it": 4.3,
    "google/gemma-3-12b-it": 12.2,
    "google/gemma-3-27b-it": 27.4,
}


def model_tag(model_name: str) -> str:
    """Filesystem-safe short tag, e.g. google/gemma-3-12b-it -> gemma-3-12b-it."""
    return model_name.split("/")[-1]


def params_billions(model_name: str) -> float:
    """Parameter count in billions for the scaling x-axis (known map, else parse `<n>b`)."""
    if model_name in GEMMA_PARAMS_B:
        return GEMMA_PARAMS_B[model_name]
    m = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    return float(m.group(1)) if m else float("nan")


# --------------------------------------------------------------------------- base pass (per model)
def run_base(config_path: str, model_name: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    seed = int(cfg.get("seed", 0))
    n_images = limit_override or int(cfg.get("n_images", 40))

    # Full context bank on both polarities + neutral + no-context — the override metric needs
    # (positive image × negative ctx) and (negative image × positive ctx); the whole bank guards
    # against any single sentence driving a size's result.
    conditions = [("none", "none", None)]
    conditions += [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
    conditions += [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
    conditions += [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]

    frame = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    sel = select_extreme_rows(frame, n_images)
    bridge = boot_gemma(model_name, device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    multi = {w: r for w, r in verify_label_tokenization(bridge.tokenizer).items() if not r["single_token"]}

    rows, n_skip = [], 0
    for _, r in tqdm(list(sel.iterrows()), desc=f"scaling {model_tag(model_name)}"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for cond, ctx_id, sentence in conditions:
            inputs = build_image_inputs(bridge, img, prompt=context_prompt(sentence))
            with torch.no_grad():
                logits = bridge.run_with_hooks(
                    inputs["input_ids"], pixel_values=inputs["pixel_values"], fwd_hooks=[])
            last = logits[0, -1]
            val = valence_score(last, tok_ids)
            lp = emotion_logprobs(last, tok_ids)
            rows.append({"model": model_name, "image_path": r["image_path"],
                         "image_valence": float(r["valence"]), "image_group": r["image_group"],
                         "condition": cond, "context_id": ctx_id, "context": sentence or "",
                         "text_code": TEXT_CODE[cond], "valence": val,
                         **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    out_pq = STAGE_F_DIR / f"conflict_scaling_{model_tag(model_name)}.parquet"
    df.to_parquet(out_pq)
    metrics = {
        "run": run_stamp(), "git": git_hash(), "model": model_name,
        "params_billions": params_billions(model_name), "seed": seed,
        "n_images": int(sel.shape[0] - n_skip), "n_skipped": n_skip, "n_forwards": int(len(rows)),
        "n_conditions": len(conditions), "text_code": TEXT_CODE, "tokenization_multi_token": multi,
    }
    save_json(metrics, STAGE_F_DIR / f"conflict_scaling_{model_tag(model_name)}_metrics.json")
    print(f"\nStage F scaling [{model_name}] — {metrics['n_images']} images x "
          f"{len(conditions)} conditions = {len(rows)} forwards ({n_skip} skipped).")
    if multi:
        print(f"  WARNING multi-token labels (first sub-token scored): {list(multi)}")
    print(f"  data -> {out_pq}")
    print("  NEXT: run the other size(s), then: python -m src.experiments.stage_f_scaling --reanalyze")
    return metrics


# --------------------------------------------------------------------------- analysis (CPU)
def analyze() -> dict:
    """Glob every per-model scaling parquet, compute the shared override gap, order by param count."""
    from .analyze_stage_f import _flip_override
    pqs = sorted(STAGE_F_DIR.glob("conflict_scaling_*.parquet"))
    if not pqs:
        raise FileNotFoundError(
            f"no conflict_scaling_*.parquet in {STAGE_F_DIR} — run a base pass per model first.")
    per_model = []
    for pq in pqs:
        df = pd.read_parquet(pq)
        model = str(df["model"].iloc[0]) if "model" in df.columns else pq.stem
        flip = _flip_override(df)
        if not flip:
            continue
        per_model.append({"model": model, "params_billions": params_billions(model),
                          "n_images": int(df["image_path"].nunique()), **flip})
    per_model.sort(key=lambda d: d["params_billions"])
    return {"per_model": per_model,
            "trend": _trend(per_model),
            "n_models": len(per_model)}


def _trend(per_model: list[dict]) -> dict:
    """Scale trend, reported HONESTLY — the gap alone is misleading when a component saturates.

    The dominance GAP can narrow even while the negative-override effect STRENGTHENS, if positive
    override rises faster and/or negative override hits its ceiling. So we surface the component
    deltas (neg-override, pos-override) alongside the gap, flag a near-ceiling component, and check
    whether the smallest/largest gap CIs even separate before calling the gap change 'significant'.
    """
    if len(per_model) < 2:
        return {}
    lo, hi = per_model[0], per_model[-1]
    dgap = hi["dominance_gap"] - lo["dominance_gap"]
    dneg = hi["neg_ctx_overrides_pos_img"] - lo["neg_ctx_overrides_pos_img"]
    dpos = hi["pos_ctx_overrides_neg_img"] - lo["pos_ctx_overrides_neg_img"]
    lo_ci, hi_ci = lo["dominance_gap_ci95"], hi["dominance_gap_ci95"]
    gaps_overlap = not (lo_ci[0] > hi_ci[1] or hi_ci[0] > lo_ci[1])  # crude significance guard
    gap_dir = "grows" if dgap > 0.03 else "narrows" if dgap < -0.03 else "flat"
    return {"smallest": lo["model"], "largest": hi["model"],
            "gap_smallest": lo["dominance_gap"], "gap_largest": hi["dominance_gap"],
            "delta_gap": dgap, "delta_neg_override": dneg, "delta_pos_override": dpos,
            "gap_direction": gap_dir, "gap_cis_overlap": gaps_overlap,
            "neg_override_near_ceiling": max(m["neg_ctx_overrides_pos_img"] for m in per_model) > 0.9,
            "all_gaps_clear_zero": all(m["dominance_gap_ci95"][0] > 0 for m in per_model)}


def _plot(per_model: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [m["params_billions"] for m in per_model]
    gap = [m["dominance_gap"] for m in per_model]
    gap_err = [[m["dominance_gap"] - m["dominance_gap_ci95"][0] for m in per_model],
               [m["dominance_gap_ci95"][1] - m["dominance_gap"] for m in per_model]]
    neg = [m["neg_ctx_overrides_pos_img"] for m in per_model]
    pos = [m["pos_ctx_overrides_neg_img"] for m in per_model]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, neg, "o--", color="#c0392b", alpha=.7, label="neg ctx overrides positive image")
    ax.plot(x, pos, "s--", color="#2980b9", alpha=.7, label="pos ctx overrides negative image")
    ax.errorbar(x, gap, yerr=gap_err, fmt="D-", color="#111", capsize=4, lw=2,
                label="dominance gap (neg − pos)  [95% CI, clustered over images]")
    ax.axhline(0, color="grey", lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}B"))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())  # no stray 6×10^0 label
    ax.set_xlim(min(x) * 0.85, max(x) * 1.18)
    ax.set_xlabel("Gemma-3 model size (parameters, log scale)")
    ax.set_ylabel("override rate (argmax-emotion category)")
    ax.set_ylim(-0.05, 1.0)
    n = per_model[0]["n_images"] if per_model else 0
    ax.set_title(f"Cross-modal negativity dominance vs Gemma-3 model scale\n"
                 f"EMOTIC test extremes (n≈{n} images), full context bank, "
                 f"argmax-emotion override", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reanalyze() -> dict:
    ensure_dirs()
    out = analyze()
    save_json(out, STAGE_F_DIR / "conflict_scaling_analysis.json")

    print("\nStage F scaling — cross-modal negativity dominance vs Gemma-3 size:")
    print(f"  {'model':<22} {'params':>7} {'neg>pos_img':>11} {'pos>neg_img':>11} {'gap':>7}  {'95% CI':>16}")
    for m in out["per_model"]:
        ci = m["dominance_gap_ci95"]
        print(f"  {model_tag(m['model']):<22} {m['params_billions']:>6.1f}B "
              f"{m['neg_ctx_overrides_pos_img']:>10.0%} {m['pos_ctx_overrides_neg_img']:>11.0%} "
              f"{m['dominance_gap']:>+7.0%}  [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    t = out.get("trend", {})
    if t:
        print(f"\n  TREND (smallest → largest): dominance gap {t['gap_smallest']:+.0%} → "
              f"{t['gap_largest']:+.0%} (Δ {t['delta_gap']:+.0%}, gap {t['gap_direction']}"
              f"{'; CIs OVERLAP — Δgap not clearly significant' if t['gap_cis_overlap'] else ''}).")
        print(f"        components: neg-override Δ {t['delta_neg_override']:+.0%}, "
              f"pos-override Δ {t['delta_pos_override']:+.0%}"
              f"{'  [neg-override near ceiling >90%: gap compressed from above]' if t['neg_override_near_ceiling'] else ''}.")
        print(f"        dominance {'PERSISTS at every scale (all CIs clear 0)' if t['all_gaps_clear_zero'] else 'NOT robust — a size includes 0'}.")
    if len(out["per_model"]) >= 2:
        fig_path = FIGURES_DIR / "stage_f_scaling.png"
        _plot(out["per_model"], fig_path)
        print(f"  figure -> {fig_path}")
    else:
        print("  (need ≥2 sizes for the scaling figure — run another --model pass.)")
    print(f"  analysis -> {STAGE_F_DIR/'conflict_scaling_analysis.json'}")
    return out


DEFAULT_MODEL = "google/gemma-3-4b-it"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — Gemma-3 4B→12B scaling of negativity dominance")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemma-3 multimodal hub id (4b/12b/27b)")
    ap.add_argument("--reanalyze", action="store_true",
                    help="aggregate all saved per-model parquets into the scaling curve (CPU, no model)")
    ap.add_argument("--limit", type=int, default=None, help="EMOTIC image count")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze()
    else:
        run_base(args.config, args.model, limit_override=args.limit)


if __name__ == "__main__":
    main()
