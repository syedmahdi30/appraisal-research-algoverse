"""Stage E, step 1 — build all six text-derived Δμ appraisal directions (pilot-plan-stage-e-f.md).

Extends the Stage A v2 / Stage D difference-of-means recipe from the valence pair to ALL SIX
appraisals, at resid_post L18, from the same 1,200 crowd-enVENT train examples Stage D used.
These are the frozen, text-learned handles that stage_e_combo.py combines into appraisal-theory
emotion arms. Nothing here touches an image — the cross-modal claim stays text -> image.

Saves results/stage_e/directions.npz:
  appraisals : (6,)   ordered appraisal names (row order of every matrix below)
  dmu        : (6, d) Δμ_a = mean(act|rating>=4) - mean(act|rating<=2) at resid_post L18
  norms      : (6,)   ‖Δμ_a‖
  cos        : (6, 6) pairwise cosine of the UNIT directions
  layer      : ()     the injection layer (18)

Also prints the cosine matrix and flags any |cos| > cos_flag_threshold — flagged pairs use the
Gram-Schmidt fallback when combined in stage_e_combo.py (E3). Text-only; runs anywhere with the
model + crowd-enVENT (no EMOTIC needed).
"""
from __future__ import annotations

import argparse

import numpy as np

from ..bridge.boot import boot_gemma
from ..data.crowd_envent import load_split, sample_tak_subset
from ..paths import STAGE_E_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_a_steering_v2 import diff_of_means, extract_resid


def build_directions(bridge, appraisals, layer, n_dir, seed):
    """Return (dmu [k, d], norms [k], cos [k, k]) for the appraisals at `layer` resid_post."""
    dtr = sample_tak_subset(load_split("train", seed=seed), seed=seed).head(n_dir)
    acts = extract_resid(bridge, dtr["text"].tolist(), [layer], "build-dirs")[layer]
    rows = []
    for a in appraisals:
        v = diff_of_means(acts, dtr[a].to_numpy())
        if v is None:
            raise ValueError(f"not enough high/low examples for {a} — widen n_dir")
        rows.append(v.astype(np.float32))
    dmu = np.stack(rows)
    norms = np.linalg.norm(dmu, axis=1)
    units = dmu / norms[:, None]
    cos = units @ units.T
    return dmu, norms, cos, len(dtr)


def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("steering_layers", [18])[0])
    appraisals = list(cfg.get("appraisals", [
        "pleasantness", "unpleasantness", "suddenness",
        "predict_event", "self_responsblt", "other_responsblt",
    ]))
    n_dir = limit_override or int(cfg.get("n_dir", 1200))
    seed = int(cfg.get("seed", 0))
    thr = float(cfg.get("cos_flag_threshold", 0.6))

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    dmu, norms, cos, n_used = build_directions(bridge, appraisals, layer, n_dir, seed)

    # off-diagonal entangled pairs (|cos| > threshold) -> Gram-Schmidt fallback in E3.
    flagged = [[appraisals[i], appraisals[j], float(cos[i, j])]
               for i in range(len(appraisals)) for j in range(i + 1, len(appraisals))
               if abs(cos[i, j]) > thr]

    np.savez(
        STAGE_E_DIR / "directions.npz",
        appraisals=np.array(appraisals), dmu=dmu, norms=norms, cos=cos, layer=layer,
    )
    metrics = {
        "run": run_stamp(), "git": git_hash(), "layer": layer, "n_dir": n_used,
        "seed": seed, "appraisals": appraisals,
        "norms": {a: float(n) for a, n in zip(appraisals, norms)},
        "cos_matrix": cos.tolist(), "cos_flag_threshold": thr,
        "entangled_pairs": flagged,
    }
    save_json(metrics, STAGE_E_DIR / "directions_metrics.json")

    print(f"\nStage E directions — resid_post L{layer}, {n_used} train examples.\n")
    hdr = "".join(f"{a[:8]:>9s}" for a in appraisals)
    print(f"{'':>10s}{hdr}")
    for i, a in enumerate(appraisals):
        print(f"{a[:10]:>10s}" + "".join(f"{cos[i, j]:+9.2f}" for j in range(len(appraisals))))
    if flagged:
        print("\n  FLAGGED |cos| > "
              f"{thr}: " + "; ".join(f"{p[0]}~{p[1]} ({p[2]:+.2f})" for p in flagged)
              + "  -> Gram-Schmidt fallback in stage_e_combo")
    else:
        print(f"\n  No pair exceeds |cos| = {thr}; all combos use the default (sum-of-units) rule.")
    print(f"\n  directions -> {STAGE_E_DIR/'directions.npz'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage E step 1 — build six Δμ appraisal directions")
    ap.add_argument("--config", default="config/stage_e.yaml")
    ap.add_argument("--limit", type=int, default=None, help="shrink direction-build (dry run)")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit)


if __name__ == "__main__":
    main()
