"""Label-balance sensitivity: does the closed vocabulary's polarity imbalance drive the result?

CPU-only. The fixed 13-label list is 4 positive against 7 negative, so every readout in the
paper sits on an asymmetric partition: the argmax has nearly twice as many ways to land on a
negative label, and P(neg) sums seven terms against four. This re-scores the reported contrasts
on a BALANCED vocabulary, all 35 four-negative subsets, renormalizing within the surviving eight
labels, and reports the distribution.

    python -m src.experiments.analyze_label_balance
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from ..paths import STAGE_F_DIR
from .common import git_hash, run_stamp, save_json
from .shared import reporting as R
from .shared.reporting import NEGATIVE_LABELS, POSITIVE_LABELS, flip_override, minimal_pair_asymmetry

MINIMAL_PQ = "conflict_qwen3-vl-8b-instruct_minimal.parquet"
VARIED_PQ = "conflict_qwen.parquet"


def _balanced_valence(df: pd.DataFrame, negatives: tuple[str, ...]) -> pd.DataFrame:
    """Re-score bounded valence over {4 positives} u {4 chosen negatives}, renormalized."""
    pos = np.exp(df[[f"lp_{w}" for w in POSITIVE_LABELS]].to_numpy()).sum(axis=1)
    neg = np.exp(df[[f"lp_{w}" for w in negatives]].to_numpy()).sum(axis=1)
    out = df.copy()
    out["valence"] = (pos - neg) / (pos + neg)
    return out


def run() -> dict:
    subsets = list(itertools.combinations(NEGATIVE_LABELS, 4))

    minimal = pd.read_parquet(STAGE_F_DIR / MINIMAL_PQ)
    full_contrast = minimal_pair_asymmetry(minimal)
    graded = [minimal_pair_asymmetry(_balanced_valence(minimal, s)) for s in subsets]
    g_val = np.array([r["paired_asymmetry"] for r in graded])
    g_lo = np.array([r["ci95"][0] for r in graded])

    # The argmax space itself must shrink for the categorical readout, so patch the label tuple.
    varied = pd.read_parquet(STAGE_F_DIR / VARIED_PQ)
    full_gap = flip_override(varied)
    original = R.EMOTION_LABELS
    cat = []
    try:
        for s in subsets:
            R.EMOTION_LABELS = tuple(POSITIVE_LABELS) + s
            r = flip_override(varied)
            cat.append((float(r["dominance_gap"]), float(r["dominance_gap_ci95"][0])))
    finally:
        R.EMOTION_LABELS = original
    c_val = np.array([x[0] for x in cat])
    c_lo = np.array([x[1] for x in cat])

    metrics = {
        "run": run_stamp(), "git": git_hash(), "n_subsets": len(subsets),
        "within_item_contrast": {
            "full_vocabulary": full_contrast["paired_asymmetry"],
            "balanced_min": float(g_val.min()), "balanced_median": float(np.median(g_val)),
            "balanced_max": float(g_val.max()),
            "n_positive": int((g_val > 0).sum()), "n_ci_excludes_zero": int((g_lo > 0).sum()),
        },
        "uncorrected_override_gap": {
            "full_vocabulary": float(full_gap["dominance_gap"]),
            "balanced_min": float(c_val.min()), "balanced_median": float(np.median(c_val)),
            "balanced_max": float(c_val.max()),
            "n_positive": int((c_val > 0).sum()), "n_ci_excludes_zero": int((c_lo > 0).sum()),
        },
    }
    save_json(metrics, STAGE_F_DIR / "label_balance_sensitivity.json")

    w = metrics["within_item_contrast"]; o = metrics["uncorrected_override_gap"]
    print(f"\nLabel-balance sensitivity ({len(subsets)} balanced 4-vs-4 vocabularies)\n")
    print(f"  within-item contrast   full {w['full_vocabulary']:+.3f} | balanced median "
          f"{w['balanced_median']:+.3f}  range [{w['balanced_min']:+.3f}, {w['balanced_max']:+.3f}]")
    print(f"                         positive {w['n_positive']}/{len(subsets)}, "
          f"CI excludes zero {w['n_ci_excludes_zero']}/{len(subsets)}")
    print(f"  override gap (uncorr.) full {o['full_vocabulary']:+.3f} | balanced median "
          f"{o['balanced_median']:+.3f}  range [{o['balanced_min']:+.3f}, {o['balanced_max']:+.3f}]")
    print(f"                         positive {o['n_positive']}/{len(subsets)}, "
          f"CI excludes zero {o['n_ci_excludes_zero']}/{len(subsets)}")
    print(f"\n  summary -> {STAGE_F_DIR/'label_balance_sensitivity.json'}")
    return metrics


if __name__ == "__main__":
    run()
