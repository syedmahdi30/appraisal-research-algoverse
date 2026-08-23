"""Bootstrap intervals for the patching tables (CPU, reads saved parquets/metrics only).

The GPU runners report recovery as a ratio of sums, sum(patched - neg) / sum(pos - neg), which is
the right estimator here: per-row ratios are unstable because an individual row's denominator
(pos - neg) can sit near zero. This module bootstraps that same estimator over the resampling unit
the rest of the paper uses -- the image -- so the two patching tables can carry intervals like every
other table.

Cross-image recovery already ships with `val_ci95` / `probe_ci95` in the per-band metrics files; the
runner computed them and they were simply never printed. This module surfaces those and recomputes
the one band whose per-row parquet survived, as a cross-check that the two agree.

Usage:
  python -m src.experiments.patching_intervals
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..paths import STAGE_F_DIR

N_BOOT, SEED = 2000, 0   # the paper's bootstrap settings elsewhere

SAME_IMAGE = [("Pair 1 (championship/funeral)", "patching_hf_pair1.parquet"),
              ("Pair 2 (wonderful/devastating)", "patching_hf.parquet")]
SAME_GROUPS = ["question", "suffix_delim", "text_all", "image", "bos", "prefix_delim"]
BANDS = [("0--12 (early)", "cross_patching_hf_0-12.json"),
         ("13--17 (mid)", "cross_patching_hf_13-17.json"),
         ("18--28 (late)", "cross_patching_hf_18-28.json")]


def _ratio_of_sums(num: np.ndarray, den: np.ndarray) -> float:
    return float(num.sum() / den.sum())


def bootstrap_recovery(df: pd.DataFrame, group: str, readout: str = "probe") -> dict:
    """Recovery + percentile interval, resampling images with replacement."""
    den = (df[f"pos_{readout}"] - df[f"neg_{readout}"]).to_numpy(float)
    num = (df[f"patch_{group}_{readout}"] - df[f"neg_{readout}"]).to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        i = rng.integers(0, len(den), len(den))
        boot[k] = _ratio_of_sums(num[i], den[i])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"recovery": _ratio_of_sums(num, den), "ci95": [float(lo), float(hi)], "n": int(len(den))}


def same_image() -> dict:
    out = {}
    for label, fname in SAME_IMAGE:
        df = pd.read_parquet(STAGE_F_DIR / fname)
        out[label] = {g: bootstrap_recovery(df, g) for g in SAME_GROUPS}
    return out


def cross_image() -> dict:
    """Read the intervals the runner already computed, per layer band."""
    out = {}
    for label, fname in BANDS:
        p = STAGE_F_DIR / fname
        if not p.exists():
            continue
        r = json.loads(p.read_text())["recovery"]
        out[label] = {g: {"recovery": r[g]["val"], "ci95": r[g]["val_ci95"]}
                      for g in ("image", "text_all", "all") if g in r}
    return out


def _fmt(d: dict) -> str:
    return f"{100*d['recovery']:6.1f}%  [{100*d['ci95'][0]:+6.1f}, {100*d['ci95'][1]:+6.1f}]"


def main() -> None:
    si = same_image()
    print(f"\nSame-image patching (probe readout, layers 13--17), {N_BOOT} resamples over images:")
    for label, groups in si.items():
        print(f"  {label}")
        for g, d in groups.items():
            print(f"    {g:14s} {_fmt(d)}   n={d['n']}")

    ci = cross_image()
    print("\nCross-image patching (behavioral valence), intervals as computed by the runner:")
    for label, groups in ci.items():
        print(f"  {label}")
        for g, d in groups.items():
            print(f"    {g:14s} {_fmt(d)}")

    # cross-check: the 18--28 band is the one whose per-row parquet survived
    pq = STAGE_F_DIR / "cross_patching_hf.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
        print("\nCross-check, 18--28 band recomputed from its surviving parquet:")
        for g in ("image", "text_all", "all"):
            print(f"    {g:14s} {_fmt(bootstrap_recovery(df, g, readout='val'))}")

    json.dump({"same_image": si, "cross_image": ci, "n_boot": N_BOOT, "seed": SEED},
              open(STAGE_F_DIR / "patching_intervals.json", "w"), indent=1)
    print(f"\n  data -> {STAGE_F_DIR/'patching_intervals.json'}")


if __name__ == "__main__":
    main()
