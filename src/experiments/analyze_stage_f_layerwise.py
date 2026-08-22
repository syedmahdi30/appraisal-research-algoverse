"""Stage F — scale-free re-localization from the layerwise parquet (CPU, no GPU rerun).

The raw `stage_f_layerwise` projections grow with residual-stream norm (±3 at L13 → ±57 at L33), so
raw neg−pos separations and the "attn-write peak" are inflated by depth, not signal. This reads the
per-image projections in `layerwise.parquet` and reports a SCALE-FREE paired effect size per layer:
    d(L) = mean_img(neg − pos) / std_img(neg − pos)     (paired over images; contexts averaged per polarity)
The onset (first layer reaching 50% of peak |d|) is robust; the effect-size PEAK says whether the
divergence genuinely amplifies late or merely rides the growing norm.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..paths import STAGE_F_DIR, ensure_dirs
from .common import run_stamp, save_json


def _paired_d(piv: pd.DataFrame) -> tuple[float, float, int]:
    """Paired effect size of (negative − positive) over images at one layer; contexts averaged."""
    g = piv.groupby(["image_path", "condition"])["resid_proj"].mean().unstack()
    if "negative" not in g or "positive" not in g:
        return float("nan"), float("nan"), 0
    both = g[["negative", "positive"]].dropna()
    diff = (both["negative"] - both["positive"]).to_numpy()
    if len(diff) < 2:
        return float("nan"), float("nan"), len(diff)
    sd = diff.std(ddof=1)
    return float(diff.mean()), float(diff.mean() / sd) if sd > 0 else float("nan"), len(diff)


def run(config_path: str | None = None, parquet: str | None = None) -> dict:
    ensure_dirs()
    # A custom parquet writes a stem-matched *_normalized.json so a raw-HF re-score never clobbers
    # the original analysis — the same per-run-path discipline the runners use.
    pq = STAGE_F_DIR / (parquet or "layerwise.parquet")
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run stage_f_layerwise first.")
    df = pd.read_parquet(pq)
    layers = sorted(int(x) for x in df["layer"].unique())
    crit = 18  # Stage A critical / read-out layer (sign anchor for the divergence)

    raw_sep, eff_d, ns = {}, {}, {}
    for L in layers:
        m, d, n = _paired_d(df[df["layer"] == L])
        raw_sep[L], eff_d[L], ns[L] = m, d, n

    dvals = np.array([eff_d[L] for L in layers], dtype=float)
    absraw = np.abs(np.array([raw_sep[L] for L in layers], dtype=float))
    # Early layers can have a large effect size from near-ZERO variance (a tiny mean over a tiny std),
    # which is a degenerate artifact, not signal — the L0 |d| blow-up. Gate candidate layers on BOTH
    # (a) a raw-separation noise floor (3x the median |raw| over the first 8, pre-mixing, layers) and
    # (b) sign-consistency with the read-out band (mean d over layers >= critical), so a wrong-sign
    # low-variance spike cannot become the peak.
    floor = 3.0 * float(np.median(absraw[:8])) if len(absraw) >= 8 else 0.0
    late_sign = np.sign(np.nanmean([eff_d[L] for L in layers if L >= crit]) or -1.0)
    cand = [L for L in layers if absraw[layers.index(L)] >= floor
            and np.sign(dvals[layers.index(L)]) == late_sign]
    if not cand:
        cand = layers
    peak_layer = max(cand, key=lambda L: abs(eff_d[L]))
    peak_d = float(eff_d[peak_layer])
    entry_layer = min(cand)                                   # first layer clearing the noise floor
    d_entry, d18 = float(eff_d[entry_layer]), eff_d.get(18, float("nan"))
    amp = (abs(peak_d) / abs(d_entry)) if d_entry else float("nan")   # scale-free late amplification

    band = ("early" if entry_layer < len(layers) // 3 else
            "mid" if entry_layer < 2 * len(layers) // 3 else "late")
    summary = (f"ENTRY (first layer clearing the raw noise floor {floor:.3f}, read-out sign) at layer "
               f"{entry_layer} (d={d_entry:+.2f}); effect-size PEAK |d|={abs(peak_d):.2f} at layer "
               f"{peak_layer}; |d| at read-out L18 = {abs(d18):.2f}. The divergence ENTERS in the "
               f"{band} band (~L{entry_layer}) and {'AMPLIFIES through the late layers' if amp > 1.5 else 'stays roughly flat'} "
               f"(peak/entry = {amp:.1f}× effect size, scale-free). Probe patch target = L{entry_layer}; "
               f"behavioral-valence carrier extends through ~L{peak_layer} (patch band must reach it).")

    metrics = {
        "run": run_stamp(), "n_layers": len(layers), "n_images_paired": ns.get(peak_layer),
        "effect_size_d": {int(L): eff_d[L] for L in layers},
        "raw_mean_sep": {int(L): raw_sep[L] for L in layers},
        "noise_floor": floor, "read_out_sign": float(late_sign),
        "entry_layer": int(entry_layer), "peak_layer": int(peak_layer), "peak_d": peak_d,
        "d_at_entry": float(d_entry), "d_at_L18": float(d18) if not np.isnan(d18) else None,
        "peak_over_entry": float(amp) if not np.isnan(amp) else None, "summary": summary,
    }
    onset_layer = entry_layer  # back-compat name used in the print loop
    save_json(metrics, STAGE_F_DIR / f"{pq.stem}_normalized.json")

    print(f"\nStage F layerwise (scale-free) — {len(layers)} layers, paired over "
          f"{ns.get(peak_layer)} images.\n")
    print(f"  {'L':>3s} {'raw neg-pos':>11s} {'effect d':>9s}")
    for L in layers:
        mark = "  <- onset" if L == onset_layer else "  <- peak |d|" if L == peak_layer else ""
        print(f"  {L:>3d} {raw_sep[L]:>+11.3f} {eff_d[L]:>+9.2f}{mark}")
    print(f"\n  {summary}")
    print(f"  metrics -> {STAGE_F_DIR / f'{pq.stem}_normalized.json'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — scale-free layerwise re-localization (CPU)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--parquet", default=None,
                    help="alternate layerwise parquet under results/stage_f (e.g. layerwise_hf.parquet)")
    args = ap.parse_args()
    run(args.config, parquet=args.parquet)


if __name__ == "__main__":
    main()
