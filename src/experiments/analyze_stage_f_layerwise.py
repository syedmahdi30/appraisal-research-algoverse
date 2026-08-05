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


def run(config_path: str | None = None) -> dict:
    ensure_dirs()
    pq = STAGE_F_DIR / "layerwise.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run stage_f_layerwise first.")
    df = pd.read_parquet(pq)
    layers = sorted(int(x) for x in df["layer"].unique())

    raw_sep, eff_d, ns = {}, {}, {}
    for L in layers:
        m, d, n = _paired_d(df[df["layer"] == L])
        raw_sep[L], eff_d[L], ns[L] = m, d, n

    dvals = np.array([eff_d[L] for L in layers], dtype=float)
    absd = np.abs(np.nan_to_num(dvals))
    peak_i = int(np.argmax(absd))
    peak_layer, peak_d = layers[peak_i], float(dvals[peak_i])
    thr = 0.5 * absd[peak_i]
    onset_layer = layers[peak_i]
    for i, L in enumerate(layers):
        if absd[i] >= thr and np.sign(dvals[i]) == np.sign(peak_d):
            onset_layer = L
            break
    # late amplification? compare |d| at the read-out layer (18) vs the peak
    d18 = eff_d.get(18, float("nan"))
    amp = (abs(peak_d) / abs(d18)) if d18 and not np.isnan(d18) and d18 != 0 else float("nan")

    band = ("early" if onset_layer < len(layers) // 3 else
            "mid" if onset_layer < 2 * len(layers) // 3 else "late")
    summary = (f"onset (|d|≥50% of peak) at layer {onset_layer}; effect-size PEAK |d|={abs(peak_d):.2f} "
               f"at layer {peak_layer}; |d| at read-out L18 = {abs(d18):.2f} "
               f"(peak/L18 = {amp:.2f}× ⇒ {'genuine late amplification' if amp > 1.3 else 'no real late amplification — raw growth was mostly residual-norm scale'}). "
               f"Context enters the read-out in the {band} band. Patch target = layer {onset_layer}.")

    metrics = {
        "run": run_stamp(), "n_layers": len(layers), "n_images_paired": ns.get(peak_layer),
        "effect_size_d": {int(L): eff_d[L] for L in layers},
        "raw_mean_sep": {int(L): raw_sep[L] for L in layers},
        "onset_layer": int(onset_layer), "peak_layer": int(peak_layer), "peak_d": peak_d,
        "d_at_L18": float(d18) if not np.isnan(d18) else None,
        "peak_over_L18": float(amp) if not np.isnan(amp) else None, "summary": summary,
    }
    save_json(metrics, STAGE_F_DIR / "layerwise_normalized.json")

    print(f"\nStage F layerwise (scale-free) — {len(layers)} layers, paired over "
          f"{ns.get(peak_layer)} images.\n")
    print(f"  {'L':>3s} {'raw neg-pos':>11s} {'effect d':>9s}")
    for L in layers:
        mark = "  <- onset" if L == onset_layer else "  <- peak |d|" if L == peak_layer else ""
        print(f"  {L:>3d} {raw_sep[L]:>+11.3f} {eff_d[L]:>+9.2f}{mark}")
    print(f"\n  {summary}")
    print(f"  metrics -> {STAGE_F_DIR/'layerwise_normalized.json'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — scale-free layerwise re-localization (CPU)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
