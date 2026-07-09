"""Analyze Stage A results (docs/experiment-1.md, analysis-rules.md).

Reads results/stage_a/metrics.json (no model / re-run needed) and produces:
  - a probe-vs-baseline table per appraisal at its best MHSA layer
  - a localization verdict (mid-layer peak AND val_r2 clearly above shuffled baseline)
  - layer-wise figures (val_r2 vs shuffled baseline) -> results/figures/

Every reported r2 is shown next to its shuffled-label baseline, as the rules require.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # headless (Colab / no display)
import matplotlib.pyplot as plt

from ..data import APPRAISAL_DISPLAY
from ..paths import FIGURES_DIR, STAGE_A_DIR

TAP = "hook_attn_out"  # MHSA — the tap the localization hypothesis is about
MID_LAYER_BAND = (10, 24)  # "mid-network" for a 34-layer model


def _int_keyed(d: dict) -> dict[int, float]:
    """metrics.json stores layer keys as strings; return {int_layer: value}."""
    return {int(k): float(v) for k, v in d.items()}


def load_metrics() -> dict:
    path = STAGE_A_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run Stage A first.")
    with open(path) as f:
        return json.load(f)


def summarize(metrics: dict) -> list[dict]:
    """Per appraisal: best MHSA layer, its val_r2, the baseline there, and the gap."""
    rows = []
    for appraisal, taps in metrics["best_layer"].items():
        layer = taps[TAP]["layer"]
        r2 = taps[TAP]["val_r2"]
        base = _int_keyed(metrics["shuffled_baseline_val_r2"][TAP][appraisal])[layer]
        rows.append({
            "appraisal": appraisal,
            "display": APPRAISAL_DISPLAY.get(appraisal, appraisal),
            "layer": layer,
            "val_r2": r2,
            "baseline_r2": base,
            "delta": r2 - base,
            "mid_layer": MID_LAYER_BAND[0] <= layer <= MID_LAYER_BAND[1],
        })
    return sorted(rows, key=lambda r: -r["val_r2"])


def verdict(rows: list[dict]) -> str:
    """Localization gate: peaks mid-network AND clearly beat the shuffled baseline."""
    strong = [r for r in rows if r["mid_layer"] and r["delta"] > 0.10]
    if len(strong) >= 4:
        return (f"SUPPORTS localization: {len(strong)}/{len(rows)} appraisals peak mid-network "
                f"(layers {MID_LAYER_BAND[0]}-{MID_LAYER_BAND[1]}) and beat the shuffled baseline "
                f"by >0.10 r2. This is the Tak-style mid-layer, MHSA-dominant signature.")
    if len(strong) >= 1:
        return (f"MIXED: only {len(strong)}/{len(rows)} appraisals show a clear mid-layer effect "
                f"above baseline. Report per-appraisal; do not generalize.")
    return "FAILS to support localization: no appraisal clearly beats the baseline mid-network."


def plot(metrics: dict, rows: list[dict]) -> str:
    """Grid of val_r2 vs shuffled baseline across layers, per appraisal (MHSA tap)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, r in zip(axes.flat, rows):
        a = r["appraisal"]
        lr2 = _int_keyed(metrics["layerwise_val_r2"][TAP][a])
        base = _int_keyed(metrics["shuffled_baseline_val_r2"][TAP][a])
        layers = sorted(lr2)
        ax.plot(layers, [lr2[l] for l in layers], "-o", ms=3, label="probe")
        ax.plot(layers, [base[l] for l in layers], "--", color="gray", label="shuffled baseline")
        ax.axvline(r["layer"], color="red", lw=1, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{r['display']}  (best L{r['layer']}, r²={r['val_r2']:.2f})", fontsize=10)
        ax.set_xlabel("layer"); ax.set_ylabel("val r²")
        ax.legend(fontsize=7)
    fig.suptitle("Stage A — appraisal probe r² vs layer (MHSA output, val split)", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "stage_a_localization.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def latex_table(rows: list[dict]) -> str:
    """LaTeX table of the localization result (probe r2 next to shuffled baseline)."""
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Appraisal & Layer & val $r^2$ & shuffled & $\Delta$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['display']} & {r['layer']} & {r['val_r2']:.3f} & "
                     f"{r['baseline_r2']:.3f} & {r['delta']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    metrics = load_metrics()
    rows = summarize(metrics)

    print(f"Stage A analysis  (n_train={metrics['n_train']} n_val={metrics['n_val']}, "
          f"tap={TAP}, critical_layer={metrics.get('critical_layer')})\n")
    print(f"{'appraisal':22s} {'layer':>5s} {'val_r2':>7s} {'baseline':>9s} {'Δ':>7s}")
    for r in rows:
        print(f"{r['display']:22s} {r['layer']:5d} {r['val_r2']:7.3f} "
              f"{r['baseline_r2']:9.3f} {r['delta']:7.3f}")

    fig = plot(metrics, rows)
    print(f"\nfigure -> {fig}")
    print(f"\nVERDICT: {verdict(rows)}")

    # persist a compact, paper-friendly summary + a reproducible LaTeX table
    out = STAGE_A_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump({"rows": rows, "verdict": verdict(rows)}, f, indent=2)
    tex = STAGE_A_DIR / "localization_table.tex"
    tex.write_text(latex_table(rows))
    print(f"summary -> {out}\nlatex   -> {tex}")


if __name__ == "__main__":
    main()
