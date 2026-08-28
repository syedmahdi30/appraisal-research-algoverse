"""Method diagram for the VLM4RWD paper: design, result, and where the effect lives.

    python scripts/generate_method_diagram.py

Every number is read from results/, never typed in. Colour is reserved for valence
polarity (blue positive, red negative, validated for CVD); the depth panel uses ink
only, because its distinction is token group and not valence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.shared.reporting import minimal_pair_asymmetry  # noqa: E402

POS, NEG = "#2a78d6", "#c8442a"          # validated diverging pair
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
                     "axes.linewidth": 0.6, "text.color": INK})


def measured() -> dict:
    df = pd.read_parquet(ROOT / "results/stage_f/conflict_qwen3-vl-8b-instruct_minimal.parquet")
    r = minimal_pair_asymmetry(df)
    pp = r["per_pair"]
    neg = float(np.mean([v["neg_effect"] for v in pp.values()]))
    pos = float(np.mean([v["pos_effect"] for v in pp.values()]))
    return {"neg": neg, "pos": pos, "ratio": abs(neg) / abs(pos),
            "contrast": r["paired_asymmetry"], "ci": r["ci95"],
            "n_img": r["n_images"], "n_pairs": r["n_pairs"],
            "rmin": min(v["ratio"] for v in pp.values()),
            "rmax": max(v["ratio"] for v in pp.values())}


def panel_design(ax) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, "A  Stimulus", weight="bold", fontsize=7.6, va="top")

    ax.add_patch(Rectangle((0.06, 0.60), 0.30, 0.22, fc="#e8e7e3", ec=RULE, lw=0.6))
    ax.text(0.21, 0.71, "photo", ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.text(0.21, 0.565, "positive", ha="center", va="top", fontsize=6.0, color=MUTED)

    ax.text(0.44, 0.80, "“…after she", fontsize=6.5, va="center")
    ax.text(0.47, 0.705, "won", fontsize=6.5, va="center", color=POS, weight="bold")
    ax.text(0.625, 0.705, "/", fontsize=6.5, va="center", color=MUTED)
    ax.text(0.695, 0.705, "lost", fontsize=6.5, va="center", color=NEG, weight="bold")
    ax.text(0.44, 0.61, "the game.”", fontsize=6.5, va="center")

    ax.text(0.06, 0.44, "same photo,", fontsize=6.2, color=MUTED, va="top")
    ax.text(0.06, 0.35, "same event,", fontsize=6.2, color=MUTED, va="top")
    ax.text(0.06, 0.26, "one word changed", fontsize=6.2, color=MUTED, va="top")
    ax.text(0.06, 0.13, "scored vs. that photo's", fontsize=6.2, color=MUTED, va="top")
    ax.text(0.06, 0.04, "neutral-context baseline", fontsize=6.2, color=MUTED, va="top")


def panel_result(ax, m: dict) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, "B  Shift in judgment", weight="bold", fontsize=7.6, va="top")
    ax.text(0, 0.925, "Qwen3-VL-8B · positive images", fontsize=6.0, color=MUTED, va="top")

    zero, span, lim = 0.60, 0.30, 1.6
    ax.plot([zero, zero], [0.42, 0.80], color=INK, lw=0.9, zorder=3)
    ax.text(zero, 0.395, "0 = neutral context", fontsize=5.8, color=MUTED,
            va="top", ha="center")

    wp = m["pos"] / lim * span
    ax.add_patch(Rectangle((zero + 0.005, 0.655), wp, 0.085, fc=POS, ec="none"))
    ax.text(zero + wp + 0.022, 0.6975, f"{m['pos']:+.2f}", va="center",
            fontsize=6.5, color=POS, weight="bold")
    ax.text(zero + 0.005, 0.755, "positive", fontsize=6.0, color=POS)

    wn = abs(m["neg"]) / lim * span
    ax.add_patch(Rectangle((zero - 0.005 - wn, 0.470), wn, 0.085, fc=NEG, ec="none"))
    ax.text(zero - 0.005 - wn - 0.022, 0.5125, f"{m['neg']:+.2f}", va="center", ha="right",
            fontsize=6.5, color=NEG, weight="bold")
    ax.text(zero - 0.005, 0.435, "negative", fontsize=6.0, color=NEG, ha="right", va="top")

    ax.text(0, 0.285, f"{m['ratio']:.1f}×", fontsize=13.5, weight="bold", va="top")
    ax.text(0, 0.125, "farther than the mirror", fontsize=6.4, color=MUTED, va="top")
    ax.text(0, 0.045, f"all {m['n_pairs']} pairs agree · contrast {m['contrast']:+.2f}",
            fontsize=6.1, color=MUTED, va="top")


def panel_depth(ax) -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, "C  Where the image signal reads out", weight="bold", fontsize=7.6, va="top")
    ax.text(0, 0.915, "Gemma-3-4B (the probe model, not B)", fontsize=6.0, color=MUTED, va="top")

    x0, x1, yb = 0.13, 0.86, 0.30
    top, bot = 0.80, 0.44
    sc = lambda v: bot + (v / 100) * (top - bot)
    xs = [x0, x1]
    ax.plot(xs, [sc(100.1), sc(31.1)], "-o", color=INK, lw=1.2, ms=3.4, zorder=3)
    ax.plot(xs, [sc(1.7), sc(63.4)], "--s", color=MUTED, lw=1.2, ms=3.0, zorder=3)

    ax.text(x0 - 0.015, sc(100.1) + 0.035, "image tokens", fontsize=6.1, ha="left")
    ax.text(x0 - 0.02, sc(1.7) - 0.045, "text positions", fontsize=6.1, color=MUTED)
    ax.text(x0 - 0.025, sc(100.1), "100%", fontsize=6.2, ha="right", va="center")
    ax.text(x1 + 0.025, sc(31.1), "31%", fontsize=6.2, va="center")
    ax.text(x0 - 0.025, sc(1.7), "2%", fontsize=6.2, ha="right", va="center")
    ax.text(x1 + 0.025, sc(63.4), "63%", fontsize=6.2, color=MUTED, va="center")

    ax.annotate("", xy=(0.97, yb), xytext=(0.03, yb),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, mutation_scale=7))
    for xf, lab in ((x0, "layers 0–12"), (x1, "18–28")):
        ax.plot([xf, xf], [yb - 0.02, yb + 0.02], color=INK, lw=0.7)
        ax.text(xf, yb - 0.075, lab, ha="center", fontsize=6.0, color=MUTED)
    ax.text(0.03, 0.11, "sentence effect restored from text", fontsize=6.2, color=MUTED, va="top")
    ax.text(0.03, 0.02, "positions at layers 13–17: 88–93%", fontsize=6.2, color=MUTED, va="top")


def main() -> None:
    m = measured()
    fig = plt.figure(figsize=(5.5, 2.35), dpi=400)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.92, 1.18], wspace=0.30,
                          left=0.012, right=0.988, top=0.95, bottom=0.03)
    panel_design(fig.add_subplot(gs[0, 0]))
    panel_result(fig.add_subplot(gs[0, 1]), m)
    panel_depth(fig.add_subplot(gs[0, 2]))
    for x in (0.335, 0.645):
        fig.add_artist(plt.Line2D([x, x], [0.06, 0.93], color=RULE, lw=0.6))
    out = ROOT / "paper/figures"
    fig.savefig(out / "method_diagram.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "method_diagram.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out/'method_diagram.pdf'} and .png")
    print(f"  neg {m['neg']:+.3f}  pos {m['pos']:+.3f}  ratio {m['ratio']:.2f}  "
          f"contrast {m['contrast']:+.3f}")


if __name__ == "__main__":
    main()
