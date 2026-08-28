"""Method overview figure for the VLM valence-conflict paper.

    python scripts/generate_method_diagram.py

The figure shows the controlled stimulus, the two experimental paths, and the
headline token-localization result from activation patching.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]

POS, NEG = "#2F75B5", "#D45A3A"
INK, MUTED, RULE = "#171717", "#606770", "#B8BDC3"
PALE, BLUE_WASH, SAND_WASH = "#F4F5F6", "#F4F8FC", "#FBF7F0"
BLUE_RULE, SAND_RULE = "#86ADD1", "#C8A77B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "font.weight": "normal",
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _round_box(ax, x, y, width, height, *, face, edge, radius=0.018, lw=0.9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color=INK, lw=0.9, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def _draw_stimulus(ax):
    _round_box(ax, 0.018, 0.17, 0.235, 0.66, face="#FCFCFC", edge=RULE, radius=0.025, lw=1.0)
    ax.text(0.035, 0.785, "Matched input", fontsize=8.8, weight="bold", va="center")

    ax.add_patch(Rectangle((0.035, 0.43), 0.074, 0.22, facecolor=PALE, edgecolor=RULE, linewidth=0.8))
    ax.add_patch(Circle((0.072, 0.565), 0.017, facecolor="#BFC4C9", edgecolor="none"))
    ax.add_patch(Ellipse((0.072, 0.505), 0.050, 0.055, facecolor="#BFC4C9", edgecolor="none"))

    contexts = [
        (0.615, "won  (+)", "#EAF3FB", POS),
        (0.510, "neutral", "#F0F1F2", MUTED),
        (0.405, "lost  (-)", "#FCEFEA", NEG),
    ]
    for y, label, face, edge in contexts:
        _round_box(ax, 0.122, y, 0.112, 0.075, face=face, edge=edge, radius=0.012, lw=0.8)
        ax.text(0.178, y + 0.0375, label, fontsize=8.0, color=edge, weight="bold", ha="center", va="center")

    ax.text(0.035, 0.315, "same image", fontsize=8.0, color=MUTED, va="center")
    ax.text(0.035, 0.255, "controlled contexts", fontsize=8.0, color=MUTED, va="center")
    ax.text(0.035, 0.195, "neutral baseline", fontsize=8.0, color=MUTED, va="center")


def _draw_behavioral_lane(ax):
    _round_box(ax, 0.310, 0.555, 0.675, 0.370, face=BLUE_WASH, edge=BLUE_RULE, radius=0.024, lw=1.0)
    ax.text(0.335, 0.885, "A", fontsize=9.0, weight="bold", va="center")
    ax.text(0.360, 0.885, "Behavioral test", fontsize=9.0, weight="bold", va="center")

    nodes = [
        (0.340, 0.660, 0.120, "4 VLMs"),
        (0.495, 0.660, 0.135, "Emotion\nprobabilities"),
        (0.665, 0.660, 0.140, "Neutral\ncorrection"),
        (0.840, 0.660, 0.125, "3 readouts"),
    ]
    for x, y, width, label in nodes:
        _round_box(ax, x, y, width, 0.145, face="white", edge=RULE, radius=0.015, lw=0.8)
        ax.text(x + width / 2, y + 0.0725, label, fontsize=8.0, weight="bold", ha="center", va="center")

    for start, end in [
        ((0.460, 0.7325), (0.495, 0.7325)),
        ((0.630, 0.7325), (0.665, 0.7325)),
        ((0.805, 0.7325), (0.840, 0.7325)),
    ]:
        _arrow(ax, start, end, color=POS)


def _draw_mechanistic_lane(ax):
    _round_box(ax, 0.310, 0.050, 0.675, 0.430, face=SAND_WASH, edge=SAND_RULE, radius=0.024, lw=1.0)
    ax.text(0.335, 0.445, "B", fontsize=9.0, weight="bold", va="center")
    ax.text(0.360, 0.445, "Mechanistic test", fontsize=9.0, weight="bold", va="center")

    method_x, method_w = 0.340, 0.285
    method_rows = [
        (0.320, 0.090, "Text-trained probe", "Gemma", POS),
        (0.200, 0.105, "Activation patching", "Gemma · Qwen · LLaVA", "#7A7F85"),
        (0.080, 0.090, "Activation steering", "Gemma", "#A46F35"),
    ]
    for y, height, label, model_label, accent in method_rows:
        _round_box(ax, method_x, y, method_w, height, face="white", edge=RULE, radius=0.012, lw=0.8)
        ax.add_patch(Circle((method_x + 0.019, y + height / 2), 0.0075, facecolor=accent, edgecolor="none"))
        ax.text(method_x + 0.037, y + height - 0.0225, label, fontsize=8.0, va="center")
        ax.text(method_x + 0.037, y + 0.0225, model_label, fontsize=8.0, color=MUTED, va="center")

    output_x, output_w = 0.670, 0.290
    simple_outputs = [
        (0.327, "cross-modal readout", POS),
        (0.087, "causal output shift", "#A46F35"),
    ]
    for y, label, accent in simple_outputs:
        _arrow(ax, (method_x + method_w, y + 0.030), (output_x, y + 0.030), color=accent)
        _round_box(ax, output_x, y, output_w, 0.060, face="white", edge=RULE, radius=0.012, lw=0.8)
        ax.text(output_x + output_w / 2, y + 0.030, label, fontsize=8.0, ha="center", va="center")

    # The measured quantity leads; the image row is an arithmetic zero, not a finding.
    # Image positions precede the context and the image is held fixed, so the patch copies a
    # value onto itself. Labelling it in the panel keeps it from reading as a result.
    result_y, result_h = 0.162, 0.148
    _arrow(ax, (method_x + method_w, result_y + result_h / 2), (0.650, result_y + result_h / 2), color="#7A7F85")
    _round_box(ax, 0.650, result_y, 0.310, result_h, face="white", edge=RULE, radius=0.012, lw=0.8)
    ax.text(0.672, 0.285, "text 62-82%", fontsize=8.0, color=POS, va="center")
    ax.text(0.672, 0.236, "image 0%", fontsize=8.0, color=MUTED, va="center")
    ax.text(0.672, 0.187, "alignment check", fontsize=8.0, color=MUTED, style="italic", va="center")
    for index in range(4):
        token_x = 0.838 + index * 0.024
        ax.add_patch(Rectangle((token_x, 0.272), 0.016, 0.026, facecolor="#9DC2E2", edgecolor=POS, linewidth=0.6))
        ax.add_patch(Rectangle((token_x, 0.223), 0.016, 0.026, facecolor="#EFF0F1", edgecolor=MUTED, linewidth=0.6))


def build_figure(_metrics: dict | None = None):
    """Build a compact method-only figure at the paper's text width."""
    fig = plt.figure(figsize=(5.5, 2.5), dpi=400)
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _draw_stimulus(ax)
    _draw_behavioral_lane(ax)
    _draw_mechanistic_lane(ax)

    ax.add_patch(Circle((0.282, 0.500), 0.007, facecolor=INK, edgecolor="none"))
    _arrow(ax, (0.253, 0.500), (0.280, 0.500))
    _arrow(ax, (0.286, 0.505), (0.330, 0.735), color=POS, connectionstyle="arc3,rad=-0.08")
    _arrow(ax, (0.286, 0.495), (0.330, 0.255), color="#A46F35", connectionstyle="arc3,rad=0.08")
    return fig


def write_outputs(fig, out: Path) -> tuple[Path, Path, Path]:
    """Write the paper PDF, high-resolution preview, and editable SVG."""
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "method_diagram.pdf"
    png = out / "method_diagram.png"
    svg = out / "method_diagram.svg"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
    return pdf, png, svg


def main() -> None:
    fig = build_figure()
    out = ROOT / "paper/figures"
    write_outputs(fig, out)
    plt.close(fig)
    print(f"wrote {out/'method_diagram.pdf'}, .png, and .svg")


if __name__ == "__main__":
    main()
