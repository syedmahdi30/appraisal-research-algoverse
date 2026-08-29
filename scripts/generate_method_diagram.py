"""Method overview figure for the VLM valence-conflict paper.

    python scripts/generate_method_diagram.py

Ports the "Method Diagram" Claude Design canvas into matplotlib so the figure
emits a vector PDF cropped to its own bounding box. Exporting the canvas
directly went through macOS print-to-PDF, which paginates onto US Letter and
silently clipped roughly 40% of the artwork, including every result.

The canvas is authored 2200px wide with 22px body text. Placed at \\linewidth
(5.5in) that renders near 4pt, so the type is re-tuned here rather than scaled.
Nothing sits below 8pt. Measured on DejaVu Sans, 8pt regular runs 17.4 characters
per inch and 8pt bold caps only 12.8, so headings run out of room well before body
text does and every string below is written to its own column's budget:

    stimulus text  1.47in -> 25 chars     result panel  1.86in -> 32 regular / 23 bold
    scorer box     0.99in -> 17 chars     arm-B panel   1.14in -> 19 chars

The canvas's three footer lines move into the LaTeX caption, where they cost no
figure height.

Numbers shown are guarded by tests/test_method_diagram_numbers.py, which
recomputes each from results/ and fails when the drawing goes stale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]

# Modernist design-system tokens, from the canvas stylesheet.
INK, RED = "#201e1d", "#ec3013"
MID, LIGHT = "#605d5d", "#9b9797"
SURFACE, EDGE, RULE, PALE_RED = "#eae9e9", "#bab6b6", "#d7d3d3", "#f0b3a6"

W_IN, H_IN = 5.5, 3.2
ASPECT = H_IN / W_IN  # multiply an x-extent by this for a visually square box

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # Archivo is not bundled; same modernist grotesque
    "font.size": 8.0,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _box(ax, x, y, w, h, *, edge=INK, lw=1.3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=edge, linewidth=lw))


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.003, head_width=0.018,
                            head_length=0.009, length_includes_head=True,
                            facecolor=INK, edgecolor="none"))


def _rule(ax, x0, x1, y, color=PALE_RED):
    ax.plot([x0, x1], [y, y], color=color, lw=0.8, solid_capstyle="butt")


def _draw_stimulus(ax):
    _box(ax, 0.028, 0.600, 0.372, 0.370)

    icon_w = 0.055
    icon_h = icon_w / ASPECT
    ix, iy = 0.058, 0.848
    ax.add_patch(Rectangle((ix, iy), icon_w, icon_h, facecolor=SURFACE, edgecolor=EDGE, lw=0.9))
    cx = ix + icon_w / 2
    ax.add_patch(Circle((cx, iy + icon_h * 0.68), icon_w * 0.20, facecolor=LIGHT, edgecolor="none"))
    ax.add_patch(Polygon([(ix + icon_w * 0.14, iy + icon_h * 0.10),
                          (ix + icon_w * 0.86, iy + icon_h * 0.10),
                          (ix + icon_w * 0.74, iy + icon_h * 0.46),
                          (ix + icon_w * 0.26, iy + icon_h * 0.46)],
                         closed=True, facecolor=LIGHT, edgecolor="none"))
    ax.text(cx, 0.832, "POSITIVE\nPHOTO", fontsize=8.0, weight="bold", color=LIGHT,
            ha="center", va="top", linespacing=1.25)

    tx = 0.152
    ax.text(tx, 0.922, "“…moments after", fontsize=8.5, va="center")
    ax.text(tx, 0.862, "WON", fontsize=8.5, weight="bold", color=INK, va="center",
            bbox=dict(boxstyle="square,pad=0.25", facecolor=SURFACE, edgecolor="none"))
    ax.text(tx + 0.062, 0.862, "/", fontsize=8.5, weight="bold", color=LIGHT, va="center")
    ax.text(tx + 0.084, 0.862, "LOST", fontsize=8.5, weight="bold", color="white", va="center",
            bbox=dict(boxstyle="square,pad=0.25", facecolor=INK, edgecolor="none"))
    ax.text(tx, 0.802, "the championship", fontsize=8.5, va="center")
    ax.text(tx, 0.742, "game.”", fontsize=8.5, va="center")


def _draw_scorer(ax):
    x, w = 0.432, 0.180
    _box(ax, x, 0.655, w, 0.250)
    mid = x + w / 2
    ax.text(mid, 0.872, "VLM", fontsize=9.0, weight="bold", ha="center", va="center")
    ax.text(mid, 0.762, "4 VLMs\ncomplete-label\nlog-probs, 13\nemotion words",
            fontsize=8.0, color=MID, ha="center", va="center", linespacing=1.35)


def _draw_behavioral_result(ax):
    x, w = 0.645, 0.350
    _box(ax, x, 0.600, w, 0.370, edge=RED, lw=1.6)
    left, right = x + 0.014, x + w - 0.014

    ax.text(left, 0.955, "MEAN SHIFT VS NEUTRAL\nQWEN3-VL-8B",
            fontsize=8.0, weight="bold", color=MID, va="top", linespacing=1.3)

    bar_x, bar_w = left + 0.064, right - left - 0.064
    for y, label, frac, color in ((0.855, "WON", 0.22, INK), (0.812, "LOST", 0.96, RED)):
        ax.text(left, y, label, fontsize=8.0, weight="bold", color=color, va="center")
        ax.add_patch(Rectangle((bar_x, y - 0.014), bar_w * frac, 0.028,
                               facecolor=color, edgecolor="none"))

    _rule(ax, left, right, 0.788)
    ax.text(left, 0.762, "4–5× larger for LOST, all 6", fontsize=8.0, va="center")

    _rule(ax, left, right, 0.736)
    ax.text(left, 0.712, "joy → sadness", fontsize=9.0, weight="bold", va="center")
    ax.text(left, 0.648, "Top label, same photo, one\nword changed. 57 of 62 flip.",
            fontsize=8.0, color=MID, va="center", linespacing=1.3)


def _draw_patch_operation(ax):
    _box(ax, 0.028, 0.050, 0.272, 0.400)

    for bx, label, color, filled in ((0.052, "Donor", INK, True), (0.176, "Recipient", RED, False)):
        rect = dict(facecolor=INK, edgecolor="none") if filled else dict(facecolor="none", edgecolor=RED, linewidth=1.6)
        ax.add_patch(Rectangle((bx, 0.352), 0.096, 0.052, **rect))
        ax.text(bx + 0.048, 0.322, label, fontsize=8.5, weight="bold", color=color,
                ha="center", va="center")
    ax.text(0.100, 0.288, "positive\ncontext", fontsize=8.0, color=MID, ha="center", va="top",
            linespacing=1.3)
    ax.text(0.224, 0.288, "negative\ncontext", fontsize=8.0, color=MID, ha="center", va="top",
            linespacing=1.3)

    _arrow(ax, 0.046, 0.078, 0.148)
    ax.text(0.090, 0.148, "activations copied,\none token group",
            fontsize=8.0, color=MID, va="center", linespacing=1.3)


def _draw_mechanistic_results(ax):
    panels = [
        (0.338, "PROBE", "ρ = +0.510", "text-trained probe\ntracks image\nvalence", None),
        (0.557, "PATCHING", "62–82%", "of context effect\nrestored at text\npositions.",
         "0% at image\npositions, an\nalignment check"),
        (0.776, "STEERING", "+0.335", "slope, still moves\nthe answer under\nconflict", None),
    ]
    for x, title, value, caption, footnote in panels:
        w = 0.219
        _box(ax, x, 0.050, w, 0.400, edge=RED, lw=1.6)
        mid = x + w / 2
        ax.text(mid, 0.412, title, fontsize=8.5, weight="bold", ha="center", va="center")
        ax.text(mid, 0.340, value, fontsize=12.0, weight="bold", ha="center", va="center")
        ax.text(mid, 0.245, caption, fontsize=8.0, color=MID, ha="center", va="center",
                linespacing=1.35)
        if footnote:
            ax.text(mid, 0.120, footnote, fontsize=8.0, color=LIGHT, ha="center", va="center",
                    linespacing=1.35)


def build_figure(_metrics: dict | None = None):
    """Build the two-arm method figure at the paper's text width."""
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=400)
    ax = fig.add_axes([0.004, 0.008, 0.992, 0.984])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.002, 0.952, "A", fontsize=13.0, weight="bold", color=RED, va="center")
    _draw_stimulus(ax)
    _arrow(ax, 0.404, 0.428, 0.785)
    _draw_scorer(ax)
    _arrow(ax, 0.616, 0.641, 0.785)
    _draw_behavioral_result(ax)

    _rule(ax, 0.002, 0.998, 0.520, color=RULE)

    ax.text(0.002, 0.432, "B", fontsize=13.0, weight="bold", color=RED, va="center")
    _draw_patch_operation(ax)
    _arrow(ax, 0.304, 0.334, 0.250)
    _draw_mechanistic_results(ax)
    return fig


def write_outputs(fig, out: Path) -> tuple[Path, Path, Path]:
    """Write the paper PDF, a high-resolution preview, and an editable SVG."""
    out.mkdir(parents=True, exist_ok=True)
    pdf, png, svg = (out / f"method_diagram.{ext}" for ext in ("pdf", "png", "svg"))
    for path in (pdf, png, svg):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    return pdf, png, svg


def main() -> None:
    fig = build_figure()
    out = ROOT / "paper/figures"
    write_outputs(fig, out)
    plt.close(fig)
    print(f"wrote {out/'method_diagram.pdf'}, .png, and .svg")


if __name__ == "__main__":
    main()
