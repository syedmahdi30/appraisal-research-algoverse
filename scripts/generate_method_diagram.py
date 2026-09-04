"""Method overview figure for the VLM valence-conflict paper.

    python scripts/generate_method_diagram.py

Ports the "Method Diagram" Claude Design canvas into matplotlib so the figure
emits a vector PDF cropped to its own bounding box. Exporting the canvas
directly went through macOS print-to-PDF, which paginates onto US Letter and
silently clipped roughly 40% of the artwork, including every result.

Panel A shows BOTH conflict directions, because the paper's headline is the
mirror contrast, not the within-positive-image ratio. A figure showing only a
positive photograph would illustrate the comparison an external reviewer flagged
as non-symmetric: there the negative sentence conflicts while the positive one
agrees. Rows are negative-text-on-positive-photo and positive-text-on-negative-
photo, each against that photograph's own neutral baseline, bars drawn to scale.

Placed at \\linewidth (5.5in) nothing sits below 8pt, which
tests/test_generate_method_diagram.py enforces along with label overlap and
canvas bounds. Measured on DejaVu Sans, 8pt regular runs 17.4 characters per inch
and 8pt bold caps only 12.8, so headings run out of room well before body text
does and every string is written to its own column's budget:

    stimulus text  1.66in -> 28 chars     result panel  2.11in -> 36 regular / 27 bold
    scorer box     0.77in -> 13 chars     arm-B panel   1.20in -> 19 chars

Values are full-precision constants rounded at render time. Storing a pre-rounded
constant is a trap: the crossed upper bound is 0.82534, and round(0.825, 2) is
0.82 in binary floating point, so a truncated constant printed an interval the
paper does not report.

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


# Panel A shows BOTH conflict directions, because the paper's headline is the mirror contrast
# (|drop| - |rise|), not the within-positive-image ratio. A figure showing only a positive photo
# would illustrate the comparison an external reviewer flagged as non-symmetric: there the negative
# sentence conflicts while the positive one agrees.
# Full precision, straight from the estimators. Storing a pre-rounded constant is a trap: the
# crossed upper bound is 0.82534, and round(0.825, 2) is 0.82 in binary floating point, so a
# truncated constant printed an interval the paper does not report. Round at render time only.
DROP, RISE = -1.4781706356769568, 0.9817493939096368   # matched set, Qwen3-VL-8B, vs neutral
MIRROR = 0.49642124176732005
MIRROR_CI = (0.11387484159848477, 0.8253392213196612)
FLIP_HI, FLIP_LO = 0.9354838709677419, 0.5277777777777778   # override rate per direction

ROW_HI, ROW_LO = 0.885, 0.690        # y centres of the two conflict directions
BOX_H = 0.170


def _photo_icon(ax, x, y_centre, label, tint):
    """A placeholder person on a photo card. EMOTIC images are not redistributable."""
    w = 0.055
    h = w / ASPECT
    x0, y0 = x, y_centre - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=SURFACE, edgecolor=EDGE, lw=0.9))
    cx = x0 + w / 2
    ax.add_patch(Circle((cx, y0 + h * 0.68), w * 0.20, facecolor=tint, edgecolor="none"))
    ax.add_patch(Polygon([(x0 + w * 0.14, y0 + h * 0.10), (x0 + w * 0.86, y0 + h * 0.10),
                          (x0 + w * 0.74, y0 + h * 0.46), (x0 + w * 0.26, y0 + h * 0.46)],
                         closed=True, facecolor=tint, edgecolor="none"))
    ax.text(cx, y0 - 0.016, label, fontsize=8.0, weight="bold", color=tint,
            ha="center", va="top")


def _sentence(ax, y_centre, word, word_face):
    """The matched pair: identical frame and event, one valence word swapped."""
    x, w = 0.126, 0.302
    _box(ax, x, y_centre - BOX_H / 2, w, BOX_H, edge=EDGE, lw=1.0)
    left = x + 0.016
    ax.text(left, y_centre + 0.052, "\u201c\u2026moments after they", fontsize=8.5, va="center")
    ax.text(left, y_centre, word, fontsize=9.0, weight="bold", color=word_face["fg"], va="center",
            bbox=dict(boxstyle="square,pad=0.30", facecolor=word_face["bg"], edgecolor="none"))
    ax.text(left, y_centre - 0.052, "the championship game.\u201d", fontsize=8.5, va="center")


def _draw_stimulus(ax):
    ax.text(0.0790, 0.936, "PHOTO", fontsize=8.0, weight="bold", color=LIGHT, ha="center",
            va="bottom")
    ax.text(0.8045, 0.974, "neutral", fontsize=8.0, color=MID, ha="center", va="bottom")
    _photo_icon(ax, 0.030, ROW_HI, "POSITIVE", LIGHT)
    _photo_icon(ax, 0.030, ROW_LO, "NEGATIVE", LIGHT)
    _sentence(ax, ROW_HI, "LOST", {"fg": "white", "bg": RED})
    _sentence(ax, ROW_LO, "WON", {"fg": "white", "bg": INK})


def _draw_scorer(ax):
    """One scorer, both directions: the only thing that differs between rows is the stimulus."""
    x, w = 0.450, 0.140
    top, bottom = ROW_HI + BOX_H / 2, ROW_LO - BOX_H / 2
    _box(ax, x, bottom, w, top - bottom)
    mid = x + w / 2
    ax.text(mid, ROW_HI + 0.040, "VLM", fontsize=9.5, weight="bold", ha="center", va="center")
    ax.text(mid, (top + bottom) / 2 - 0.005, "4 VLMs\nlog-probs\n13 emotion\nwords",
            fontsize=8.0, color=MID, ha="center", va="center", linespacing=1.30)


def _draw_behavioral_result(ax):
    """Each row's signed shift against that photograph's own neutral baseline, on a shared axis.

    A magnitude bar would discard the sign, which is the whole point: the two directions move the
    judgment opposite ways and by different amounts. Both rows share one scale and one neutral tick,
    so their lengths are directly comparable by eye.
    """
    x, w = 0.613, 0.383
    track_left, track_right = x + 0.030, x + w - 0.030
    neutral = (track_left + track_right) / 2
    unit = (neutral - track_left) / abs(DROP)      # |drop| reaches the end of the track

    for y, value, colour, flip, note, direction in (
            (ROW_HI, DROP, RED, FLIP_HI, "joy \u2192 sadness", "toward negative"),
            (ROW_LO, RISE, INK, FLIP_LO, "sadness \u2192 joy", "toward positive")):
        _box(ax, x, y - BOX_H / 2, w, BOX_H, edge=colour, lw=1.4)
        axis_y = y + 0.052
        ax.plot([track_left, track_right], [axis_y, axis_y], color=EDGE, lw=0.8,
                solid_capstyle="butt", zorder=1)
        ax.plot([neutral, neutral], [axis_y - 0.016, axis_y + 0.016], color=MID, lw=0.9, zorder=2)
        ax.add_patch(FancyArrow(neutral, axis_y, value * unit, 0, width=0.004,
                                head_width=0.020, head_length=0.010,
                                length_includes_head=True, facecolor=colour, edgecolor="none",
                                zorder=3))

        left = x + 0.016
        shown = f"{value:+.2f}".replace("-", "\u2212")
        ax.text(left, y - 0.006, shown, fontsize=11.0, weight="bold", color=colour, va="center")
        ax.text(left + 0.105, y - 0.006, direction, fontsize=8.0, color=colour, va="center")
        ax.text(left, y - 0.060, f"top label: {note}, {flip:.0%}",
                fontsize=8.0, color=MID, va="center")


def _draw_headline(ax):
    """The symmetric comparison the two rows exist to support."""
    y = 0.556
    ax.text(0.030, y, "MIRROR CONTRAST", fontsize=8.0, weight="bold", color=MID, va="center")
    ax.text(0.262, y, f"+{MIRROR:.3f}", fontsize=10.5, weight="bold", color=RED, va="center")
    # Round before formatting: f"{0.825:.2f}" yields 0.82 on the float representation, which would
    # print an interval the paper does not report. The paper says [+0.11, +0.83].
    lo, hi = (round(v, 2) for v in MIRROR_CI)
    ax.text(0.382, y, f"[+{lo:.2f}, +{hi:.2f}]", fontsize=8.0, color=MID, va="center")
    ax.text(0.996, y, "within positive photos: 4\u20135\u00d7, all 6 pairs",
            fontsize=8.0, color=MID, ha="right", va="center")


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

    ax.text(0.002, 0.975, "A", fontsize=13.0, weight="bold", color=RED, va="center")
    _draw_stimulus(ax)
    for row in (ROW_HI, ROW_LO):
        _arrow(ax, 0.430, 0.447, row)
        _arrow(ax, 0.592, 0.610, row)
    _draw_scorer(ax)
    _draw_behavioral_result(ax)
    _draw_headline(ax)

    _rule(ax, 0.002, 0.998, 0.512, color=RULE)

    ax.text(0.002, 0.470, "B", fontsize=13.0, weight="bold", color=RED, va="center")
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
