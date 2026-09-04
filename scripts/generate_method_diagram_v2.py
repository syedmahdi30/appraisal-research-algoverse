"""Draw the revised method schematic as editable SVG, vector PDF, and PNG.

Run: python3 scripts/generate_method_diagram_v2.py
Coordinates are typographic points, so type sizes are actual print sizes.
The two behavioral magnitudes reuse the audited original figure's constants.
Illustrations and analysis glyphs are schematic, never experimental samples.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D

try:   # importable both as `python scripts/generate_method_diagram_v2.py` and as a module
    from generate_method_diagram import DROP, RISE, MIRROR, MIRROR_CI
except ModuleNotFoundError:   # pragma: no cover - depends on how the caller set sys.path
    from scripts.generate_method_diagram import DROP, RISE, MIRROR, MIRROR_CI

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE = "#253446", "#586779", "#CCD5DD"
POS, NEG, BLUE = "#167D8D", "#BE553D", "#5366A0"
PALE_POS, PALE_NEG, PALE_BLUE = "#EAF5F4", "#FBEEEA", "#EFF1F8"
W, H = 396, 230


def label(ax, x, y, text, size=8, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", linespacing=1.35)


def box(ax, x, y, w, h, fill="white", edge=LINE, radius=3, lw=.7):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=INK, lw=.9, style="-|>", rad=0):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=7, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(p)


def curve(ax, points, color, lw=1.2):
    ax.add_patch(PathPatch(MPath(points, [MPath.MOVETO] + [MPath.CURVE3] * 2),
                           fill=False, edgecolor=color, linewidth=lw, capstyle="round"))


def portrait(ax, x, y, positive=True):
    """Original vector illustration, not a reproduced EMOTIC photograph."""
    c, pale = (POS, PALE_POS) if positive else (NEG, PALE_NEG)
    box(ax, x, y, 39, 34, pale, c, radius=3)
    ax.plot([x+3,x+36],[y+29,y+29],color=LINE,lw=.7)
    ax.plot([x+30,x+30],[y+8,y+27],color=LINE,lw=1)
    ax.add_patch(Ellipse((x+30,y+9),10,11,facecolor=LINE,edgecolor="none"))
    for px,py,col,s in [(10,12,c,1),(22,16,MUTED,.75)]:
        ax.add_patch(Circle((x+px,y+py),2.6*s,facecolor=col,edgecolor="none"))
        box(ax,x+px-2.5*s,y+py+3.4*s,5*s,8*s,col,"none",radius=1)
        for dx in [-1.5,1.5]:
            ax.plot([x+px+dx*s,x+px+dx*1.4*s],
                    [y+py+10*s,y+py+16*s],color=col,lw=1.4*s)


def tokens(ax, x, y, fills, w=9, h=9, gap=3):
    for i, fill in enumerate(fills):
        box(ax, x+i*(w+gap), y, w, h, fill, fill if fill != "white" else LINE, radius=1.3, lw=.65)


def build_figure():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
                         "pdf.fonttype":42, "svg.fonttype":"none"})
    fig = plt.figure(figsize=(W/72,H/72), dpi=180, facecolor="white")
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")

    label(ax, 3, 9, "A", 10, weight="bold")
    label(ax, 18, 9, "Behavioral experiment", 10, weight="bold")
    label(ax, 393, 9, "4 models; Qwen shown", 8, MUTED, ha="right")

    # Matched-context examples. Both signs of image valence are represented;
    # each effect is measured against that image's neutral-context baseline.
    for y, positive, word, c, pale in [(30,True,"lost",NEG,PALE_NEG),
                                      (79,False,"won",POS,PALE_POS)]:
        portrait(ax, 5, y, positive)
        label(ax, 24.5, y+41, "Positive" if positive else "Negative", 8,
              POS if positive else NEG, ha="center")
        label(ax, 55, y+17, "+", 12, MUTED, ha="center")
        box(ax, 66, y+1, 99, 33, pale, edge="none")
        label(ax, 73, y+10, "“…they")
        label(ax, 112, y+10, word, color=c, weight="bold")
        label(ax, 73, y+25, "the championship.”")
        arrow(ax, 168, y+17, 184, y+17, MUTED)

    # Shared forward-pass model. Layer glyphs are illustrative, not circuitry.
    box(ax, 186, 29, 59, 86, PALE_BLUE, BLUE, radius=4)
    label(ax, 215.5, 42, "VLM", 11, weight="bold", ha="center")
    for y in [57,66,75]:
        box(ax, 201, y, 29, 5, "white", BLUE, radius=1.4, lw=.65)
    label(ax, 215.5, 96, "Emotion", 8, ha="center")
    label(ax, 215.5, 107, "scores", 8, ha="center")

    label(ax, 325, 29, "Shift vs. neutral", 8, weight="bold", ha="center")
    for cy, value, c, text in [(47,DROP,NEG,"toward negative"),
                               (96,RISE,POS,"toward positive")]:
        arrow(ax, 247, cy, 264, cy, MUTED)
        # Each row has its own neutral-context baseline; signed shifts are to scale.
        origin = 327
        ax.plot([274,382],[cy,cy],color=LINE,lw=.75)
        ax.plot([origin,origin],[cy-4,cy+4],color=MUTED,lw=.75)
        arrow(ax, origin, cy, origin+value*32, cy, c, lw=2)
        label(ax, 325, cy+15, f"{value:+.2f}".replace("-","−")+"  "+text,
              8, c, ha="center")
    label(ax, 75, 125, "Mirror contrast", 8, weight="bold")
    label(ax, 157, 125, f"+{MIRROR:.3f}", 9, BLUE, weight="bold")
    lo, hi = MIRROR_CI
    label(ax, 207, 125, f"95% CI [+{lo:.2f}, +{hi:.2f}]", 8, MUTED)

    ax.plot([3,393],[137,137],color=LINE,lw=.7)
    label(ax, 3, 150, "B", 10, weight="bold")
    label(ax, 18, 150, "Analysis experiments", 10, weight="bold")
    label(ax, 393, 150, "Separate runs", 8, MUTED, ha="right")
    lower_start = set(ax.get_children())

    # Three operations, not three result dashboards.
    for x, title in [(3,"Probe"),(138,"Patch"),(273,"Steer")]:
        label(ax, x+58, 170, title, 9, weight="bold", ha="center")
    for x in [130,265]:
        ax.plot([x,x],[165,246],color=LINE,lw=.6)

    # Frozen text-trained probe applied to image-conditioned states.
    label(ax, 6, 188, "Text", 8, MUTED)
    arrow(ax, 30, 188, 49, 188, MUTED)
    box(ax, 54, 182, 51, 13, PALE_BLUE, edge="none")
    label(ax, 79.5, 188.5, "Fit probe", 8, BLUE, ha="center")
    arrow(ax, 79.5, 197, 79.5, 211, BLUE)
    label(ax, 6, 203, "Image states", 8, MUTED)
    tokens(ax, 6, 214, [POS,POS,BLUE], w=7,h=9,gap=3)
    arrow(ax, 37, 218.5, 52, 218.5, MUTED)
    box(ax, 54, 212, 51, 13, "white", BLUE)
    label(ax, 79.5, 218.5, "Frozen", 8, BLUE, ha="center")
    arrow(ax, 108, 218.5, 115, 218.5, BLUE)
    label(ax, 120, 218.5, "v", 8, BLUE, ha="center")
    label(ax, 61, 240, "Read image valence", 8, MUTED, ha="center")

    # Explicit before/after: only the selected donor states replace target states.
    label(ax, 140, 188, "Donor", 8, POS)
    tokens(ax, 205, 183, [LINE,POS,POS,LINE], w=8,h=10,gap=3)
    box(ax, 214, 181, 23, 14, "none", POS, radius=2)
    arrow(ax, 225.5, 197, 225.5, 212, POS, lw=1.4)
    label(ax, 205, 204, "copy", 8, MUTED, ha="right")
    tokens(ax, 141, 216, [NEG,NEG,NEG,NEG], w=8,h=10,gap=3)
    arrow(ax, 186, 221, 201, 221, MUTED)
    tokens(ax, 205, 216, [NEG,POS,POS,NEG], w=8,h=10,gap=3)
    box(ax, 214, 214, 23, 14, "none", POS, radius=2)
    label(ax, 161.5, 240, "Target", 8, NEG, ha="center")
    label(ax, 225.5, 240, "Patched", 8, MUTED, ha="center")

    # A mean-difference text direction is added, distinct from probe weights.
    label(ax, 331, 188, "Text-derived direction", 8, MUTED, ha="center")
    label(ax, 294, 201, "h", 8, MUTED, ha="center")
    label(ax, 347, 201, "d", 8, BLUE, ha="center")
    label(ax, 384, 201, "h′", 8, MUTED, ha="center")
    tokens(ax, 280, 211, [LINE,LINE,BLUE], w=8,h=10,gap=3)
    label(ax, 322, 216, "+", 12, BLUE, ha="center")
    arrow(ax, 337, 222, 349, 208, BLUE, lw=1.4)
    arrow(ax, 359, 216, 374, 216, MUTED)
    tokens(ax, 379, 211, [POS], w=10,h=10)
    label(ax, 332, 240, "Measure output shift", 8, MUTED, ha="center")

    # Compress geometry while preserving actual font point sizes.
    compact = Affine2D().translate(0,-150).scale(1,.8).translate(0,150)
    for artist in set(ax.get_children()) - lower_start:
        artist.set_transform(compact + ax.transData)

    return fig


def validate(fig):
    """Check actual print size, canvas containment and text collisions."""
    fig.canvas.draw()
    ax=fig.axes[0]; renderer=fig.canvas.get_renderer()
    labels=[(t,t.get_window_extent(renderer)) for t in ax.texts]
    assert min(t.get_fontsize() for t,_ in labels)>=8
    canvas=fig.bbox
    for text,b in labels:
        assert canvas.contains(b.x0,b.y0) and canvas.contains(b.x1,b.y1), text.get_text()
    for i,(a,ba) in enumerate(labels):
        for b,bb in labels[i+1:]:
            if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>1 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>1:
                raise AssertionError(f"Overlapping labels: {a.get_text()} / {b.get_text()}")


if __name__ == "__main__":
    fig=build_figure(); validate(fig)
    out=ROOT/"paper"/"figures"
    for ext in ["pdf","svg","png"]:
        path=out/f"method_diagram_v2.{ext}"
        fig.savefig(path,dpi=450,facecolor="white")
        print(path)
    plt.close(fig)
