"""Guards for the shipping method figure (the v2 design).

`generate_method_diagram_v2` draws in typographic points rather than the 0-1 canvas the original
used, so the geometry checks here derive their tolerances from the axes' own limits instead of
assuming a unit square. What is checked is the same: nothing below 8pt, nothing off-canvas, nothing
overlapping, and the numbers printed in the drawing still traceable to results/.

The v2 module imports DROP, RISE, MIRROR and MIRROR_CI from `generate_method_diagram`, so the values
stay under the manifest's guard in tests/test_method_diagram_numbers.py; what is added here is that
the shipping PDF actually prints them, and that the manuscript actually includes this figure rather
than the superseded one.
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from scripts import generate_method_diagram_v2 as diagram

MIN_POINT_SIZE = 8.0
ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "paper/figures/method_diagram_v2.pdf"
MANUSCRIPT = ROOT / "paper/neurips_2026.tex"
MANIFEST = json.loads((ROOT / "paper/figures/method_diagram.numbers.json").read_text())


@pytest.fixture
def drawn():
    fig = diagram.build_figure()
    fig.canvas.draw()
    try:
        yield fig, fig.axes[0], fig.canvas.get_renderer()
    finally:
        plt.close(fig)


def _labels(axis, renderer):
    for text in axis.texts:
        if text.get_visible() and text.get_text().strip():
            yield text, text.get_window_extent(renderer=renderer)


def _span(axis):
    """Drawing extent in the axes' own units, whatever those are."""
    (x0, x1), (y0, y1) = sorted(axis.get_xlim()), sorted(axis.get_ylim())
    return x1 - x0, y1 - y0


# --------------------------------------------------------------------------- geometry
def test_no_type_below_the_print_legibility_floor(drawn):
    _, axis, renderer = drawn
    sizes = {text.get_fontsize() for text, _ in _labels(axis, renderer)}
    assert min(sizes) >= MIN_POINT_SIZE, (
        f"smallest type is {min(sizes)}pt; below {MIN_POINT_SIZE}pt is unreadable in print"
    )


def test_the_figure_ships_at_the_text_width(drawn):
    """Coordinates are points, so the drawing must be one text column wide."""
    fig, _, _ = drawn
    width_inches, _ = fig.get_size_inches()
    assert width_inches == pytest.approx(5.5, abs=0.02)


def test_every_label_stays_on_the_canvas(drawn):
    fig, axis, renderer = drawn
    canvas = fig.bbox
    outside = [
        text.get_text()[:40]
        for text, box in _labels(axis, renderer)
        if not (canvas.contains(box.x0, box.y0) and canvas.contains(box.x1, box.y1))
    ]
    assert outside == []


def test_labels_do_not_overlap_each_other(drawn):
    """Tolerance scales with the drawing, since these coordinates are points, not fractions."""
    _, axis, renderer = drawn
    width, _ = _span(axis)
    slack = 0.002 * width
    labels = list(_labels(axis, renderer))
    collisions = []
    for index, (left, left_box) in enumerate(labels):
        for right, right_box in labels[index + 1:]:
            dx = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            dy = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if dx > slack and dy > slack:
                collisions.append((left.get_text()[:24], right.get_text()[:24]))
    assert collisions == []


def test_the_generator_self_validates(drawn):
    """The module ships its own validate(); it must agree with these tests, not contradict them."""
    fig, _, _ = drawn
    diagram.validate(fig)


# --------------------------------------------------------------------------- what it prints
def _figure_text():
    if shutil.which("pdftotext") is None:
        pytest.skip("poppler's pdftotext not available")
    if not FIGURE.exists():
        pytest.skip("figure not built")
    return subprocess.run(["pdftotext", str(FIGURE), "-"], capture_output=True, text=True).stdout


def test_shipping_figure_prints_the_audited_values():
    """The values are imported from the audited module, so the drawing must show them unchanged."""
    text = _figure_text()
    low, high = (round(v, 2) for v in diagram.MIRROR_CI)
    expected = [
        f"+{diagram.MIRROR:.3f}",
        f"[+{low:.2f}, +{high:.2f}]",
        f"{diagram.DROP:+.2f}".replace("-", "−"),
        f"{diagram.RISE:+.2f}",
    ]
    missing = [value for value in expected if value not in text]
    assert not missing, f"shipping figure does not print {missing}"


def test_shipping_figure_shows_both_conflict_directions():
    text = _figure_text()
    for token in ("Positive", "Negative", "Mirror contrast", "toward negative", "toward positive"):
        assert token in text, f"figure no longer shows {token!r}"


def test_printed_values_agree_with_the_numbers_manifest():
    """Ties the drawing back to results/ through the manifest the other suite checks."""
    assert diagram.MIRROR == pytest.approx(MANIFEST["mirror_contrast"]["shown"], abs=5e-4)
    assert round(diagram.DROP, 2) == pytest.approx(MANIFEST["direction_effects"]["shown_drop"])
    assert round(diagram.RISE, 2) == pytest.approx(MANIFEST["direction_effects"]["shown_rise"])


# --------------------------------------------------------------------------- which figure ships
def test_the_manuscript_includes_this_figure_not_the_superseded_one():
    """Two generators exist; only one is the shipping figure, and the paper must say which."""
    source = MANUSCRIPT.read_text()
    assert "figures/method_diagram_v2.pdf" in source
    assert "{figures/method_diagram.pdf}" not in source
