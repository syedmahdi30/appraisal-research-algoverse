"""Geometry and legibility invariants for the method figure.

These are deliberately design-agnostic. What the figure *claims* is guarded by
tests/test_method_diagram_numbers.py, which checks every printed number against
results/. What is guarded here is that the drawing is physically readable at the
size it ships: nothing below 8pt, nothing overlapping, and no label straddling
the edge of the box that is supposed to contain it.

The straddle check exists because porting the design from a 2200px canvas to a
5.5in figure silently overflowed six labels out of their boxes; at a glance the
figure still looked plausible.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Rectangle

from scripts import generate_method_diagram as diagram

MIN_POINT_SIZE = 8.0
TEXT_WIDTH_INCHES = 5.5


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
            yield text, text.get_window_extent(renderer=renderer).transformed(
                axis.transData.inverted()
            )


def _containers(axis):
    """Rectangles drawn as outlines are boxes meant to contain their contents."""
    for patch in axis.patches:
        if isinstance(patch, Rectangle) and patch.get_facecolor()[3] == 0:
            box = patch.get_bbox()
            if box.width > 0.05 and box.height > 0.05:
                yield patch, box


def test_figure_is_text_width_and_single_axes(drawn):
    fig, _, _ = drawn
    width, height = fig.get_size_inches()
    assert width == TEXT_WIDTH_INCHES
    assert len(fig.axes) == 1
    assert height <= 4.0, "a taller figure costs a page the build cannot spare"


def test_no_type_below_the_print_legibility_floor(drawn):
    _, axis, renderer = drawn
    sizes = {text.get_fontsize() for text, _ in _labels(axis, renderer)}
    assert min(sizes) >= MIN_POINT_SIZE, (
        f"smallest type is {min(sizes)}pt; below {MIN_POINT_SIZE}pt is unreadable in print"
    )


def test_labels_stay_within_the_canvas(drawn):
    _, axis, renderer = drawn
    outside = [
        text.get_text()[:40]
        for text, box in _labels(axis, renderer)
        if box.x0 < -0.01 or box.x1 > 1.01 or box.y0 < -0.01 or box.y1 > 1.01
    ]
    assert outside == []


def test_labels_do_not_overlap_each_other(drawn):
    _, axis, renderer = drawn
    labels = list(_labels(axis, renderer))
    collisions = []
    for index, (left, left_box) in enumerate(labels):
        for right, right_box in labels[index + 1:]:
            dx = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            dy = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if dx > 0.002 and dy > 0.002:
                collisions.append((left.get_text()[:28], right.get_text()[:28]))
    assert collisions == []


def test_no_label_straddles_the_box_meant_to_hold_it(drawn):
    """A label must be wholly inside a container box or wholly outside it."""
    _, axis, renderer = drawn
    straddles = []
    for _, container in _containers(axis):
        for text, box in _labels(axis, renderer):
            overlaps = (
                box.x1 > container.x0 and box.x0 < container.x1
                and box.y1 > container.y0 and box.y0 < container.y1
            )
            contained = (
                box.x0 >= container.x0 - 0.004 and box.x1 <= container.x1 + 0.004
                and box.y0 >= container.y0 - 0.004 and box.y1 <= container.y1 + 0.004
            )
            if overlaps and not contained:
                straddles.append(text.get_text()[:34])
    assert straddles == [], f"labels crossing a box edge: {straddles}"


def test_write_outputs_exports_pdf_png_and_svg(tmp_path):
    fig = diagram.build_figure()
    try:
        paths = diagram.write_outputs(fig, tmp_path)
        assert {path.suffix for path in paths} == {".pdf", ".png", ".svg"}
        assert (tmp_path / "method_diagram.pdf").read_bytes().startswith(b"%PDF")
        assert (tmp_path / "method_diagram.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (tmp_path / "method_diagram.svg").read_text()
    finally:
        plt.close(fig)


def test_labels_do_not_collide_with_filled_marks(drawn):
    """Bars and chips are patches, so the text-vs-text overlap check cannot see them."""
    _, axis, renderer = drawn
    filled = [
        patch.get_bbox()
        for patch in axis.patches
        if isinstance(patch, Rectangle) and patch.get_facecolor()[3] > 0
    ]
    collisions = []
    for text, box in _labels(axis, renderer):
        if text.get_bbox_patch() is not None:
            continue  # chips intentionally sit on their own background
        for mark in filled:
            dx = min(box.x1, mark.x1) - max(box.x0, mark.x0)
            dy = min(box.y1, mark.y1) - max(box.y0, mark.y0)
            if dx > 0.002 and dy > 0.002:
                collisions.append(text.get_text()[:28])
    assert collisions == [], f"labels overrun a filled mark: {collisions}"
