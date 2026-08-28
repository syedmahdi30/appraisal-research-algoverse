import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts import generate_method_diagram as diagram


SAMPLE_METRICS = {
    "neg": -1.48,
    "pos": 0.31,
    "ratio": 4.8,
    "contrast": 1.15,
    "ci": (0.94, 1.34),
    "n_img": 62,
    "n_pairs": 6,
    "rmin": 4.3,
    "rmax": 5.5,
}


def test_method_figure_separates_behavioral_and_mechanistic_flows_without_results():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        labels = {
            text.get_text()
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        }
        joined = " ".join(" ".join(label.split()) for label in labels)

        assert "Behavioral test" in labels
        assert "Mechanistic test" in labels
        assert "Neutral correction" in joined
        assert "Text-trained probe" in joined
        assert "Activation patching" in joined
        assert "Activation steering" in joined
        assert all(result not in joined for result in ("4.8x", "88-93%", "+0.31", "-1.48"))
    finally:
        plt.close(fig)


def test_method_boxes_show_current_model_coverage():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        texts = [
            text
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        ]
        labels = [" ".join(text.get_text().split()) for text in texts]

        assert labels.count("Gemma") == 2
        assert labels.count("Gemma · Qwen · LLaVA") == 1
        assert "Gemma-3-4B" not in labels

        methods = {
            label: next(text for text in texts if text.get_text() == label)
            for label in ("Text-trained probe", "Activation patching", "Activation steering")
        }
        model_labels = [text for text in texts if "Gemma" in text.get_text()]
        for method in methods.values():
            assert any(abs(method.get_position()[1] - model.get_position()[1]) <= 0.065 for model in model_labels)
    finally:
        plt.close(fig)


def test_patching_visual_reports_token_level_recovery_in_the_right_direction():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        labels = {
            " ".join(text.get_text().split())
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        }

        assert "image 0%" in labels
        assert "text 62-82%" in labels
    finally:
        plt.close(fig)


def test_behavioral_lane_names_the_three_readouts_without_inventing_a_score():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        joined = " ".join(
            " ".join(text.get_text().split())
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        )

        assert "3 readouts" in joined
        assert "Asymmetry score" not in joined
    finally:
        plt.close(fig)


def test_input_scope_does_not_claim_every_context_is_a_one_word_pair():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        labels = {
            text.get_text()
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        }

        assert "controlled contexts" in labels
        assert "one-word pair" not in labels
    finally:
        plt.close(fig)


def test_mechanistic_panel_uses_bold_only_for_its_heading():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        texts = {
            text.get_text(): text
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        }
        internal_labels = (
            "Text-trained probe",
            "Activation patching",
            "Activation steering",
            "cross-modal readout",
            "image 0%",
            "text 62-82%",
            "causal output shift",
        )

        assert texts["Mechanistic test"].get_fontweight() == "bold"
        assert all(texts[label].get_fontweight() == "normal" for label in internal_labels)
    finally:
        plt.close(fig)


def test_build_figure_enforces_compact_legible_paper_layout():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        width, height = fig.get_size_inches()
        visible_text = [
            text
            for axis in fig.axes
            for text in axis.texts
            if text.get_visible() and text.get_text().strip()
        ]

        assert width == 5.5
        assert height <= 2.6
        assert len(fig.axes) == 1
        assert min(text.get_fontsize() for text in visible_text) >= 8.0
    finally:
        plt.close(fig)


def test_write_outputs_exports_pdf_png_and_editable_svg(tmp_path):
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        paths = diagram.write_outputs(fig, tmp_path)

        assert {path.suffix for path in paths} == {".pdf", ".png", ".svg"}
        assert all(path.stat().st_size > 1_000 for path in paths)
        assert (tmp_path / "method_diagram.pdf").read_bytes().startswith(b"%PDF")
        assert (tmp_path / "method_diagram.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (tmp_path / "method_diagram.svg").read_text()
    finally:
        plt.close(fig)


def test_visible_labels_stay_inside_their_panel():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        escaped = []

        for panel_index, axis in enumerate(fig.axes):
            for text in axis.texts:
                if not text.get_visible() or not text.get_text().strip():
                    continue
                bounds = text.get_window_extent(renderer=renderer).transformed(axis.transAxes.inverted())
                if bounds.x0 < -0.01 or bounds.x1 > 1.01 or bounds.y0 < -0.01 or bounds.y1 > 1.01:
                    escaped.append((panel_index, text.get_text()))

        assert escaped == []
    finally:
        plt.close(fig)


def test_visible_labels_do_not_overlap_each_other():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        collisions = []

        for panel_index, axis in enumerate(fig.axes):
            labels = [
                (text, text.get_window_extent(renderer=renderer))
                for text in axis.texts
                if text.get_visible() and text.get_text().strip()
            ]
            for index, (left, left_box) in enumerate(labels):
                for right, right_box in labels[index + 1:]:
                    overlap_x = max(0, min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0))
                    overlap_y = max(0, min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0))
                    if overlap_x * overlap_y > 4:
                        collisions.append((panel_index, left.get_text(), right.get_text()))

        assert collisions == []
    finally:
        plt.close(fig)


def test_shared_input_labels_stay_inside_the_stimulus_block():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        fig.canvas.draw()
        axis = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        escaped = []

        for text in axis.texts:
            x, _ = text.get_position()
            if x >= 0.253 or not text.get_text().strip():
                continue
            bounds = text.get_window_extent(renderer=renderer).transformed(axis.transData.inverted())
            if bounds.x1 > 0.253:
                escaped.append(text.get_text())

        assert escaped == []
    finally:
        plt.close(fig)


def test_token_row_labels_clear_their_token_blocks():
    fig = diagram.build_figure(SAMPLE_METRICS)
    try:
        fig.canvas.draw()
        axis = fig.axes[0]
        renderer = fig.canvas.get_renderer()

        row_colors = {}
        for label, token_y in (("image 0%", 0.256), ("text 62-82%", 0.210)):
            text = next(item for item in axis.texts if item.get_text() == label)
            text_bounds = text.get_window_extent(renderer=renderer).transformed(axis.transData.inverted())
            token_blocks = [
                patch
                for patch in axis.patches
                if isinstance(patch, Rectangle)
                and abs(patch.get_y() - token_y) < 0.001
                and patch.get_x() >= 0.8
            ]

            assert token_blocks
            assert text_bounds.x1 + 0.008 <= min(block.get_x() for block in token_blocks)
            row_colors[label] = token_blocks[0].get_facecolor()

        image_color = row_colors["image 0%"]
        text_color = row_colors["text 62-82%"]
        assert max(text_color[:3]) - min(text_color[:3]) > max(image_color[:3]) - min(image_color[:3])
    finally:
        plt.close(fig)
