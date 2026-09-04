"""Reviewer-control stimuli and analysis wiring (no model, no images required).

The load-bearing property here is COMPARABILITY: a control run only says something about the
published result if it is scored by the same estimators on stimuli that reduce to the published ones
in the baseline variant. Two guards enforce that — frame/question index 0 must reproduce the
published stimuli byte-for-byte, and the analysis path must reproduce the published statistics from
the published parquet.
"""
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.conflict_contexts import (CONTEXT_FRAMES, MINIMAL_PAIRS, NEUTRAL_CONTEXTS,
                                        QUESTION_VARIANTS, build_conditions,
                                        build_frame_conditions, frame_sentence)
from src.experiments.shared.readouts import QUESTION

MATCHED_PQ = Path("results/stage_f/conflict_qwen3-vl-8b-instruct_minimal.parquet")


# --------------------------------------------------------------------------- stimuli
def test_frame_zero_reproduces_published_pairs():
    """Frame 0 must be the published sentence, or the sweep has no baseline to compare against."""
    for i, (positive, negative, _swap) in enumerate(MINIMAL_PAIRS):
        assert frame_sentence(0, i, "positive") == positive
        assert frame_sentence(0, i, "negative") == negative


def test_question_zero_is_the_published_question():
    assert QUESTION_VARIANTS[0][1] == QUESTION


def test_frame_conditions_match_the_minimal_bank_shape():
    """Every frame must carry the same conditions as the published minimal bank."""
    published = build_conditions("minimal")
    for index in range(len(CONTEXT_FRAMES)):
        conditions = build_frame_conditions(index)
        assert len(conditions) == len(published)
        assert [c for c, _, _ in conditions] == [c for c, _, _ in published]
        assert [i for _, i, _ in conditions] == [i for _, i, _ in published]


def test_neutral_contexts_are_not_reframed():
    """Neutral is the per-image baseline; re-framing it would move baseline and effect together."""
    for index in range(len(CONTEXT_FRAMES)):
        neutral = [s for cond, _, s in build_frame_conditions(index) if cond == "neutral"]
        assert neutral == list(NEUTRAL_CONTEXTS)


def test_every_frame_changes_only_the_valence_word_within_a_pair():
    """A frame is only valid if its own positive/negative members still differ minimally."""
    for index in range(len(CONTEXT_FRAMES)):
        for pair in range(len(MINIMAL_PAIRS)):
            positive = frame_sentence(index, pair, "positive").split()
            negative = frame_sentence(index, pair, "negative").split()
            assert len(positive) == len(negative), (index, pair)
            differing = [k for k, (a, b) in enumerate(zip(positive, negative)) if a != b]
            assert len(differing) == 1, (index, pair, differing)


# --------------------------------------------------------------------------- person grounding
def _row(bbox):
    return pd.Series({"bbox": bbox})


def test_crop_returns_the_box_region_with_margin():
    from src.experiments.stage_f_controls import CROP_MARGIN, ground_person
    image = Image.new("RGB", (400, 300), (10, 10, 10))
    cropped = ground_person(image, _row([100, 100, 200, 200]), "crop")
    expected = 100 + 2 * int(CROP_MARGIN * 100)
    assert cropped.size == (expected, expected)
    assert cropped.size != image.size


def test_crop_clips_to_the_image_and_keeps_frame_on_degenerate_boxes():
    from src.experiments.stage_f_controls import ground_person
    image = Image.new("RGB", (50, 50), (10, 10, 10))
    assert ground_person(image, _row([0, 0, 50, 50]), "crop").size == (50, 50)
    assert ground_person(image, _row([10, 10, 11, 11]), "crop").size == (50, 50)


def test_box_preserves_size_but_changes_pixels():
    from src.experiments.stage_f_controls import ground_person
    image = Image.new("RGB", (400, 300), (10, 10, 10))
    marked = ground_person(image, _row([100, 100, 200, 200]), "box")
    assert marked.size == image.size
    assert list(marked.getdata()) != list(image.getdata())


def test_missing_bbox_leaves_the_image_untouched():
    from src.experiments.stage_f_controls import ground_person
    image = Image.new("RGB", (60, 60), (10, 10, 10))
    for bbox in (None, "not-a-box", [1, 2, 3]):
        assert ground_person(image, _row(bbox), "crop").size == image.size
    assert ground_person(image, _row([1, 1, 5, 5]), "none") is image


# --------------------------------------------------------------------------- analysis wiring
@pytest.mark.skipif(not MATCHED_PQ.exists(), reason="published matched parquet not staged")
def test_analysis_reproduces_the_published_matched_statistics():
    """The control analyzer must recover the paper's numbers from the paper's own parquet.

    Guards against the estimator swap that this module got wrong once: `asymmetry_vs_floor` gives
    +0.478 on this data because it differences cell means, while the paper averages per photograph
    and reports +0.496. Using the wrong one would make every control silently non-comparable.
    """
    from src.experiments.stage_f_controls import _analyze
    df = pd.read_parquet(MATCHED_PQ).assign(variant="original", grounding="none")
    metrics = _analyze(df, "test", "controls_pytest", {})
    variant = metrics["per_variant"]["original"]

    assert variant["minimal_pair_asymmetry"]["paired_asymmetry"] == pytest.approx(1.148, abs=5e-4)
    within_ci = variant["minimal_pair_asymmetry"]["ci95"]
    assert within_ci[0] == pytest.approx(0.943, abs=2e-3)
    assert within_ci[1] == pytest.approx(1.344, abs=2e-3)

    assert variant["mirror_contrast"]["asymmetry_index"] == pytest.approx(0.496, abs=5e-4)
    crossed = variant["mirror_contrast"]["ci95_crossed"]
    assert crossed[0] == pytest.approx(0.114, abs=5e-3)
    assert crossed[1] == pytest.approx(0.825, abs=5e-3)
    assert variant["mirror_contrast"]["crossed_clears_zero"] is True

    assert variant["override_gap"]["dominance_gap"] == pytest.approx(0.408, abs=1e-3)


@pytest.mark.skipif(not MATCHED_PQ.exists(), reason="published matched parquet not staged")
def test_variants_are_scored_independently():
    """Per-variant statistics must not bleed across variants when both are in one frame."""
    from src.experiments.stage_f_controls import _analyze
    df = pd.read_parquet(MATCHED_PQ)
    doubled = pd.concat([df.assign(variant="a"), df.assign(variant="b")])
    metrics = _analyze(doubled, "test", "controls_pytest", {})
    a = metrics["per_variant"]["a"]["mirror_contrast"]["asymmetry_index"]
    b = metrics["per_variant"]["b"]["mirror_contrast"]["asymmetry_index"]
    assert a == pytest.approx(b)
    assert a == pytest.approx(0.496, abs=5e-4)


# --------------------------------------------------------------------------- coverage gate
def test_coverage_gate_aborts_when_images_are_missing():
    """The first frame sweep scored 4 of 150 images and printed an ordinary-looking table."""
    from src.experiments.stage_f_controls import check_coverage
    with pytest.raises(RuntimeError, match="ABORT"):
        check_coverage(n_scored=4, n_selected=150, n_missing=146, allow_missing=False)


def test_coverage_gate_allows_a_complete_run():
    from src.experiments.stage_f_controls import check_coverage
    check_coverage(n_scored=150, n_selected=150, n_missing=0, allow_missing=False)


def test_coverage_gate_tolerates_a_few_unreadable_images():
    from src.experiments.stage_f_controls import MAX_MISSING_FRACTION, check_coverage
    check_coverage(n_scored=147, n_selected=150,
                   n_missing=int(MAX_MISSING_FRACTION * 150), allow_missing=False)


def test_allow_missing_downgrades_the_abort_to_a_warning(capsys):
    from src.experiments.stage_f_controls import check_coverage
    check_coverage(n_scored=4, n_selected=150, n_missing=146, allow_missing=True)
    assert "ABORT" in capsys.readouterr().out


# --------------------------------------------------------------------------- image root override
@pytest.mark.skipif(not Path("data/processed/emotic_test.parquet").exists(),
                    reason="processed EMOTIC parquet not staged")
def test_images_root_rebuilds_paths_from_folder_and_filename():
    """The parquet's absolute Colab paths must be re-rootable, exactly, for any mount point."""
    from src.experiments.stage_f_controls import resolve_image_paths
    df = pd.read_parquet("data/processed/emotic_test.parquet").head(20)
    rerooted = resolve_image_paths(df, "/somewhere/emotic")
    for original, rebuilt, folder, filename in zip(df.image_path, rerooted.image_path,
                                                   df.folder, df.filename):
        assert rebuilt == f"/somewhere/emotic/{folder}/{filename}"
        assert original.endswith(f"{folder}/{filename}")   # the join the reroot relies on
    assert resolve_image_paths(df, "/somewhere/emotic/").image_path.iloc[0] == \
        rerooted.image_path.iloc[0]                        # trailing slash tolerated


@pytest.mark.skipif(not Path("data/processed/emotic_test.parquet").exists(),
                    reason="processed EMOTIC parquet not staged")
def test_no_images_root_leaves_paths_alone():
    from src.experiments.stage_f_controls import resolve_image_paths
    df = pd.read_parquet("data/processed/emotic_test.parquet").head(5)
    assert list(resolve_image_paths(df, None).image_path) == list(df.image_path)


# --------------------------------------------------------------------------- paired variant delta
@pytest.mark.skipif(not MATCHED_PQ.exists(), reason="published matched parquet not staged")
def test_identical_variants_give_a_degenerate_paired_interval():
    """The pairing is the point: the same data twice must differ by exactly zero.

    An unpaired bootstrap would return a nonzero-width interval here, which is precisely the error
    that reading two separate intervals for overlap invites.
    """
    from src.experiments.stage_f_controls import paired_variant_delta
    df = pd.read_parquet(MATCHED_PQ)
    doubled = pd.concat([df.assign(variant="none"), df.assign(variant="box")], ignore_index=True)
    result = paired_variant_delta(doubled, "none", "box")

    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    assert result["delta_ci95_paired"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert result["reproduces"] is True
    assert result["base_mirror"] == pytest.approx(result["other_mirror"])


@pytest.mark.skipif(not MATCHED_PQ.exists(), reason="published matched parquet not staged")
def test_paired_delta_detects_a_shifted_variant():
    """A variant with a genuinely smaller drop must show a negative, zero-excluding change."""
    from src.experiments.stage_f_controls import paired_variant_delta
    df = pd.read_parquet(MATCHED_PQ)
    weakened = df.copy()
    conflict = (weakened.image_group == "positive") & (weakened.condition == "negative")
    weakened.loc[conflict, "valence"] = weakened.loc[conflict, "valence"] * 0.5
    doubled = pd.concat([df.assign(variant="none"), weakened.assign(variant="box")],
                        ignore_index=True)
    result = paired_variant_delta(doubled, "none", "box")

    assert result["delta"] < 0
    assert result["delta_ci95_paired"][1] < 0, "a halved drop must exclude zero"
    assert result["reproduces"] is False


@pytest.mark.skipif(not MATCHED_PQ.exists(), reason="published matched parquet not staged")
def test_paired_delta_absent_without_a_none_baseline():
    from src.experiments.stage_f_controls import _analyze
    df = pd.read_parquet(MATCHED_PQ)
    doubled = pd.concat([df.assign(variant="a"), df.assign(variant="b")], ignore_index=True)
    assert "paired_delta" not in _analyze(doubled, "test", "controls_pytest", {})
