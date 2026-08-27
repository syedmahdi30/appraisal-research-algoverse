import pandas as pd
import pytest

from src.experiments.stage_f_llava import DEFAULT_MODEL
from src.experiments.stage_f_llava_patching import (
    _analyze,
    _artifact_paths,
    _parse_layer_band,
    _validated_groups,
)


def _segment(*, image_start=2, image_length=10, question_start=14, n=20):
    import numpy as np

    image = np.arange(image_start, image_start + image_length)
    question = np.arange(question_start, question_start + 2)
    return {
        "image": image,
        "context": np.arange(image_start + image_length, question_start),
        "question": question,
        "template": np.array([0, 1, question_start, question_start + 1, n - 1]),
        "n": n,
        "img_len": image_length,
        "question_ok": True,
        "image_ok": True,
    }


def test_llava_patching_artifacts_are_model_specific_and_do_not_overwrite_behavior_runs():
    default_data, default_metrics = _artifact_paths(DEFAULT_MODEL)
    next_data, next_metrics = _artifact_paths("llava-hf/llava-v1.6-mistral-7b-hf")

    assert default_data.name == "patching_llava_sequence.parquet"
    assert default_metrics.name == "patching_llava_sequence_metrics.json"
    assert next_data.name == "patching_llava-v1-6-mistral-7b-hf_sequence.parquet"
    assert next_metrics.name == "patching_llava-v1-6-mistral-7b-hf_sequence_metrics.json"
    assert len({default_data, default_metrics, next_data, next_metrics}) == 4


def test_default_layer_band_is_proportional_and_overrides_are_inclusive():
    assert _parse_layer_band(None, 32) == list(range(11, 30))
    assert _parse_layer_band("4-6", 32) == [4, 5, 6]

    with pytest.raises(ValueError, match="outside decoder depth"):
        _parse_layer_band("30-32", 32)
    with pytest.raises(ValueError, match="START-END"):
        _parse_layer_band("bad", 32)


def test_segmentation_must_contain_expanded_image_and_all_aligned_groups():
    groups = _validated_groups(_segment(), _segment(question_start=15, n=21))

    assert set(groups) == {
        "image", "question", "bos", "prefix_delim", "suffix_delim", "structure", "text_all"
    }

    with pytest.raises(RuntimeError, match="image placeholder was not expanded"):
        _validated_groups(_segment(image_length=1), _segment(image_length=1))


def test_analysis_records_sequence_scoring_and_recovery_metadata(tmp_path):
    frame = pd.DataFrame({
        "pos_val": [1.0, 1.0],
        "neg_val": [-1.0, -1.0],
        **{
            f"patch_{group}_val": [0.0, 0.0]
            for group in (
                "image", "question", "bos", "prefix_delim", "suffix_delim", "structure",
                "text_all",
            )
        },
    })

    metrics = _analyze(
        frame,
        model_name=DEFAULT_MODEL,
        patch_layers=[11, 12],
        n_layers=32,
        donor_context="positive",
        recipient_context="negative",
        n_skipped=1,
        data_path=tmp_path / "rows.parquet",
        metrics_path=tmp_path / "metrics.json",
        n_boot=20,
    )

    assert metrics["score_mode"] == "sequence"
    assert metrics["score_rule"] == "sum of teacher-forced conditional token log probabilities"
    assert metrics["n_images"] == 2
    assert metrics["patch_layers"] == [11, 12]
    assert metrics["recovery"]["image"]["val"] == pytest.approx(0.5)
    assert metrics["n_skipped"] == 1
    assert (tmp_path / "metrics.json").exists()
