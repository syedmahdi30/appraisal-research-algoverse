import pandas as pd

from src.data.labels import EMOTION_LABELS
from src.experiments.stage_f_llava import (
    _artifact_paths,
    _content_mean_frame,
    _sequence_columns,
)


def test_sequence_artifacts_cannot_overwrite_first_subtoken_results():
    """Reusing the legacy paths would destroy the published comparison instead of preserving it."""
    legacy = _artifact_paths("first-subtoken", text_only=False)
    sequence = _artifact_paths("sequence", text_only=False)

    assert legacy[0].name == "conflict_llava.parquet"
    assert legacy[1].name == "conflict_llava_metrics.json"
    assert sequence[0].name == "conflict_llava_sequence.parquet"
    assert sequence[1].name == "conflict_llava_sequence_metrics.json"
    assert legacy != sequence


def test_sequence_columns_keep_sum_primary_and_mean_separate():
    """Putting mean scores in lp_* would silently analyze the sensitivity as the primary rule."""
    result = {
        "sequence_sum": {
            "valence": 0.25,
            "logprobs": {label: -float(i + 1) for i, label in enumerate(EMOTION_LABELS)},
            "scores": {label: -float(i + 10) for i, label in enumerate(EMOTION_LABELS)},
        },
        "content_mean": {
            "valence": -0.5,
            "logprobs": {label: -float(i + 20) for i, label in enumerate(EMOTION_LABELS)},
            "scores": {label: -float(i + 30) for i, label in enumerate(EMOTION_LABELS)},
        },
    }

    columns = _sequence_columns(result)

    assert columns["valence"] == 0.25
    assert columns["valence_content_mean"] == -0.5
    assert columns["lp_joy"] == result["sequence_sum"]["logprobs"]["joy"]
    assert columns["lp_content_mean_joy"] == result["content_mean"]["logprobs"]["joy"]
    assert columns["score_sequence_sum_joy"] == result["sequence_sum"]["scores"]["joy"]
    assert columns["score_content_mean_joy"] == result["content_mean"]["scores"]["joy"]


def test_content_mean_frame_replaces_every_primary_readout_column():
    """Leaving even one lp_* column unchanged would mix two scoring rules in the override rate."""
    row = {"valence": 0.1, "valence_content_mean": -0.2}
    for index, label in enumerate(EMOTION_LABELS):
        row[f"lp_{label}"] = -float(index)
        row[f"lp_content_mean_{label}"] = -float(index + 100)

    mean_df = _content_mean_frame(pd.DataFrame([row]))

    assert mean_df.loc[0, "valence"] == -0.2
    for index, label in enumerate(EMOTION_LABELS):
        assert mean_df.loc[0, f"lp_{label}"] == -float(index + 100)
