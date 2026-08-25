import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.labels import EMOTION_LABELS
from src.experiments.shared.reporting import (
    arbitration,
    content_mean_frame,
    correlation,
    flip_override,
    image_discriminability,
    polarity_auc,
    sequence_result_columns,
    text_only_readouts,
    token_budget_trends,
)


def _lp(winner):
    return {f"lp_{label}": 0.0 if label == winner else -10.0 for label in EMOTION_LABELS}


def test_correlation_filters_nonfinite_pairs_and_keeps_schema():
    result = correlation([1.0, 2.0, float("nan"), 3.0], [2.0, 4.0, 99.0, 6.0])

    assert result["n"] == 3
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == 1.0


def test_arbitration_aligns_each_steered_row_to_its_own_cell_baseline():
    frame = pd.DataFrame(
        [
            {"beta": 0, "valence": 1.0, "probe_readout": 10.0},
            {"beta": -1, "valence": 0.0, "probe_readout": 9.0},
            {"beta": 1, "valence": 3.0, "probe_readout": 11.0},
            {"beta": 0, "valence": 2.0, "probe_readout": 20.0},
            {"beta": -1, "valence": 0.0, "probe_readout": 18.0},
            {"beta": 1, "valence": 4.0, "probe_readout": 22.0},
        ]
    )

    result = arbitration(frame, [-1, 1])

    assert result["valence"] == {-1: -1.5, 1: 2.0}
    assert result["probe"] == {-1: -1.5, 1: 1.5}
    assert result["valence_slope"] == pytest.approx(1.75)
    assert result["probe_slope"] == pytest.approx(1.5)


def test_qwen_runner_import_does_not_require_stage_c_sklearn_dependency():
    script = """
import builtins
import sys
import types

real_import = builtins.__import__
# Torch is a declared Qwen dependency but is not part of this import-boundary assertion.
sys.modules["torch"] = types.ModuleType("torch")

def import_without_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ModuleNotFoundError("simulated requirements-qwen environment: no sklearn")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_sklearn
import src.experiments.stage_f_qwen
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_polarity_auc_keeps_stage_c_calculation():
    result = polarity_auc(range(10), np.array([0.0] * 5 + [1.0] * 5))

    assert result == {"n": 10, "n_pos": 5, "n_neg": 5, "auc": 1.0}


def test_flip_override_and_discriminability_schema():
    rows = []
    for image, group, valence, condition, winner in (
        ("pos", "positive", 0.8, "none", "joy"),
        ("pos", "positive", -0.4, "negative", "sadness"),
        ("neg", "negative", -0.8, "none", "sadness"),
        ("neg", "negative", 0.4, "positive", "joy"),
    ):
        rows.append({"image_path": image, "image_group": group, "valence": valence,
                     "condition": condition, **_lp(winner)})
    frame = pd.DataFrame(rows)
    override = flip_override(frame, n_boot=20, seed=7)
    assert override["neg_ctx_overrides_pos_img"] == 1.0
    assert override["pos_ctx_overrides_neg_img"] == 1.0
    assert set(override) == {
        "neg_ctx_overrides_pos_img", "pos_ctx_overrides_neg_img", "dominance_gap",
        "dominance_gap_ci95", "n_pos_images", "n_neg_images",
    }
    discrim = image_discriminability(frame)
    assert discrim["discriminability_gap"] == pytest.approx(1.6)
    assert discrim["auc"] == 1.0


def test_text_only_and_trend_outputs_keep_existing_keys():
    rows = []
    for condition, value, winner in (("positive", 0.5, "joy"),
                                     ("negative", -0.75, "sadness"),
                                     ("neutral", 0.0, "neutral")):
        rows.append({"condition": condition, "valence": value, **_lp(winner)})
    readouts = text_only_readouts(pd.DataFrame(rows))
    assert set(readouts) == {"saturation_frac", "n_rows", "bounded_valence", "unbounded_margin"}
    assert readouts["bounded_valence"]["ratio_raw"] == pytest.approx(1.5)

    table = pd.DataFrame([
        {"model": "m", "source": "a", "image_tokens": 10, "override_gap": 0.2,
         "ci_lo": 0.1, "ci_hi": 0.3, "auc": 0.95},
        {"model": "m", "source": "b", "image_tokens": 20, "override_gap": 0.25,
         "ci_lo": 0.15, "ci_hi": 0.35, "auc": 0.96},
    ])
    trends = token_budget_trends(table)
    assert trends["within_model"][0]["all_cis_overlap"] is True


def test_sequence_transformations_preserve_primary_and_sensitivity_columns():
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

    columns = sequence_result_columns(result)
    mean_frame = content_mean_frame(pd.DataFrame([columns]))

    assert columns["valence"] == 0.25
    assert columns["lp_joy"] == result["sequence_sum"]["logprobs"]["joy"]
    assert mean_frame.loc[0, "valence"] == -0.5
    assert mean_frame.loc[0, "lp_joy"] == result["content_mean"]["logprobs"]["joy"]
