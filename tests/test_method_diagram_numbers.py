"""Guard the numbers printed inside the method figure.

The figure is a hand-maintained design asset: no script redraws it when results
move, and results have moved twice during drafting. These tests recompute every
number the figure asserts from results/ and fail if the drawing has gone stale.

A failure here means the FIGURE is wrong, not the expectation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "paper/figures/method_diagram.numbers.json").read_text())


def _dig(obj, dotted):
    for key in dotted.split("."):
        obj = obj[key]
    return obj


def _load(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present; results/ is git-ignored")
    return json.loads(path.read_text())


def _parquet(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present; results/ is git-ignored")
    return pd.read_parquet(path)


def test_probe_rho_matches_source():
    spec = MANIFEST["probe_rho"]
    actual = _dig(_load(spec["source"]), spec["path"])
    assert abs(round(actual, 3) - spec["shown"]) < spec["tol"], (
        f"figure shows rho={spec['shown']}, source has {actual:.4f}"
    )


def test_steering_slope_matches_source():
    spec = MANIFEST["steering_slope"]
    actual = _dig(_load(spec["source"]), spec["path"])
    assert abs(round(actual, 3) - spec["shown"]) < spec["tol"], (
        f"figure shows slope={spec['shown']}, source has {actual:.4f}"
    )


def test_unique_photograph_count_matches_source():
    spec = MANIFEST["unique_photographs_patching"]
    assert _dig(_load(spec["source"]), spec["path"]) == spec["shown"]


def test_matched_pair_ratios_span_the_shown_range():
    spec = MANIFEST["matched_pair_ratios"]
    per_pair = _dig(_load(spec["source"]), spec["path"])
    ratios = [v["ratio"] for v in per_pair.values()]
    assert round(min(ratios), 1) == spec["shown_low"]
    assert round(max(ratios), 1) == spec["shown_high"]


def test_patching_recovery_spans_the_shown_range():
    """62-82% is a range across the three patched models on one common readout."""
    spec = MANIFEST["patching_text_all_pct"]
    from src.experiments.shared.patching import collapse_duplicate_image_rows

    values, image_values = [], []
    for source in spec["sources"]:
        recovery = _load(source)["recovery"]
        values.append(recovery["text_all"]["val"] * 100)
        image_values.append(recovery["image"]["val"] * 100)

    frame = collapse_duplicate_image_rows(_parquet(spec["gemma_parquet"]))
    denominator = (frame["pos_val"] - frame["neg_val"]).to_numpy(float)
    for group, sink in (("text_all", values), ("image", image_values)):
        numerator = (frame[f"patch_{group}_val"] - frame["neg_val"]).to_numpy(float)
        sink.append(float(numerator.sum() / denominator.sum()) * 100)

    assert int(np.floor(min(values))) == spec["shown_low"], (
        f"figure shows {spec['shown_low']}%, models span {min(values):.1f}-{max(values):.1f}%"
    )
    assert int(np.ceil(max(values))) == spec["shown_high"], (
        f"figure shows {spec['shown_high']}%, models span {min(values):.1f}-{max(values):.1f}%"
    )

    tolerance = MANIFEST["patching_image_pct"]["tol_pct"]
    assert max(abs(v) for v in image_values) < tolerance, (
        "image-position recovery is no longer ~0%; the figure calls it an alignment check"
    )


def test_joy_to_sadness_flip_count():
    spec = MANIFEST["joy_to_sadness"]
    frame = _parquet(spec["source"])
    label_columns = [c for c in frame.columns if c.startswith("lp_")]
    frame = frame.assign(
        top=frame[label_columns].idxmax(axis=1).str.replace("lp_", "", regex=False)
    )
    pair = frame[(frame.image_group == "positive") & (frame.context_id == spec["context_id"])]
    tops = pair.pivot_table(index="image_path", columns="condition", values="top", aggfunc="first")

    assert len(tops) == spec["shown_total"], (
        f"figure shows {spec['shown_total']} positive images, data has {len(tops)}"
    )
    flips = int(((tops["positive"] == "joy") & (tops["negative"] == "sadness")).sum())
    assert flips == spec["shown_flips"], (
        f"figure shows joy->sadness on {spec['shown_flips']}, data gives {flips}"
    )
