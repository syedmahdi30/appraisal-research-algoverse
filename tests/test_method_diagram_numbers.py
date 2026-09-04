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


# --------------------------------------------------------------------------- panel A, both directions
def _matched_scored(rel):
    """The matched parquet with the readouts the published estimators expect."""
    from src.experiments.analyze_stage_f_unbounded import add_readouts
    return add_readouts(_parquet(rel))


def test_mirror_contrast_and_interval_match_source():
    """Panel A's headline. Guards the estimator too, not only the value.

    `asymmetry_vs_floor` gives +0.478 on this data because it differences cell means, while the
    paper and the figure report the per-photograph average, +0.496. A figure drawn from the wrong
    estimator would agree with nothing in the paper.
    """
    from src.experiments.analyze_stage_f_unbounded import mirror_contrast
    spec = MANIFEST["mirror_contrast"]
    result = mirror_contrast(_matched_scored(spec["source"]), "valence")

    assert abs(result["asymmetry_index"] - spec["shown"]) < spec["tol"], (
        f"figure shows {spec['shown']}, source gives {result['asymmetry_index']:.4f}"
    )
    low, high = result["ci95_crossed"]
    assert abs(round(low, 2) - spec["shown_ci"][0]) < spec["tol_ci"]
    assert abs(round(high, 2) - spec["shown_ci"][1]) < spec["tol_ci"]
    assert low > 0, "figure prints an interval excluding zero"


def test_direction_effects_match_source():
    """The two bars are drawn to scale, so both magnitudes have to be right."""
    from src.experiments.analyze_stage_f_unbounded import mirror_contrast
    spec = MANIFEST["direction_effects"]
    result = mirror_contrast(_matched_scored(spec["source"]), "valence")

    assert abs(result["drop"] - spec["shown_drop"]) < spec["tol"], (
        f"figure shows {spec['shown_drop']}, source gives {result['drop']:.4f}"
    )
    assert abs(result["rise"] - spec["shown_rise"]) < spec["tol"], (
        f"figure shows {spec['shown_rise']}, source gives {result['rise']:.4f}"
    )
    assert abs(result["drop"]) > abs(result["rise"]), (
        "the figure draws the negative-text bar longer than the positive-text bar"
    )


def test_override_rates_match_source():
    from src.experiments.analyze_stage_f_unbounded import override_gap
    spec = MANIFEST["override_rates"]
    result = override_gap(_matched_scored(spec["source"]))

    assert abs(result["neg_ctx_overrides_pos_img"] - spec["shown_neg_on_pos"]) < spec["tol"]
    assert abs(result["pos_ctx_overrides_neg_img"] - spec["shown_pos_on_neg"]) < spec["tol"]


# --------------------------------------------------------------------------- figure vs manifest
def _figure_text():
    """Text actually rendered inside the figure PDF."""
    import shutil
    import subprocess
    if shutil.which("pdftotext") is None:
        pytest.skip("poppler's pdftotext not available")
    pdf = ROOT / "paper/figures/method_diagram.pdf"
    if not pdf.exists():
        pytest.skip("figure not built")
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout


def test_figure_prints_the_values_the_manifest_claims():
    """Close the loop: the manifest asserts what the figure shows, so check the figure shows it.

    Without this the guard is one-sided. It caught nothing when the crossed upper bound rendered as
    +0.82 while the paper and the manifest both said +0.83 -- the constant in the generator had been
    stored pre-rounded, and round(0.825, 2) is 0.82 in binary floating point.
    """
    text = _figure_text()
    mirror = MANIFEST["mirror_contrast"]
    low, high = mirror["shown_ci"]
    expected = [
        f"+{mirror['shown']:.3f}",
        f"[+{low:.2f}, +{high:.2f}]",
        f"{MANIFEST['direction_effects']['shown_drop']:+.2f}".replace("-", "−"),
        f"{MANIFEST['direction_effects']['shown_rise']:+.2f}",
        f"{MANIFEST['override_rates']['shown_neg_on_pos']:.0%}",
        f"{MANIFEST['override_rates']['shown_pos_on_neg']:.0%}",
        f"+{MANIFEST['probe_rho']['shown']:.3f}",
        f"+{MANIFEST['steering_slope']['shown']:.3f}",
        f"{MANIFEST['patching_text_all_pct']['shown_low']}–"
        f"{MANIFEST['patching_text_all_pct']['shown_high']}%",
    ]
    missing = [value for value in expected if value not in text]
    assert not missing, f"figure PDF does not print {missing}"


def test_figure_shows_both_conflict_directions():
    """Panel A exists to show the symmetric comparison, not only the positive-photo arm."""
    text = _figure_text()
    for token in ("POSITIVE", "NEGATIVE", "MIRROR CONTRAST", "joy → sadness",
                  "sadness → joy"):
        assert token in text, f"figure no longer shows {token!r}"
