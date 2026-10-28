"""Camera-ready readout grid: anchors against the published matched set, and the new-run guards.

The grid is only trustworthy if its published cells reproduce the paper before the new cells are
read, so the anchors are pinned here. The corrected categorical gap's crossed interval is pinned too:
it is printed in the paper and previously had no code path that regenerated it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.conflict_contexts import (MINIMAL_PAIRS, NEUTRAL_BANK_B, NEUTRAL_CONTEXTS,
                                        build_equal_room_conditions, build_neutral_bank_conditions,
                                        build_screen_conditions)
from src.experiments.analyze_matched_readouts import (equal_room, matched_grid, merge_neutral_bank,
                                                      photo_key, screen_photos)

MATCHED_PQ = Path("results/stage_f/conflict_qwen3-vl-8b-instruct_minimal.parquet")
needs_matched = pytest.mark.skipif(not MATCHED_PQ.exists(), reason="matched parquet not on disk")


@pytest.fixture(scope="module")
def grid():
    return matched_grid(pd.read_parquet(MATCHED_PQ))


# --------------------------------------------------------------------------- published anchors
@needs_matched
def test_bounded_mirror_reproduces_the_published_value(grid):
    from src.experiments.analyze_stage_f_unbounded import add_readouts as legacy, mirror_contrast
    assert grid["valence"]["mirror"]["estimate"] == pytest.approx(0.4964, abs=1e-3)
    # The published interval came from independent per-group pair draws in the older module.
    df = pd.read_parquet(MATCHED_PQ)
    published = mirror_contrast(legacy(df[df["condition"] != "none"]), "valence")
    assert published["ci95_crossed"] == pytest.approx([0.114, 0.825], abs=2e-3)


@needs_matched
def test_within_item_reproduces_the_published_value(grid):
    assert grid["valence"]["within_pos"]["estimate"] == pytest.approx(1.148, abs=1e-3)


@needs_matched
def test_corrected_category_gap_point_estimate(grid):
    assert grid["category"]["mirror"]["estimate"] == pytest.approx(0.2308, abs=1e-3)


def test_shared_pairs_are_drawn_once_for_both_groups():
    """The same six pairs appear in both groups, so a replicate must resample them jointly."""
    from src.experiments.analyze_matched_readouts import _replicate
    g = lambda: {"neg": np.zeros((5, 6)), "pos": np.zeros((5, 6)), "neutral": np.zeros((5, 2)),
                 "context_ids": [f"mp{i}" for i in range(6)]}
    rng = np.random.default_rng(1)
    for _ in range(20):
        Pd, Nd = _replicate(rng, g(), g(), True, True)
        assert list(Pd[2]) == list(Nd[2]) and list(Pd[3]) == list(Nd[3])
    P, N = g(), g()
    N["context_ids"] = [f"p{i}" for i in range(6)]   # varied set: different sentences per group
    draws = [_replicate(rng, P, N, True, False) for _ in range(20)]
    assert any(list(a[2]) != list(b[2]) for a, b in draws)


@needs_matched
def test_within_item_reverses_on_negative_images(grid):
    """The reason the within-item contrast cannot carry a valence claim: it tracks conflict."""
    for value in ("valence", "sum_lo", "best_lo"):
        assert grid[value]["within_pos"]["estimate"] > 0
        assert grid[value]["within_neg"]["estimate"] < 0


# --------------------------------------------------------------------------- equal-room estimator
def _synthetic(pull_neg: float, pull_pos: float, n_photos: int = 20) -> pd.DataFrame:
    rows = []
    for k in range(n_photos):
        path = f"/x/emodb_small/images/{k}.jpg"
        for i in range(10):
            rows.append(dict(image_path=path, image_group="mid", condition="neutral",
                             context_id=f"zb{i}", valence=0.01 * (i - 4.5)))
        for i in range(6):
            rows.append(dict(image_path=path, image_group="mid", condition="negative",
                             context_id=f"mp{i}", valence=-pull_neg))
            rows.append(dict(image_path=path, image_group="mid", condition="positive",
                             context_id=f"mp{i}", valence=pull_pos))
    df = pd.DataFrame(rows)
    for col in [f"lp_{w}" for w in __import__("src.data.labels", fromlist=["x"]).EMOTION_LABELS]:
        df[col] = np.log(1 / 13)
    return df


def test_equal_room_is_zero_when_both_sentences_pull_equally():
    out = equal_room(_synthetic(0.5, 0.5), tuple(f"zb{i}" for i in range(10)), threshold=np.inf,
                     n_boot=200)
    assert out["valence"]["equal_room"]["estimate"] == pytest.approx(0.0, abs=1e-9)


def test_room_adjusted_contrast_removes_a_baseline_driven_imbalance():
    """Pure geometry: each member moves a fixed fraction of its room, so the raw contrast is driven
    by where the baseline sits and the zero-baseline intercept is ~0."""
    rows = []
    for k, b in enumerate(np.linspace(-0.3, 0.3, 13)):
        path = f"/x/emodb_small/images/{k}.jpg"
        for i in range(4):
            rows.append(dict(image_path=path, image_group="mid", condition="neutral",
                             context_id=f"zb{i}", valence=b))
        for i in range(6):
            rows.append(dict(image_path=path, image_group="mid", condition="negative",
                             context_id=f"mp{i}", valence=b - 0.5 * (b + 1)))
            rows.append(dict(image_path=path, image_group="mid", condition="positive",
                             context_id=f"mp{i}", valence=b + 0.5 * (1 - b)))
    df = pd.DataFrame(rows)
    from src.data.labels import EMOTION_LABELS
    for w in EMOTION_LABELS:
        df[f"lp_{w}"] = np.log(1 / 13)
    out = equal_room(df, tuple(f"zb{i}" for i in range(4)), threshold=np.inf, n_boot=50)
    assert out["valence"]["at_zero_baseline"]["estimate"] == pytest.approx(0.0, abs=1e-9)


def test_equal_room_is_positive_when_negative_pulls_harder():
    out = equal_room(_synthetic(0.6, 0.3), tuple(f"zb{i}" for i in range(10)), threshold=np.inf,
                     n_boot=200)
    assert out["valence"]["equal_room"]["estimate"] == pytest.approx(0.3, abs=1e-9)


def test_screen_is_deterministic_and_thresholded():
    screen = pd.DataFrame({"image_path": [f"p{k}" for k in range(10)],
                           "valence": [-0.9, -0.2, 0.0, 0.1, 0.29, 0.31, 0.5, 0.9, -0.3, 0.25]})
    kept = screen_photos(screen, 0.3, None)
    assert sorted(kept) == ["p1", "p2", "p3", "p4", "p8", "p9"]
    assert screen_photos(screen, 0.3, 3) == screen_photos(screen, 0.3, 3)
    assert len(screen_photos(screen, 0.3, 3)) == 3


# --------------------------------------------------------------------------- stimuli
def test_bank_b_is_disjoint_from_the_published_neutral_pair():
    assert len(set(NEUTRAL_BANK_B)) == len(NEUTRAL_BANK_B) == 10
    assert not set(NEUTRAL_BANK_B) & set(NEUTRAL_CONTEXTS)
    assert all(s.startswith("This photo was taken ") for s in NEUTRAL_BANK_B)
    assert [i for _, i, _ in build_neutral_bank_conditions()] == [f"zb{i}" for i in range(10)]


def test_equal_room_measurement_never_uses_the_screening_sentences():
    """Screen on the published pair, measure on bank B: selection must not define the baseline."""
    screen = {s for _, _, s in build_screen_conditions() if s}
    measure = {s for c, _, s in build_equal_room_conditions() if c == "neutral"}
    assert screen == set(NEUTRAL_CONTEXTS)
    assert measure == set(NEUTRAL_BANK_B)
    pairs = [i for c, i, _ in build_equal_room_conditions() if c in ("positive", "negative")]
    assert sorted(set(pairs)) == [f"mp{i}" for i in range(len(MINIMAL_PAIRS))]


# --------------------------------------------------------------------------- merge guard
def _bank_rows(prefix, overrides=None, drop=()):
    """One photo, both published neutral rows plus all ten bank-B rows, with log-probs."""
    from src.data.labels import EMOTION_LABELS
    overrides = overrides or {}
    rows = []
    for cid in ("z0", "z1", *[f"zb{i}" for i in range(10)]):
        if cid in drop:
            continue
        lp = {f"lp_{w}": np.log(1 / 13) for w in EMOTION_LABELS}
        lp["lp_joy"] = overrides.get(cid, np.log(1 / 13))
        rows.append(dict(image_path=f"{prefix}/emotic/mscoco/images/a.jpg", image_group="positive",
                         condition="neutral", context_id=cid, valence=0.0, **lp))
    return pd.DataFrame(rows)


def _published():
    return _bank_rows("/content/old", drop=[f"zb{i}" for i in range(10)])


def test_merge_maps_remounted_paths_and_checks_reproduction():
    merged, check = merge_neutral_bank(_published(), _bank_rows("/content/new"))
    assert check["max_abs_valence_diff"] == 0.0 and check["max_abs_sum_lo_diff"] == 0.0
    zb = merged[merged["context_id"] == "zb0"]
    assert list(zb["image_path"]) == ["/content/old/emotic/mscoco/images/a.jpg"]
    assert photo_key(zb["image_path"].iloc[0]) == "mscoco/images/a.jpg"


def test_merge_refuses_a_log_odds_drift_that_bounded_valence_hides():
    with pytest.raises(ValueError, match="does not reproduce"):
        merge_neutral_bank(_published(), _bank_rows("/b", overrides={"z0": -1.0}))


def test_merge_refuses_incomplete_coverage():
    with pytest.raises(ValueError, match="coverage"):
        merge_neutral_bank(_published(), _bank_rows("/b", drop=("zb3",)))
    with pytest.raises(ValueError, match="coverage"):
        merge_neutral_bank(_published(), _bank_rows("/b", drop=("z1",)))


def test_merge_refuses_non_finite_log_probs():
    with pytest.raises(ValueError, match="non-finite"):
        merge_neutral_bank(_published(), _bank_rows("/b", overrides={"zb0": np.nan}))


# --------------------------------------------------------------------------- generation parser
@pytest.mark.parametrize("text, label, polarity", [
    ("Joy, not sadness.", "joy", "positive"),
    ("Based on the photo, the person feels sadness rather than joy.", "sadness", "negative"),
    ("Neutral.", "neutral", "neither"),
    ("Surprise!", "surprise", "neither"),
    ("Based on the visual evidence in the photograph", "other", "other"),
])
def test_generation_parser_takes_the_earliest_label_and_matches_forced_choice_polarity(
        text, label, polarity):
    from src.experiments.stage_f_controls import parse_generated
    assert parse_generated(text) == (label, polarity)


def test_mid_valence_pool_is_one_row_per_photo_and_deterministic():
    from src.experiments.stage_f_controls import select_mid_valence_pool
    df = pd.DataFrame({"image_path": [f"p{k // 2}" for k in range(40)],
                       "valence": [float(1 + (k % 10)) for k in range(40)]})
    a = select_mid_valence_pool(df, 8)
    assert a["image_path"].is_unique
    assert a["valence"].between(4, 7).all()
    assert list(a["image_path"]) == list(select_mid_valence_pool(df, 8)["image_path"])
    assert set(a["image_group"]) == {"mid"}


EMOTIC_PQ = Path("data/processed/emotic_test.parquet")


@pytest.mark.skipif(not (EMOTIC_PQ.exists() and MATCHED_PQ.exists()), reason="data not on disk")
def test_pinned_photo_index_is_exactly_the_published_matched_set():
    """The neutral-bank run must score the photos that were published, not a re-selection."""
    from src.experiments.stage_f_controls import select_published_photos
    sel = select_published_photos(pd.read_parquet(EMOTIC_PQ))
    published = pd.read_parquet(MATCHED_PQ)
    published = published[published["condition"] == "none"]
    assert set(sel["image_path"].map(photo_key)) == set(published["image_path"].map(photo_key))
    for group in ("positive", "negative"):
        assert (sel["image_group"] == group).sum() == \
            published[published["image_group"] == group]["image_path"].nunique()
