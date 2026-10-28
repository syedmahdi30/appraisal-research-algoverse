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
    m = grid["valence"]["mirror"]
    assert m["estimate"] == pytest.approx(0.4964, abs=1e-3)
    assert m["ci95_crossed"] == pytest.approx([0.114, 0.825], abs=2e-3)


@needs_matched
def test_within_item_reproduces_the_published_value(grid):
    assert grid["valence"]["within_pos"]["estimate"] == pytest.approx(1.148, abs=1e-3)


@needs_matched
def test_corrected_category_gap_has_a_code_path_for_its_crossed_interval(grid):
    c = grid["category"]["mirror"]
    assert c["estimate"] == pytest.approx(0.2308, abs=1e-3)
    assert c["ci95_crossed"] == pytest.approx([-0.026, 0.455], abs=2e-3)


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
def _rows(prefix, cid, v):
    return dict(image_path=f"{prefix}/emotic/mscoco/images/a.jpg", image_group="positive",
                condition="neutral", context_id=cid, valence=v)


def test_merge_maps_remounted_paths_and_checks_reproduction():
    matched = pd.DataFrame([_rows("/content/old", "z0", 0.9), _rows("/content/old", "z1", 0.3)])
    bank = pd.DataFrame([_rows("/content/new", "z0", 0.9), _rows("/content/new", "z1", 0.3),
                         _rows("/content/new", "zb0", 0.5)])
    merged, check = merge_neutral_bank(matched, bank)
    assert check["max_abs_valence_diff"] == 0.0
    zb = merged[merged["context_id"] == "zb0"]
    assert list(zb["image_path"]) == ["/content/old/emotic/mscoco/images/a.jpg"]
    assert photo_key(zb["image_path"].iloc[0]) == "mscoco/images/a.jpg"


def test_merge_refuses_a_session_that_does_not_reproduce():
    matched = pd.DataFrame([_rows("/a", "z0", 0.9), _rows("/a", "z1", 0.3)])
    bank = pd.DataFrame([_rows("/b", "z0", 0.7), _rows("/b", "z1", 0.3), _rows("/b", "zb0", 0.5)])
    with pytest.raises(ValueError, match="does not reproduce"):
        merge_neutral_bank(matched, bank)


def test_mid_valence_pool_is_one_row_per_photo_and_deterministic():
    from src.experiments.stage_f_controls import select_mid_valence_pool
    df = pd.DataFrame({"image_path": [f"p{k // 2}" for k in range(40)],
                       "valence": [float(1 + (k % 10)) for k in range(40)]})
    a = select_mid_valence_pool(df, 8)
    assert a["image_path"].is_unique
    assert a["valence"].between(4, 7).all()
    assert list(a["image_path"]) == list(select_mid_valence_pool(df, 8)["image_path"])
    assert set(a["image_group"]) == {"mid"}
