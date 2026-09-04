"""EMOTIC image-root discovery (synthetic tree — no real images needed)."""
import importlib.util
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "locate_emotic", Path(__file__).resolve().parents[1] / "scripts" / "locate_emotic.py")
locate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(locate)


def _rows():
    return pd.DataFrame([
        {"folder": "framesdb/images", "filename": "a.jpg"},
        {"folder": "framesdb/images", "filename": "b.jpg"},
        {"folder": "mscoco/images", "filename": "c.jpg"},
    ])


def _stage(root: Path, rows: pd.DataFrame, folders):
    for _, r in rows.iterrows():
        if r.folder not in folders:
            continue
        path = root / r.folder / r.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_finds_a_nested_root(tmp_path):
    root = tmp_path / "deep" / "nest" / "emotic"
    _stage(root, _rows(), {"framesdb/images", "mscoco/images"})
    assert root in locate.candidate_roots(tmp_path)


def test_coverage_reports_per_corpus_shortfall(tmp_path):
    """The realistic partial case: EMOTIC ships framesdb, mscoco images do not."""
    root = tmp_path / "emotic"
    rows = _rows()
    _stage(root, rows, {"framesdb/images"})
    fraction, per = locate.coverage(rows, root)
    assert fraction == pytest_approx(2 / 3)
    assert per["framesdb/images"] == (2, 2)
    assert per["mscoco/images"] == (0, 1)


def test_coverage_is_zero_for_an_unrelated_root(tmp_path):
    fraction, _ = locate.coverage(_rows(), tmp_path / "nothing-here")
    assert fraction == 0.0


def test_no_candidates_when_nothing_is_staged(tmp_path):
    (tmp_path / "unrelated" / "images").mkdir(parents=True)
    assert locate.candidate_roots(tmp_path) == []


def pytest_approx(value):
    import pytest
    return pytest.approx(value)
