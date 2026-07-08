"""crowd-enVENT split + subset logic (synthetic corpus — no download needed)."""
import numpy as np
import pandas as pd

from src.data import crowd_envent as ce
from src.data.labels import APPRAISAL_TARGETS, EMOTION_LABELS


def _fake_corpus(n_per_emotion: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for emo in EMOTION_LABELS:
        for _ in range(n_per_emotion):
            row = {"text": "some event", "emotion": emo}
            for a in APPRAISAL_TARGETS:
                row[a] = rng.integers(1, 6)
            rows.append(row)
    return pd.DataFrame(rows)


def test_make_splits_sizes_and_disjoint():
    # 13 emotions x ~508 ≈ 6604; use exactly 6600 to hit canonical sizes.
    df = _fake_corpus().sample(n=6600, random_state=0).reset_index(drop=True)
    splits = ce.make_splits(df, seed=0)
    assert len(splits["train"]) == 4320
    assert len(splits["val"]) == 1080
    assert len(splits["test"]) == 1200
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(df)


def test_make_splits_deterministic():
    df = _fake_corpus().sample(n=6600, random_state=0).reset_index(drop=True)
    a = ce.make_splits(df, seed=0)["train"]
    b = ce.make_splits(df, seed=0)["train"]
    assert a.equals(b)


def test_sample_tak_subset_caps_guilt_and_shame():
    df = _fake_corpus(n_per_emotion=600)
    sub = ce.sample_tak_subset(df, seed=0)
    counts = sub["emotion"].value_counts()
    assert counts["guilt"] == 250 and counts["shame"] == 250
    assert counts["joy"] == 500
