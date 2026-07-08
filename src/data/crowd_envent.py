"""crowd-enVENT loader (text, Stage A).

Free download: https://www.romanklinger.de/data-sets/crowd-enVent2023.zip
`scripts/download_data.py --dataset crowd-envent` extracts it to:
    data/raw/corpus/crowd-enVent_generation.tsv   (6,600 self-annotated experiencer rows)
    data/raw/corpus/crowd-enVent_validation.tsv    (reader re-annotations; not used here)
    data/raw/predictions/...                        (paper model outputs; not used here)

Verified columns (generation.tsv, 6600x61): `generated_text`, `emotion`, and the 1-5
appraisal ratings incl. our six targets (APPRAISAL_TARGETS) — see docs/datasets.md.

SPLIT CAVEAT: the download ships ONE corpus, not separate train/val/test files. We derive
a deterministic emotion-stratified split at the canonical sizes (4320/1080/1200) with a
fixed seed. This is reproducible but is NOT guaranteed identical to the paper's partition;
report it as a caveat. `SEED` fixes it so every call/stage sees the same split.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..paths import RAW_DIR
from .labels import APPRAISAL_TARGETS

CORPUS_DIR = RAW_DIR / "corpus"
GENERATION_FILE = CORPUS_DIR / "crowd-enVent_generation.tsv"

SPLITS = ("train", "val", "test")
SPLIT_SIZES = {"train": 4320, "val": 1080, "test": 1200}  # sum = 6600
SEED = 0

TEXT_COL = "generated_text"
EMOTION_COL = "emotion"


def load_generation() -> pd.DataFrame:
    """Load the full generation corpus, normalized to text/emotion/appraisal columns."""
    if not GENERATION_FILE.exists():
        raise FileNotFoundError(
            f"{GENERATION_FILE} not found. Run: python scripts/download_data.py --dataset crowd-envent"
        )
    df = pd.read_csv(GENERATION_FILE, sep="\t")
    missing = [c for c in (TEXT_COL, EMOTION_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns {missing} not in {GENERATION_FILE.name}")

    keep = [TEXT_COL, EMOTION_COL] + [c for c in APPRAISAL_TARGETS if c in df.columns]
    out = df[keep].rename(columns={TEXT_COL: "text", EMOTION_COL: "emotion"})
    return out.dropna(subset=["text", "emotion"]).reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Deterministic emotion-stratified split into train/val/test at the canonical sizes."""
    train, temp = train_test_split(
        df, train_size=SPLIT_SIZES["train"], stratify=df["emotion"], random_state=seed,
    )
    val, test = train_test_split(
        temp, train_size=SPLIT_SIZES["val"], stratify=temp["emotion"], random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def load_split(split: str, seed: int = SEED) -> pd.DataFrame:
    """Load one split of the (deterministically re-derived) crowd-enVENT partition."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return make_splits(load_generation(), seed=seed)[split]


def sample_tak_subset(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Tak et al. sampling: 500/emotion, except guilt and shame (250 each).

    Requires an `emotion` column; per-class sampling is capped at the available count.
    """
    if "emotion" not in df.columns:
        raise KeyError("sample_tak_subset needs an 'emotion' column")
    caps = {"guilt": 250, "shame": 250}
    parts = []
    for emo, group in df.groupby("emotion"):
        n = min(caps.get(emo, 500), len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
