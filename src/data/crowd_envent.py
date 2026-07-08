"""crowd-enVENT loader (text, Stage A).

Free download: https://www.romanklinger.de/data-sets/crowd-enVent2023.zip
Expected layout after `scripts/download_data.py --dataset crowd-envent`:
    data/raw/crowd-enVent2023/   (extracted TSVs)
Predefined splits: train 4,320 / val 1,080 / test 1,200 (docs/datasets.md).

The exact TSV filenames in the release can vary; `_find_split_file` matches on the
split name so we don't hardcode a brittle path. Appraisal columns are the 1-5 Likert
ratings; `APPRAISAL_TARGETS` names the six we probe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import RAW_DIR
from .labels import APPRAISAL_TARGETS

ENVENT_DIR = RAW_DIR / "crowd-enVent2023"
SPLITS = ("train", "val", "test")


def _find_split_file(split: str, root: Path = ENVENT_DIR) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python scripts/download_data.py --dataset crowd-envent"
        )
    # Match files like *train*.tsv / *validation*.tsv; 'val' also matches 'validation'.
    needle = "valid" if split == "val" else split
    matches = sorted(p for p in root.rglob("*.tsv") if needle in p.name.lower())
    if not matches:
        raise FileNotFoundError(f"no *{needle}*.tsv under {root}")
    return matches[0]


def load_split(split: str, text_col: str = "generated_text") -> pd.DataFrame:
    """Load one split as a DataFrame with a normalized `text` column.

    Keeps the emotion label and the appraisal-target columns when present. The source
    column names differ slightly across releases; `text_col` can be overridden.
    """
    path = _find_split_file(split)
    df = pd.read_csv(path, sep="\t")

    if text_col not in df.columns:
        # Fall back to the first text-like column.
        candidates = [c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()]
        if not candidates:
            raise KeyError(f"no text column found in {path.name}; columns={list(df.columns)}")
        text_col = candidates[0]

    df = df.rename(columns={text_col: "text"})
    keep = ["text"]
    if "emotion" in df.columns:
        keep.append("emotion")
    keep += [c for c in APPRAISAL_TARGETS if c in df.columns]
    return df[keep].reset_index(drop=True)


def sample_tak_subset(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Tak et al. sampling: 500/emotion, except guilt and shame (250 each).

    Requires an `emotion` column. Sampling is capped at the available count per class.
    """
    if "emotion" not in df.columns:
        raise KeyError("sample_tak_subset needs an 'emotion' column")
    caps = {"guilt": 250, "shame": 250}
    parts = []
    for emo, group in df.groupby("emotion"):
        n = min(caps.get(emo, 500), len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
