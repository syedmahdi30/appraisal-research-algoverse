"""Deterministic EMOTIC valence-extreme sampling helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_valence_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[np.isfinite(df["valence"].to_numpy(dtype=float))].sort_values("valence", kind="stable")


def select_extreme_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the positive extreme first, then the negative extreme, tagged by group."""
    ordered = _finite_valence_rows(df)
    k = n // 2
    low = ordered.head(k).assign(image_group="negative")
    high = ordered.tail(k).assign(image_group="positive")
    return pd.concat([high, low]).reset_index(drop=True)


def select_ranked_pairs(df: pd.DataFrame, n_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return descending-positive and ascending-negative rows for rank-aligned pairing."""
    ordered = _finite_valence_rows(df)
    positive = ordered.tail(n_pairs).sort_values("valence", ascending=False, kind="stable").reset_index(drop=True)
    negative = ordered.head(n_pairs).sort_values("valence", kind="stable").reset_index(drop=True)
    return positive, negative
