"""Canonical filesystem locations for the project.

Import these instead of hardcoding paths so scripts, experiments, and tests agree.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
DOCS_DIR = ROOT / "docs"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
STAGE_A_DIR = RESULTS_DIR / "stage_a"
STAGE_C_DIR = RESULTS_DIR / "stage_c"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the writable output trees if they do not already exist."""
    for d in (RAW_DIR, PROCESSED_DIR, STAGE_A_DIR, STAGE_C_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
