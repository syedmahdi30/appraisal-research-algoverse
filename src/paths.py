"""Canonical filesystem locations for the project.

Import these instead of hardcoding paths so scripts, experiments, and tests agree.
"""
from __future__ import annotations

import os
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
STAGE_D_DIR = RESULTS_DIR / "stage_d"
STAGE_E_DIR = RESULTS_DIR / "stage_e"
STAGE_F_DIR = RESULTS_DIR / "stage_f"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the writable output trees if they do not already exist.

    `colab_bootstrap.py --drive` replaces `results/` and `data/processed/` with symlinks into Google
    Drive. When the Drive FUSE mount drops mid-session those symlinks dangle, and `mkdir(exist_ok=True)`
    then raises a bare `FileExistsError` because it re-raises when the path is not a directory --
    pointing at the wrong problem entirely. Creating the directory would be worse than failing: it
    would shadow the Drive link and write results somewhere they are silently lost on the next mount.
    """
    for d in (RAW_DIR, PROCESSED_DIR, STAGE_A_DIR, STAGE_C_DIR, STAGE_D_DIR,
              STAGE_E_DIR, STAGE_F_DIR, FIGURES_DIR):
        if d.is_symlink() and not d.exists():
            raise RuntimeError(
                f"{d} is a dangling symlink -> {os.readlink(d)}. The Google Drive mount has "
                f"dropped (this is also what produces 'OSError: [Errno 107] Transport endpoint is "
                f"not connected'). Remount it and re-run:\n"
                f"    from google.colab import drive; drive.mount('/content/drive', force_remount=True)")
        d.mkdir(parents=True, exist_ok=True)
