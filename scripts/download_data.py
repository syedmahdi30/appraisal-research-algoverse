#!/usr/bin/env python
"""Fetch/prepare datasets into data/raw/. See docs/datasets.md.

crowd-enVENT is a free direct download and is fully automated. EMOTIC is gated: this
script only extracts an archive you have already obtained via the signed form — it never
fabricates access.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import RAW_DIR, ensure_dirs

ENVENT_URL = "https://www.romanklinger.de/data-sets/crowd-enVent2023.zip"


def download_crowd_envent() -> Path:
    ensure_dirs()
    dest_zip = RAW_DIR / "crowd-enVent2023.zip"
    if not dest_zip.exists():
        print(f"downloading {ENVENT_URL}")
        urllib.request.urlretrieve(ENVENT_URL, dest_zip)
    print(f"extracting {dest_zip.name}")
    with zipfile.ZipFile(dest_zip) as z:
        z.extractall(RAW_DIR)
    print(f"crowd-enVENT ready under {RAW_DIR}")
    return RAW_DIR


def prepare_emotic(archive: str | None, annotations: str | None) -> Path:
    dest = RAW_DIR / "emotic"
    if archive is None:
        raise SystemExit(
            "EMOTIC is gated. Submit the form at https://s3.sunai.uoc.edu/emotic/download.html, "
            "then re-run: python scripts/download_data.py --dataset emotic "
            "--archive /path/to/emotic.zip --annotations /path/to/Annotations.mat"
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting EMOTIC images from {archive}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    print(f"EMOTIC images under {dest}")

    if annotations is None:
        print("no --annotations Annotations.mat given; images extracted but Stage C needs the "
              "labels. Re-run with --annotations /path/to/Annotations.mat to build the tables.")
        return dest

    # Parse Annotations.mat -> data/processed/emotic_{split}.parquet.
    from src.data.emotic import convert_mat_to_parquet

    print(f"converting annotations {annotations}")
    counts = convert_mat_to_parquet(annotations)
    print(f"EMOTIC ready: {counts}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/prepare datasets")
    ap.add_argument("--dataset", required=True, choices=["crowd-envent", "emotic"])
    ap.add_argument("--archive", help="path to the EMOTIC images zip (gated)")
    ap.add_argument("--annotations", help="path to EMOTIC Annotations.mat (gated)")
    args = ap.parse_args()

    if args.dataset == "crowd-envent":
        download_crowd_envent()
    else:
        prepare_emotic(args.archive, args.annotations)


if __name__ == "__main__":
    main()
