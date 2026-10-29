"""Restore results/ from the public bundle in release/results/.

The bundle ships model outputs only. Columns that carried EMOTIC's own annotations were replaced by
`<column>_emotic_row`, a row position in `data/processed/emotic_test.parquet`. With EMOTIC access
(see README, Data), this script joins them back so every analysis runs on the original schema.

Without EMOTIC, pass --no-emotic: files are copied with the index columns left in place. The
analyses that use only `image_group` and the label log-probabilities -- including every number in
the readout table (Table 2) -- run unchanged; those that need the human valence rating do not.

    python scripts/rehydrate_results.py [--no-emotic] [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "release" / "results", ROOT / "results"
EMOTIC = ROOT / "data" / "processed" / "emotic_test.parquet"
SUFFIX = "_emotic_row"


def photo_key(path) -> str:
    return "/".join(str(path).split("/")[-3:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-emotic", action="store_true", help="copy without restoring EMOTIC columns")
    ap.add_argument("--force", action="store_true", help="overwrite existing files in results/")
    a = ap.parse_args()
    emotic = None
    if not a.no_emotic:
        if not EMOTIC.exists():
            sys.exit(f"{EMOTIC} missing. Build it with scripts/download_data.py --dataset emotic, "
                     f"or pass --no-emotic.")
        emotic = pd.read_parquet(EMOTIC)
        emotic_keys = emotic["image_path"].map(photo_key).to_numpy()
        emotic_val = emotic["valence"].astype(float).to_numpy()
    n = 0
    for src in sorted(p for p in SRC.rglob("*") if p.is_file()):
        dst = DST / src.relative_to(SRC)
        if dst.exists() and not a.force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".parquet" and emotic is not None:
            df = pd.read_parquet(src)
            for col in [c for c in df.columns if c.endswith(SUFFIX)]:
                name = col[: -len(SUFFIX)]
                path_col = {"donor_valence": "donor_path",
                            "recipient_valence": "recipient_path"}.get(name, "image_path")
                rows = df[col].to_numpy()
                # The index is only trusted if it points at the same photograph.
                bad = [k for k, (p, r) in enumerate(zip(df[path_col], rows))
                       if photo_key(p) != emotic_keys[r]]
                if bad:
                    sys.exit(f"{src}: {len(bad)} rows point at a different photograph; your "
                             f"emotic_test.parquet does not match the one the bundle was built on")
                df[name] = emotic_val[rows]
                df = df.drop(columns=[col])
            df.to_parquet(dst)
        else:
            shutil.copy2(src, dst)
        n += 1
    print(f"{n} files -> {DST}" + (" (EMOTIC columns NOT restored)" if a.no_emotic else ""))


if __name__ == "__main__":
    main()
