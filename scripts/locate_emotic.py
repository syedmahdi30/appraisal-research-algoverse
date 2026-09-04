#!/usr/bin/env python3
"""Find the EMOTIC image root on this machine and print the exact --images-root to use.

`data/processed/emotic_test.parquet` stores absolute `/content/...` paths from the original Colab run,
so any other layout resolves nothing and every control run aborts on coverage. Guessing the root wastes
a GPU session; this searches for it and reports per-corpus coverage of the images the controls actually
need.

    python scripts/locate_emotic.py                     # search /content (Colab default)
    python scripts/locate_emotic.py --search-root ~/data

Prints nothing but findings — it never writes or moves files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments.shared.sampling import select_extreme_rows  # noqa: E402

CORPORA = ("framesdb", "mscoco", "emodb_small", "ade20k")
# EMOTIC ships these two; mscoco/ade20k carry annotations only, and their images come from the
# source datasets. A correct EMOTIC-only extraction therefore cannot reach full coverage.
SHIPPED_WITH_EMOTIC = ("framesdb", "emodb_small")
MAX_DEPTH = 10


def candidate_roots(search_root: Path) -> list[Path]:
    """Directories that contain at least one EMOTIC corpus subdirectory."""
    found: set[Path] = set()
    base_depth = len(search_root.parts)
    for dirpath, dirnames, _files in os.walk(search_root, followlinks=False):
        here = Path(dirpath)
        if len(here.parts) - base_depth >= MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if any(name in CORPORA for name in dirnames):
            found.add(here)
    return sorted(found)


def coverage(rows: pd.DataFrame, root: Path) -> tuple[float, dict[str, tuple[int, int]]]:
    """Fraction of `rows` whose image exists under `root`, plus per-corpus (present, needed)."""
    paths = root.as_posix().rstrip("/") + "/" + rows["folder"].astype(str) + "/" \
        + rows["filename"].astype(str)
    ok = paths.map(os.path.exists)
    per: dict[str, tuple[int, int]] = {}
    for folder, group in rows.assign(_ok=ok.to_numpy()).groupby("folder"):
        per[str(folder)] = (int(group["_ok"].sum()), int(len(group)))
    return float(ok.mean()) if len(rows) else 0.0, per


def main() -> None:
    ap = argparse.ArgumentParser(description="Locate the EMOTIC image root")
    ap.add_argument("--search-root", default="/content", help="directory to search (default /content)")
    ap.add_argument("--parquet", default="data/processed/emotic_test.parquet")
    ap.add_argument("--n-images", type=int, default=150, help="selection size the controls use")
    args = ap.parse_args()

    parquet = Path(args.parquet)
    if not parquet.exists():
        raise SystemExit(f"missing {parquet} — stage the processed EMOTIC parquet first.")
    full = pd.read_parquet(parquet).reset_index(drop=True)
    selected = select_extreme_rows(full, args.n_images)

    baked = float(selected["image_path"].map(os.path.exists).mean())
    print(f"selected rows: {len(selected)} annotations over "
          f"{selected['image_path'].nunique()} images")
    print(f"baked-in paths resolve: {baked:.0%}"
          f"{'  (nothing to do)' if baked > 0.95 else ''}")
    print("\ncorpora the selection needs:")
    for folder, group in selected.groupby("folder"):
        corpus = str(folder).split("/")[0]
        ships = "ships with EMOTIC" if corpus in SHIPPED_WITH_EMOTIC else "SEPARATE download"
        print(f"  {str(folder):22s} {len(group):4d} rows   ({ships})")

    search_root = Path(os.path.expanduser(args.search_root))
    if not search_root.exists():
        raise SystemExit(f"\nsearch root {search_root} does not exist — pass --search-root")
    print(f"\nsearching {search_root} for EMOTIC corpus directories ...")
    roots = candidate_roots(search_root)
    if not roots:
        print("  none found. The images are not under this search root.")
        print("  EMOTIC needs a signed request form; if the archive was never downloaded in this")
        print("  session, no --images-root will help. Re-download or mount the drive holding it.")
        return

    scored = []
    for root in roots:
        fraction, per = coverage(selected, root)
        scored.append((fraction, root, per))
    scored.sort(key=lambda item: -item[0])

    print(f"  {len(roots)} candidate root(s), best first:\n")
    for fraction, root, per in scored[:5]:
        print(f"  {fraction:6.0%}  {root}")
        for folder, (present, needed) in sorted(per.items()):
            flag = "" if present == needed else "   <-- incomplete"
            print(f"            {folder:22s} {present:3d}/{needed:3d}{flag}")

    best_fraction, best_root, _ = scored[0]
    print()
    if best_fraction == 0:
        print("No candidate root resolves any selected image. The directories exist but the files")
        print("under them do not match the parquet — check that the archive finished extracting.")
        return
    print(f"Use:  --images-root {best_root}")
    if best_fraction > 0.95:
        print("Full coverage: run the controls normally.")
    else:
        print(f"Coverage is {best_fraction:.0%}, below the 5% miss the runner allows, so add")
        print("--allow-missing. For --person that stays valid because the ungrounded baseline is")
        print("paired on the same images; report it as a paired contrast on n images, not as a")
        print("replication of the published number.")


if __name__ == "__main__":
    main()
