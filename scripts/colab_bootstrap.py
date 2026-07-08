#!/usr/bin/env python
"""One-shot Colab session setup (deps + optional Drive persistence + env check).

IMPORTANT: `google.colab.drive.mount` and `google.colab.userdata` only work inside a
notebook CELL (they need the live kernel); they crash in a `!python` subprocess. So this
script does NOT mount Drive or read Secrets itself. Do those in a cell FIRST — then this
script (and every other `!python` call) inherits the mounted drive and the HF_TOKEN env:

    # --- run this in a notebook cell, once per session ---
    from google.colab import userdata, drive
    import os
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # token from the Colab key icon
    drive.mount("/content/drive")

Then, from the repo root:
    !python scripts/colab_bootstrap.py            # deps + env check
    !python scripts/colab_bootstrap.py --drive    # + persist results/ & data/processed to Drive
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE_PROJECT = "/content/drive/MyDrive/algoverse-appraisal"  # persistent home on Drive


def _on_colab() -> bool:
    return Path("/content").exists()


def install_requirements() -> None:
    # torch is pinned >=2.2 (lower bound only) so Colab's CUDA build satisfies it and pip
    # won't swap it out — reinstalling torch on Colab often breaks CUDA.
    req = ROOT / "requirements.txt"
    print(f"[deps] pip install -r {req}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)


def check_hf_token() -> bool:
    if os.environ.get("HF_TOKEN"):
        print("[token] HF_TOKEN present in env")
        return True
    print("[token] HF_TOKEN NOT set. In a NOTEBOOK CELL (not !python) run:\n"
          "    from google.colab import userdata; import os\n"
          "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
          "  (add the token first via the Colab key icon, name HF_TOKEN, notebook access ON).\n"
          "  Gemma is gated and will not load without it.")
    return False


def link_drive() -> None:
    """Symlink the small persistent dirs to Drive — IF Drive is already mounted in a cell.

    Persists only `results/` and `data/processed/` (small, expensive to recompute).
    `data/raw/` stays on fast local disk. Does nothing (with guidance) if Drive isn't
    mounted, rather than crashing — mounting needs the notebook kernel.
    """
    if not Path("/content/drive/MyDrive").exists():
        print("[drive] /content/drive not mounted. In a NOTEBOOK CELL run:\n"
              "    from google.colab import drive; drive.mount('/content/drive')\n"
              "  then re-run with --drive. Skipping persistence for now.")
        return
    for sub in ("results", "data/processed"):
        target = Path(DRIVE_PROJECT) / sub
        target.mkdir(parents=True, exist_ok=True)
        link = ROOT / sub
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        elif link.exists():  # replace the empty local dir with a symlink
            shutil.rmtree(link, ignore_errors=True)
        link.symlink_to(target, target_is_directory=True)
        print(f"[drive] {sub}/ -> {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Colab session bootstrap")
    ap.add_argument("--drive", action="store_true", help="persist results/ + data/processed to Drive")
    ap.add_argument("--skip-deps", action="store_true", help="skip pip install (already done)")
    args = ap.parse_args()

    if not _on_colab():
        print("[warn] this does not look like a Colab runtime; continuing anyway")

    if not args.skip_deps:
        install_requirements()
    check_hf_token()
    if args.drive:
        link_drive()

    print("\n[check] running scripts/check_environment.py ...\n")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_environment.py")])
    print("\nBootstrap done. Next: !python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
