#!/usr/bin/env python
"""One-shot Colab session setup. Run once at the start of every Colab session.

Colab runtimes are EPHEMERAL — packages, model weights, and outputs are wiped when the
runtime recycles. This script rebuilds the session:
  1. confirm we're on a Colab GPU runtime
  2. install requirements.txt (skips torch — keep Colab's CUDA-matched build)
  3. load HF_TOKEN from Colab Secrets (the key icon), fall back to env
  4. optionally mount Google Drive and redirect data/ + results/ there so they persist
  5. run scripts/check_environment.py

Usage (from the repo root on the runtime):
    !python scripts/colab_bootstrap.py                # outputs stay on ephemeral disk
    !python scripts/colab_bootstrap.py --drive        # persist data/ + results/ to Drive
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE_PROJECT = "/content/drive/MyDrive/algoverse-appraisal"  # persistent home on Drive


def _on_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def install_requirements() -> None:
    # torch is pinned >=2.2 (lower bound only) so Colab's CUDA build satisfies it and pip
    # won't swap it out — reinstalling torch on Colab often breaks CUDA.
    req = ROOT / "requirements.txt"
    print(f"[deps] pip install -r {req}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)


def load_hf_token() -> bool:
    if os.environ.get("HF_TOKEN"):
        print("[token] HF_TOKEN already set in env")
        return True
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[token] loaded HF_TOKEN from Colab Secrets")
            return True
    except Exception as e:  # noqa: BLE001 - secret missing / not on Colab
        print(f"[token] could not read Colab Secret: {e}")
    print("[token] HF_TOKEN NOT set — add it via the Colab key icon (name: HF_TOKEN, "
          "notebook access ON). Gemma is gated and will not load without it.")
    return False


def mount_drive_and_link() -> None:
    """Persist the SMALL, worth-keeping dirs to Drive via symlink.

    Only `results/` and `data/processed/` are persisted — they're small and expensive to
    recompute. `data/raw/` stays on Colab's fast local disk (large image sets read slowly
    from mounted Drive; raw archives live in Drive and are re-extracted per session).
    """
    import shutil

    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
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
    ap.add_argument("--drive", action="store_true", help="persist data/ + results/ to Google Drive")
    ap.add_argument("--skip-deps", action="store_true", help="skip pip install (already done)")
    args = ap.parse_args()

    if not _on_colab():
        print("[warn] this does not look like a Colab runtime; continuing anyway")

    if not args.skip_deps:
        install_requirements()
    load_hf_token()
    if args.drive:
        mount_drive_and_link()

    print("\n[check] running scripts/check_environment.py ...\n")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_environment.py")])
    print("\nBootstrap done. Next: !python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
