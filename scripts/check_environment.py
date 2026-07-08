#!/usr/bin/env python
"""Report the environment contract without loading any model (runs anywhere).

Checks Python, torch/CUDA, transformers, transformer-lens versions, HF_TOKEN presence,
and dataset directory status. Prints a Ready / Ready-with-warnings / Blocked verdict.
See docs/setup.md and the experiment-setup skill.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _version(mod: str) -> str | None:
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return None


def main() -> int:
    warnings: list[str] = []
    blockers: list[str] = []

    print(f"python           : {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        blockers.append("Python >= 3.11 required")

    torch_v = _version("torch")
    print(f"torch            : {torch_v}")
    if torch_v is None:
        blockers.append("torch not installed (pip install -r requirements.txt)")
    else:
        import torch
        cuda = torch.cuda.is_available()
        print(f"cuda available   : {cuda}")
        if cuda:
            print(f"cuda device      : {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA not available — Gemma smoke test / experiments need a GPU")

    tf_v = _version("transformers")
    tl_v = _version("transformer_lens")
    print(f"transformers     : {tf_v}")
    print(f"transformer_lens : {tl_v}")
    if tf_v is None:
        blockers.append("transformers not installed")
    if tl_v is None:
        blockers.append("transformer-lens not installed")

    for mod in ("sklearn", "pandas", "numpy", "yaml", "PIL"):
        if _version(mod) is None and importlib.util.find_spec(mod) is None:
            blockers.append(f"{mod} not installed")

    has_token = bool(os.environ.get("HF_TOKEN"))
    print(f"HF_TOKEN set      : {has_token}")
    if not has_token:
        warnings.append("HF_TOKEN not set — Gemma is gated and will not load")

    envent = ROOT / "data" / "raw" / "crowd-enVent2023"
    emotic = ROOT / "data" / "raw" / "emotic"
    print(f"crowd-enVENT      : {'present' if envent.exists() else 'missing'} ({envent})")
    print(f"EMOTIC            : {'present' if emotic.exists() else 'missing'} ({emotic})")
    if not envent.exists():
        warnings.append("crowd-enVENT missing — python scripts/download_data.py --dataset crowd-envent")
    if not emotic.exists():
        warnings.append("EMOTIC missing — gated; submit the form (see docs/datasets.md)")

    print("\n--- verdict ---")
    if blockers:
        print("BLOCKED:")
        for b in blockers:
            print(f"  - {b}")
        return 1
    if warnings:
        print("READY WITH WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        return 0
    print("READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
