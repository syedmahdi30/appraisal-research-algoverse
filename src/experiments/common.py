"""Shared helpers for experiment runners: config loading and result serialization."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: str | Path) -> None:
    """Write a metrics dict as pretty JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_probes(probes, path: str | Path) -> None:
    """Persist fitted probes (coef, intercept, steering vectors) as a single .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        names=np.array(probes.names),
        coef=probes.coef,
        intercept=probes.intercept,
        z_unit=probes.z_unit,
        alpha=probes.alpha,
    )


def load_probes(path: str | Path):
    """Load probes saved by `save_probes` into an AppraisalProbes."""
    from ..probes.train import AppraisalProbes

    d = np.load(path, allow_pickle=True)
    return AppraisalProbes(
        names=list(d["names"]),
        coef=d["coef"],
        intercept=d["intercept"],
        z_unit=d["z_unit"],
        alpha=float(d["alpha"]),
    )


def run_stamp() -> str:
    """UTC timestamp for tagging run outputs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")
