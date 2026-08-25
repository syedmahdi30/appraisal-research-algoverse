"""Pure artifact keys, paths, collision guards, and provenance metadata for experiment runners."""
from __future__ import annotations

import re
from pathlib import Path

from ..common import git_hash, run_stamp


def model_key(model_name: str) -> str:
    """Filesystem-safe key for the final component of a model identifier."""
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower().split("/")[-1]).strip("-")


def run_key_suffix(style: str = "chat", bank: str = "full") -> str:
    """Style/bank key tail, in its historical ordering."""
    suffix = "" if style == "chat" else f"_{style}"
    return suffix if bank == "full" else suffix + f"_{bank}"


def token_budget_key(model_name: str, max_side: int | None,
                     style: str = "chat", bank: str = "full") -> str:
    """Historical model, token-budget, style, and bank artifact key.

    Every component is in the key so that runs cannot collide. `bank` in particular: a minimal-pair
    run and a full-bank run of the same model are different experiments, and a shared path would
    silently overwrite one with the other — the failure mode that destroyed three published numbers
    before per-run paths were introduced. Do not shorten this key to "simplify" a filename.
    """
    key = model_key(model_name)
    if max_side:
        key = f"{key}_px{max_side}"
    return key + run_key_suffix(style, bank)


def token_budget_artifact_paths(root: Path, model_name: str, max_side: int | None,
                                style: str = "chat", bank: str = "full", *,
                                text_only: bool = False) -> tuple[Path, Path]:
    """Return the historical token-budget data and metrics paths for one run."""
    prefix = "text_only" if text_only else "conflict"
    key = token_budget_key(
        model_name, None if text_only else max_side, style=style, bank=bank
    )
    stem = f"{prefix}_{key}"
    return root / f"{stem}.parquet", root / f"{stem}_metrics.json"


def token_budget_metric_paths(root: Path, model_name: str, style: str = "chat",
                              bank: str = "full") -> list[Path]:
    """Return this model's base metrics, limited to the requested context bank."""
    core = model_key(model_name)
    tail = run_key_suffix(style, bank)
    exact = root / f"conflict_{core}{tail}_metrics.json"
    runs = [exact] if exact.exists() else []
    # The glob alone is not enough: for the full bank `tail` is empty, so `_px*_metrics.json` also
    # swallows `_px448_minimal_metrics.json`. Anchor the budget tag to digits and the tail to the end.
    pattern = re.compile(rf"^conflict_{re.escape(core)}_px\d+{re.escape(tail)}_metrics\.json$")
    return runs + sorted(path for path in root.glob(f"conflict_{core}_px*_metrics.json")
                         if pattern.match(path.name))


def llava_artifact_paths(root: Path, score_mode: str, text_only: bool, model_name: str,
                         default_model: str) -> tuple[Path, Path]:
    """Return LLaVA's historical data and metrics paths for one scoring configuration."""
    if score_mode not in {"first-subtoken", "sequence"}:
        raise ValueError(f"unknown score mode: {score_mode}")
    prefix = "text_only" if text_only else "conflict"
    suffix = "" if score_mode == "first-subtoken" else "_sequence"
    key = "llava" if model_name == default_model else model_key(model_name)
    stem = f"{prefix}_{key}{suffix}"
    return root / f"{stem}.parquet", root / f"{stem}_metrics.json"


def ensure_output_available(path: Path, force: bool, message: str) -> None:
    """Raise the caller's exact collision message unless replacement is explicitly allowed."""
    if path.exists() and not force:
        raise FileExistsError(message)


def artifact_metadata(**fields) -> dict:
    """Insert stable run provenance before the caller-supplied metric fields."""
    return {"run": run_stamp(), "git": git_hash(), **fields}
