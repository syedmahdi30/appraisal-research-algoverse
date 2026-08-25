from pathlib import Path

import pytest

from src.experiments.shared import artifacts
from src.experiments.shared.artifacts import (
    ensure_output_available,
    llava_artifact_paths,
    token_budget_key,
    token_budget_metric_paths,
)


def test_token_budget_keys_and_bank_filtered_glob(tmp_path):
    model = "Qwen/Qwen3-VL-8B-Instruct"
    assert token_budget_key(model, None) == "qwen3-vl-8b-instruct"
    assert token_budget_key(model, 448, "legacy", "minimal") == (
        "qwen3-vl-8b-instruct_px448_legacy_minimal"
    )
    for name in (
        "conflict_qwen3-vl-8b-instruct_metrics.json",
        "conflict_qwen3-vl-8b-instruct_px448_metrics.json",
        "conflict_qwen3-vl-8b-instruct_px448_minimal_metrics.json",
    ):
        (tmp_path / name).write_text("{}")
    assert [path.name for path in token_budget_metric_paths(tmp_path, model)] == [
        "conflict_qwen3-vl-8b-instruct_metrics.json",
        "conflict_qwen3-vl-8b-instruct_px448_metrics.json",
    ]


def test_llava_paths_preserve_legacy_names_and_reject_unknown_mode(tmp_path):
    default = "llava-hf/llava-1.5-7b-hf"
    paths = llava_artifact_paths(tmp_path, "sequence", False, default, default)
    assert paths == (
        tmp_path / "conflict_llava_sequence.parquet",
        tmp_path / "conflict_llava_sequence_metrics.json",
    )
    with pytest.raises(ValueError, match="unknown score mode"):
        llava_artifact_paths(tmp_path, "average", False, default, default)


def test_collision_guard_preserves_the_callers_exact_error(tmp_path):
    output = tmp_path / "published.parquet"
    output.write_bytes(b"published")
    message = f"{output} already exists — refusing to overwrite a completed run."
    with pytest.raises(FileExistsError, match="refusing to overwrite") as error:
        ensure_output_available(output, force=False, message=message)
    assert str(error.value) == message
    ensure_output_available(output, force=True, message=message)


def test_artifact_metadata_preserves_provenance_key_order(monkeypatch):
    monkeypatch.setattr(artifacts, "run_stamp", lambda: "run-1")
    monkeypatch.setattr(artifacts, "git_hash", lambda: "abc123")
    metadata = artifacts.artifact_metadata(model="model-id", n_rows=4)
    assert list(metadata) == ["run", "git", "model", "n_rows"]
    assert metadata == {"run": "run-1", "git": "abc123", "model": "model-id", "n_rows": 4}
