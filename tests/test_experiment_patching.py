import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.labels import EMOTION_LABELS
from src.experiments.shared.patching import (
    aligned_patch_groups,
    behavioral_same_image_recovery,
    bridge_patch_hook,
    cross_image_groups,
    cross_image_recovery,
    find_subsequence,
    probe_recovery_valid,
    same_image_recovery,
    segment_prompt_positions,
    stash_activation,
)
from src.experiments.shared.hf_runtime import patch_residuals
from src.experiments.shared.readouts import bridge_probe_and_logits, bridge_probe_readout
from src.experiments.shared.reporting import (
    cross_image_metrics,
    cross_image_verdict,
    print_cross_image_report,
    same_image_verdict,
)


def _segment(context_start, question_start, n=14):
    return {
        "image": np.array([2, 3, 4]),
        "context": np.arange(context_start, question_start),
        "question": np.array([question_start, question_start + 1]),
        "n": n,
    }


class _Tokenizer:
    def __init__(self, encoded_question):
        self.encoded_question = encoded_question

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.encoded_question)


def _token_ids():
    return {label: index for index, label in enumerate(EMOTION_LABELS)}


def _import_with_blocked_package(module_name, blocked_package):
    script = f"""
import importlib
import importlib.abc
import sys
import types

sys.modules["torch"] = types.ModuleType("torch")

class BlockedPackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {blocked_package!r} or fullname.startswith({blocked_package!r} + "."):
            raise ModuleNotFoundError("blocked import dependency: " + fullname)
        return None

sys.meta_path.insert(0, BlockedPackageFinder())
importlib.import_module({module_name!r})
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )


def test_cross_hf_import_does_not_require_transformerbridge_package():
    result = _import_with_blocked_package(
        "src.experiments.stage_f_cross_patching_hf", "src.bridge"
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", [
    "src.experiments.stage_f_patching_hf",
    "src.experiments.stage_f_cross_patching_hf",
])
def test_raw_hf_patching_imports_do_not_require_sklearn(module_name):
    result = _import_with_blocked_package(module_name, "sklearn")

    assert result.returncode == 0, result.stderr


def test_find_subsequence_and_segment_prompt_positions_preserve_token_boundaries():
    token_values = [10, 11, 99, 99, 99, 20, 21, 30, 31, 40, 41]
    input_ids = torch.tensor([token_values])

    segment = segment_prompt_positions(
        _Tokenizer([30, 31]), input_ids, "Question?", expected_image_tokens=3
    )

    assert find_subsequence(token_values, [30, 31]) == 7
    assert find_subsequence(token_values, []) is None
    assert list(segment) == [
        "image", "context", "question", "template", "n", "img_len", "question_ok", "image_ok"
    ]
    assert segment["image"].tolist() == [2, 3, 4]
    assert segment["context"].tolist() == [5, 6]
    assert segment["question"].tolist() == [7, 8]
    assert segment["template"].tolist() == [0, 1, 7, 8, 9, 10]
    assert segment["img_len"] == 3
    assert segment["question_ok"] is True
    assert segment["image_ok"] is True


def test_segment_prompt_positions_preserves_fallback_and_variable_image_count_semantics():
    token_values = [10, 11, 99, 99, 99, *range(20, 35)]
    segment = segment_prompt_positions(
        _Tokenizer([88]), torch.tensor([token_values]), "Missing?", expected_image_tokens=None
    )

    assert segment["context"].tolist() == [5, 6, 7]
    assert segment["question"].dtype == np.dtype(int)
    assert segment["question"].tolist() == []
    assert segment["question_ok"] is False
    assert segment["image_ok"] is True


def test_aligned_groups_exclude_context_and_final_query_token():
    donor = _segment(5, 7, n=12)
    recipient = _segment(5, 8, n=13)
    groups, ok = aligned_patch_groups(donor, recipient)
    assert list(groups) == [
        "image", "question", "structure", "bos", "prefix_delim", "suffix_delim", "text_all"
    ]
    assert all(ok[name] for name in ("image", "question", "structure", "text_all"))
    assert groups["image"][0].tolist() == [2, 3, 4]
    assert 5 not in groups["text_all"][0]
    assert 11 not in groups["text_all"][0]
    assert 12 not in groups["text_all"][1]


def test_cross_groups_and_bootstrap_recovery_are_deterministic():
    segment = _segment(5, 7, n=12)
    groups, ok = cross_image_groups(segment, expected_image_tokens=3)
    assert list(groups) == ["image", "context", "question", "structure", "text_all", "all"]
    assert all(ok.values())
    assert 11 not in groups["all"]

    frame = pd.DataFrame({
        "pos_probe": [2.0, 4.0], "neg_probe": [0.0, 0.0],
        "pos_val": [1.0, 1.0], "neg_val": [-1.0, -1.0],
        **{f"patch_{name}_probe": [1.0, 2.0] for name in groups},
        **{f"patch_{name}_val": [0.0, 0.0] for name in groups},
    })
    recovery = cross_image_recovery(frame, tuple(groups), n_boot=20, seed=3)
    repeated = cross_image_recovery(frame, tuple(groups), n_boot=20, seed=3)
    assert recovery == repeated
    assert recovery["image"]["probe"] == pytest.approx(0.5)
    assert recovery["image"]["val"] == pytest.approx(0.5)
    assert recovery["image"]["probe_ci95"] == recovery["context"]["probe_ci95"]
    assert probe_recovery_valid([13, 14, 17], 18) is True
    assert probe_recovery_valid([18], 18) is False


def test_same_image_recovery_preserves_schema_and_group_order():
    frame = pd.DataFrame({
        "pos_probe": [4.0, 6.0], "neg_probe": [0.0, 2.0],
        "pos_val": [1.0, 1.0], "neg_val": [-1.0, -1.0],
        "patch_second_probe": [2.0, 4.0], "patch_second_val": [0.0, 0.0],
        "patch_first_probe": [1.0, 3.0], "patch_first_val": [-0.5, -0.5],
    })

    recovery = same_image_recovery(frame, ("second", "first"))

    assert list(recovery) == ["pos_probe", "neg_probe", "pos_val", "neg_val", "second", "first"]
    assert recovery["second"] == {"probe": pytest.approx(0.5), "val": pytest.approx(0.5)}
    assert recovery["first"] == {"probe": pytest.approx(0.25), "val": pytest.approx(0.25)}


def test_behavioral_recovery_bootstraps_paired_image_rows():
    """Dropping baseline pairing would give the wrong recovery under heterogeneous image gaps."""
    frame = pd.DataFrame({
        "pos_val": [1.0, 3.0],
        "neg_val": [0.0, 1.0],
        "patch_image_val": [0.5, 2.0],
        "patch_text_all_val": [1.0, 3.0],
    })

    recovery = behavioral_same_image_recovery(
        frame, ("image", "text_all"), n_boot=20, seed=7
    )

    assert recovery["pos_val"] == 2.0
    assert recovery["neg_val"] == 0.5
    assert recovery["image"]["val"] == pytest.approx(0.5)
    assert recovery["text_all"]["val"] == pytest.approx(1.0)
    assert len(recovery["image"]["ci95"]) == 2


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])


class _TinyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()


def test_raw_hf_patch_broadcasts_donor_prompt_state_to_every_label_batch_row():
    """Patching only row zero would leave 12 of 13 teacher-forced labels unpatched."""
    model = _TinyVLM()
    donor = {0: torch.tensor([
        [10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]
    ])}
    recipient = torch.zeros((3, 5, 2))

    with patch_residuals(
        model,
        donor,
        donor_indices=[1, 3],
        recipient_indices=[0, 2],
        patch_all_batch_rows=True,
    ):
        patched = model.language_model.layers[0](recipient)

    assert patched[:, 0, :].tolist() == [[20.0, 21.0]] * 3
    assert patched[:, 2, :].tolist() == [[40.0, 41.0]] * 3
    assert patched[:, 1, :].tolist() == [[0.0, 0.0]] * 3


def test_same_image_verdict_preserves_sink_interpretation():
    recovery = {
        group: {"probe": value, "val": value}
        for group, value in {
            "image": 0.0, "question": 0.2, "bos": 0.0, "prefix_delim": 0.0,
            "suffix_delim": 0.6, "structure": 0.6, "text_all": 0.8,
        }.items()
    }

    verdict = same_image_verdict(recovery)

    assert "IMAGE tokens causally INERT" in verdict
    assert "dominant sink = suffix-delims (60%)" in verdict
    assert "~20% remains in the unpatched CONTEXT tokens" in verdict


def test_bridge_hooks_preserve_batch_and_position_indexing():
    activation = torch.zeros((2, 3, 2), dtype=torch.float32)
    donor = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
    patched = bridge_patch_hook([1], donor)(activation, hook=None)

    assert patched[0, 1].tolist() == [5.0, 6.0]
    assert patched[0, 0].tolist() == [0.0, 0.0]
    assert patched[1, 1].tolist() == [0.0, 0.0]

    store = {}
    assert stash_activation(store)(patched, hook=None) is patched
    assert store["act"].data_ptr() == patched.data_ptr()


class _Bridge:
    def __init__(self):
        self.activation = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]]])
        self.logits = torch.zeros((1, 3, len(EMOTION_LABELS)))
        self.logits[0, -1, EMOTION_LABELS.index("joy")] = 3.0
        self.seen_hook_names = []

    def run_with_hooks(self, input_ids, pixel_values, fwd_hooks):
        assert input_ids.shape[-1] == 3
        assert pixel_values == "pixels"
        for name, hook in fwd_hooks:
            self.seen_hook_names.append(name)
            if name == "tap":
                hook(self.activation, hook=None)
            else:
                hook(self.activation, hook=None)
        return self.logits


def test_bridge_readouts_use_the_last_prompt_token_and_same_forward_hooks():
    bridge = _Bridge()
    ids = torch.tensor([[5, 6, 7]])
    coef = np.array([2.0, -1.0])
    token_ids = _token_ids()
    extra_seen = []

    def extra_hook(activation, hook):
        extra_seen.append(activation.shape)
        return activation

    probe, valence = bridge_probe_readout(
        bridge, ids, "pixels", "tap", token_ids, coef, 0.5,
        extra_hooks=[("extra", extra_hook)],
    )
    assert probe == pytest.approx(2.5)
    assert valence > 0
    assert bridge.seen_hook_names == ["tap", "extra"]
    assert extra_seen == [torch.Size([1, 3, 2])]

    probe, valence, logprobs = bridge_probe_and_logits(
        bridge, ids, "pixels", "tap", coef, 0.5, token_ids,
        steering_hooks=[("steer", extra_hook)],
    )
    assert probe == pytest.approx(2.5)
    assert valence > 0
    assert list(logprobs) == list(EMOTION_LABELS)
    assert max(logprobs, key=logprobs.get) == "joy"


def test_runner_facades_accept_legacy_bridge_helper_keywords():
    from src.experiments import stage_f_attribution, stage_f_conflict, stage_f_patching

    bridge = _Bridge()
    ids = torch.tensor([[5, 6, 7]])
    coef = np.array([2.0, -1.0])
    tok_ids = _token_ids()

    probe, valence = stage_f_patching._readout(
        bridge=bridge, ids=ids, pv="pixels", name="tap", tok_ids=tok_ids,
        coef=coef, inter=0.5, extra_hooks=None,
    )
    assert probe == pytest.approx(2.5)
    assert valence > 0

    probe, valence, logprobs = stage_f_conflict._probe_and_logits(
        bridge=bridge, ids=ids, pv="pixels", name="tap", coef=coef, inter=0.5,
        tok_ids=tok_ids, steer_hooks=None,
    )
    assert probe == pytest.approx(2.5)
    assert valence > 0
    assert max(logprobs, key=logprobs.get) == "joy"
    assert stage_f_attribution._find_subseq(hay=[1, 2, 3], needle=[2, 3]) == 1


def test_cross_image_reporting_preserves_metrics_and_print_contract(capsys, tmp_path):
    recovery = {
        "pos_probe": 2.0, "neg_probe": 0.0, "pos_val": 1.0, "neg_val": -1.0,
        "n_pairs": 2,
        **{
            group: {
                "probe": 0.75, "probe_ci95": [0.5, 1.0],
                "val": 0.5, "val_ci95": [0.25, 0.75],
            }
            for group in ("image", "context", "question", "structure", "text_all", "all")
        },
    }
    metrics = cross_image_metrics(
        recovery, [13, 14, 17], 18, "neutral sentence", "neutral", 2, 1, 0,
        run_stamp="run-1", git_hash="abc123",
    )

    assert list(metrics) == [
        "run", "git", "critical_layer", "patch_layers", "n_pairs", "n_skipped",
        "n_segmentation_dropped", "context_polarity", "context", "recovery", "probe_valid",
        "verdict", "design",
    ]
    assert metrics["run"] == "run-1"
    assert metrics["git"] == "abc123"
    assert metrics["probe_valid"] is True
    assert metrics["verdict"] == cross_image_verdict(recovery, [13, 14, 17], 18)

    data_path = tmp_path / "cross.parquet"
    metrics_path = tmp_path / "cross_metrics.json"
    print_cross_image_report(recovery, metrics, [13, 14, 17], data_path, metrics_path)
    output = capsys.readouterr().out
    assert "Stage F CROSS-IMAGE patching" in output
    assert f"data -> {data_path}" in output
    assert f"metrics -> {metrics_path}" in output
