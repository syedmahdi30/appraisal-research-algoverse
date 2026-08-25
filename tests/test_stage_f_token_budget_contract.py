import sys

import pandas as pd
import pytest

from src.data.conflict_contexts import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS
from src.data.labels import EMOTION_LABELS
from src.experiments import stage_f_token_budget as token_budget
from src.experiments.shared import artifacts, hf_runtime, reporting
from src.experiments.shared.artifacts import run_key_suffix, token_budget_key
from src.experiments.shared.reporting import text_only_readouts, token_budget_trends


QWEN = "Qwen/Qwen3-VL-8B-Instruct"


def test_parser_preserves_option_order_flags_defaults_and_choices():
    parser = token_budget.build_parser()

    assert [action.option_strings for action in parser._actions] == [
        ["-h", "--help"],
        ["--config"],
        ["--model"],
        ["--max-side"],
        ["--limit"],
        ["--force"],
        ["--text-only"],
        ["--bank"],
        ["--prompt-style"],
        ["--show-prompt"],
        ["--reanalyze"],
        ["--aggregate"],
    ]
    args = parser.parse_args([])
    assert vars(args) == {
        "config": "config/stage_f.yaml",
        "model": QWEN,
        "max_side": None,
        "limit": None,
        "force": False,
        "text_only": False,
        "bank": "full",
        "prompt_style": "chat",
        "show_prompt": False,
        "reanalyze": False,
        "aggregate": False,
    }
    configured = parser.parse_args([
        "--config", "custom.yaml", "--model", "model/id", "--max-side", "448",
        "--limit", "3", "--force", "--text-only", "--bank", "minimal",
        "--prompt-style", "legacy", "--show-prompt", "--reanalyze", "--aggregate",
    ])
    assert vars(configured) == {
        "config": "custom.yaml",
        "model": "model/id",
        "max_side": 448,
        "limit": 3,
        "force": True,
        "text_only": True,
        "bank": "minimal",
        "prompt_style": "legacy",
        "show_prompt": True,
        "reanalyze": True,
        "aggregate": True,
    }


def test_base_and_trend_column_order_is_pinned():
    assert token_budget.BASE_RESULT_COLUMNS == [
        "image_path", "image_valence", "image_group", "condition", "context_id",
        "context", "text_code", "probe_readout", "valence",
        *[f"lp_{label}" for label in EMOTION_LABELS],
    ]
    assert token_budget.TOKEN_BUDGET_TREND_COLUMNS == [
        "source", "model", "bank", "max_side", "image_tokens", "image_token_fraction",
        "discriminability_gap", "auc", "text_only_ratio", "override_gap", "ci_lo", "ci_hi",
    ]


def test_shared_artifact_paths_keep_base_and_text_only_names(tmp_path):
    assert artifacts.token_budget_artifact_paths(tmp_path, QWEN, 448) == (
        tmp_path / "conflict_qwen3-vl-8b-instruct_px448.parquet",
        tmp_path / "conflict_qwen3-vl-8b-instruct_px448_metrics.json",
    )
    assert artifacts.token_budget_artifact_paths(
        tmp_path, QWEN, 448, style="legacy", bank="minimal", text_only=True
    ) == (
        tmp_path / "text_only_qwen3-vl-8b-instruct_legacy_minimal.parquet",
        tmp_path / "text_only_qwen3-vl-8b-instruct_legacy_minimal_metrics.json",
    )


def test_shared_reporting_helpers_keep_analysis_order_and_text_only_math(monkeypatch):
    monkeypatch.setattr(reporting, "image_discriminability", lambda _df: {"d": 1})
    monkeypatch.setattr(reporting, "asymmetry_vs_floor", lambda _df: {"a": 2})
    monkeypatch.setattr(reporting, "flip_override", lambda _df: {"f": 3})
    fields = reporting.token_budget_analysis_fields(
        pd.DataFrame([{"image_path": "one.jpg"}]), "model/id", {"image_tokens": 12}, 448,
        multi={"joy": {"single_token": False}}, n_skipped=2,
    )
    assert list(fields) == [
        "model", "max_side", "read_out", "n_images", "n_rows", "n_skipped", "image_tokens",
        "image_discriminability", "asymmetry_vs_floor", "flip_override",
        "tokenization_multi_token",
    ]
    assert fields["n_images"] == 1

    frame = pd.DataFrame([
        {"condition": "none", "valence": -0.1},
        {"condition": "neutral", "valence": 0.1},
        {"condition": "positive", "valence": 0.5},
        {"condition": "negative", "valence": -0.7},
    ])
    summary = reporting.text_only_control_summary(frame)
    assert list(summary) == [
        "neutral_baseline", "none_baseline", "pos_effect", "neg_effect", "pos_raw", "neg_raw",
        "text_only_ratio_vs_neutral", "text_only_ratio_raw", "reference_ratio",
    ]
    assert summary == pytest.approx({
        "neutral_baseline": 0.1,
        "none_baseline": -0.1,
        "pos_effect": 0.4,
        "neg_effect": -0.8,
        "pos_raw": 0.5,
        "neg_raw": -0.7,
        "text_only_ratio_vs_neutral": 2.0,
        "text_only_ratio_raw": 1.4,
        "reference_ratio": 1.4,
    })
    assert reporting.cross_modal_amplification(None, 1.0) == "no base run to compare"
    assert reporting.cross_modal_amplification(1.3, 1.0) == (
        "CROSS-MODAL amplification (image inflates the ratio)"
    )
    assert reporting.cross_modal_amplification(1.2, 1.0) == "STIMULUS confound (ratios match)"
    assert reporting.cross_modal_amplification(0.5, 1.0) == "image dampens (reversed)"


class CapturingProcessor:
    def __init__(self):
        self.messages = None
        self.chat_calls = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.chat_calls += 1
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered-chat"

    def __call__(self, **kwargs):
        return kwargs


def test_chat_prompt_structure_and_wording_are_pinned_for_both_families():
    processor = CapturingProcessor()
    encoded = token_budget.build_inputs(
        processor, "IMAGE", "They won.", family="qwen", style="chat"
    )
    assert processor.messages == [{
        "role": "user",
        "content": [
            {"type": "image", "image": "IMAGE"},
            {"type": "text", "text": (
                "Context: They won. What single emotion is this person feeling?"
            )},
        ],
    }]
    assert encoded == {
        "return_tensors": "pt", "text": ["rendered-chat"],
        "images": ["IMAGE"], "padding": True,
    }

    encoded = token_budget.build_inputs(
        processor, "IMAGE", None, family="llava", style="chat"
    )
    assert processor.messages == [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What single emotion is this person feeling?"},
        ],
    }]
    assert encoded == {
        "return_tensors": "pt", "text": "rendered-chat", "images": ["IMAGE"],
    }


def test_legacy_prompt_keeps_the_published_gemma_scaffold():
    processor = CapturingProcessor()

    encoded = token_budget.build_inputs(
        processor, "IMAGE", "They won.", family="qwen", style="legacy"
    )

    assert processor.chat_calls == 0
    assert encoded == {
        "return_tensors": "pt",
        "text": (
            "<start_of_turn>user\n<start_of_image>Context: They won. "
            "What single emotion is this person feeling?<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "images": ["IMAGE"],
    }


def test_condition_order_and_ids_are_pinned_for_both_banks():
    full = token_budget._conditions("full")
    assert [(condition, context_id) for condition, context_id, _ in full] == [
        ("none", "none"),
        *[("positive", f"p{i}") for i in range(6)],
        *[("negative", f"n{i}") for i in range(6)],
        ("neutral", "z0"),
        ("neutral", "z1"),
    ]
    assert full[1][2] == POSITIVE_CONTEXTS[0]
    assert full[7][2] == NEGATIVE_CONTEXTS[0]

    minimal = token_budget._conditions("minimal")
    assert [(condition, context_id) for condition, context_id, _ in minimal] == [
        ("none", "none"),
        *[
            item
            for i in range(6)
            for item in (("positive", f"mp{i}"), ("negative", f"mp{i}"))
        ],
        ("neutral", "z0"),
        ("neutral", "z1"),
    ]


def test_existing_base_parquet_is_never_overwritten_or_followed_by_model_loading(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(token_budget, "STAGE_F_DIR", tmp_path)
    output = tmp_path / "conflict_qwen3-vl-8b-instruct.parquet"
    output.write_bytes(b"published")
    monkeypatch.setattr(
        token_budget,
        "load_vlm",
        lambda *_args, **_kwargs: pytest.fail("collision guard must precede model loading"),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite") as error:
        token_budget.run_base("config/stage_f.yaml", QWEN, None, force=False)

    assert str(error.value) == (
        f"{output} already exists — refusing to overwrite a completed run. Pass --force to "
        "replace it, or change --max-side / --model so the run gets its own key."
    )
    assert output.read_bytes() == b"published"


def test_existing_text_only_parquet_is_never_overwritten_or_followed_by_model_loading(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(token_budget, "STAGE_F_DIR", tmp_path)
    output = tmp_path / "text_only_qwen3-vl-8b-instruct.parquet"
    output.write_bytes(b"published")
    monkeypatch.setattr(
        token_budget,
        "load_vlm",
        lambda *_args, **_kwargs: pytest.fail("collision guard must precede model loading"),
    )

    with pytest.raises(FileExistsError) as error:
        token_budget.run_text_only("config/stage_f.yaml", QWEN, force=False)

    assert str(error.value) == f"{output} already exists — pass --force to replace it."
    assert output.read_bytes() == b"published"


class FakeImage:
    def __init__(self, name):
        self.name = name
        self.size = (32, 24)

    def convert(self, mode):
        assert mode == "RGB"
        return self


class FakeInputIds:
    def __init__(self, length):
        self.shape = (1, length)


class RuntimeProcessor(CapturingProcessor):
    tokenizer = object()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages

    def __call__(self, **kwargs):
        return {
            "input_ids": FakeInputIds(20 if kwargs.get("images") else 5),
            "_image": kwargs.get("images", [None])[0] if kwargs.get("images") else None,
            "_text": kwargs["text"],
        }


def test_shared_image_token_measurement_keeps_exact_fields():
    measured = hf_runtime.image_prompt_token_counts(
        RuntimeProcessor(), FakeImage("image.jpg"), "qwen", "chat", token_budget.build_inputs
    )

    assert measured == {
        "image_tokens": 15,
        "prompt_tokens_with_image": 20,
        "prompt_tokens_text_only": 5,
        "image_token_fraction": 0.75,
        "expansion_ok": True,
        "note": "",
    }


def test_base_rows_stay_image_major_then_condition_major(tmp_path, monkeypatch):
    source = pd.DataFrame({
        "image_path": ["negative.jpg", "positive.jpg"],
        "valence": [-3.0, 3.0],
    })
    written = {}

    monkeypatch.setattr(token_budget, "STAGE_F_DIR", tmp_path)
    monkeypatch.setattr(token_budget, "load_config", lambda _path: {"n_images": 2})
    monkeypatch.setattr(token_budget, "ensure_dirs", lambda: None)
    monkeypatch.setattr(token_budget.pd, "read_parquet", lambda _path: source.copy())
    monkeypatch.setattr(token_budget.Image, "open", lambda path: FakeImage(path))
    monkeypatch.setattr(
        token_budget, "load_vlm", lambda _model, _max_side: (object(), RuntimeProcessor(), "qwen")
    )
    monkeypatch.setattr(
        token_budget, "first_content_token_ids",
        lambda _processor: {label: i for i, label in enumerate(EMOTION_LABELS)},
    )
    monkeypatch.setattr(token_budget, "verify_label_tokenization", lambda _tokenizer: {})
    monkeypatch.setattr(token_budget, "tqdm", lambda rows, **_kwargs: rows)

    def fake_readout(_model, inputs, _token_ids):
        image = inputs["_image"]
        rendered = inputs["_text"][0]
        prompt = rendered[0]["content"][-1]["text"]
        base = 0.4 if image.name == "positive.jpg" else -0.4
        if any(sentence in prompt for sentence in POSITIVE_CONTEXTS):
            value = base + 0.2
        elif any(sentence in prompt for sentence in NEGATIVE_CONTEXTS):
            value = base - 0.2
        else:
            value = base
        winner = "joy" if value >= 0 else "sadness"
        return value, {label: (0.0 if label == winner else -10.0) for label in EMOTION_LABELS}

    monkeypatch.setattr(token_budget, "model_readout", fake_readout)
    monkeypatch.setattr(
        token_budget, "artifact_metadata", lambda **fields: {"run": "run-1", "git": "abc", **fields}
    )
    monkeypatch.setattr(token_budget, "save_json", lambda _data, _path: None)
    monkeypatch.setattr(token_budget, "_print", lambda _metrics: None)

    def capture_parquet(frame, path, *args, **kwargs):
        written["path"] = path
        written["frame"] = frame.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", capture_parquet)

    token_budget.run_base("config/stage_f.yaml", QWEN, None)

    frame = written["frame"]
    conditions = [condition for condition, _, _ in token_budget._conditions("full")]
    assert written["path"] == tmp_path / "conflict_qwen3-vl-8b-instruct.parquet"
    assert list(frame.columns) == token_budget.BASE_RESULT_COLUMNS
    assert frame["image_path"].tolist() == ["positive.jpg"] * 15 + ["negative.jpg"] * 15
    assert frame["condition"].tolist() == conditions + conditions


def test_main_preserves_dispatch_precedence_and_argument_forwarding(monkeypatch):
    calls = []
    monkeypatch.setattr(
        token_budget, "show_prompt",
        lambda *args, **kwargs: calls.append(("show_prompt", args, kwargs)),
    )
    monkeypatch.setattr(
        token_budget, "aggregate",
        lambda *args, **kwargs: calls.append(("aggregate", args, kwargs)),
    )
    monkeypatch.setattr(
        token_budget, "reanalyze_text_only",
        lambda *args, **kwargs: calls.append(("reanalyze_text_only", args, kwargs)),
    )
    monkeypatch.setattr(
        token_budget, "run_text_only",
        lambda *args, **kwargs: calls.append(("run_text_only", args, kwargs)),
    )
    monkeypatch.setattr(
        token_budget, "reanalyze",
        lambda *args, **kwargs: calls.append(("reanalyze", args, kwargs)),
    )
    monkeypatch.setattr(
        token_budget, "run_base",
        lambda *args, **kwargs: calls.append(("run_base", args, kwargs)),
    )

    cases = [
        (
            ["--show-prompt", "--aggregate", "--text-only", "--reanalyze", "--max-side", "448"],
            ("show_prompt", (QWEN, 448), {}),
        ),
        (
            ["--aggregate", "--text-only", "--reanalyze"],
            ("aggregate", (), {}),
        ),
        (
            ["--text-only", "--reanalyze", "--max-side", "448", "--prompt-style", "legacy",
             "--bank", "minimal"],
            ("reanalyze_text_only", (QWEN,), {"style": "legacy", "bank": "minimal"}),
        ),
        (
            ["--text-only", "--force", "--prompt-style", "legacy", "--bank", "minimal"],
            ("run_text_only", ("config/stage_f.yaml", QWEN),
             {"force": True, "style": "legacy", "bank": "minimal"}),
        ),
        (
            ["--reanalyze", "--max-side", "224", "--bank", "minimal"],
            ("reanalyze", (QWEN, 224), {"bank": "minimal"}),
        ),
        (
            ["--config", "custom.yaml", "--model", "model/id", "--max-side", "896",
             "--limit", "9", "--force", "--prompt-style", "legacy", "--bank", "minimal"],
            ("run_base", ("custom.yaml", "model/id", 896),
             {"limit_override": 9, "force": True, "style": "legacy", "bank": "minimal"}),
        ),
    ]

    for argv, expected in cases:
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["stage_f_token_budget", *argv])
        token_budget.main()
        assert calls == [expected]


def test_named_compatibility_facades_keep_their_public_contract(monkeypatch):
    assert token_budget._key_suffix is run_key_suffix
    assert token_budget.slug is token_budget_key
    assert token_budget._text_only_readouts is text_only_readouts
    assert token_budget._trends is token_budget_trends
    for name in (
        "_conditions", "_base_runs_for", "_analyze", "run_base", "run_text_only",
        "reanalyze_text_only", "reanalyze", "aggregate", "show_prompt", "main",
    ):
        assert callable(getattr(token_budget, name))

    monkeypatch.setattr(
        token_budget, "artifact_metadata", lambda **fields: {"run": "run-1", "git": "abc", **fields}
    )
    metrics = token_budget._analyze(
        pd.DataFrame(), "model/id", {"image_tokens": 12}, 448,
        multi={"joy": {"single_token": False}}, n_skipped=2,
    )
    assert list(metrics) == [
        "run", "git", "model", "max_side", "read_out", "n_images", "n_rows", "n_skipped",
        "image_tokens", "image_discriminability", "asymmetry_vs_floor", "flip_override",
        "tokenization_multi_token",
    ]
    assert metrics["n_images"] == 0
    assert metrics["n_rows"] == 0
    assert metrics["n_skipped"] == 2
