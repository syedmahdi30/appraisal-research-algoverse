# Experiment Codebase Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicated and private cross-runner experiment helpers with focused shared modules while preserving every existing CLI, artifact, prompt, seed, schema, and backend behavior.

**Architecture:** Existing `src.experiments.stage_*` modules remain executable façades. They call downward into a lightweight `src/experiments/shared/` package for pure readouts, deterministic sampling, artifact naming, reporting, patching primitives, and raw-HF runtime helpers; shared modules never import a runner. TransformerBridge orchestration and raw Hugging Face orchestration remain separate paths.

**Tech Stack:** Python 3.11+, PyTorch, pandas, NumPy, SciPy, scikit-learn, Hugging Face Transformers, TransformerBridge, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-experiment-codebase-refactor-design.md`

## Global Compatibility Contract

- Preserve existing module entry points and external import paths.
- Preserve CLI flags, defaults, accepted values, exit behavior, and help-level semantics.
- Preserve artifact filenames, directory layout, JSON keys, Parquet columns, and column order.
- Preserve prompt text, context construction, condition ordering, label ordering, and scoring rules.
- Preserve random seeds, sampling behavior, selected examples, and overwrite/collision guards.
- Preserve model identifiers, loading options, device/dtype decisions, and hook locations.
- Preserve summary formulas, result-row construction, aggregation, and provenance metadata.
- Keep TransformerBridge and raw Hugging Face execution visibly separate.
- Shared modules must not import runner modules.
- Shared-package imports must not load models, initialize accelerators, or import runner modules.
- Do not reformat unrelated files or touch the pre-existing untracked workspace files.
- Every task ends with focused tests passing and one coherent commit.

## File Map

**Create**

- `src/experiments/shared/__init__.py` — package marker with no eager imports.
- `src/experiments/shared/readouts.py` — raw-HF closed-vocabulary emotion scoring and prompt text.
- `src/experiments/shared/sampling.py` — deterministic valence-extreme row selection.
- `src/experiments/shared/artifacts.py` — stable model/run keys and artifact path construction.
- `src/experiments/shared/reporting.py` — pure Stage C/Stage F metrics and result transformations.
- `src/experiments/shared/patching.py` — prompt segmentation, patch groups, hooks, and recovery math.
- `src/experiments/shared/hf_runtime.py` — raw-HF model/layer/tap/encoding/residual utilities.
- `tests/test_shared_readouts_sampling.py`
- `tests/test_experiment_artifacts.py`
- `tests/test_experiment_reporting.py`
- `tests/test_experiment_patching.py`
- `tests/test_hf_runtime.py`
- `tests/test_stage_f_token_budget_contract.py`
- `tests/test_experiment_boundaries.py`

**Modify**

- Raw-HF readout consumers: `stage_f_qwen.py`, `stage_f_llava.py`, `stage_f_token_budget.py`, `stage_f_qwen_patching.py`, `stage_d_steering_hf.py`, `stage_f_patching_hf.py`, `stage_f_cross_patching_hf.py`, `stage_f_arbitration_hf.py`, `diagnose_image_pathway.py`.
- Sampling consumers: `stage_f_conflict.py`, `stage_f_qwen.py`, `stage_f_patching_hf.py`, `stage_f_cross_patching_hf.py`, and Stage F runners currently importing `select_extreme_images`.
- Reporting consumers: `stage_c_transfer.py`, `stage_c_transfer_hf.py`, `stage_c_caption.py`, `analyze_stage_f.py`, `stage_f_qwen.py`, `stage_f_llava.py`, `stage_f_token_budget.py`, `stage_f_cross_patching.py`, `stage_f_cross_patching_hf.py`.
- Patching consumers: `stage_f_attribution.py`, `stage_f_patching.py`, `stage_f_patching_hf.py`, `stage_f_cross_patching.py`, `stage_f_cross_patching_hf.py`, `stage_f_qwen_patching.py`.
- Raw-HF runtime consumers: `stage_c_transfer_hf.py`, `stage_d_steering_hf.py`, `stage_f_patching_hf.py`, `stage_f_cross_patching_hf.py`, `stage_f_layerwise_hf.py`, `stage_f_arbitration_hf.py`, `stage_f_token_budget.py`.
- Compatibility-only callers: `stage_f_prompts.py` and any module found by the final private-import scan.

---

### Task 1: Shared raw-HF readouts and deterministic sampling

**Files:**

- Create: `src/experiments/shared/__init__.py`
- Create: `src/experiments/shared/readouts.py`
- Create: `src/experiments/shared/sampling.py`
- Create: `tests/test_shared_readouts_sampling.py`
- Modify: `src/experiments/stage_f_qwen.py:34-130`
- Modify: raw-HF modules importing readout helpers from `stage_f_qwen.py`
- Modify: Stage F modules implementing or importing valence-extreme selection

**Interfaces:**

- Produces: `user_text(context_sentence: str | None) -> str`
- Produces: `first_content_token_ids(processor) -> dict[str, int]`
- Produces: `closed_vocab_valence(logits_last, token_ids) -> float`
- Produces: `closed_vocab_logprobs(logits_last, token_ids) -> dict[str, float]`
- Produces: `model_readout(model, inputs, token_ids) -> tuple[float, dict[str, float]]`
- Produces: `select_extreme_rows(df: pandas.DataFrame, n: int) -> pandas.DataFrame`
- Produces: `select_ranked_pairs(df: pandas.DataFrame, n_pairs: int) -> tuple[pandas.DataFrame, pandas.DataFrame]`
- Compatibility: `stage_f_qwen` continues to expose `emotion_token_ids`, `valence_score`, `emotion_logprobs`, `readout`, `_user_text`, and `select_extreme_images`.

- [ ] **Step 1: Write failing readout and sampling tests**

```python
# tests/test_shared_readouts_sampling.py
import math

import pandas as pd
import pytest
import torch

from src.data.labels import EMOTION_LABELS
from src.experiments.shared.readouts import (
    closed_vocab_logprobs,
    closed_vocab_valence,
    first_content_token_ids,
    user_text,
)
from src.experiments.shared.sampling import select_extreme_rows, select_ranked_pairs


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        label = text.strip()
        index = EMOTION_LABELS.index(label)
        return [900, 100 + index]

    def decode(self, ids):
        return " " if ids == [900] else EMOTION_LABELS[ids[0] - 100]


class FakeProcessor:
    tokenizer = FakeTokenizer()


def test_readout_contract_and_prompt_text():
    ids = first_content_token_ids(FakeProcessor())
    assert ids == {label: 100 + i for i, label in enumerate(EMOTION_LABELS)}
    assert user_text(None) == "What single emotion is this person feeling?"
    assert user_text("They won.") == "Context: They won. What single emotion is this person feeling?"

    logits = torch.zeros(256)
    logits[ids["joy"]] = 2.0
    logprobs = closed_vocab_logprobs(logits, ids)
    assert sum(math.exp(value) for value in logprobs.values()) == pytest.approx(1.0)
    assert closed_vocab_valence(logits, ids) > 0


def test_extreme_selection_preserves_published_order_and_pairing():
    df = pd.DataFrame({"image_path": [f"p{i}" for i in range(6)],
                       "valence": [-3, -2, -1, 1, 2, 3]})
    selected = select_extreme_rows(df, 4)
    assert selected["image_path"].tolist() == ["p4", "p5", "p0", "p1"]
    assert selected["image_group"].tolist() == ["positive", "positive", "negative", "negative"]

    positive, negative = select_ranked_pairs(df, 2)
    assert positive["image_path"].tolist() == ["p5", "p4"]
    assert negative["image_path"].tolist() == ["p0", "p1"]
```

- [ ] **Step 2: Run the new tests and confirm the package is missing**

Run: `pytest tests/test_shared_readouts_sampling.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.experiments.shared'`.

- [ ] **Step 3: Implement the shared functions by extracting existing behavior unchanged**

```python
# src/experiments/shared/readouts.py
from __future__ import annotations

import torch

from ...data.labels import EMOTION_LABELS

QUESTION = "What single emotion is this person feeling?"
POSITIVE = ("joy", "pride", "relief", "trust")
NEGATIVE = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")


def user_text(context_sentence: str | None) -> str:
    context = "" if not context_sentence else f"Context: {context_sentence} "
    return f"{context}{QUESTION}"


def first_content_token_ids(processor) -> dict[str, int]:
    tokenizer = processor.tokenizer
    token_ids = {}
    for label in EMOTION_LABELS:
        encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        token_ids[label] = next(
            (token for token in encoded if tokenizer.decode([token]).strip()),
            encoded[0] if encoded else -1,
        )
    distinct = len(set(token_ids.values()))
    if distinct < len(EMOTION_LABELS):
        raise ValueError(
            f"emotion label token ids collapsed ({distinct}/{len(EMOTION_LABELS)} distinct) — "
            f"the read-out would be degenerate. Tokenizer {type(tokenizer).__name__}; "
            "inspect encode(' joy')."
        )
    return token_ids


def closed_vocab_logprobs(logits_last, token_ids) -> dict[str, float]:
    index = torch.tensor([token_ids[label] for label in EMOTION_LABELS], device=logits_last.device)
    values = torch.log_softmax(logits_last[index].float(), dim=-1)
    return {label: float(values[i]) for i, label in enumerate(EMOTION_LABELS)}


def closed_vocab_valence(logits_last, token_ids) -> float:
    index = torch.tensor([token_ids[label] for label in EMOTION_LABELS], device=logits_last.device)
    probabilities = torch.softmax(logits_last[index].float(), dim=-1)
    by_label = {label: probabilities[i].item() for i, label in enumerate(EMOTION_LABELS)}
    return sum(by_label[label] for label in POSITIVE) - sum(by_label[label] for label in NEGATIVE)


def model_readout(model, inputs, token_ids):
    moved = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**moved)
    logits_last = output.logits[0, -1].float()
    return closed_vocab_valence(logits_last, token_ids), closed_vocab_logprobs(logits_last, token_ids)
```

Implement `sampling.py` by moving the finite filtering, stable `sort_values("valence")`, positive-first concatenation, and descending-positive/ascending-negative pair ordering exactly from the current runners. Do not add an even-`n` validation because the current behavior intentionally selects `2 * (n // 2)` rows.

- [ ] **Step 4: Add compatibility aliases and migrate callers**

```python
# src/experiments/stage_f_qwen.py
from .shared.readouts import (
    QUESTION,
    closed_vocab_logprobs as emotion_logprobs,
    closed_vocab_valence as valence_score,
    first_content_token_ids as emotion_token_ids,
    model_readout as readout,
    user_text as _user_text,
)
from .shared.sampling import select_extreme_rows


def select_extreme_images(n: int) -> pd.DataFrame:
    frame = pd.read_parquet(PROCESSED_DIR / "emotic_test.parquet")
    return select_extreme_rows(frame, n)
```

Change raw-HF consumers to import the public shared names directly. Keep bridge-only `stage_a_steering.emotion_token_ids` in bridge runners because its tokenizer contract differs. Replace selection bodies with `select_extreme_rows` or `select_ranked_pairs` after loading the same split from the same source.

- [ ] **Step 5: Run focused and existing readout tests**

Run: `pytest tests/test_shared_readouts_sampling.py tests/test_multitoken_scoring.py tests/test_stage_f_llava_scoring.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the shared readout/sampling slice**

```bash
git add src/experiments/shared tests/test_shared_readouts_sampling.py src/experiments
git commit -m "refactor: share experiment readouts and sampling"
```

---

### Task 2: Stable artifact keys and paths

**Files:**

- Create: `src/experiments/shared/artifacts.py`
- Create: `tests/test_experiment_artifacts.py`
- Modify: `src/experiments/stage_f_token_budget.py:78-106,311-337`
- Modify: `src/experiments/stage_f_llava.py:40-56`
- Test: `tests/test_stage_f_keys.py`
- Test: `tests/test_stage_f_llava_scoring.py`

**Interfaces:**

- Produces: `model_key(model_name: str) -> str`
- Produces: `run_key_suffix(style: str = "chat", bank: str = "full") -> str`
- Produces: `token_budget_key(model_name, max_side, style="chat", bank="full") -> str`
- Produces: `token_budget_metric_paths(root, model_name, style="chat", bank="full") -> list[Path]`
- Produces: `llava_artifact_paths(root, score_mode, text_only, model_name, default_model) -> tuple[Path, Path]`
- Produces: `ensure_output_available(path: Path, force: bool, message: str) -> None`
- Produces: `artifact_metadata(**fields) -> dict` with `run` and `git` inserted first.
- Compatibility: `stage_f_token_budget.slug`, `_key_suffix`, `_base_runs_for`, and `stage_f_llava._artifact_paths` remain callable with their current signatures.

- [ ] **Step 1: Write failing direct tests for the shared artifact API**

```python
# tests/test_experiment_artifacts.py
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
    assert paths == (tmp_path / "conflict_llava_sequence.parquet",
                     tmp_path / "conflict_llava_sequence_metrics.json")
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
```

- [ ] **Step 2: Run the new tests and confirm the module is missing**

Run: `pytest tests/test_experiment_artifacts.py -v`

Expected: import fails because `shared.artifacts` does not exist.

- [ ] **Step 3: Extract key/path construction without changing strings**

```python
# src/experiments/shared/artifacts.py
from __future__ import annotations

import re
from pathlib import Path

from ..common import git_hash, run_stamp


def model_key(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower().split("/")[-1]).strip("-")


def run_key_suffix(style: str = "chat", bank: str = "full") -> str:
    suffix = "" if style == "chat" else f"_{style}"
    return suffix if bank == "full" else suffix + f"_{bank}"


def token_budget_key(model_name: str, max_side: int | None,
                     style: str = "chat", bank: str = "full") -> str:
    key = model_key(model_name)
    if max_side:
        key = f"{key}_px{max_side}"
    return key + run_key_suffix(style, bank)


def ensure_output_available(path: Path, force: bool, message: str) -> None:
    if path.exists() and not force:
        raise FileExistsError(message)


def artifact_metadata(**fields) -> dict:
    return {"run": run_stamp(), "git": git_hash(), **fields}
```

Move the existing anchored regex from `_base_runs_for` into `token_budget_metric_paths`; accept `root: Path` explicitly so tests and callers do not monkeypatch a module global. Move the exact `_artifact_paths` branching into `llava_artifact_paths` with the output root and default model passed explicitly.

- [ ] **Step 4: Keep runner façades and route them to the shared functions**

```python
# stage_f_token_budget.py
_key_suffix = run_key_suffix
slug = token_budget_key


def _base_runs_for(model_name: str, style: str = "chat", bank: str = "full") -> list[Path]:
    return token_budget_metric_paths(STAGE_F_DIR, model_name, style, bank)


# stage_f_llava.py
def _artifact_paths(score_mode: str, *, text_only: bool, model_name: str = DEFAULT_MODEL):
    return llava_artifact_paths(
        STAGE_F_DIR, score_mode, text_only, model_name, DEFAULT_MODEL
    )
```

Also replace LLaVA's import of `stage_f_token_budget.slug` with a direct import of `model_key`/`token_budget_key` from `shared.artifacts`; the compatibility wrapper remains only for external callers and existing tests.

Route token-budget base/text-only collision checks through `ensure_output_available`, passing their current complete error strings unchanged. Keep each check before model loading and before any write.

Use `artifact_metadata` for the Stage F metric dictionaries touched by this refactor, preserving the current field order after `run`/`git`. Keep `common.save_json` as the serializer so formatting and filesystem behavior do not change.

- [ ] **Step 5: Run every artifact compatibility test**

Run: `pytest tests/test_experiment_artifacts.py tests/test_stage_f_keys.py tests/test_stage_f_llava_scoring.py -v`

Expected: exact historical filenames and keys still pass.

- [ ] **Step 6: Commit the artifact slice**

```bash
git add src/experiments/shared/artifacts.py src/experiments/stage_f_token_budget.py src/experiments/stage_f_llava.py tests
git commit -m "refactor: centralize experiment artifact keys"
```

---

### Task 3: Shared reporting and metric transformations

**Files:**

- Create: `src/experiments/shared/reporting.py`
- Create: `tests/test_experiment_reporting.py`
- Modify: `src/experiments/stage_c_transfer.py:88-158,270-300`
- Modify: `src/experiments/stage_c_transfer_hf.py:52 and metric call sites`
- Modify: `src/experiments/stage_c_caption.py:41 and correlation call sites`
- Modify: `src/experiments/analyze_stage_f.py:35-142,198-305`
- Modify: `src/experiments/stage_f_qwen.py:173-208`
- Modify: `src/experiments/stage_f_llava.py:58-76,121-180`
- Modify: `src/experiments/stage_f_token_budget.py:211-244,429-602`

**Interfaces:**

- Produces Stage C metrics: `correlation`, `shared_emotic_label`, `polarity_vector`, `polarity_auc`, `random_direction_controls`, `transfer_verdict`.
- Produces Stage F metrics: `flip_override`, `minimal_pair_asymmetry`, `asymmetry_vs_floor`.
- Produces transformations: `sequence_result_columns`, `content_mean_frame`, `image_discriminability`, `text_only_readouts`, `token_budget_trends`.
- Compatibility: old private names in `stage_c_transfer`, `analyze_stage_f`, `stage_f_llava`, and `stage_f_token_budget` remain aliases for callers outside the repository during the migration.

- [ ] **Step 1: Write characterization tests against representative frames**

```python
# tests/test_experiment_reporting.py
import pandas as pd
import pytest

from src.data.labels import EMOTION_LABELS
from src.experiments.shared.reporting import (
    correlation,
    flip_override,
    image_discriminability,
    text_only_readouts,
    token_budget_trends,
)


def _lp(winner):
    return {f"lp_{label}": 0.0 if label == winner else -10.0 for label in EMOTION_LABELS}


def test_flip_override_and_discriminability_schema():
    rows = []
    for image, group, valence, condition, winner in (
        ("pos", "positive", 0.8, "none", "joy"),
        ("pos", "positive", -0.4, "negative", "sadness"),
        ("neg", "negative", -0.8, "none", "sadness"),
        ("neg", "negative", 0.4, "positive", "joy"),
    ):
        rows.append({"image_path": image, "image_group": group, "valence": valence,
                     "condition": condition, **_lp(winner)})
    frame = pd.DataFrame(rows)
    override = flip_override(frame, n_boot=20, seed=7)
    assert override["neg_ctx_overrides_pos_img"] == 1.0
    assert override["pos_ctx_overrides_neg_img"] == 1.0
    assert set(override) == {
        "neg_ctx_overrides_pos_img", "pos_ctx_overrides_neg_img", "dominance_gap",
        "dominance_gap_ci95", "n_pos_images", "n_neg_images",
    }
    discrim = image_discriminability(frame)
    assert discrim["discriminability_gap"] == pytest.approx(1.6)
    assert discrim["auc"] == 1.0


def test_text_only_and_trend_outputs_keep_existing_keys():
    rows = []
    for condition, value, winner in (("positive", 0.5, "joy"),
                                     ("negative", -0.75, "sadness"),
                                     ("neutral", 0.0, "neutral")):
        rows.append({"condition": condition, "valence": value, **_lp(winner)})
    readouts = text_only_readouts(pd.DataFrame(rows))
    assert set(readouts) == {"saturation_frac", "n_rows", "bounded_valence", "unbounded_margin"}
    assert readouts["bounded_valence"]["ratio_raw"] == pytest.approx(1.5)

    table = pd.DataFrame([
        {"model": "m", "source": "a", "image_tokens": 10, "override_gap": 0.2,
         "ci_lo": 0.1, "ci_hi": 0.3, "auc": 0.95},
        {"model": "m", "source": "b", "image_tokens": 20, "override_gap": 0.25,
         "ci_lo": 0.15, "ci_hi": 0.35, "auc": 0.96},
    ])
    trends = token_budget_trends(table)
    assert trends["within_model"][0]["all_cis_overlap"] is True
```

- [ ] **Step 2: Run the new reporting tests and confirm failure**

Run: `pytest tests/test_experiment_reporting.py -v`

Expected: import fails because `shared.reporting` does not exist.

- [ ] **Step 3: Move scientific formulas byte-for-byte before renaming them**

In `reporting.py`, first move the current bodies and constants without algebraic cleanup. The first function is copied exactly except for its public name:

```python
POSITIVE_LABELS = ("joy", "pride", "relief", "trust")
NEGATIVE_LABELS = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")

def correlation(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(pred) & np.isfinite(target)
    if finite.sum() < 3 or np.std(pred[finite]) == 0 or np.std(target[finite]) == 0:
        return {"n": int(finite.sum()), "pearson": None, "spearman": None}
    return {
        "n": int(finite.sum()),
        "pearson": float(pearsonr(pred[finite], target[finite])[0]),
        "spearman": float(spearmanr(pred[finite], target[finite])[0]),
    }

FUNCTION_RELOCATIONS = (
    ("stage_c_transfer.py", "_shared_label", "shared_emotic_label"),
    ("stage_c_transfer.py", "_polarity", "polarity_vector"),
    ("stage_c_transfer.py", "_auc", "polarity_auc"),
    ("stage_c_transfer.py", "_random_controls", "random_direction_controls"),
    ("stage_c_transfer.py", "_verdict", "transfer_verdict"),
    ("analyze_stage_f.py", "_flip_override", "flip_override"),
    ("analyze_stage_f.py", "_minimal_pair_asymmetry", "minimal_pair_asymmetry"),
    ("analyze_stage_f.py", "_asymmetry_vs_floor", "asymmetry_vs_floor"),
)
```

For each tuple in `FUNCTION_RELOCATIONS`, copy the complete named source function to `reporting.py`, change only its function name to the third value, and preserve defaults, finite filtering, bootstrap construction, percentile bounds, thresholds, returned keys, and exception behavior. The tuple is an implementation checklist in the plan, not code to add to the production module. Move required private helpers such as the Stage F cell-mean calculation into `reporting.py` under public descriptive names rather than importing `analyze_stage_f`.

- [ ] **Step 4: Extract the smaller transformations with exact return schemas**

Move `_sequence_columns` and `_content_mean_frame` from LLaVA to `sequence_result_columns` and `content_mean_frame`. Move `image_discriminability`, `_text_only_readouts`, and `_trends` from the token-budget runner to public names. Keep all dict insertion order and existing column replacement order unchanged.

- [ ] **Step 5: Replace private cross-runner imports and retain compatibility aliases**

```python
# stage_c_transfer.py
from .shared.reporting import (
    correlation as _corr,
    polarity_auc as _auc,
    polarity_vector as _polarity,
    random_direction_controls as _random_controls,
    shared_emotic_label as _shared_label,
    transfer_verdict as _verdict,
)

# analyze_stage_f.py
from .shared.reporting import (
    asymmetry_vs_floor as _asymmetry_vs_floor,
    flip_override as _flip_override,
    minimal_pair_asymmetry as _minimal_pair_asymmetry,
)
```

Update `stage_c_caption.py`, `stage_c_transfer_hf.py`, Qwen, LLaVA, and token-budget callers to import public functions directly from `shared.reporting`. Existing analyzers may keep aliases so external import paths remain valid, but no runner may import those aliases from another runner.

- [ ] **Step 6: Run reporting, analysis, and artifact tests**

Run: `pytest tests/test_experiment_reporting.py tests/test_stage_f_llava_scoring.py tests/test_stage_f_keys.py -v`

Expected: all tests pass with unchanged keys and filenames.

- [ ] **Step 7: Commit the reporting slice**

```bash
git add src/experiments/shared/reporting.py src/experiments/stage_c_transfer.py src/experiments/stage_c_transfer_hf.py src/experiments/stage_c_caption.py src/experiments/analyze_stage_f.py src/experiments/stage_f_qwen.py src/experiments/stage_f_llava.py src/experiments/stage_f_token_budget.py tests/test_experiment_reporting.py
git commit -m "refactor: centralize experiment reporting metrics"
```

---

### Task 4: Shared patching primitives and bridge readout helpers

**Files:**

- Create: `src/experiments/shared/patching.py`
- Create: `tests/test_experiment_patching.py`
- Modify: `src/experiments/shared/readouts.py`
- Modify: `src/experiments/shared/reporting.py`
- Modify: `src/experiments/stage_f_attribution.py:53-116`
- Modify: `src/experiments/stage_f_patching.py:37-103,208-240`
- Modify: `src/experiments/stage_f_cross_patching.py:47-88,173-260`
- Modify: `src/experiments/stage_f_cross_patching_hf.py:49-53 and call sites`
- Modify: `src/experiments/stage_f_patching_hf.py:57-59 and call sites`
- Modify: `src/experiments/stage_f_qwen_patching.py:67-176,276-315`
- Modify: `src/experiments/stage_f_conflict.py:57-83`
- Modify: `src/experiments/stage_f_prompts.py:42 and call sites`

**Interfaces:**

- Produces: `find_subsequence(haystack, needle) -> int | None`
- Produces: `segment_prompt_positions(tokenizer, input_ids, question, expected_image_tokens) -> dict`
- Produces: `aligned_patch_groups(donor_segment, recipient_segment) -> tuple[dict, dict]`
- Produces: `cross_image_groups(segment, expected_image_tokens=256) -> tuple[dict, dict]`
- Produces: `stash_activation(store)`, `bridge_patch_hook(recipient_indices, donor_values)`
- Produces: `same_image_recovery(df, groups) -> dict`
- Produces: `cross_image_recovery(df, groups, n_boot=2000, seed=0) -> dict`
- Produces: `probe_recovery_valid(patch_layers, critical_layer) -> bool`
- Produces in `readouts.py`: `bridge_probe_readout(bridge, input_ids, pixel_values, tap_name, token_ids, coef, intercept, extra_hooks=None) -> tuple[float, float]`
- Produces in `readouts.py`: `bridge_probe_and_logits(bridge, input_ids, pixel_values, tap_name, coef, intercept, token_ids, steering_hooks=None) -> tuple[float, float, dict[str, float]]`
- Produces in `reporting.py`: public cross-image metric and print helpers used by both backends.

- [ ] **Step 1: Write failing segmentation, grouping, and recovery tests**

```python
# tests/test_experiment_patching.py
import numpy as np
import pandas as pd
import pytest

from src.experiments.shared.patching import (
    aligned_patch_groups,
    cross_image_groups,
    cross_image_recovery,
    probe_recovery_valid,
)


def _segment(context_start, question_start, n=14):
    return {
        "image": np.array([2, 3, 4]),
        "context": np.arange(context_start, question_start),
        "question": np.array([question_start, question_start + 1]),
        "n": n,
    }


def test_aligned_groups_exclude_context_and_final_query_token():
    donor = _segment(5, 7, n=12)
    recipient = _segment(5, 8, n=13)
    groups, ok = aligned_patch_groups(donor, recipient)
    assert all(ok[name] for name in ("image", "question", "structure", "text_all"))
    assert groups["image"][0].tolist() == [2, 3, 4]
    assert 5 not in groups["text_all"][0]
    assert 11 not in groups["text_all"][0]
    assert 12 not in groups["text_all"][1]


def test_cross_groups_and_bootstrap_recovery_are_deterministic():
    segment = _segment(5, 7, n=12)
    groups, ok = cross_image_groups(segment, expected_image_tokens=3)
    assert all(ok.values())
    assert 11 not in groups["all"]

    frame = pd.DataFrame({
        "pos_probe": [2.0, 4.0], "neg_probe": [0.0, 0.0],
        "pos_val": [1.0, 1.0], "neg_val": [-1.0, -1.0],
        **{f"patch_{name}_probe": [1.0, 2.0] for name in groups},
        **{f"patch_{name}_val": [0.0, 0.0] for name in groups},
    })
    recovery = cross_image_recovery(frame, tuple(groups), n_boot=20, seed=3)
    assert recovery["image"]["probe"] == pytest.approx(0.5)
    assert recovery["image"]["val"] == pytest.approx(0.5)
    assert probe_recovery_valid([13, 14, 17], 18) is True
    assert probe_recovery_valid([18], 18) is False
```

- [ ] **Step 2: Run the patching tests and confirm failure**

Run: `pytest tests/test_experiment_patching.py -v`

Expected: import fails because `shared.patching` does not exist.

- [ ] **Step 3: Extract segmentation and grouping without changing token semantics**

Implement `segment_prompt_positions` from the current Gemma and Qwen segmenters with explicit parameters:

```python
def segment_prompt_positions(tokenizer, input_ids, question: str,
                             expected_image_tokens: int | None) -> dict:
    token_values = input_ids[0].tolist()
    n_tokens = len(token_values)
    best_start, best_length = 0, 0
    start = 0
    while start < n_tokens:
        end = start
        while end + 1 < n_tokens and token_values[end + 1] == token_values[start]:
            end += 1
        length = end - start + 1
        if length > best_length:
            best_start, best_length = start, length
        start = end + 1

    image_end = best_start + best_length
    image = np.arange(best_start, image_end)
    question_start, question_length = None, 0
    for anchor in (" " + question, question):
        encoded = tokenizer.encode(anchor, add_special_tokens=False)
        question_start = find_subsequence(token_values, encoded)
        if question_start is not None:
            question_length = len(encoded)
            break
    if question_start is None or question_start <= image_end:
        question_start, question_length = max(image_end, n_tokens - 12), 0

    context = np.arange(image_end, question_start)
    question_positions = (
        np.arange(question_start, min(question_start + question_length, n_tokens))
        if question_length else np.array([], dtype=int)
    )
    excluded = set(image.tolist()) | set(context.tolist())
    template = np.array([index for index in range(n_tokens) if index not in excluded])
    image_ok = (
        best_length == expected_image_tokens
        if expected_image_tokens is not None else bool(best_length)
    )
    return {
        "image": image,
        "context": context,
        "question": question_positions,
        "template": template,
        "n": n_tokens,
        "img_len": int(best_length),
        "question_ok": bool(question_length),
        "image_ok": image_ok,
    }
```

The bridge wrapper passes `expected_image_tokens=256`; Qwen passes `None`. Preserve the current fallback `max(image_end, n - 12)` and the decoded diagnostic fields on the bridge wrapper. Move the exact current `_aligned_groups` and `_cross_groups` bodies into their public shared names; parameterize only the expected image-token count.

- [ ] **Step 4: Extract recovery and hook primitives unchanged**

Move the current same-image mean recovery and cross-image clustered-bootstrap recovery bodies. Preserve shared bootstrap indices across token groups, seed `0`, `n_boot=2000`, percentile `[2.5, 97.5]`, query-token exclusion, and every returned key. Move `_stash_hook` and the TransformerBridge `_patch_hook` without changing hook signatures or tensor indexing.

- [ ] **Step 5: Extract bridge readout functions without mixing backends**

```python
def bridge_probe_readout(bridge, input_ids, pixel_values, tap_name,
                         token_ids, coef, intercept, extra_hooks=None):
    store = {}
    hooks = [(tap_name, stash_activation(store)), *list(extra_hooks or [])]
    with torch.no_grad():
        logits = bridge.run_with_hooks(input_ids, pixel_values=pixel_values, fwd_hooks=hooks)
    activation = store["act"][0, input_ids.shape[-1] - 1].float().cpu().numpy()
    probe = float(predict(activation[None, :], coef, intercept)[0])
    return probe, closed_vocab_valence(logits[0, -1], token_ids)
```

Add the log-probability-returning variant for `stage_f_conflict`/`stage_f_prompts`. These functions stay in `readouts.py` and call bridge APIs directly; they do not call `hf_runtime.py`.

- [ ] **Step 6: Move shared cross-image reporting helpers**

Move `_probe_valid`, `_verdict`, `_metrics`, and `_print` from `stage_f_cross_patching.py` into public reporting/patching names. Pass `run_stamp`, `git_hash`, and output paths into reporting boundaries where required so the shared module does not import a runner or rely on mutable runner globals. Update both bridge and raw-HF cross-patching runners to call the same functions.

- [ ] **Step 7: Replace private runner imports and keep local aliases only for compatibility**

Update every listed patching consumer to import from `shared.patching`, `shared.readouts`, or `shared.reporting`. `stage_f_attribution.segment_positions`, `stage_f_patching._aligned_groups`, and existing private recovery names may remain aliases in their defining façades, but no other runner may import those aliases.

- [ ] **Step 8: Run patching and hook tests**

Run: `pytest tests/test_experiment_patching.py tests/test_hooks.py -v`

Expected: all tests pass; no GPU or checkpoint is loaded.

- [ ] **Step 9: Commit the patching slice**

```bash
git add src/experiments/shared src/experiments/stage_f_attribution.py src/experiments/stage_f_patching.py src/experiments/stage_f_patching_hf.py src/experiments/stage_f_cross_patching.py src/experiments/stage_f_cross_patching_hf.py src/experiments/stage_f_qwen_patching.py src/experiments/stage_f_conflict.py src/experiments/stage_f_prompts.py tests/test_experiment_patching.py
git commit -m "refactor: share activation patching primitives"
```

---

### Task 5: Raw Hugging Face runtime boundary

**Files:**

- Create: `src/experiments/shared/hf_runtime.py`
- Create: `tests/test_hf_runtime.py`
- Modify: `src/experiments/stage_c_transfer_hf.py:55-136`
- Modify: `src/experiments/stage_d_steering_hf.py`
- Modify: `src/experiments/stage_f_patching_hf.py:64-176`
- Modify: `src/experiments/stage_f_cross_patching_hf.py`
- Modify: `src/experiments/stage_f_layerwise_hf.py`
- Modify: `src/experiments/stage_f_arbitration_hf.py`
- Modify: `src/experiments/stage_f_token_budget.py:108-153`

**Interfaces:**

- Produces: `find_language_layers(model, verbose=True) -> torch.nn.ModuleList`
- Produces: `last_token_tap(model, layer, tap)` context manager
- Produces: `capture_residuals(model, layers)` context manager
- Produces: `patch_residuals(model, donor, donor_indices, recipient_indices)` context manager
- Produces: `load_gemma_hf(model_name)` and `load_vlm(model_name, max_side=None)`
- Produces: `encode_image_prompt(processor, image, prompt, device) -> dict`
- Produces: `resize_long_side(image, max_side) -> PIL.Image.Image`
- Compatibility: `stage_c_transfer_hf.find_lm_layers`, `last_token_tap`, `load_hf`, and `stage_f_patching_hf` raw-HF helpers remain callable aliases.

- [ ] **Step 1: Write failing runtime tests with toy modules**

```python
# tests/test_hf_runtime.py
import torch

from src.experiments.shared.hf_runtime import (
    capture_residuals,
    find_language_layers,
    patch_residuals,
)


class Block(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = torch.nn.Module()
        self.vision_tower.layers = torch.nn.ModuleList([Block()])
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([Block(), Block()])

    def forward(self, hidden):
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        return hidden


def test_language_layer_discovery_excludes_vision_and_caches_residuals():
    model = ToyModel()
    assert find_language_layers(model, verbose=False) is model.language_model.layers
    hidden = torch.zeros(1, 3, 2)
    with capture_residuals(model, [0, 1]) as captured:
        model(hidden)
    assert captured[0].shape == (3, 2)
    assert captured[1].shape == (3, 2)


def test_residual_patch_changes_only_recipient_positions():
    model = ToyModel()
    hidden = torch.zeros(1, 3, 2)
    donor = {0: torch.tensor([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])}
    with patch_residuals(model, donor, [1], [2]):
        output = model(hidden)
    assert output[0, 0].tolist() == [2.0, 2.0]
    assert output[0, 2].tolist() == [7.0, 7.0]
```

- [ ] **Step 2: Run the new runtime tests and confirm failure**

Run: `pytest tests/test_hf_runtime.py -v`

Expected: import fails because `shared.hf_runtime` does not exist.

- [ ] **Step 3: Move language-layer discovery and hook context managers**

Copy `find_lm_layers`, `_submodule`, `last_token_tap`, `resid_capture_full`, and `patch_resid` into their public shared names. Preserve:

```python
_LANGUAGE_LAYER_CACHE: dict[int, torch.nn.ModuleList] = {}
VISION_PATH_MARKERS = ("vision", "siglip", "vision_tower")
```

Keep tuple outputs intact, clone patched hidden states before writing, remove every hook in `finally`, and keep cache keys based on `id(model)`. Do not route Qwen's family-specific `decoder_layers` through this helper unless its toy-free import and real module-name search are already proven equivalent.

- [ ] **Step 4: Move raw-HF loading, encoding, and resize helpers**

Move current bodies without changing loader classes or arguments:

```python
def load_gemma_hf(model_name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    return model, AutoProcessor.from_pretrained(model_name)
```

`load_vlm` preserves Qwen3/Qwen2.5 class dispatch, `torch_dtype="auto"`, `device_map="auto"`, processor `max_pixels`, and the current fallback warning. `resize_long_side` preserves aspect-ratio arithmetic and `Image.BICUBIC`. `encode_image_prompt` preserves processor keyword shape and `.to(device)` behavior.

- [ ] **Step 5: Update raw-HF callers and leave compatibility aliases**

All raw-HF runners import public runtime helpers directly. `stage_c_transfer_hf` and `stage_f_patching_hf` retain aliases for external callers, but no runner imports through those façades. TransformerBridge modules must not import `hf_runtime.py`.

- [ ] **Step 6: Run runtime, hook, and import smoke tests**

Run: `pytest tests/test_hf_runtime.py tests/test_hooks.py -v`

Run: `python -c "import src.experiments.shared.hf_runtime as h; assert not hasattr(h, 'bridge')"`

Expected: all tests pass and importing the module performs no model load.

- [ ] **Step 7: Commit the raw-HF boundary**

```bash
git add src/experiments/shared/hf_runtime.py src/experiments/stage_c_transfer_hf.py src/experiments/stage_d_steering_hf.py src/experiments/stage_f_patching_hf.py src/experiments/stage_f_cross_patching_hf.py src/experiments/stage_f_layerwise_hf.py src/experiments/stage_f_arbitration_hf.py src/experiments/stage_f_token_budget.py tests/test_hf_runtime.py
git commit -m "refactor: isolate shared raw hf runtime helpers"
```

---

### Task 6: Reduce the token-budget runner to a stable façade

**Files:**

- Create: `tests/test_stage_f_token_budget_contract.py`
- Modify: `src/experiments/stage_f_token_budget.py`
- Modify: `src/experiments/shared/artifacts.py`
- Modify: `src/experiments/shared/reporting.py`
- Modify: `src/experiments/shared/hf_runtime.py`

**Interfaces:**

- Produces in runner: `build_parser() -> argparse.ArgumentParser`
- Preserves: `_conditions`, `_key_suffix`, `slug`, `_base_runs_for`, `_analyze`, `_text_only_readouts`, `_trends`, and every existing top-level run/reanalysis function as compatibility façades.
- Consumes: shared artifact, readout, sampling, reporting, and raw-HF interfaces from Tasks 1-5.

- [ ] **Step 1: Write failing CLI, prompt, row-schema, and overwrite tests**

```python
# tests/test_stage_f_token_budget_contract.py
from pathlib import Path

import pandas as pd
import pytest

from src.data.labels import EMOTION_LABELS
from src.experiments import stage_f_token_budget as token_budget


def test_parser_preserves_flags_and_defaults():
    parser = token_budget.build_parser()
    args = parser.parse_args([])
    assert args.config == "config/stage_f.yaml"
    assert args.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert args.max_side is None
    assert args.limit is None
    assert args.force is False
    assert args.text_only is False
    assert args.bank == "full"
    assert args.prompt_style == "chat"
    assert args.show_prompt is False
    assert args.reanalyze is False
    assert args.aggregate is False


def test_base_row_column_order_is_pinned():
    expected = [
        "image_path", "image_valence", "image_group", "condition", "context_id",
        "context", "text_code", "probe_readout", "valence",
        *[f"lp_{label}" for label in EMOTION_LABELS],
    ]
    assert token_budget.BASE_RESULT_COLUMNS == expected


class CapturingProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered-chat"

    def __call__(self, **kwargs):
        return kwargs


def test_chat_prompt_structure_and_wording_are_pinned():
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


def test_existing_parquet_is_never_overwritten_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(token_budget, "STAGE_F_DIR", tmp_path)
    output = tmp_path / "conflict_qwen3-vl-8b-instruct.parquet"
    output.write_bytes(b"published")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        token_budget.run_base(
            "config/stage_f.yaml", "Qwen/Qwen3-VL-8B-Instruct", None, force=False
        )
    assert output.read_bytes() == b"published"
```

- [ ] **Step 2: Run the contract tests and confirm the intentional failure**

Run: `pytest tests/test_stage_f_token_budget_contract.py -v`

Expected: fails because `build_parser` and `BASE_RESULT_COLUMNS` are not yet defined; the overwrite test must not reach model loading.

- [ ] **Step 3: Extract parser construction without changing dispatch**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage F — visual token budget vs textual override")
    parser.add_argument("--config", default="config/stage_f.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-side", type=int, default=None,
                        help="downscale images so the long side <= N (moves the token budget on "
                             "dynamic-resolution models); omit for the model's native handling")
    parser.add_argument("--limit", type=int, default=None, help="EMOTIC image count")
    parser.add_argument("--force", action="store_true", help="overwrite an existing run for this key")
    parser.add_argument("--text-only", action="store_true",
                        help="image-ablated stimulus control (keyed by model and --bank; ignores "
                             "--max-side)")
    parser.add_argument("--bank", choices=("full", "minimal"), default="full",
                        help="context bank: 'full' (6 pos / 6 neg / 2 neutral) or 'minimal' (6 "
                             "token-matched valence-only pairs, the valence-vs-event-content control). "
                             "The bank is part of the run key, so the two never overwrite each other.")
    parser.add_argument("--prompt-style", choices=("chat", "legacy"), default="chat",
                        help="chat = processor.apply_chat_template (portable); legacy = the hand-written "
                             "Gemma scaffold used for the PUBLISHED Gemma run (Gemma only)")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print both prompt styles side by side and exit (processor only, no GPU)")
    parser.add_argument("--reanalyze", action="store_true",
                        help="recompute from the saved parquet (CPU). With --text-only, recomputes the "
                             "stimulus control on both the bounded and the unbounded scale.")
    parser.add_argument("--aggregate", action="store_true",
                        help="build the cross-run trend table (CPU)")
    return parser
```

`main()` becomes `args = build_parser().parse_args()` followed by the existing branch order unchanged.

- [ ] **Step 4: Make row and aggregate schemas explicit**

```python
BASE_RESULT_COLUMNS = [
    "image_path", "image_valence", "image_group", "condition", "context_id",
    "context", "text_code", "probe_readout", "valence",
    *[f"lp_{label}" for label in EMOTION_LABELS],
]

TOKEN_BUDGET_TREND_COLUMNS = [
    "source", "model", "bank", "max_side", "image_tokens", "image_token_fraction",
    "discriminability_gap", "auc", "text_only_ratio", "override_gap", "ci_lo", "ci_hi",
]
```

Construct DataFrames with these columns after preserving existing row dict insertion order. This pins empty-run schemas as well as populated-run schemas.

- [ ] **Step 5: Replace utility bodies with compatibility delegates**

Keep orchestration (`run_base`, `run_text_only`, `reanalyze*`, `aggregate`, `show_prompt`, `main`) in the runner. Replace metric, trend, key, sampling, raw-HF load, resize, and readout bodies with direct shared calls. Prompt construction stays in this runner because chat/legacy wording is experiment-specific. Preserve condition loop order and all output-path checks before model loading.

- [ ] **Step 6: Run token-budget and artifact contract tests**

Run: `pytest tests/test_stage_f_token_budget_contract.py tests/test_stage_f_keys.py tests/test_experiment_artifacts.py tests/test_experiment_reporting.py -v`

Expected: all tests pass; the overwrite test proves no checkpoint load occurs before collision protection.

- [ ] **Step 7: Exercise CLI help without loading models**

Run: `python -m src.experiments.stage_f_token_budget --help`

Expected: exit code 0 and the same eleven options currently exposed by the module.

- [ ] **Step 8: Commit the façade cleanup**

```bash
git add src/experiments/stage_f_token_budget.py src/experiments/shared tests/test_stage_f_token_budget_contract.py
git commit -m "refactor: simplify the token budget runner"
```

---

### Task 7: Enforce dependency direction and verify the repository contract

**Files:**

- Create: `tests/test_experiment_boundaries.py`
- Modify: any `src/experiments/*.py` still importing a private name from another runner
- Modify: only compatibility aliases or imports needed to make the guard pass

**Interfaces:**

- Consumes: all shared interfaces from Tasks 1-6.
- Produces: a permanent AST-level guard against shared-to-runner dependencies and private runner-to-runner imports.

- [ ] **Step 1: Write the dependency-boundary test**

```python
# tests/test_experiment_boundaries.py
import ast
from pathlib import Path


EXPERIMENTS = Path("src/experiments")


def _imports(path):
    return [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ImportFrom)]


def test_shared_modules_never_import_runner_modules():
    violations = []
    for path in sorted((EXPERIMENTS / "shared").glob("*.py")):
        for node in _imports(path):
            module = node.module or ""
            if module.startswith("stage_") or module.startswith("analyze_stage_"):
                violations.append(f"{path}:{node.lineno} imports {module}")
    assert violations == []


def test_runners_never_import_private_names_from_other_runners():
    violations = []
    for path in sorted(EXPERIMENTS.glob("*.py")):
        for node in _imports(path):
            module = node.module or ""
            private = [alias.name for alias in node.names if alias.name.startswith("_")]
            if private and (module.startswith("stage_") or module.startswith("analyze_stage_")):
                violations.append(f"{path}:{node.lineno} {module}: {', '.join(private)}")
    assert violations == []
```

- [ ] **Step 2: Run the guard and list every remaining edge**

Run: `pytest tests/test_experiment_boundaries.py -v`

Expected before the last cleanup: any missed private edge is printed with its exact file and line; after cleanup both tests pass.

- [ ] **Step 3: Replace remaining private imports with shared public imports**

The known baseline edges that must be gone are:

```text
stage_c_caption -> stage_c_transfer._corr
stage_c_transfer_hf -> stage_c_transfer._auc/_corr/_polarity/_random_controls/_shared_label/_verdict
stage_f_cross_patching -> stage_f_attribution._stash_hook
stage_f_cross_patching -> stage_f_patching._patch_hook/_readout
stage_f_cross_patching_hf -> stage_f_cross_patching private metrics/group helpers
stage_f_cross_patching_hf -> stage_f_patching_hf._TokShim
stage_f_llava -> stage_f_qwen._user_text
stage_f_patching -> stage_f_attribution._stash_hook
stage_f_patching_hf -> stage_f_patching private group/recovery/verdict helpers
stage_f_prompts -> stage_f_conflict._probe_and_logits
stage_f_token_budget -> stage_f_qwen._user_text
```

Do not satisfy the guard by renaming private helpers public inside runners. Move the concept into the correct shared module and import it from there.

- [ ] **Step 4: Run the complete unit suite**

Run: `pytest tests/ -v`

Expected: all existing and new tests pass.

- [ ] **Step 5: Compile every experiment module**

Run: `python -m compileall -q src/experiments`

Expected: exit code 0 and no syntax errors.

- [ ] **Step 6: Run no-model CLI smoke tests**

```bash
python -m src.experiments.stage_f_qwen --help
python -m src.experiments.stage_f_llava --help
python -m src.experiments.stage_f_token_budget --help
python -m src.experiments.stage_f_conflict --help
python -m src.experiments.stage_f_patching --help
python -m src.experiments.stage_f_patching_hf --help
python -m src.experiments.stage_f_cross_patching --help
python -m src.experiments.stage_f_cross_patching_hf --help
python -m src.experiments.analyze_stage_f --help
```

Expected: every command exits 0 without loading a checkpoint or initializing CUDA.

- [ ] **Step 7: Review the diff for compatibility and unrelated changes**

Run: `git diff --check`

Run: `git diff --stat HEAD~7..HEAD`

Run: `git status --short`

Expected: no whitespace errors; only the planned source/tests/docs are changed; the pre-existing untracked files remain unmodified and uncommitted.

- [ ] **Step 8: Record model-dependent verification separately**

Do not download checkpoints during the default verification. In the handoff, list the existing project smoke path for a configured GPU environment:

```bash
python scripts/smoke_test.py
```

State explicitly whether it was run. If it was not run because models/data/GPU are unavailable, report that as an unexecuted hardware smoke test rather than a passing test.

- [ ] **Step 9: Commit the dependency guard and final cleanup**

```bash
git add tests/test_experiment_boundaries.py src/experiments
git commit -m "test: enforce experiment dependency boundaries"
```

- [ ] **Step 10: Perform final verification before claiming completion**

Run: `pytest tests/ -q`

Run: `python -m compileall -q src/experiments`

Run: `git status --short`

Expected: tests and compilation pass; status contains only the user's pre-existing untracked files.
