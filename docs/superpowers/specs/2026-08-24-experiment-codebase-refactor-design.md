# Behavior-Preserving Experiment Codebase Refactor

**Date:** 2026-08-24  
**Status:** Approved design; implementation not started

## Problem

The experiment code works, but several runner modules have accumulated multiple responsibilities and depend on private helpers defined in other runners. The clearest example is `src/experiments/stage_f_token_budget.py`, which currently combines model dispatch, prompt construction, experiment execution, analysis, persistence, aggregation, and CLI handling. Related Stage F runners repeat or cross-import logic for scoring, image selection, patch grouping, recovery metrics, result rows, and artifact handling.

This structure makes otherwise local changes risky: a helper that looks runner-specific may be an undocumented dependency of another experiment, and duplicated implementations can drift. The refactor should improve those boundaries without changing scientific behavior or invalidating existing commands and artifacts.

## Goals

1. Give shared experiment concepts one clear implementation and home.
2. Keep runner modules readable as experiment-specific orchestration and stable CLI façades.
3. Replace private runner-to-runner imports with imports from an explicit shared package.
4. Make the behavior that affects reproducibility easy to characterize and test.
5. Reduce the size and responsibility count of the largest Stage F modules incrementally.

## Non-goals

- Redesigning experiments, prompts, metrics, datasets, or paper claims.
- Introducing a universal model or experiment-runner framework.
- Unifying TransformerBridge and raw Hugging Face execution paths.
- Renaming commands, changing output locations, or revising artifact schemas.
- Performing broad formatting, style-only rewrites, or unrelated cleanup.
- Changing dependency versions or model-loading policy.

## Compatibility contract

The refactor must preserve the following observable behavior unless a separately reviewed bug fix explicitly changes it:

- Existing module entry points and import paths used outside the implementation.
- CLI flags, defaults, accepted values, exit behavior, and help-level semantics.
- Artifact filenames, directory layout, JSON keys, Parquet columns and column order.
- Prompt text, context construction, condition ordering, label ordering, and scoring rules.
- Random seeds, sampling behavior, selected examples, and overwrite/collision guards.
- Model identifiers, loading options, device and dtype decisions, and hook locations.
- Summary formulas, result-row construction, aggregation, and provenance metadata.
- The separation between TransformerBridge orchestration and raw Hugging Face hooks.

Public runner modules remain executable façades. Existing private helpers may temporarily re-export extracted implementations when needed to preserve internal or test imports during migration.

## Proposed structure

Create `src/experiments/shared/` as a small package of focused helpers:

```text
src/experiments/
├── shared/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── hf_runtime.py
│   ├── patching.py
│   ├── readouts.py
│   ├── reporting.py
│   └── sampling.py
├── stage_f_token_budget.py
├── stage_f_qwen.py
├── stage_f_llava.py
└── ... existing runners
```

### `artifacts.py`

Own deterministic output paths, artifact keys, collision/overwrite checks, run metadata, and common serialization guards. It must not decide experiment conditions or scientific content.

### `readouts.py`

Own reusable emotion-label log-probability calculations, valence summaries, tokenization validation, and other pure transformations from logits/scores to established readouts. Prompt construction remains in the relevant runner unless it is already identical and demonstrably context-free.

### `sampling.py`

Own deterministic selection helpers such as extreme-image selection. Callers continue to supply data, labels, counts, and seeds explicitly.

### `patching.py`

Own layer parsing, token-span segmentation/alignment, patch-group construction, hook primitives, and recovery calculations that are truly common across Stage F patching experiments. It must not choose a backend or silently normalize backend-specific tensor layouts.

### `hf_runtime.py`

Own raw Hugging Face utilities that are shared by Qwen/LLaVA-style runners: device resolution, model/processor loading helpers, encoding conventions, and layer discovery. TransformerBridge code does not depend on this module.

### `reporting.py`

Own repeated summary and result-row transformations once their equality has been characterized. Plot-specific presentation stays with analysis scripts unless the plotting code is both identical and behaviorally covered.

## Dependency rules

The intended dependency direction is:

```text
runner / CLI façade
        ↓
shared experiment helpers
        ↓
existing data, probe, and bridge primitives
```

- Shared modules must not import runner modules.
- Runner modules may import shared modules and lower-level project packages.
- Raw Hugging Face helpers and TransformerBridge helpers remain separate dependency branches.
- Shared functions should receive important choices explicitly instead of reading runner globals.
- Pure calculations should remain pure; model state, filesystem writes, and CLI parsing stay at orchestration boundaries.
- Extraction is justified by a real repeated concept, not merely similar-looking lines.

## Migration sequence

### Phase 1: Characterize behavior

Add focused tests that lock down the existing contract before moving code. Cover artifact names and schemas, prompts and condition ordering, selection determinism, patch grouping/alignment, recovery calculations, summary metrics, and representative CLI parsing. Where full model execution is impractical, test pure helpers with small tensors and fixtures and record representative output structures.

### Phase 2: Extract low-risk pure helpers

Move readout, sampling, artifact-key, and metadata helpers into the shared package. Update callers one concept at a time. Preserve temporary compatibility aliases where existing tests or imports require them.

### Phase 3: Extract patching primitives

Move only the patch alignment, grouping, hook, and recovery primitives proven equivalent by tests. Keep backend-specific activation access and hook registration in their existing backend paths.

### Phase 4: Decompose token-budget orchestration

Reduce `stage_f_token_budget.py` to its CLI, experiment-specific decisions, and top-level orchestration. Delegate shared calculations and persistence mechanics without changing its executable module, argument surface, prompt construction, condition order, or outputs.

### Phase 5: Remove runner-to-runner private imports

Update Qwen, LLaVA, cross-patching, conflict, and analysis modules to import the extracted shared concepts. Keep façades or aliases only where required for compatibility.

### Phase 6: Remove proven duplication

Delete old private implementations only after all callers have moved and equivalence tests pass. Avoid speculative abstractions and leave near-duplicates alone when their scientific or backend semantics differ.

### Phase 7: Verify the repository contract

Run the full test suite, focused import and CLI smoke tests, compile checks, and schema comparisons against representative fixtures or existing artifacts. Review the final diff for unintended formatting or behavior changes.

## Testing strategy

Tests should be added or strengthened at three levels:

1. **Pure unit tests:** scoring, valence aggregation, sampling, layer parsing, patch groups, recovery, artifact keys, and result-row transformations.
2. **Contract tests:** CLI defaults, prompt strings, condition order, output filenames, JSON keys, Parquet column order, and overwrite behavior.
3. **Smoke tests:** import every touched runner, invoke `--help`, and run the smallest available no-download or fixture-backed execution paths.

Verification commands will be chosen from the repository's existing environment, with at least:

```bash
pytest tests/
python -m compileall src/experiments
```

Any model-dependent smoke test must be reported separately from the default test suite so that unavailable checkpoints or hardware do not hide failures in the local refactor.

## Error handling and provenance

- Existing validation errors and overwrite protections remain visible; shared helpers must not catch and suppress them.
- New shared functions validate their own structural assumptions with actionable messages.
- Artifact metadata continues to record the same experiment and revision information.
- A missing optional backend or model should fail at the same orchestration boundary as before, not during import of the shared package.
- Shared-package imports must remain lightweight and must not load models or initialize accelerator state.

## Incremental review and rollback

Each migration phase should be independently reviewable and leave the test suite passing. A phase should move one coherent concept, update its callers, and remove the old implementation only after verification. This keeps rollback local and avoids a single repository-wide rewrite.

## Success criteria

The refactor is complete when:

- No experiment runner imports private helpers from another runner.
- Shared scoring, sampling, artifact, and patching concepts have one tested implementation where their semantics are truly identical.
- `stage_f_token_budget.py` is primarily orchestration rather than a collection of unrelated utilities.
- All compatibility-contract tests and the existing test suite pass.
- Existing CLI and artifact consumers require no changes.
- TransformerBridge and raw Hugging Face execution remain visibly separate.
- The final diff contains no unrelated mass formatting or paper/content edits.
