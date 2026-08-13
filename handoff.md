# Handoff — Stage F multi-model robustness (Qwen port)

_Written 2026-08-10. Everything below is committed and pushed to `origin/main` (tip `94c2a47`);
working tree is clean. GPU runs happen on Colab A100 (Qwen) — no GPU/data locally._

## Goal
Take the **Stage F cross-modal negativity-dominance** result (negative text overrides a positive
image; image tokens causally inert; carried by the assistant-turn scaffold on Gemma-3-4B) from
"one model" to **submission-ready robustness** for a NeurIPS 2026 workshop (deadline ~last week of
Aug). This session's target = **multi-model replication on Qwen3-VL-8B** (the #1 robustness ask from
the 2026-08-07 team meeting), folded into the team write-up + shareable artifact.

## Current State
- **Multi-model robustness COMPLETE and clean.** Both the effect AND the mechanism replicate on
  Qwen3-VL-8B (different architecture), all run on Colab A100 this session:
  - **Effect:** within-positive-image neg/pos ratio **3.64× (Qwen) vs 3.58× (Gemma)**; text-only raw
    |neg|/|pos| **1.00 vs 1.06** (symmetric → cross-modal on both). Matched calibration-free override
    gap **+39% [Qwen 76/37] vs +64% [Gemma 84/21]**, both CIs ≫ 0.
  - **Mechanism:** **image tokens causally INERT on Qwen (0%)** — replicates Gemma. TWIST: carrier is
    **distributed** on Qwen (question 12% / suffix 6% each small, but text_all 65% — super-additive
    redundancy) vs **concentrated** on Gemma (turn scaffold alone 65%). High-level mechanism
    model-general; fine-grained locus model-specific.
- **Deliverables all updated & consistent:** `docs/stage-f-mechanism.md` (has a Multi-model
  robustness section w/ both effect + mechanism tables), the shareable artifact
  (https://claude.ai/code/artifact/de18e776-7a83-41c2-bc52-53db22b58dcc — republish same path to
  update), and project memory ([[algoverse-stage-ef-pilot]] item (5)).
- All testable logic validated locally (segmentation, alignment, recovery, verdict, CIs). Model
  forwards ran on Colab; outputs pasted by the user and interpreted.

## Active Files
- `src/experiments/stage_f_qwen.py` — Qwen base pass + `--text-only` + `--reanalyze` (CPU). Behavioral
  valence only (no probe). Writes `conflict_qwen.parquet` in the Gemma schema so shared analyzers work.
- `src/experiments/stage_f_qwen_patching.py` — Qwen carrier experiment (raw HF forward hooks, no
  bridge/probe). `--reanalyze` (CPU) recomputes recovery + bootstrap CIs. Handles variable image-token
  counts; `decoder_layers()` resolves the LM layer path; broad default band `[13,33]/36`.
- `src/experiments/analyze_stage_f.py` — Gemma analyzer; this session ADDED `_flip_override` (shared
  cross-model override metric) wired into `run()`.
- `docs/stage-f-mechanism.md` — the team write-up (now includes Qwen).
- `requirements-qwen.txt` — Qwen venv (transformers 4.57.x); added pandas/pyarrow/scipy/tqdm.
- Scratchpad artifact source: `<scratchpad>/stage-f-mechanism.html` (republish to the same URL).

## Changes Made
- Built `stage_f_qwen.py` (base + text-only) — raw HF, behavioral valence, Gemma-schema parquet.
- Added diagnostics after live-run surprises: per-context breakdown + argmax emotion + **raw ratio**
  (Qwen floors its no-info baseline to sadness, breaking the vs-neutral ratio); per-cell valence table
  (image-moves-it check); **saturation-robust override metric** (Qwen saturates valence to ±1).
- Added `_flip_override` to `analyze_stage_f.py` — argmax-emotion valence-category override, **shared**
  by Gemma + Qwen (calibration-free), with clustered bootstrap CI; wired into both pipelines.
- Built `stage_f_qwen_patching.py` — full carrier port + bootstrap CIs + `--reanalyze`.
- Filled matched cross-model numbers into doc/artifact/memory; added Qwen mechanism result.
- 13 commits `43b3741 → 94c2a47`.

## Failed Attempts
- **Naive text-only |neg|/|pos| ratio on Qwen = 0.00 (degenerate).** Qwen answers *sadness* with full
  confidence for `none`/neutral contexts (weekday→boredom), flooring the neutral baseline. Fix: use the
  **RAW ratio (vs 0)** — 1.00, symmetric, matches Gemma. Lesson: the vs-neutral baseline is unreliable
  on a model that floors its no-information prior.
- **Graded asymmetry-vs-floor is fragile on Qwen** (saturates ±1, floors negative images). At n=20 it
  looked symmetric (noise); at n=150 it clears 0. Don't trust the graded metric at small n on a
  confident model — the **override rate** is the robust primary.
- **Sign-of-valence flip-rate is NOT cross-model comparable** — Gemma's behavioral valence is
  negatively skewed, so "valence<0" isn't a clean flip. Switched to the **argmax-emotion category**
  (calibration-free) so both models use one metric.
- Expected the Qwen carrier to also concentrate in the turn scaffold (like Gemma). It does **not** —
  single text groups recover little but text_all recovers 65% (super-additive = distributed/redundant).
  Not a bug (validated union logic); a real model difference.

## Next Steps
1. **(optional) Get exact Qwen patching CIs** — `python -m src.experiments.stage_f_qwen_patching
   --reanalyze` (CPU, in the Qwen venv) to print bootstrap CIs on image (~0%) and text_all (~65%); slot
   into the doc/artifact if a reviewer wants them.
2. **(optional) Qwen layer-entry localization** — the one open mechanism gap (does Qwen also enter
   ~L13?). Needs a **probe-free layer lens** (project each layer onto the joy−sadness *unembedding*
   direction, since Qwen has no probe). Small build; closes the last "Gemma-only" caveat.
3. **Pivot to desk work (recommended, higher-leverage w/ ~3 wks left):** NeurIPS workshop shortlist +
   novelty/related-works map for negativity dominance; **sink-token literature** (attention sinks —
   Xiao; massive activations — Sun; register tokens — Darcet) to explain the concentrated-vs-distributed
   carrier and the "why ~10 turn tokens" action item.
4. **(deferred robustness) 3-seed repeat** — still the one un-closed caveat in every threats section;
   blocker remains the seed-suffix output plumbing (runners clobber across seeds) — never built.
5. **(stretch) third model** — Gemma-3-12B or Gemma-4 (what Arnav uses); reuse `stage_f_qwen.py`
   (`--model`) for a Qwen2.5-VL cross-check, or port the same pattern.
