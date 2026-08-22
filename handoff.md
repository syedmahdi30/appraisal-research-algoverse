# Handoff — bridge bug found; foundation re-verified; paper rewrite pending

_Written 2026-08-22 ~20:20. Branch `main`, tip `aff848a`, pushed. 15 commits this session.
Working tree clean except pre-existing untracked files (`PAPER-CONTEXT.md`, `PROJECT-INSTRUCTIONS.md`,
`session*.md`, `graphify-out/`, PDFs, `paper/neurips_2026old.tex`). Deadlines: Interp4Discovery Aug 29,
VLM4RWD Aug 30._

**Full session record: `docs/bridge-bug-2026-08-22.md`. Read that before this file.**

## Goal

Get the Stage F cross-modal negativity-dominance paper submission-ready for a NeurIPS 2026 workshop.
This session turned into validating whether the results are real: a re-implementation failed to
reproduce the headline Gemma number, which exposed a TransformerBridge bug and forced a re-run of
every Gemma image-conditioned result.

## Current State

- **TransformerBridge computes a different forward from raw HF for Gemma-3 on byte-identical inputs.**
  Text path exact (behavioural valence |Δ| = 0.0011); multimodal path is not (0/5 argmax agreement,
  6.15 nats). Internal representations are sound (`resid_post` cosine ≥ 0.978 through L32, probe site
  0.980) — **the corruption is at the output**. Gemma is the only model that used the bridge.
- **Organising principle for the damage: DIFFERENTIAL measures survive, ABSOLUTE/categorical ones die.**
  Correlations, slopes and ratios-of-differences subtract off the offset the bug introduces; an
  override rate depends on where the model sits, which is what the bug moves.
- **Re-verified on raw HF (all pass):**
  - Stage A — text path exact, untouched.
  - Stage C read-out transfer — ρ **+0.510** vs published +0.507, AUC 0.912 vs 0.898, n=7,280.
  - Stage D causal steering — pleasantness slope **+0.3360** vs published +0.3293; unpleasantness
    −0.3156; suddenness −0.0776; random −0.0350. 9.6× random, 4.3× specificity.
- **Casualty:** Gemma conflict override gap **+65% → +12% (CI [−0%, +24%])**. Its cell means and the
  mirror contrast go with it.
- **Unaffected (always raw HF):** Qwen +39%, LLaVA-NeXT +19%, LLaVA-1.5 −13%.
- **Two hypotheses falsified this session:** the architecture-boundary claim and the visual-token-budget
  hypothesis (details in §Failed Attempts).
- **The paper is NOT yet edited.** `paper/neurips_2026.tex` still asserts the +65%, the falsified LLaVA
  boundary, and the "~1.8×" magnitude claim. Deliberate — framing decision comes first.
- **Artifact risk:** `results/` is git-ignored. This session's outputs (`metrics_hf.json`,
  `steering_metrics_hf.json`, `tap_verification_hf.json`, `dir_verification_hf.json`, all
  `conflict_*_metrics.json`) exist **only on the Colab runtime**. Numbers are in the doc; files are not
  backed up.

## Active Files

- `docs/bridge-bug-2026-08-22.md` — the session record: evidence chain, triage table, all numbers.
- `src/experiments/diagnose_image_pathway.py` — bridge vs raw HF parity; `--layer-scan`.
- `src/experiments/stage_c_transfer_hf.py` — Stage C on raw HF; `--verify-tap`. **Run, survives.**
- `src/experiments/stage_d_steering_hf.py` — Stage D on raw HF; `--verify-dirs`. **Run, survives.**
- `src/experiments/stage_f_token_budget.py` — new-model runner, per-run output paths, `--text-only`,
  `--prompt-style`, `--show-prompt`.
- `src/experiments/analyze_stage_f_unbounded.py` — unbounded log-odds readout + crossed bootstrap.
- `src/experiments/analyze_judge_robustness.py` — judge sweep (bugfix applied this session).
- `paper/neurips_2026.tex` — cleaned up earlier this session; **numbers now stale**. Overleaf only.

## Changes Made

1. Merged `tae-experiment-planning` into `main` and switched to working from `main`.
2. Paper cleanup pass + Codex reviewer pass (language, method clarity, resolved TODOs, fixed 3.56 vs
   3.58, `ada2025` → `zhang2026anydepth`).
3. Ran the judge-robustness sweep; fixed `_published_gap_beside` hardcoding `conflict_analysis.json`
   (Qwen/LLaVA were being anchored to Gemma's gap).
4. New `analyze_stage_f_unbounded.py`: unbounded log-odds readout + crossed (image × sentence) bootstrap.
5. New `stage_f_token_budget.py` with per-run output paths (fixes the fixed-path clobbering that
   previously destroyed three published numbers); later added `--text-only`, `--prompt-style`,
   `--show-prompt`, and a rewritten `aggregate()` after its first trend statistic proved misleading.
6. Ran LLaVA-NeXT, the Qwen resolution sweep, and re-ran Gemma + LLaVA-1.5 through the new runner.
7. New `diagnose_image_pathway.py` (+ text-only parity, per-stack image influence, `--layer-scan`).
8. New `stage_c_transfer_hf.py` and `stage_d_steering_hf.py` — raw-HF ports with site verification.
9. Cached the LM-layer lookup (`_LAYER_CACHE`) — steering was re-walking the module tree ~3,600× and
   printing each time.
10. Wrote `docs/bridge-bug-2026-08-22.md`; updated project memory throughout.

## Failed Attempts

- **The architecture-boundary claim is FALSIFIED.** LLaVA-NeXT (`llava-v1.6-mistral-7b`, 2147 image
  tokens) *shows* the asymmetry: gap **+19% CI [+8,+30]**. Same linear-projector family as LLaVA-1.5,
  so "linear-projector design ⇒ no asymmetry" is wrong. Do not resubmit it.
- **The visual-token-budget hypothesis is FALSIFIED.** Qwen sweep with identical weights was flat
  (128 tok +42%, 262 tok +37%/+39%, all CIs overlapping, discriminability AUC held at .983–.986), and
  a matched-budget pair kills it independently: Gemma 256 tok → +65% vs Qwen 262 tok → +39%.
- **Prompt template was NOT the cause of the Gemma gap.** `--prompt-style legacy` reproduces the
  published scaffold byte-for-byte (one duplicated `<bos>` aside) and still returns +12%.
- **Preprocessing was NOT the cause.** `pixel_values` identical to `max|Δ| = 0.000000`.
- **Do not port a bridge-fitted probe without verifying the tap.** The plausible `self_attn` scores
  r² **−6.26**; the correct site is `post_attention_layernorm` (r² +0.634). A wrong tap looks exactly
  like "cross-modal transfer failed".
- **Three of the assistant's own summary statistics were wrong**, each initially convincing: pooled
  Pearson r = −0.993 "supports token budget"; `max|Δ|/RMS` layer scan reporting catastrophic
  divergence (Gemma outlier dims — 5% change in 1 of 2,560 coords gives max/RMS ≈ 2.5 while cosine
  stays 1.000000); a 0.99 cosine cut calling 0.980 "disagreement". **When a summary statistic
  contradicts a large direct measurement, distrust the statistic.**
- **EMOTIC annotates per person**, so `image_path` recurs — a head slice returns the same photo
  repeatedly. Never key or head-slice on it.
- Do not revive the TAE measurement-validity framing (mentor called it overreach).

## Next Steps

1. **Copy `results/` off Colab** (`cp -r results/stage_c results/stage_d results/stage_f
   /content/drive/MyDrive/...`) — the artifacts are not backed up anywhere.
2. **Re-score same-image patching on raw HF** to confirm the 57–65% turn-boundary result. Expected to
   survive (a ratio of differences AND probe-scored at L18) — confirm rather than assume.
3. **Rewrite the paper**: retract the +65% Gemma override gap; promote Qwen to primary model (+39%,
   never touched the bridge); drop the "~1.8×" magnitude claim from the abstract (it flips sign on an
   unbounded readout); rewrite §7.1 and the "Why would LLaVA differ?" discussion for the falsified
   boundary; re-word LLaVA-1.5 as a null rather than a reversal.
4. **Decide the venue framing with Sneheel** — two ruled-out architectural explanations plus a
   measurement-validity story fit Interp4Discovery's negative-results track (Aug 29) better than
   VLM4RWD's grounding framing (Aug 30).
5. Run `llava-v1.6-vicuna-7b` to close the NeXT language-backbone confound.
6. Add the responsible-use statement (desk-reject trigger at Interp4Discovery).
7. Re-run the three overwritten-provenance experiments with per-run output paths.
