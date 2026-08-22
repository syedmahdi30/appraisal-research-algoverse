# Handoff — TransformerBridge multimodal bug found; Gemma results under re-run

_Written 2026-08-22 (~05:00). Branch `main` (merged + pushed, tip `a0c93b2`). Working tree: only
pre-existing untracked files (`PAPER-CONTEXT.md`, `PROJECT-INSTRUCTIONS.md`, `session*.md`,
`graphify-out/`, PDFs). Deadlines: Interp4Discovery Aug 29, VLM4RWD Aug 30._

## Read this first

**`docs/bridge-bug-2026-08-22.md`** — the full findings write-up for this session, with every number,
the evidence chain, the triage table, and the re-run plan. This handoff is only the orientation layer.

## Current State

- **TransformerBridge computes a different forward from raw HF for Gemma-3 on byte-identical
  inputs.** Text path is exact (valence |Δ| = 0.0011); the multimodal path is not (0/5 argmax
  agreement, 6.15 nats). Internal representations are sound (resid cosine ≥ 0.978 through L32, probe
  site 0.980) — **the corruption is at the output**. Gemma is the only model that used the bridge.
- **Headline casualty:** Gemma conflict override gap **+65% → +12% (CI [−0%, +24%])**.
- **Foundation holds:** Stage A exact; **Stage C re-run on raw HF reproduces** (ρ +0.510 vs published
  +0.507, AUC 0.912 vs 0.898, 7,280 images).
- **Unaffected:** Qwen (+39%), LLaVA-NeXT (+19%), LLaVA-1.5 (−13%) — all raw HF throughout.
- **Two hypotheses falsified this session:** the architecture-boundary claim (LLaVA-NeXT shows the
  effect) and the visual-token-budget hypothesis (flat Qwen sweep + matched-budget pair).
- **Paper is NOT yet edited.** `paper/neurips_2026.tex` still asserts the falsified LLaVA boundary,
  the +65%, and the "~1.8×" magnitude claim. Deliberate — the framing decision comes first.

## Next Steps

1. **Stage D re-run** — `python -m src.experiments.stage_d_steering_hf --verify-dirs` then the sweep.
   The causal capstone; published slope +0.329 was read off the corrupted output.
2. **Re-score same-image patching on raw HF** — expected to survive (probe-scored at L18), confirm.
3. **Paper rewrite** — retract +65%, promote Qwen to primary, drop "1.8×", rewrite §7.1 + Discussion.
4. `llava-v1.6-vicuna-7b` to close the NeXT backbone confound.
5. Consider **Interp4Discovery (Aug 29) as primary** — two ruled-out architectural explanations plus a
   measurement-validity story fit its negative-results track better than VLM4RWD's framing.

## Active Files

- `docs/bridge-bug-2026-08-22.md` — **the session record. Start here.**
- `src/experiments/stage_c_transfer_hf.py` — Stage C on raw HF (`--verify-tap`). **Done, survives.**
- `src/experiments/stage_d_steering_hf.py` — Stage D on raw HF (`--verify-dirs`). **Built, not run.**
- `src/experiments/diagnose_image_pathway.py` — bridge vs raw HF parity, `--layer-scan`.
- `src/experiments/stage_f_token_budget.py` — new-model runner, per-run paths, `--text-only`.
- `src/experiments/analyze_stage_f_unbounded.py` — unbounded readout + crossed bootstrap.
- `paper/neurips_2026.tex` — cleaned up, **numbers now stale**. Compiles on Overleaf only.

## Failed Attempts / do not repeat

- **Do not port a bridge-fitted probe without verifying the tap.** The obvious `self_attn` scores
  r² −6.26; the real site is `post_attention_layernorm` (r² +0.634). A wrong tap looks exactly like
  "transfer failed".
- **Do not revive** the TAE framing, the architecture-boundary claim, or the token-budget hypothesis.
- Three of the assistant's summary statistics were wrong this session (pooled r = −0.993;
  `max|Δ|/RMS` layer scan; a 0.99 cosine threshold). When a statistic contradicts a large direct
  measurement, distrust the statistic.
- EMOTIC annotates per person → `image_path` recurs; never key or head-slice on it.

## Artifact risk

`results/` is git-ignored. This session's outputs (`metrics_hf.json`, `tap_verification_hf.json`, all
`conflict_*_metrics.json` from the token-budget runs) live **only on the Colab runtime**. Copy them to
Drive or the repo-adjacent `results/results/` tree before that runtime is recycled — the numbers are
recorded in the doc, but the artifacts are not backed up.
