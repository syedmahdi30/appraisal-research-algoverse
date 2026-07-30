# Stage E + Stage F Pilot Spec (hand to Claude Code)

Purpose: test for signal on two experiments, one A100 session each, reusing the Stage A-D machinery.

- **Stage E: appraisal-specific emotion steering.** Do combined text-derived appraisal directions produce appraisal-theory-predicted *specific* emotions under image input (anger, guilt, fear, pride), not just a valence shift?
- **Stage F: modality conflict.** When image and text carry conflicting appraisal cues, which modality does the shared appraisal readout follow, and can Δμ steering arbitrate?

These are pilots. Optimize for a clean signal / partial / null verdict, not significance. Full controls (lexical-frequency partialling, multi-seed, Qwen, larger n) come after signal.

Scope guard: no new models, no retraining, no re-fitting probes on image data. Probes and directions stay frozen from Stage A. Layer 18 only.

---

## 0. Shared ground rules (do not deviate)

1. Boot: `TransformerBridge.boot_transformers("google/gemma-3-4b-it", device="cuda", dtype=torch.bfloat16)`; assert `bridge.cfg.is_multimodal`.
2. Every image forward under `torch.no_grad()` (the SigLIP tower's 4096-patch eager attention OOMs a 40GB A100 otherwise).
3. Steering site: `blocks.18.hook_resid_post`, last prompt token only. Position = `input_ids.shape[-1] - 1`. Never hardcode token positions.
4. Readout: logits at the last prompt token, no generation. Capture (a) log-probs for all 13 single-token emotion labels from `src/data/labels.py`, (b) valence = P(positive) − P(negative) exactly as Stage D computes it. At startup, verify single-token status for every label in Gemma's tokenizer; if a target (e.g. guilt, pride) is multi-token, score its first token and log a warning in metrics.json.
5. Directions in natural units: Δμ_a = mean(act | rating ≥ 4) − mean(act | rating ≤ 2) at L18 resid_post, last token, from the same 1,200 crowd-enVENT train examples Stage D used. β = multiples of the low-to-high shift.
6. Cast bf16 activations to fp32 before numpy. Seed 0 everywhere. Persist per-forward rows to parquet plus a metrics.json (include config, β grid, git hash). Figures to `results/figures/`.
7. Compute the β=0 baseline forward once per image and reuse it across all arms.

---

## Stage E: appraisal-specific emotion steering

### E1. Build all six Δμ directions

New script `src/experiments/stage_e_directions.py`:

- Extend the `stage_a_steering_v2` direction builder from the current directions to all six appraisals: `pleasantness, unpleasantness, suddenness, predict_event, self_responsblt, other_responsblt`. Reuse cached Stage A fp32 activation matrices from Drive if present; otherwise re-run the 1,200 text forwards (cheap, text-only).
- Save `results/stage_e/directions.npz`: each Δμ, its norm, and the 6×6 pairwise cosine matrix of the unit directions.
- Print the cosine matrix. Flag any pair with |cos| > 0.6; flagged pairs use the orthogonalized fallback in E3.

### E2. Empirical appraisal-to-emotion prediction matrix

New CPU script `src/experiments/analyze_appraisal_profiles.py`:

- From the crowd-enVENT train split: `prof = df.groupby("emotion")[APPRAISAL_COLS].mean()`, then z-score each appraisal column across the 13 emotions.
- For a signed combo C: `score(e) = sum over a in C of sign_a * z[e, a]`. Predicted target = argmax over emotions; record the runner-up.
- Cross-check against the theory table below. If the empirical argmax disagrees with theory for any arm, use the empirical target and log the change. Save `results/stage_e/appraisal_profiles.json`.

Theory predictions (Smith & Ellsworth 1985 style):

| Arm | Combo (signs) | Predicted target | Negative prediction |
|---|---|---|---|
| A1 | +unpleasantness +other_responsblt | anger | not guilt/shame |
| A2 | +unpleasantness +self_responsblt | guilt (shame runner-up) | not anger |
| A3 | +unpleasantness +suddenness | fear (surprise runner-up) | not pride |
| A4 | +pleasantness +self_responsblt | pride | not guilt |
| A5 | +pleasantness +suddenness | surprise or joy | not sadness |

Control arms:

| Arm | Definition | Purpose |
|---|---|---|
| N1 | +pleasantness +other_responsblt | must NOT raise anger (valence-flipped A1) |
| N2 | +unpleasantness +predict_event | weak/no specific target expected; sanity |
| S1-S6 | each appraisal alone | "combination beats components" claim |
| R | sum of two orthogonal random unit vectors, scaled to combo norm | perturbation null |
| A1-raw | raw Δμ sum for A1 (unrescaled) | magnitude sensitivity check |

### E3. Combination rule

Default for all combo arms: unit-normalize each Δμ, sum the two units, rescale the sum so its norm equals the mean of the two component Δμ norms. Steer with `act[:, pos, :] += beta * combo`.

- Entangled fallback (only for pairs flagged in E1): Gram-Schmidt the second unit direction against the first before summing.

### E4. Pilot run

New script `src/experiments/stage_e_combo.py`, mirroring `stage_d_steering.py`:

- Images: first 30 of the Stage D 150 (EMOTIC test, seed 0). Prompt: `IMAGE_EMOTION_PROMPT`, unchanged.
- Arms: A1-A5, N1, N2, S1-S6, R, A1-raw (15 arms).
- β grid: {−3, −2, −1, +1, +2, +3}, plus the shared β=0 baseline.
- Persist per (image, arm, β): 13 label log-probs, valence, arm metadata → `results/stage_e/combo_pilot.parquet`.
- Budget: ~30 × 15 × 6 + 30 ≈ 2,700 forwards, below the Stage D run. One session.

### E5. Analysis and decision D1

New CPU script `src/experiments/analyze_stage_e.py`:

- Per arm: mean Δ log-prob per emotion vs β (relative to β=0); target-emotion slope; target's rank among the 13 gainers at β=+3; win-rate = fraction of images where the target is the top gainer at β=+3 (chance ≈ 7.7%).
- Figure `results/figures/stage_e_combo_pilot.png`: one panel per congruent arm, target Δ log-prob vs β, with the two component singles and R overlaid.
- **SIGNAL** (proceed to full run) if, for at least 3 of A1-A5: target slope positive and monotone (Spearman ρ over the 6 β points ≥ 0.8), combo target-gain at β=+3 exceeds the best single-component target-gain, win-rate ≥ 15%, and N1 does not raise anger.
- **PARTIAL** if only valence-linked emotions move (joy/sadness up or down, no specificity): retry with β = ±4 and inspect A1-raw before concluding; if still flat, the result demotes to "shared valence axis, no specific-emotion synthesis" and we pivot the headline.
- **NULL** if combos ≤ singles ≤ random: first verify Stage D pleasantness reproduces on the same 30 images (sanity gate); if it does, report the combo-null honestly and stop Stage E.

### E6. Deferred to the full run (post-signal)

150-300 images; lexical-frequency confound control (regress per-emotion Δ log-prob on emotion-token corpus frequency, report residualized effects); 3 seeds; JS divergence of the full 13-way distribution; complete incongruent grid.

---

## Stage F: modality conflict

### F1. Context bank and prompt

New module `src/data/conflict_contexts.py` with generic one-sentence contexts that fit any person photo:

- Negative (6): "This photo was taken moments after they received devastating news." / "...moments before the accident." / "...at the funeral of a close friend." / "...just after they lost their job." / "...during the worst week of their life." / "...right after a painful goodbye."
- Positive (6): "...moments after they won the championship." / "...at their surprise birthday party." / "...just after they got the job they wanted." / "...on the best day of their life." / "...moments after hearing wonderful news." / "...at a celebration held in their honor."
- Neutral (2): "This photo was taken on a weekday." / "This photo was taken indoors."

Prompt: same builder as Stage C/D, context inserted between the image and the question:
`<start_of_image> Context: {ctx} What single emotion is this person feeling?`
Last-token readout position unchanged (`input_ids.shape[-1] - 1`).

### F2. Pilot run

New script `src/experiments/stage_f_conflict.py`:

- Images: from the Stage C per-image parquet, the 20 highest-valence and 20 lowest-valence test images (40 total). Log image ids and their EMOTIC valence.
- Conditions per image: no-context, one random positive ctx, one random negative ctx, one neutral ctx (log ctx_id). 4 forwards per image, 160 total.
- Measure per forward: (a) frozen Stage A pleasantness probe readout at `blocks.18.hook_attn_out`, last token (Stage C machinery, unchanged); (b) behavioral valence from logits; (c) 13 log-probs.
- Persist → `results/stage_f/conflict_pilot.parquet`.

### F3. Arbitration pass

`stage_f_conflict.py --arbitrate`: on the 80 incongruent cells only (pos-image + neg-ctx, neg-image + pos-ctx), sweep the pleasantness Δμ at β {−3..+3}. 80 × 6 = 480 forwards. Question: does steering move the behavioral valence through the conflict regardless of which modality was winning?

### F4. Analysis and decision D2

New CPU script `src/experiments/analyze_stage_f.py`:

- OLS on the 160 base rows: `probe_readout ~ z(image_valence) + text_code` with text coded −1/0/+1. Same regression for behavioral valence. Report standardized coefficients β_img, β_txt and the dominance ratio |β_txt| / |β_img|.
- Figure `results/figures/stage_f_conflict.png`: 2×2 cell means (probe readout and behavioral valence) plus the arbitration curves.
- **SIGNAL** if both coefficients are sign-correct and incongruent cells separate from congruent cells in the predicted direction. Any stable dominance pattern (image-led, text-led, or mixed) is a finding. Arbitration signal: behavioral valence slope under steering within ±50% of the Stage D single-direction slope (≈0.33 per β) and the probe readout crossing zero on incongruent cells.
- **NULL** if the context sentences leave both readouts unchanged: first verify the context is attended at all (retry one stronger variant that names "this person" inside the context); if still flat, report that image dominates and text context is ignored at L18, which is itself reportable.

---

## Run order (Colab, per docs/colab.md bootstrap)

```bash
# Session 1: Stage E
python -m src.experiments.stage_e_directions
python -m src.experiments.analyze_appraisal_profiles
python -m src.experiments.stage_e_combo --limit 30
python -m src.experiments.analyze_stage_e

# Session 2: Stage F
python -m src.experiments.stage_f_conflict --limit 40
python -m src.experiments.stage_f_conflict --arbitrate
python -m src.experiments.analyze_stage_f
```

Each session: mount Drive and load HF_TOKEN in a notebook cell first, then `colab_bootstrap.py --drive`, then `smoke_test.py`, then the stage commands.

## Deliverables back to me, per stage

metrics.json, the parquet (keep per-image rows, do not aggregate them away), one figure, and a 5-sentence verdict: signal / partial / null, which criterion decided it, and the recommended next step.
