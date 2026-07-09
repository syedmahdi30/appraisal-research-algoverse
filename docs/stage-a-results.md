# Stage A results — text-side appraisal replication (Gemma-3-4B)

Paper-ready summary of the text-only replication gate. Both halves pass: appraisal
**read-out replicates strongly** (mid-layer, MHSA-dominant, well above shuffled baselines),
and **difference-of-means steering is causal** (theory-predicted valence shifts, flat random
control). A key methodological finding: the read-out (probe) direction does *not* steer —
the difference-of-means direction does.

## Setup
- **Model:** `google/gemma-3-4b-it` via TransformerBridge (bf16), 34 layers, d_model 2560.
  Read-out at the language-model blocks `hook_resid_post / hook_attn_out / hook_mlp_out`,
  last prompt token.
- **Data:** crowd-enVENT generation corpus (6,600 self-annotated experiencer events).
  Deterministic emotion-stratified split at the canonical 4,320 / 1,080 / 1,200 sizes
  (fixed seed; see caveats — not the paper's exact partition). Six appraisal targets on the
  1–5 Likert scale: pleasantness, unpleasantness, suddenness, event predictability (`predict_event`),
  own responsibility (`self_responsblt`), others' responsibility (`other_responsblt`).
- **Probes:** per-appraisal ridge regression on standardized last-token activations, alpha
  chosen by RidgeCV per appraisal. Layer/tap selected on **val**; unique-effect steering
  vectors z_a = (I − P_−a)v_a built at the critical layer and frozen for Stage C.
- **Steering:** add β·‖residual‖·z_a_unit at the critical layer's residual stream, last
  token; β as a fraction of the mean residual norm. Control = norm-matched random direction.
  Readout = closed-vocab emotion **valence** score P(positive) − P(negative) over the 13
  single-token emotion labels.

## Result 1 — Read-out replicates (SUPPORTS localization)
All six appraisals peak in the **mid-network (layers 17–21 of 34)** at the **MHSA output**
(`hook_attn_out`), with val r² far above the shuffled-label baseline (near 0). This is the
Tak et al. mid-layer, MHSA-dominant localization signature. Critical layer = 18.
n_train = 4,320, n_val = 1,080.

| Appraisal | Best MHSA layer | val r² | shuffled baseline | Δ |
|---|---:|---:|---:|---:|
| Pleasantness | 18 | 0.641 | 0.007 | 0.635 |
| Unpleasantness | 18 | 0.600 | −0.041 | 0.641 |
| Own Responsibility | 21 | 0.442 | −0.012 | 0.454 |
| Suddenness | 17 | 0.315 | −0.003 | 0.318 |
| Others' Responsibility | 19 | 0.313 | −0.020 | 0.332 |
| Event Predictability | 17 | 0.238 | −0.030 | 0.268 |

Figure: `results/figures/stage_a_localization.png` (val r² vs layer, probe vs shuffled
baseline, per appraisal). Pleasantness/unpleasantness are the strongest and cleanest —
relevant because pleasantness is the Stage C bridge to EMOTIC valence.

```latex
\begin{tabular}{lrrrr}
\toprule
Appraisal & Layer & val $r^2$ & shuffled & $\Delta$ \\
\midrule
Pleasantness           & 18 & 0.641 &  0.007 & 0.635 \\
Unpleasantness         & 18 & 0.600 & -0.041 & 0.641 \\
Own Responsibility     & 21 & 0.442 & -0.012 & 0.454 \\
Suddenness             & 17 & 0.315 & -0.003 & 0.318 \\
Others' Responsibility & 19 & 0.313 & -0.020 & 0.332 \\
Event Predictability   & 17 & 0.238 & -0.030 & 0.268 \\
\bottomrule
\end{tabular}
```

## Result 2 — Difference-of-means steering is causal (SUPPORTS)
Steering by **β·Δμ_a** at layer 18 (Δμ_a = mean activation for high-rated minus low-rated
examples; natural units, β = multiples of the low→high shift) produced monotonic,
theory-predicted valence shifts, while a norm-matched random control stayed flat
(120 test prompts, directions estimated from 1,200 train examples). Mean Δ valence vs β:

| Direction | β=−3 | −2 | −1 | +1 | +2 | +3 | slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pleasantness (predict +) | −0.191 | −0.116 | −0.059 | +0.104 | +0.208 | +0.324 | **+0.084** |
| Unpleasantness (predict −) | +0.283 | +0.184 | +0.096 | −0.053 | −0.103 | −0.167 | **−0.074** |
| Random (control) | +0.026 | +0.017 | +0.009 | −0.004 | −0.011 | −0.018 | −0.007 |

Both appraisals move **monotonically in the theory-predicted direction** (+pleasantness raises
valence, +unpleasantness lowers it), with slopes ~10× the random control's. This supports a
**causal** role for the appraisal directions, not merely decodable ones.

### Methodological note (why v1 failed, v2 worked)
An earlier attempt (v1) steered with the **probe/ridge unique-effect vector**, scaled as a
fraction of the residual norm. It never beat the random control at any β: below ~0.02 the
nudge was swamped (Gemma's residual norm ≈ 3.7e4, and RMSNorm divides by it), in [0.02, 0.08]
the random control moved as much as the appraisals, and above ~0.1 coherence broke (chaotic,
random swings ±0.7). The fix was to change the **direction**, not the magnitude: probe weights
are optimized to *read* a feature, not *move* it, whereas the **difference-of-means** direction
is the appraisal's actual activation footprint and steers cleanly at a single layer.
Figures: `results/figures/stage_a_steering_v2.png` (v2, supports), `stage_a_steering.png` (v1, null).

## Threats to validity
- **Split.** crowd-enVENT ships one corpus; our train/val/test is a seeded stratified split,
  not the paper's exact partition — absolute r² may differ from Tak et al.
- **Single seed.** One split/seed; no cross-seed variance reported yet.
- **Correctness filtering not applied.** Probes are fit on all examples, not only those the
  model classifies correctly (a documented Tak-style filter) — a next-iteration refinement.
- **Steering metric is blunt.** Pos/neg valence collapses 13 emotions into two groups and
  excludes surprise/neutral; a per-emotion readout could be more sensitive. The clean monotonic
  signal survives this bluntness, but effect *sizes* are metric-dependent.
- **Steering shown for two appraisals.** Causal steering is demonstrated for pleasantness /
  unpleasantness (unambiguous valence predictions); the other four appraisals lack a clean
  scalar readout and are not claimed. Single layer (18) only; a multi-layer band was not needed
  but could strengthen effect size.
- **Steering direction ≠ read-out direction.** Effects require the difference-of-means
  direction; the probe direction does not steer (documented above), so "the probe is causal" is
  *not* the claim — "the appraisal's mean-shift direction is causal" is.
- **In-domain read-out.** r² is val-split text; cross-modal transfer is untested here (Stage C).

## Carried into Stage C
- Frozen probes + unique-effect vectors at layer 18 (`results/stage_a/probes.npz`).
- Stage C's primary test is **read-out transfer** (these frozen probes applied to
  image-conditioned activations), which rests on Result 1.
- Stage D (cross-modal steering) should reuse the **difference-of-means recipe** from Result 2
  (`stage_a_steering_v2`), not the probe-direction / residual-fraction approach that failed on text.

## Reproduce
```bash
python -m src.experiments.stage_a_text          # probes + layerwise metrics
python -m src.experiments.analyze_stage_a       # localization table + figure + summary.json
python -m src.experiments.stage_a_steering      # steering table + figure
```
Artifacts: `results/stage_a/{metrics.json, probes.npz, summary.json, steering_metrics.json}`,
`results/figures/stage_a_{localization,steering}.png`.
