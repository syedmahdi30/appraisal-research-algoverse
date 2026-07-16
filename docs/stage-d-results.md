# Stage D results — cross-modal causal steering (Gemma-3-4B)

Paper-ready summary of the causal capstone. A **text-derived** appraisal direction, injected
into the residual stream **under image input**, shifts the model's closed-vocab emotion
output in the theory-predicted direction: **+pleasantness raises valence (slope +0.329),
+unpleasantness lowers it (−0.309)**, both monotonic in β and ~12× a norm-matched random
null, while a non-valence specificity control (suddenness) stays small. Unlike the Stage C
read-out (correlational), a causal steering effect **cannot be explained by verbalization**
("the model is just captioning the image") — this is the strongest evidence in the project
for a shared, causally-active appraisal representation across text and vision.

## Setup
- **Model / site:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Steering at
  `blocks.18.hook_resid_post` (the Stage A critical layer), last prompt token, **under image
  conditioning** (`pixel_values`).
- **Directions (frozen, text-learned):** difference-of-means
  `Δμ_a = mean(act | rating ≥ 4) − mean(act | rating ≤ 2)` at resid_post L18, estimated from
  **1,200 crowd-enVENT train** examples. This is the **Stage A v2 recipe** — the read-out
  (probe) direction does *not* steer; the difference-of-means direction does. The cross-modal
  claim is therefore **text → image**: a direction learned entirely from text steers image
  behavior. β is in **natural units** (multiples of the low→high Δμ shift).
- **Readout:** closed-vocab emotion **valence** = P(positive) − P(negative) over the 13
  single-token emotion labels, at the last token. One forward per (image, direction, β) under
  `torch.no_grad()` — **no autoregressive generation** (fast; and the SigLIP tower's
  4096-patch attention OOMs a 40 GB A100 without `no_grad`).
- **Data:** 150 EMOTIC test images (seed 0). **Controls:** a norm-matched **random** direction
  (null) and **suddenness** (a non-valence appraisal that should not move valence — a
  specificity control).

## Result — cross-modal steering is causal (SUPPORTS)
Mean Δ valence vs β (n = 150 images; base valence mean = +0.180):

| Direction | β=−3 | β=−2 | β=−1 | β=+1 | β=+2 | β=+3 | slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Pleasantness** (predict +) | −1.049 | −0.854 | −0.466 | +0.366 | +0.622 | +0.762 | **+0.329** |
| **Unpleasantness** (predict −) | +0.716 | +0.566 | +0.323 | −0.412 | −0.793 | −1.014 | **−0.309** |
| Suddenness (specificity) | +0.202 | +0.138 | +0.075 | −0.076 | −0.156 | −0.230 | −0.073 |
| Random (null) | +0.019 | +0.037 | +0.029 | −0.039 | −0.088 | −0.127 | −0.027 |

- **Theory-predicted signs, monotonic.** +pleasantness raises valence and +unpleasantness
  lowers it, smoothly across β (±~1.0 at β=±3). The two appraisals are clean mirrors.
- **Beats the controls.** The appraisal slopes are **~12×** the random null (0.027) and
  **~4.5×** the suddenness specificity control (0.073). The effect is the *direction's*, not an
  artifact of perturbing the residual stream.
- **Single layer suffices.** Layer 18 alone produced the effect under image conditioning; no
  multi-layer band was required (unlike the fallback Stage A anticipated).

Figure: `results/figures/stage_d_steering.png` (Δ valence vs β per direction; appraisals slope
oppositely, controls flat through the middle).

## Methodological notes
- **Difference-of-means, not the probe direction.** Per the Stage A v2 finding, probe/ridge
  weights *read* a feature but do not *move* it; the mean-shift direction is the appraisal's
  activation footprint and steers. Stage D reuses this exactly, cross-modally.
- **Natural-scale β.** No arbitrary magnitude to tune — β = multiples of the low→high shift.
- **No generation.** Valence is read from the logits at the last prompt token, so each
  condition is a single forward pass; `pixel_values` flows through `run_with_hooks`.
- **The direction is text-only.** Δμ is built entirely from crowd-enVENT text activations and
  never sees an image — so a causal effect on image behavior is genuine cross-modal transfer.

## Threats to validity
- **Suddenness is not perfectly flat** (−0.073, ~2.7× the null). This is *theory-consistent*,
  not a failure: appraisals are not orthogonal, and sudden events skew unpleasant in
  crowd-enVENT, so a small negative valence leak is expected. It remains ~4.5× smaller than the
  pleasantness/unpleasantness effects.
- **Null asymmetry.** The random control is slightly asymmetric (β=+3 → −0.127) but its slope
  is tiny (−0.027, ~12× below the appraisal directions).
- **Blunt readout.** Pos/neg valence collapses 13 emotions into two groups (excludes
  surprise/neutral); the clean monotonic signal survives this, but effect *sizes* are
  metric-dependent.
- **Scope.** One seed, one layer (18), one model (Gemma-3-4B), n = 150 images; steering shown
  for the two valence-anchored appraisals (+ suddenness as a control). No person localization
  (whole image fed; "this person" without the EMOTIC bounding box), as in Stage C.

## Relationship to Stage C and the overall arc
Stage C established **correlational** cross-modal transfer that survives neutral and rich
caption controls (an *upper bound* on non-verbal signal). Stage D supplies the **causal**
complement the read-out could not: steering along a text-derived appraisal direction *drives*
image-conditioned emotion output, which no verbalization account explains. Together:

| Stage | Evidence | Result |
|---|---|---|
| A | text read-out + causal steering | appraisals decodable and causal in text |
| C | cross-modal read-out (+ caption controls) | transfer real (ρ=0.51), not merely verbalization |
| D | cross-modal steering | text appraisal directions causally steer image behavior |

## Reproduce
```bash
python -m src.experiments.stage_d_steering --limit 10   # dry run (sanity)
python -m src.experiments.stage_d_steering              # full sweep (n_images in config)
```
Artifacts: `results/stage_d/steering_metrics.json`, `results/figures/stage_d_steering.png`.
Config: `config/stage_d.yaml` (steering_layers, betas, n_dir, n_images, appraisals).

## LaTeX table
```latex
% Stage D — cross-modal steering (EMOTIC test, n=150). Slope of mean Δvalence vs β.
\begin{tabular}{lrrrrrrr}
\toprule
Direction & $\beta{=}{-}3$ & $-2$ & $-1$ & $+1$ & $+2$ & $+3$ & slope \\
\midrule
Pleasantness (pred $+$)   & $-1.049$ & $-0.854$ & $-0.466$ & $+0.366$ & $+0.622$ & $+0.762$ & $+0.329$ \\
Unpleasantness (pred $-$) & $+0.716$ & $+0.566$ & $+0.323$ & $-0.412$ & $-0.793$ & $-1.014$ & $-0.309$ \\
Suddenness (specificity)  & $+0.202$ & $+0.138$ & $+0.075$ & $-0.076$ & $-0.156$ & $-0.230$ & $-0.073$ \\
Random (null)             & $+0.019$ & $+0.037$ & $+0.029$ & $-0.039$ & $-0.088$ & $-0.127$ & $-0.027$ \\
\bottomrule
\end{tabular}
```
