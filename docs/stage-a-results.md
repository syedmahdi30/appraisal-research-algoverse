# Stage A results — text-side appraisal replication (Gemma-3-4B)

Paper-ready summary of the text-only replication gate. Two findings: appraisal **read-out
replicates strongly** (mid-layer, MHSA-dominant, well above shuffled baselines), and
single-layer additive **steering is inconclusive** (no clean effect above a norm-matched
random control). Both are reported here with equal prominence.

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

## Result 2 — Steering is INCONCLUSIVE
Adding the frozen appraisal directions at the critical layer did **not** shift emotion
valence in the theory-predicted direction above a norm-matched random control, at any β we
tested (120 test prompts, mean residual norm ≈ 37,200). Mean Δ valence vs β:

| Direction | β=−0.08 | −0.04 | −0.02 | +0.02 | +0.04 | +0.08 | slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pleasantness (predict +) | +0.096 | +0.024 | −0.024 | +0.049 | +0.076 | −0.073 | −0.60 |
| Unpleasantness (predict −) | +0.109 | +0.047 | −0.059 | +0.118 | +0.290 | +0.424 | +2.29 |
| Random (control) | +0.215 | +0.045 | +0.019 | −0.013 | −0.032 | −0.092 | −1.68 |

The **random control moves as much as the appraisal directions** (|slope| 1.68 vs 0.60/2.29),
so movement cannot be attributed to appraisal content. Pleasantness has the wrong-signed,
non-monotonic slope; unpleasantness trends up (opposite the prediction) but not clearly above
the control. Across the full range — β<0.02 swamped (shifts ~0.001), β∈[0.02,0.08] control
moves as much as signal, β>0.1 breaks coherence — **no β window isolates a causal appraisal
effect.** Figure: `results/figures/stage_a_steering.png`.

Interpretation (careful language): read-out **supports** an appraisal representation;
single-layer additive steering is **inconclusive** — consistent with Gemma-3's
massive-activation / RMSNorm regime (the residual norm ~3.7e4 swamps a single-direction nudge)
and a blunt valence metric. We do **not** claim causal steering.

## Threats to validity
- **Split.** crowd-enVENT ships one corpus; our train/val/test is a seeded stratified split,
  not the paper's exact partition — absolute r² may differ from Tak et al.
- **Single seed.** One split/seed; no cross-seed variance reported yet.
- **Correctness filtering not applied.** Probes are fit on all examples, not only those the
  model classifies correctly (a documented Tak-style filter) — a next-iteration refinement.
- **Steering metric is blunt.** Pos/neg valence collapses 13 emotions into two groups and
  excludes surprise/neutral; a per-emotion or per-appraisal readout could be more sensitive.
- **Steering method is minimal.** Single-layer additive intervention only; multi-layer
  injection or projection/clamping was not tried. The null is about *this* method, not a
  general claim that appraisal directions are non-causal.
- **In-domain read-out.** r² is val-split text; cross-modal transfer is untested here (Stage C).

## Carried into Stage C
- Frozen probes + unique-effect vectors at layer 18 (`results/stage_a/probes.npz`).
- Stage C's primary test is **read-out transfer** (these frozen probes applied to
  image-conditioned activations), which rests on the strong Result 1 — independent of the
  steering null. Cross-modal steering (Stage D) is a lower-priority bonus and inherits the
  open steering question above.

## Reproduce
```bash
python -m src.experiments.stage_a_text          # probes + layerwise metrics
python -m src.experiments.analyze_stage_a       # localization table + figure + summary.json
python -m src.experiments.stage_a_steering      # steering table + figure
```
Artifacts: `results/stage_a/{metrics.json, probes.npz, summary.json, steering_metrics.json}`,
`results/figures/stage_a_{localization,steering}.png`.
