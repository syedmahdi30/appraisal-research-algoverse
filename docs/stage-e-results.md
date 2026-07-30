# Stage E results — appraisal-specific & compositional emotion steering (Gemma-3-4B)

Paper-ready summary of the Stage E pilot. Stage D showed a text-derived **valence** direction
steers image-conditioned emotion output. Stage E asks the sharper, appraisal-theory question:
do *combinations* of text-derived appraisal directions, injected under image input, elicit the
theory-predicted **specific** emotion (anger, guilt, fear, pride, surprise) — not just a valence
shift? Two findings, both matched-norm controlled:

1. **Single-appraisal specificity (real).** Individual appraisal directions cause *specific*
   negative emotions beyond valence — **unpleasantness → fear** cleanly and monotonically.
2. **Compositional synthesis (the headline, and only visible after de-correlation).**
   **+unpleasant + other-responsibility → anger** is genuinely *compositional*: once the
   other-responsibility direction is stripped of valence, **neither component alone produces anger**
   (unpleasant→fear, other-responsibility→joy), but their **combination → anger** (97 % of images,
   beating both components at matched magnitude). This is the textbook prediction *anger = negative
   + other-caused*, realized causally and cross-modally. It is masked by raw (entangled) directions
   and appears only after valence de-correlation.

This is a **PARTIAL COMPOSITIONAL** result: one arm (anger) composes cleanly; the finer/positive
targets (guilt, pride, surprise) do not, hitting a joy attractor for positive valence.

## Setup
- **Model / site:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Steering at
  `blocks.18.hook_resid_post`, last prompt token, **under image conditioning** (`pixel_values`),
  one forward per (image, arm, β) under `torch.no_grad()` — no autoregressive generation.
- **Directions (frozen, text-learned):** difference-of-means `Δμ_a` at resid_post L18 from 1,200
  crowd-enVENT train examples, for all six appraisals (Stage A v2 / Stage D recipe). The cross-modal
  claim is **text → image**: directions never see an image.
- **Read-out:** closed-vocab **log-softmax over the 13 emotion labels** at the last token; the Stage
  E signal is Δ log-prob of the target emotion vs the per-image β=0 baseline (computed once per
  image and reused). Metrics per arm: target-emotion **slope** and monotonicity (Spearman over the 6
  β points), the target's **rank** among the 13 at β=+3, and **win-rate** = fraction of images where
  the target is the single biggest gainer at β=+3 (chance ≈ 1/13 = 7.7 %).
- **Arms (congruent theory):** A1 `+unpleasant +other-resp → anger`, A2 `+unpleasant +self-resp →
  guilt`, A3 `+unpleasant +sudden → fear`, A4 `+pleasant +self-resp → pride`, A5 `+pleasant +sudden
  → surprise`. **Controls:** N1 `+pleasant +other-resp` (must *not* raise anger), N2 `+unpleasant
  +predictable` (weak target), single appraisals S1–S6, and a two-orthogonal-random null R.
- **Combination rule:** unit-normalise each component Δμ, sum, rescale to the mean of the two
  component norms; entangled pairs (|cos|>0.6) are Gram-Schmidt orthogonalised first.
- **Matched-norm control (the key control):** for each congruent arm, also steer *each component
  direction rescaled to that combo's exact norm*, so combo-vs-single differs only in **direction**,
  not magnitude. Composition is credited only when the combo makes the target the **winning**
  emotion *above both* matched-norm singles.
- **De-correlation:** in the `decorrelate=valence` run, every **non-valence** appraisal Δμ is
  residualised against the bipolar valence axis `v = unit(Δμ_pleasant) − unit(Δμ_unpleasant)` before
  building combos (max |cos with valence| across non-valence appraisals **0.79 → 0.00**). The valence
  appraisals themselves are left raw (they *are* the axis).
- **Data:** first 30 of the Stage D 150 EMOTIC test images (seed 0). Pilot scale.

## The headline — A1 anger only composes after de-correlation
For arm A1 (`+unpleasant +other-responsibility → anger`), at **matched magnitude** (β=+3):

| A1 components / combination | unpleasant alone | other-responsibility alone | **combination** | combo beats both singles? |
|---|---|---|---|---|
| **Raw directions** | → fear (anger r2) | **→ anger** (r1) | → anger (97 % win) | **No** — one component already carries anger |
| **Valence-decorrelated** | → fear (anger r2) | **→ joy** (anger r7) | **→ anger (97 % win)** | **Yes** — neither component alone; composition |

The flip is the result. Raw `other_responsblt` Δμ is valence-contaminated (cos 0.79 with the valence
axis), so it produces anger on its own and *masks* the composition. After removing valence, the pure
other-attribution signal alone produces **joy**, unpleasantness alone produces **fear**, and only the
**combination** produces **anger** — exactly the appraisal-theory prediction, and it survives the
matched-norm control.

## Result 1 — raw run (`decorrelate=none`): single-appraisal specificity, not compositional
Per-arm at β=+3 (n=30 images; win-rate chance ≈ 7.7 %):

| Arm | Target | slope | ρ (monotone) | rank@+3 | win% | combo argmax | components alone (matched) | combo beats matched? |
|---|---|---:|---:|---:|---:|---|---|:--:|
| A1 | anger | +0.855 | +1.00 | 1 | 97 % | anger | unpleasant→fear, **other→anger** | **No** |
| A2 | guilt | +0.969 | +1.00 | 2 | 0 % | fear | unpleasant→anger, self→joy | yes¹ |
| A3 | fear | +0.883 | +1.00 | 2 | 27 % | anger | **unpleasant→fear**, sudden→anger | **No** |
| A4 | pride | +0.073 | +0.37 | 3 | 0 % | joy | pleasant→joy, self→joy | No |
| A5 | surprise | −0.022 | −0.20 | 2 | 3 % | joy | pleasant→joy, sudden→anger | yes¹ |

¹ "beats matched" is True but the combo's **winning** emotion is *not* the target (fear/joy win), so
it is not compositional synthesis of the target — the earlier lenient rank-≤2 rule over-credited
these; the strict criterion (target must be the combo argmax **and** beat both matched singles)
correctly rejects them.

- **Verdict (raw):** SINGLE-APPRAISAL SPECIFICITY, NOT COMPOSITIONAL. The specific emotions are
  carried by *one* appraisal direction (A1: other-responsibility→anger; A3: unpleasant→fear), not
  the combination — because the raw directions are valence-entangled. Compositional synthesis: 0/5.

## Result 2 — de-correlated run (`decorrelate=valence`): compositional anger (SUPPORTS)
Same arms, non-valence appraisals residualised against the valence axis (n=30):

| Arm | Target | slope | ρ | rank@+3 | win% | combo argmax | components alone (matched) | combo beats matched? |
|---|---|---:|---:|---:|---:|---|---|:--:|
| **A1** | **anger** | +0.604 | +1.00 | **1** | **97 %** | **anger** | unpleasant→fear, **other^⊥→joy** | **Yes ✓** |
| A2 | guilt | +0.849 | +1.00 | 2 | 0 % | fear | unpleasant→anger, self^⊥→trust | yes¹ |
| A3 | fear | +0.627 | +1.00 | 2 | 23 % | anger | **unpleasant→fear**, sudden^⊥→joy | No (single-appraisal) |
| A4 | pride | +0.066 | +0.77 | 2 | 0 % | joy | pleasant→joy, self^⊥→trust | yes¹ |
| A5 | surprise | −0.017 | −0.20 | 2 | 0 % | joy | pleasant→joy, sudden^⊥→joy | yes¹ |

¹ Again True on the rank test but the target is not the combo argmax — not counted as synthesis.

- **A1 anger — compositional (the result).** combo argmax = anger, rank 1, 97 % win, monotone
  (ρ=+1.00), and **beats both matched-norm singles** (unpleasant→fear, other^⊥→joy). Neither
  component alone yields anger.
- **A3 fear — single-appraisal.** unpleasant alone → fear (rank 1 at matched norm); adding
  suddenness^⊥ pushes the combo to *anger*, not the fear target — so fear is a single-appraisal
  effect, not compositional.
- **A2 guilt** rises to rank 2 via the combination but *fear* still wins; **A4 pride / A5 surprise**
  fail — positive valence collapses to a **joy attractor** (pleasant→joy dominates, and the residual
  self/suddenness signals do not redirect it).
- **Verdict (decorrelated):** PARTIAL COMPOSITIONAL — 1/5 arms (A1 anger) show genuine compositional
  synthesis; A3 shows single-appraisal fear specificity.

## Controls
- **N1 (`+pleasant +other-responsibility`, must not raise anger):** anger Δ@+3 = **−3.92** — anger is
  strongly *suppressed*, confirming the A1 anger effect is valence-gated (other-attribution raises
  anger only under negativity), not an artifact of the other-responsibility direction.
- **N2 (`+unpleasant +predictable`):** target boredom never emerges (rank 8); the combo drifts to the
  generic negative attractor — a sanity control, as intended.
- **R (two-orthogonal-random null)** and **A1-raw (unrescaled Δμ sum)** are logged in the metrics for
  magnitude/perturbation reference.

## Methodological notes
- **Matched-norm control is essential.** A single arm steers with the full Δμ while the combo is
  rescaled to the mean component norm, so comparing raw target *gains* favours the single by
  construction. Crediting composition only when the combo wins **at equal norm** removes the
  magnitude confound; the A1 result survives it.
- **De-correlation is necessary, not cosmetic.** With raw directions, `other_responsblt` Δμ ≈ a
  valence direction (cos 0.79), so it alone produces anger and the composition is invisible. The
  valence-orthogonal run is the fair test — and it is the run in which anger composes.
- **Closed-vocab, per-emotion read-out.** Log-softmax over the 13 labels (consistent with Stage D's
  valence = P[pos]−P[neg]); the specific-emotion signal is well-posed as "which of the 13 gains most."
- **Empirical target grounding.** Targets were cross-checked against the crowd-enVENT
  appraisal→emotion profile (z-scored); the self-blame arm's empirical argmax is shame, with guilt
  the runner-up (theory-consistent), and neither wins under steering.

## Threats to validity
- **One compositional arm, pilot scale.** Composition is clean for **anger only**, on **n=30**
  images, **one seed**, **one layer (18)**, one model. Promote to 150–300 images / 3 seeds before any
  headline claim; treat A3 (fear) as single-appraisal.
- **No lexical-frequency control yet.** Per-emotion Δ log-prob is not yet residualised against
  emotion-token corpus frequency; a full run must add it (a frequent token could gain more under any
  perturbation).
- **Positive-emotion failure / joy attractor.** Pride and surprise never emerge; positive valence
  collapses to joy. This is a real limit of appraisal→emotion compositionality here, not just noise.
- **Residual entanglement.** De-correlation removes the valence component but two non-valence
  appraisals can still correlate with each other (hence the Gram-Schmidt step on flagged pairs); the
  self/other/sudden directions are weaker probes to begin with (Stage A r² 0.24–0.44).
- **Magnitude-dependent emotion.** The emotion elicited by a single appraisal shifts with steering
  magnitude (unpleasant → fear at some norms, anger at others), so absolute argmaxes are
  scale-sensitive; the matched-norm comparison controls for this within an arm but the phenomenon
  itself is a caveat.
- **No person localisation.** Whole image fed ("this person" without the EMOTIC box); per-person
  duplication means the 30 rows are ~fewer distinct images (as in Stage C/D).

## Relationship to the arc
| Stage | Evidence | Result |
|---|---|---|
| A | text read-out + causal steering | appraisals decodable and causal in text |
| C | cross-modal read-out (+ caption controls) | transfer real (ρ=0.51), not mere verbalization |
| D | cross-modal **valence** steering | text valence directions causally steer image behavior |
| **E** | cross-modal **specific-emotion** steering | single appraisals cause specific emotions (unpleasant→fear); **+unpleasant +other-responsibility compose into anger** (after de-correlation) |

Stage E extends the causal claim from *valence* (D) to *specific emotions*, and from *single
directions* to a *theory-predicted composition* — the strongest appraisal-theory-grounded result in
the project, bounded honestly to the one arm that survives every control.

## Reproduce
```bash
python -m src.experiments.stage_e_directions                       # six Δμ + cosine matrix
python -m src.experiments.analyze_appraisal_profiles               # empirical appraisal→emotion targets
# raw (entangled) run:
python -m src.experiments.stage_e_combo        --limit 30
python -m src.experiments.analyze_stage_e
# valence-decorrelated run (the compositional test):
python -m src.experiments.stage_e_combo        --limit 30 --decorrelate valence
python -m src.experiments.analyze_stage_e                    --decorrelate valence
```
Artifacts: `results/stage_e/{directions.npz, appraisal_profiles.json, combo_pilot{,_valence}.parquet,
combo_pilot_metrics{,_valence}.json, combo_analysis{,_valence}.json}`,
`results/figures/stage_e_combo_pilot{,_valence}.png`. Config: `config/stage_e.yaml`
(`matched_norm_control`, `decorrelate`).

## LaTeX table
```latex
% Stage E — A1 anger composes only after valence de-correlation (EMOTIC, n=30, β=+3).
\begin{tabular}{llll c}
\toprule
Directions & unpleasant alone & other-resp.\ alone & combination & beats both? \\
\midrule
Raw (entangled)      & $\to$ fear & $\to$ \textbf{anger} & $\to$ anger (97\%) & No \\
Valence-decorrelated & $\to$ fear & $\to$ joy            & $\to$ \textbf{anger} (97\%) & \textbf{Yes} \\
\bottomrule
\end{tabular}
% Compositional synthesis: 0/5 arms (raw) -> 1/5 arms (decorrelated: A1 anger). N1 control: anger drops -3.92.
```
