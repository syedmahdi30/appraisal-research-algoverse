# Stage F results — modality conflict & steering arbitration (Gemma-3-4B)

Paper-ready summary of the **full** Stage F run (150 images, full 6+6+2 context bank). When a person
image and a one-sentence text **context** carry conflicting appraisal cues, which modality does the
shared appraisal read-out follow, and can a text-derived Δμ steering vector arbitrate? Three findings:

1. **Balanced multimodal integration (not image dominance).** With a realistic bank of contexts,
   image and text contribute **comparably** — standardized β_img ≈ 0.32–0.37 vs β_txt ≈ 0.32–0.38
   (dominance ratio 0.86 for the internal probe read-out, **1.19 for behavioral valence**, i.e. text
   slightly leads the output). Both cues are **sign-correct** for both read-outs: image and text write
   valence into the *same* representation.
2. **The context effect is a ceiling/floor pattern, not a negativity asymmetry.** Relative to a neutral
   context, context pulls valence hardest *against* the image's saturated valence — a negative context
   drops a positive face by −0.598, a positive context lifts a negative face by +0.333, while the
   with-the-grain directions have little headroom (+0.167, −0.056). The pilot's "negativity asymmetry"
   does not survive.
3. **Steering arbitrates the conflict (causal).** On the incongruent cells, the text-derived
   **pleasantness Δμ** drives behavioral valence at slope **+0.215 (~65 % of Stage D's ~0.33**, within
   ±50 %) — the shared appraisal handle moves the output through the conflict, though a strong context
   is now a real competitor.

**Verdict: SIGNAL.** Balanced integration with both cues sign-correct and causal arbitration — the
cleanest evidence in the project that image and text share one appraisal representation rather than
running modality-siloed systems.

> **Supersedes the pilot.** An earlier 40-image / one-context-per-polarity pilot reported
> *image-dominance* (ratio ~0.15–0.22) and a *negativity asymmetry*. Both were artifacts of a single,
> unusually weak positive context and a strong negative one. The full bank (6 contexts per polarity,
> 150 images) is the authoritative run; see "Why the bank matters."

## Setup
- **Model / site:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Read-out = the **frozen Stage
  A pleasantness probe** at `blocks.18.hook_attn_out`, last prompt token (Stage C machinery, never
  re-fit). Behavioral valence = closed-vocab P(positive) − P(negative) over the 13 emotion labels.
  Both captured in one forward under `torch.no_grad()`.
- **Images:** the **75 highest- and 75 lowest-EMOTIC-valence** test images (true extremes, from the
  EMOTIC test parquet). 150 persons → **121 distinct images** (EMOTIC is per-person; disclosed).
- **Context bank (full run):** 6 positive, 6 negative, 2 neutral generic one-sentence contexts, plus a
  no-context condition — **15 conditions per image**, `text_code ∈ {−1, 0, +1}`. Prompt = the Stage
  C/D image prompt with `Context: {ctx}` inserted; last-token read-out position unchanged. 150 × 15 =
  **2,250 base forwards**.
- **Arbitration pass:** incongruent cells (positive image + negative context; negative image + positive
  context), sweep the text-derived pleasantness Δμ at `blocks.18.hook_resid_post`, β ∈ {−3…+3}. 150
  cells × (6+1) = **1,050 forwards**.
- **Regression:** standardized OLS `readout ~ z(image_valence) + text_code`, separately for the probe
  read-out and behavioral valence; dominance ratio = |β_txt| / |β_img|. **No-context is excluded** and
  **neutral is the baseline** — see "The no-context baseline" below.

## Result 1 — modality integration: balanced, both sign-correct (SUPPORTS shared read-out)
Standardized OLS over the sentence-bearing conditions (no-context excluded):

| Read-out | β_img (image valence) | β_txt (text context) | \|β_txt\|/\|β_img\| | pattern |
|---|---:|---:|---:|---|
| Frozen pleasantness probe (attn_out L18) | +0.367 | +0.315 | 0.86 | balanced |
| Behavioral valence (logits) | +0.322 | +0.382 | **1.19** | balanced (text-leaning) |

- Both cues enter with the **theory-predicted sign** for **both** the internal probe read-out and the
  model output.
- The two are **comparable in magnitude** (ratios 0.86 and 1.19); for behavioral valence the text
  context slightly *leads*. The model genuinely integrates both modalities.
- **Note on cross-run comparison.** Absolute β magnitudes are not directly comparable to the pilot
  (design differs); the **dominance ratio** is the robust quantity, and it moves decisively from
  image-led (pilot 0.15–0.22) to balanced (0.86–1.19).

## Result 2 — context effect: a ceiling/floor pattern (negativity asymmetry does NOT replicate)
Mean Δ behavioral valence vs the **neutral** context, per image group × context polarity:

| Image group | + positive context | − negative context |
|---|---:|---:|
| Positive (high-valence face) | +0.167 | **−0.598** |
| Negative (low-valence face) | **+0.333** | −0.056 |

- Context pulls valence **hardest against the image's own saturated valence**: a negative context
  drops a positive face a lot (−0.598) and a positive context lifts a negative face a lot (+0.333),
  while the with-the-grain moves are small (a positive face is already near the valence ceiling; a
  negative face near the floor). This is a **ceiling/floor** pattern — largely symmetric — with a mild
  residual tendency for negative context to be the stronger mover.
- The pilot's −0.475-vs-+0.049 "negativity asymmetry" was a weak-positive-context artifact and is
  **retracted**.

### The no-context baseline is non-comparable (why the earlier "neutral anomaly" appeared)
Raw per-condition means exposed a confound: **adding *any* framing sentence — even a neutral one —
raises the read-out relative to no-context, in BOTH the probe and behavioral valence.** No-context
sits down near the negative condition:

| condition (positive images) | behavioral valence | probe read-out |
|---|---:|---:|
| no-context | −0.469 | +4.544 |
| + neutral context | −0.058 | +4.868 |
| + positive context | +0.109 | +5.145 |
| + negative context | −0.656 | +4.381 |

Because the effect is present in the **probe read-out**, it is representational (the framing sentence
changes the L18 representation), not a logit/label quirk. It is also **uniform across contexts** (no
single rogue sentence). The mitigation is baked into the analysis: **no-context is dropped from the
regression and neutral is the within-structure baseline**, so only context *polarity* varies. (The
`_condition_breakdown` diagnostic that surfaced this is retained in `analyze_stage_f`.)

## Result 3 — steering arbitrates the conflict (SUPPORTS causal shared handle)
On the incongruent cells, injecting the text-derived **pleasantness Δμ** at resid_post L18:

| Metric | slope vs β | reference |
|---|---:|---|
| **Behavioral valence** | **+0.215** | Stage D single-direction ≈ +0.33 (~65 %, within ±50 %) |
| Frozen probe read-out | +0.000 | expected ~0 — read at attn_out L18, **upstream** of the injection |

- The text-derived valence direction drives the emotion output **through the modality conflict**, the
  causal complement to Result 1. The slope is lower than the pilot's +0.35, consistent with balanced
  integration: a strong context is now a real competitor the steering must overcome.
- The probe read-out slope is **0.000 by construction** (probe site upstream of the injection);
  recorded to demonstrate the invariance, not scored.

## Methodological notes
- **Full context bank, averaged.** Six contexts per polarity remove the single-sentence sensitivity
  that skewed the pilot; results are averaged over the bank with `context_id` retained.
- **No-context excluded; neutral is the baseline.** The no-context prompt is structurally
  non-comparable (Result 2); dominance and the context effect use only sentence-bearing conditions.
- **Image valence is EMOTIC ground truth (1–10), z-scored** — an external anchor, so Result 1 is not
  circular.
- **Two read-outs, one forward:** probe (internal) and behavioral valence (output) captured together.
- **Arbitration read on behavioral valence only** — the frozen probe is upstream of the injection.
- **Behavioral valence is negatively skewed** (even happy faces read slightly negative on the
  closed-vocab P[pos]−P[neg]); only *relative* effects (regression, deltas) are interpreted.

## Threats to validity
- **Single seed / one layer / one model.** 150 persons (121 distinct images), layer 18, Gemma-3-4B,
  one seed. A 3-seed repeat is the remaining robustness step.
- **Per-person duplication, no localization.** Whole image fed with "this person" (no EMOTIC box); a
  photo can appear as both a high- and low-valence person.
- **Arbitration uses one opposing context per cell** (not the full bank) to bound compute; the slope
  may vary with context strength.
- **Residual context asymmetry.** The ceiling/floor effect is not perfectly symmetric (−0.598 vs
  +0.333); a mild negative-stronger tendency remains, to be confirmed with seeds.
- **Pilot ≠ full design.** The pilot's image-dominance/negativity-asymmetry do not survive the bank —
  a cautionary note on single-context conflict tests, not a contradiction.

## Why the bank matters (pilot → full)
| | Pilot (40 img, 1 ctx/polarity) | Full (150 img, 6+6+2 bank) |
|---|---|---|
| Probe read-out ratio | 0.22 (image-led) | 0.86 (balanced) |
| Behavioral valence ratio | 0.15 (image-led) | 1.19 (balanced, text-leaning) |
| Positive ctx on negative image | +0.049 | +0.333 (vs neutral) |
| Arbitration slope | +0.350 | +0.215 |
The pilot's single positive context was weak and its single negative context strong; averaging over a
realistic bank reveals balanced integration and a ceiling/floor context effect.

## Relationship to the arc
| Stage | Question | Result |
|---|---|---|
| C | does the text read-out survive on images? | yes (ρ=0.51), beyond captions |
| D | does a text valence direction steer image behavior? | yes (causal, slope ~0.33) |
| E | do appraisal *combinations* make specific emotions? | anger composes (after de-correlation) |
| **F** | when image & text conflict, who wins, can steering arbitrate? | **balanced integration; both cues sign-correct; pleasantness Δμ arbitrates (~65 % of Stage D)** |

Stage F is the direct test of a **shared** representation: if image and text fed separate appraisal
systems, a single text-derived direction could not arbitrate an image-driven emotion — but it does,
and the two modalities contribute comparably to the same read-out. With C/D/E this is convergent
evidence for one modality-agnostic appraisal geometry that both modalities write into and that
steering can drive.

## Reproduce
```bash
python -m src.experiments.stage_f_conflict              # base pass (150 img × 15 conditions = 2,250 fwd)
python -m src.experiments.stage_f_conflict --arbitrate  # arbitration (1,050 fwd)
python -m src.experiments.analyze_stage_f               # OLS + context effect + RAW breakdown + arbitration
```
Config: `config/stage_f.yaml` (`n_images: 150`, `full_context_bank: true`). Artifacts:
`results/stage_f/{conflict_pilot.parquet, conflict_metrics.json, arbitration_pilot.parquet,
arbitration_metrics.json, conflict_analysis.json}`, `results/figures/stage_f_conflict.png`.

## LaTeX table
```latex
% Stage F — modality conflict (EMOTIC extremes, 150 persons / 121 images, full bank, sentence-bearing conditions).
\begin{tabular}{lrrr}
\toprule
Read-out & $\beta_{\text{img}}$ & $\beta_{\text{txt}}$ & $|\beta_{\text{txt}}|/|\beta_{\text{img}}|$ \\
\midrule
Frozen pleasantness probe (attn\_out L18) & $+0.367$ & $+0.315$ & $0.86$ \\
Behavioral valence (logits)               & $+0.322$ & $+0.382$ & $1.19$ \\
\bottomrule
\end{tabular}
% Balanced integration, both sign-correct. Arbitration: pleasantness Δμ steers behavioral valence at +0.215 (~65% of Stage D 0.33).
% Context effect vs neutral (ceiling/floor): neg-ctx on positive face -0.598; pos-ctx on negative face +0.333.
```
