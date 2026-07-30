# Stage F results — modality conflict & steering arbitration (Gemma-3-4B)

Paper-ready summary of the **full** Stage F run (150 images, full 6+6+2 context bank). When a person
image and a one-sentence text **context** carry conflicting appraisal cues, which modality does the
shared appraisal read-out follow, and can a text-derived Δμ steering vector arbitrate? Three findings:

1. **Balanced multimodal integration (not image dominance).** With a realistic bank of contexts,
   image and text contribute **comparably** — standardized β_img ≈ 0.32–0.37 vs β_txt ≈ 0.31–0.37
   (dominance ratio 0.82 for the internal probe read-out, **1.16 for behavioral valence**, i.e. text
   slightly leads the output). Both cues are **sign-correct** for both read-outs: image and text write
   valence into the *same* representation.
2. **The context effect is roughly symmetric.** Both a positive context lifting a negative image
   (+0.370) and a negative context lowering a positive image (−0.187) are substantial; the pilot's
   apparent "negativity asymmetry" does **not** replicate once the full bank is used.
3. **Steering arbitrates the conflict (causal).** On the incongruent cells, the text-derived
   **pleasantness Δμ** drives behavioral valence at slope **+0.215 (~65 % of Stage D's ~0.33**, within
   ±50 %) — the shared appraisal handle moves the output through the conflict, though a strong context
   is now a real competitor.

**Verdict: SIGNAL.** Balanced integration with both cues sign-correct and causal arbitration — the
cleanest evidence in the project that image and text share one appraisal representation rather than
running modality-siloed systems.

> **Supersedes the pilot.** An earlier 40-image / one-context-per-polarity pilot reported
> *image-dominance* (ratio ~0.15–0.22) and a *negativity asymmetry*. Both were artifacts of using a
> single, unusually weak positive context and a strong negative one. The full bank (6 contexts per
> polarity, 150 images) is the authoritative run; see "Why the bank matters."

## Setup
- **Model / site:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Read-out = the **frozen Stage
  A pleasantness probe** at `blocks.18.hook_attn_out`, last prompt token (Stage C machinery, never
  re-fit). Behavioral valence = closed-vocab P(positive) − P(negative) over the 13 emotion labels.
  Both captured in one forward under `torch.no_grad()`.
- **Images:** the **75 highest- and 75 lowest-EMOTIC-valence** test images (true extremes, from the
  EMOTIC test parquet). 150 persons → **121 distinct images** (EMOTIC is per-person; disclosed).
- **Context bank (full run):** 6 positive, 6 negative, 2 neutral generic one-sentence contexts, plus a
  no-context condition — **15 conditions per image**, `text_code ∈ {−1, 0, +1}` (negative / none≡neutral
  / positive). Prompt = the Stage C/D image prompt with `Context: {ctx}` inserted; last-token read-out
  position unchanged. 150 × 15 = **2,250 base forwards**.
- **Arbitration pass:** incongruent cells (positive image + negative context; negative image + positive
  context), sweep the text-derived pleasantness Δμ at `blocks.18.hook_resid_post`, β ∈ {−3…+3}. 150
  cells × (6+1) = **1,050 forwards**.
- **Regression:** standardized OLS `readout ~ z(image_valence) + text_code`, separately for the probe
  read-out and behavioral valence; dominance ratio = |β_txt| / |β_img|.

## Result 1 — modality integration: balanced, both sign-correct (SUPPORTS shared read-out)
Standardized OLS over the 2,250 base rows:

| Read-out | β_img (image valence) | β_txt (text context) | \|β_txt\|/\|β_img\| | pattern |
|---|---:|---:|---:|---|
| Frozen pleasantness probe (attn_out L18) | +0.374 | +0.306 | 0.82 | balanced |
| Behavioral valence (logits) | +0.322 | +0.374 | **1.16** | balanced (text-leaning) |

- Both cues enter with the **theory-predicted sign** for **both** the internal probe read-out and the
  model output: higher image valence → higher read-out, more-positive context → higher read-out.
- The two are **comparable in magnitude** (ratios 0.82 and 1.16); for behavioral valence the text
  context slightly *leads*. The model genuinely integrates both modalities — it neither ignores the
  caption nor lets the image drown it out.
- **Note on cross-run comparison.** Absolute β magnitudes are not directly comparable to the pilot
  (15 within-image conditions vs 4 change the variance decomposition); the **dominance ratio** is the
  robust quantity, and it moves decisively from image-led (pilot 0.15–0.22) to balanced (0.82–1.16).

## Result 2 — context effect is roughly symmetric (negativity asymmetry does NOT replicate)
Mean Δ behavioral valence vs the no-context baseline, per image group × context polarity:

| Image group | + positive context | + negative context | + neutral context |
|---|---:|---:|---:|
| Positive (high-valence face) | +0.577 | −0.187 | +0.410 |
| Negative (low-valence face) | +0.370 | −0.019 | +0.037 |

- **Both polarities move valence substantially**, and — unlike the pilot — a positive context clearly
  lifts a negative image (**+0.370**). The pilot's −0.475-vs-+0.049 "negativity asymmetry" was a
  weak-positive-context artifact and is **retracted**.
- **A neutral-context / baseline confound (under diagnosis).** Neutral context raises valence on
  positive images (+0.410) almost as much as a positive context — implying the *presence of a framing
  sentence*, not its polarity, shifts the "vs no-context" delta. The **clean estimate of the text
  effect is therefore the OLS `text_code` coefficient** (neutral and none both coded 0), which is the
  balanced β_txt ≈ +0.31–0.37 above. `analyze_stage_f` now emits a per-condition RAW-means breakdown
  (both read-outs, per context_id) to localize whether it is the `none` baseline that is low or a
  specific neutral context that is high, and whether it appears in the probe read-out or only in
  behavioral valence. Treated as a caveat, not a headline, pending that breakdown.

## Result 3 — steering arbitrates the conflict (SUPPORTS causal shared handle)
On the incongruent cells, injecting the text-derived **pleasantness Δμ** at resid_post L18:

| Metric | slope vs β | reference |
|---|---:|---|
| **Behavioral valence** | **+0.215** | Stage D single-direction ≈ +0.33 (~65 %, within ±50 %) |
| Frozen probe read-out | +0.000 | expected ~0 — read at attn_out L18, **upstream** of the injection |

- The text-derived valence direction drives the emotion output **through the modality conflict**,
  the causal complement to Result 1's correlational integration. The slope is lower than the pilot's
  +0.35, consistent with the balanced-integration picture: a strong context is now a real competitor
  the steering must overcome.
- The probe read-out slope is **0.000 by construction** (probe site upstream of the injection);
  recorded to demonstrate the invariance, not scored.

## Methodological notes
- **Full context bank, averaged.** Six contexts per polarity remove the single-sentence sensitivity
  that skewed the pilot; results are averaged over the bank with `context_id` retained.
- **OLS `text_code` is the clean text-effect estimate**; the "vs no-context" deltas carry a
  framing-sentence confound (Result 2) and are diagnostic, not primary.
- **Image valence is EMOTIC ground truth (1–10), z-scored** — an external anchor, so Result 1 is not
  circular.
- **Two read-outs, one forward:** probe (internal) and behavioral valence (output) captured together,
  so dominance is compared at both levels.
- **Arbitration read on behavioral valence only** — the frozen probe is upstream of the injection.

## Threats to validity
- **Single seed / one layer / one model.** 150 persons (121 distinct images), layer 18, Gemma-3-4B,
  one seed. A 3-seed repeat is the remaining robustness step.
- **Per-person duplication, no localization.** Whole image fed with "this person" (no EMOTIC box); a
  photo can appear as both a high- and low-valence person.
- **Neutral/vs-none confound (Result 2)** under active diagnosis; the OLS coefficient is used as the
  clean text-effect estimate meanwhile.
- **Arbitration uses one opposing context per cell** (not the full bank) to bound compute; the slope
  may vary with context strength.
- **Pilot ≠ full design.** The pilot's image-dominance/negativity-asymmetry do not survive the bank —
  a cautionary note on single-context conflict tests, not a contradiction.

## Why the bank matters (pilot → full)
| | Pilot (40 img, 1 ctx/polarity) | Full (150 img, 6+6+2 bank) |
|---|---|---|
| Probe read-out ratio | 0.22 (image-led) | 0.82 (balanced) |
| Behavioral valence ratio | 0.15 (image-led) | 1.16 (balanced, text-leaning) |
| Positive ctx on negative image | +0.049 | +0.370 |
| Arbitration slope | +0.350 | +0.215 |
The pilot's single positive context was weak and its single negative context strong; averaging over a
realistic bank reveals balanced integration and a symmetric context effect.

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
% Stage F — modality conflict (EMOTIC extremes, 150 persons / 121 images, full bank, n=2250 base rows).
\begin{tabular}{lrrr}
\toprule
Read-out & $\beta_{\text{img}}$ & $\beta_{\text{txt}}$ & $|\beta_{\text{txt}}|/|\beta_{\text{img}}|$ \\
\midrule
Frozen pleasantness probe (attn\_out L18) & $+0.374$ & $+0.306$ & $0.82$ \\
Behavioral valence (logits)               & $+0.322$ & $+0.374$ & $1.16$ \\
\bottomrule
\end{tabular}
% Balanced integration, both sign-correct. Arbitration: pleasantness Δμ steers behavioral valence at +0.215 (~65% of Stage D 0.33).
```
