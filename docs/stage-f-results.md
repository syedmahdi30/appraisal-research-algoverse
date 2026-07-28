# Stage F results — modality conflict & steering arbitration (Gemma-3-4B)

Paper-ready summary of the Stage F pilot. When a person image and a one-sentence text **context**
carry conflicting appraisal cues, which modality does the shared appraisal read-out follow, and can
a text-derived Δμ steering vector arbitrate the conflict? Three findings:

1. **Image dominates, but text is real.** Both the frozen appraisal read-out and the model's
   behavioral valence are **image-led** (standardized β_img ≈ 0.84–0.89 vs β_txt ≈ 0.13–0.20;
   dominance ratio 0.15–0.22), yet the text context's contribution is **sign-correct and non-trivial**
   — the model does not ignore the caption.
2. **A negativity asymmetry.** The one large, clean context effect is **a negative caption on a
   positive image** (Δ valence = **−0.475**); a positive caption barely moves a negative image
   (+0.049). A grim narrative can override a smile far more than a cheerful narrative can override a
   frown.
3. **Steering arbitrates the conflict (causal).** On the incongruent cells, injecting the text-derived
   **pleasantness Δμ** drives behavioral valence at slope **+0.350 — within ±50 % of Stage D's ~0.33**
   — i.e. the shared appraisal handle moves the output through the conflict regardless of which
   modality was winning.

**Verdict: SIGNAL.** Image-led with a real, sign-correct text effect and full-strength causal
arbitration — the cleanest test in the project of a *shared* (not modality-siloed) appraisal
representation.

## Setup
- **Model / site:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Read-out = the **frozen
  Stage A pleasantness probe** at `blocks.18.hook_attn_out`, last prompt token (Stage C machinery,
  never re-fit). Behavioral valence = closed-vocab P(positive) − P(negative) over the 13 emotion
  labels at the last token. Both captured in one forward under `torch.no_grad()`.
- **Images:** the **20 highest- and 20 lowest-EMOTIC-valence** test images (true extremes, taken
  directly from the EMOTIC test parquet). 40 persons → **29 distinct images** (EMOTIC is per-person;
  disclosed under threats).
- **Contexts (one per polarity, seed 0):** positive = *"This photo was taken at a celebration held in
  their honor."*, negative = *"This photo was taken just after they lost their job."*, neutral =
  *"This photo was taken indoors."* Prompt = the Stage C/D image prompt with `Context: {ctx}` inserted
  between the image and the question; last-token read-out position unchanged.
- **Base pass:** 4 conditions per image — no-context, +positive, +negative, +neutral — coded
  `text_code ∈ {−1, 0, +1}` (negative / none≡neutral / positive). 40 × 4 = **160 forwards**.
- **Arbitration pass:** on the **incongruent cells** (positive image + negative context; negative
  image + positive context), sweep the text-derived pleasantness Δμ at `blocks.18.hook_resid_post`,
  β ∈ {−3…+3}. 40 cells × (6+1) = **280 forwards**.
- **Regression:** standardized OLS `readout ~ z(image_valence) + text_code`, run separately for the
  frozen probe read-out and behavioral valence; β's are standardized slopes, and dominance ratio =
  |β_txt| / |β_img|.

## Result 1 — modality dominance: image-led, text sign-correct (SUPPORTS shared read-out)
Standardized OLS over the 160 base rows:

| Read-out | β_img (image valence) | β_txt (text context) | \|β_txt\|/\|β_img\| | pattern |
|---|---:|---:|---:|---|
| Frozen pleasantness probe (attn_out L18) | **+0.892** | **+0.197** | 0.22 | image-led |
| Behavioral valence (logits) | **+0.843** | **+0.125** | 0.15 | image-led |

- Both cues enter with the **theory-predicted sign** (higher image valence → higher read-out; more
  positive context → higher read-out), for **both** the internal probe read-out and the model's
  output valence.
- The image cue is **~5–7× stronger** than the text cue — the model anchors on what the face shows —
  but the text contribution is far from zero: a caption measurably shifts both the internal appraisal
  read-out and the emotion output.

## Result 2 — a negativity asymmetry in the context effect
Mean Δ behavioral valence vs the no-context baseline, per image group × context polarity:

| Image group | + positive context | + negative context | + neutral context |
|---|---:|---:|---:|
| Positive (high-valence face) | +0.011 | **−0.475** | −0.012 |
| Negative (low-valence face) | +0.049 | −0.094 | −0.104 |

- The single large, clean effect is **positive image + negative context (−0.475)**: a tragic caption
  substantially reframes a happy face toward negative.
- The mirror is weak: **negative image + positive context (+0.049)** barely lifts a sad face. Positive
  captions are near-inert; a positive image is already near the valence ceiling.
- On negative images, adding *any* sentence drifts valence slightly down (neutral −0.104 ≈ negative
  −0.094), so the negative-image row is largely within pilot noise except for the small positive lift.
- **Reading:** text context arbitrates asymmetrically — negativity in language is far more potent at
  overriding the image than positivity is. (Consistent with a negativity bias; to be confirmed at
  scale with the full 6-context bank.)

## Result 3 — steering arbitrates the conflict (SUPPORTS causal shared handle)
On the incongruent cells, injecting the **text-derived pleasantness Δμ** at resid_post L18:

| Metric | slope vs β | reference |
|---|---:|---|
| **Behavioral valence** | **+0.350** | Stage D single-direction ≈ +0.33 (within ±50 %) |
| Frozen probe read-out | +0.000 | expected ~0 — read at attn_out L18, **upstream** of the injection |

- The text-derived valence direction drives the emotion output **at full Stage-D strength straight
  through the modality conflict** — the shared appraisal handle can override whichever modality was
  winning. This is the causal complement to the correlational dominance in Result 1.
- The probe read-out slope is **0.000 by construction**: the probe site (`attn_out L18`) is upstream
  of the `resid_post L18` injection, so it *cannot* reflect the steering. It is recorded to
  demonstrate the invariance (a transparency check), **not** scored as an arbitration outcome.

## Methodological notes
- **Standardized OLS** so β_img and β_txt are directly comparable; `text_code` is the appraisal
  polarity (−1/0/+1) with no-context ≡ neutral ≡ 0.
- **Image valence is EMOTIC ground truth (1–10), z-scored** — an external anchor, not the model's own
  output, so Result 1 is not circular.
- **Two read-outs, one forward:** the frozen probe (internal representation) and behavioral valence
  (output) are captured together, so dominance can be compared at both levels.
- **Arbitration read on behavioral valence only.** Because the frozen probe is upstream of the
  injection, arbitration is (and can only be) judged on the output; this was designed in, not a
  limitation discovered late.

## Threats to validity
- **Pilot scale / single seed / one layer / one model.** 40 persons (**29 distinct images**), one
  context per polarity, layer 18, Gemma-3-4B. Promote with the full 6-context bank, ≥3 seeds, and
  150+ images.
- **Per-person duplication, no localization.** Whole image fed with "this person" (no EMOTIC box);
  40 persons collapse to 29 images, and a photo can appear as both a high- and low-valence person.
- **Context coding is coarse.** `text_code ∈ {−1,0,+1}` treats all negative (all positive) contexts as
  identical and folds no-context into neutral; the full run should model context identity and average
  over the bank.
- **Negativity asymmetry is one number.** −0.475 rests on ~20 positive-image cells with a single
  negative caption; needs the full bank + seeds before it is a claim rather than an observation.
- **Probe arbitration is unmeasurable at the frozen site** (upstream of the injection); a downstream
  read-out (or a later-layer probe) would be needed to see steering move the *internal* read-out.

## Relationship to the arc
| Stage | Question | Result |
|---|---|---|
| C | does the text read-out survive on images? | yes (ρ=0.51), beyond captions |
| D | does a text valence direction steer image behavior? | yes (causal, slope ~0.33) |
| E | do appraisal *combinations* make specific emotions? | anger composes (after de-correlation) |
| **F** | when image & text conflict, who wins, and can steering arbitrate? | **image-led but text is real; pleasantness Δμ arbitrates at full strength** |

Stage F is the direct test of a **shared** representation: if image and text fed *separate* appraisal
systems, a single text-derived direction could not arbitrate an image-driven emotion — but it does,
at Stage-D magnitude. Together with C/D/E this is convergent evidence for one modality-agnostic
appraisal geometry that both modalities write into and that steering can drive.

## Reproduce
```bash
python -m src.experiments.stage_f_conflict --limit 40   # base pass (160 forwards)
python -m src.experiments.stage_f_conflict --arbitrate  # arbitration pass (280 forwards)
python -m src.experiments.analyze_stage_f               # OLS dominance + asymmetry + arbitration
```
Artifacts: `results/stage_f/{conflict_pilot.parquet, conflict_metrics.json, arbitration_pilot.parquet,
arbitration_metrics.json, conflict_analysis.json}`, `results/figures/stage_f_conflict.png`.
Config: `config/stage_f.yaml`.

## LaTeX table
```latex
% Stage F — modality conflict (EMOTIC extremes, 40 persons / 29 images, n=160 base rows).
\begin{tabular}{lrrr}
\toprule
Read-out & $\beta_{\text{img}}$ & $\beta_{\text{txt}}$ & $|\beta_{\text{txt}}|/|\beta_{\text{img}}|$ \\
\midrule
Frozen pleasantness probe (attn\_out L18) & $+0.892$ & $+0.197$ & $0.22$ \\
Behavioral valence (logits)               & $+0.843$ & $+0.125$ & $0.15$ \\
\bottomrule
\end{tabular}
% Image-led, both sign-correct. Arbitration: pleasantness Δμ steers behavioral valence at +0.350 (Stage D ~0.33).
% Asymmetry: positive image + negative context = −0.475; negative image + positive context = +0.049.
```
