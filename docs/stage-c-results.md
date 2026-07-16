# Stage C results — cross-modal appraisal read-out transfer (Gemma-3-4B)

Paper-ready summary of the cross-modal read-out gate. The frozen text-trained appraisal
probe, applied **unchanged** to Gemma's image-conditioned activations, tracks EMOTIC
ground-truth valence far above chance (**pleasantness Spearman ρ = 0.507 on the full test
split**), and this transfer is **not merely verbalization-mediated**: controlling for a
neutral *and* a rich perceptual caption of each image, the image read-out retains a
significant unique valence contribution (semipartial ρ = 0.201, p < 0.001). The claim is
kept correlational — a surviving residual is an *upper bound* on any non-verbal appraisal
representation, which the Stage D causal test is designed to resolve.

## Setup
- **Model / site:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Read-out at the
  **frozen Stage A probe site** — language-model block `blocks.18.hook_attn_out`, last prompt
  token — under image conditioning (SigLIP → projector → LM). Probes are **frozen from Stage A**
  and never re-fit on image data.
- **Data:** EMOTIC test partition (official split). Per-person rows; continuous
  Valence/Arousal/Dominance (1–10) and 26 discrete categories. Ground-truth anchor for the
  appraisal probes is **continuous valence** (pleasantness ↔ valence; unpleasantness ↔ −valence);
  the other four appraisals have **no image-side ground truth** and are not scored here.
- **Prompt:** `IMAGE_EMOTION_PROMPT` (`<start_of_image>` + "What single emotion is this person
  feeling?"). Image forwards run under `torch.no_grad()` (the SigLIP tower does 4096-patch eager
  attention; the retained graph otherwise OOMs a 40 GB A100 on the first image).
- **Metrics are scale-invariant by design.** The probe predicts crowd-enVENT pleasantness on a
  **1–5** scale; EMOTIC valence is **1–10**. Raw r² is scale-confounded (a perfect rank match
  reads as failure from the offset alone), so we report **Spearman/Pearson correlation** and a
  categorical **polarity AUC** (shared-7 emotion sign), never bare r².

## Result 1 — Read-out transfers (SUPPORTS)
The frozen pleasantness probe tracks EMOTIC valence under image conditioning; unpleasantness
mirrors it. Full test split, n = 7,280.

| Appraisal | ρ (Spearman) vs valence | Pearson | polarity AUC | vs random null | text→image retention |
|---|---:|---:|---:|---:|---:|
| Pleasantness   | **+0.507** | +0.524 | 0.898 (n=440) | p = 0.010 | 0.66 |
| Unpleasantness | **−0.448** | −0.473 | 0.123 (mirror) | p = 0.010 | 0.60 |

- **Null / control.** A 100-draw **norm-matched random-direction** null (Gemma's activations are
  strongly anisotropic → random directions correlate non-trivially with valence: mean |ρ| = 0.128,
  p95 = 0.300, max = 0.362). The probe clears all 100 draws (empirical p = 0.010).
- **Polarity AUC** (does the read-out rank positive-emotion images — shared-7 = joy — above negative
  ones — anger/disgust/fear/sadness?) = **0.898** on the 440 single-label images; unpleasantness is
  the mirror (0.123 = 1 − 0.877).
- **Retention** = |image ρ vs valence| / |text ρ vs the 1–5 rating| ≈ **0.6** — roughly 60% of the
  text-side effect survives crossing into images.

Figure: `results/figures/stage_c_readout.png` (frozen read-out vs EMOTIC valence, per appraisal).

## Result 2 — Not merely verbalization-mediated (SUPPORTS, correlational)
Read-out transfer alone cannot separate **shared representation** from **verbalization**
(the model internally captions the image, and the appraisal direction reads that implicit
text). We generate a caption per image, run it through the **text** pipeline, apply the **same
frozen probe**, and measure the image read-out's **unique** contribution to valence after
controlling for the caption (rank-based **semipartial** correlation). Two caption richness
levels bound the verbalization hypothesis (mechanism subset, n = 1,000):

| Control | image ρ | caption ρ | image↔caption ρ | **unique(image)** |
|---|---:|---:|---:|---:|
| Neutral caption ("Describe this image in one sentence.") | 0.482 | 0.379 | 0.653 | **+0.310** (p<0.001) |
| Rich caption (expression/posture/body-language, prose)  | 0.482 | 0.429 | 0.697 | **+0.256** (p<0.001) |
| **Neutral + rich jointly** | 0.482 | — | — | **+0.201** (p<0.001) |

Unpleasantness mirrors throughout: **−0.279 → −0.198 → −0.153** (all p < 0.001).

- **The progression 0.310 → 0.256 → 0.201** is the result: richer/joint caption controls absorb
  *some* of the image's advantage (caption ρ rises 0.379 → 0.429; image↔caption alignment rises
  0.653 → 0.697), but a **substantial, highly significant residual survives** controlling for a
  plain **and** a detailed perceptual caption **jointly**.
- **Interpretation.** The transfer is **not explained by verbalization** at either richness level —
  the image representation carries valence-relevant appraisal signal beyond what detailed perceptual
  description captures. This is favorable to a shared-representation account.

Figures: `results/figures/stage_c_caption_baseline.png` (neutral),
`stage_c_caption_baseline_rich.png` (rich).

## Methodological notes
- **Scale-invariant metrics, not r².** The 1–5 probe vs 1–10 valence mismatch makes raw r² read as a
  false null; Spearman/Pearson + polarity AUC are the correct, scale-free measures.
- **Anisotropic null.** A single random control is misleading here — we report the full 100-draw
  distribution and an empirical p-value.
- **Semipartial mechanism.** `part_r = (r_iv − r_cv·r_ic)/√(1−r_ic²)`, with the p-value from the
  t-test on adding the image rank to an OLS of valence rank on the caption rank(s); validated on
  synthetic signal-present / signal-absent cases. The joint control is a rank-OLS with both captions.
- **Captions must be prose.** The probe is trained on natural-sentence text; a markdown/bulleted
  caption is out-of-distribution and artificially inflates the residual. The rich prompt forces two
  or three sentences of plain prose (Gemma's meta preamble/headers are stripped).
- **Determinism / persistence.** Captions are greedy (reproducible). Per-image image + caption
  read-outs are persisted (`caption_readout{,_rich}.parquet`) so mechanism re-analyses (semipartial,
  richer captions) need no re-generation.

## Threats to validity
- **Correlational, and an upper bound.** A surviving semipartial residual bounds — but does not
  prove — a non-verbal appraisal representation; a still-richer caption could absorb more. Causal
  confirmation is Stage D. Do **not** read Result 2 as established shared multimodal geometry.
- **No appraisal ground truth on images.** Pleasantness/unpleasantness are anchored to continuous
  valence and shared-7 polarity — both proxies. The other four appraisals are untested here.
- **Lossy label mapping.** EMOTIC-26 → shared-7 and single-label filtering are lossy (polarity AUC
  rests on the 440 single-label images); the mapping is a first-class caveat.
- **No person localization.** EMOTIC ships per-person bounding boxes, but we feed the **whole image**
  and ask about "this person" without the box. For multi-person images the model cannot know which
  person, yet we compare to one person's valence — added noise the signal survived, but disclosed.
- **Anisotropy.** Random directions reach |ρ| ≈ 0.36 with valence; the probe clears the null
  (p = 0.010) but the margin over the *best* random direction is modest — hence the emphasis on the
  polarity AUC and the semipartial as corroborating, independent signals.
- **Single seed / layer / model.** One split/seed, layer 18 only, Gemma-3-4B only; the mechanism
  subset is n = 1,000 while the read-out headline is the full n = 7,280.

## Carried into Stage D
- Result 2 is correlational; **Stage D cross-modal steering** is the causal capstone — inject the
  appraisal direction under image input and measure the emotion-output shift.
- Reuse the **Stage A difference-of-means recipe** (`stage_a_steering_v2`), **not** the probe-direction
  / residual-fraction approach that failed on text. Directions and the critical layer (18) are frozen
  from Stage A.

## Reproduce
```bash
python -m src.experiments.stage_c_transfer --full            # read-out on the full test split (Result 1)
python -m src.experiments.stage_c_caption                    # neutral caption baseline (Result 2)
python -m src.experiments.stage_c_caption --style rich       # rich caption robustness (Result 2)
python -m src.experiments.analyze_stage_c_mechanism          # CPU-only combined semipartial
```
Artifacts: `results/stage_c/{metrics.json, caption_metrics.json, caption_metrics_rich.json,
caption_readout{,_rich}.parquet, mechanism_summary.json}`,
`results/figures/stage_c_{readout, caption_baseline, caption_baseline_rich}.png`.

## LaTeX tables
```latex
% Result 1 — read-out transfer (full EMOTIC test, n=7280)
\begin{tabular}{lrrrr}
\toprule
Appraisal & $\rho$ vs valence & polarity AUC & vs null & retention \\
\midrule
Pleasantness   & $+0.507$ & $0.898$ & $p{=}0.010$ & $0.66$ \\
Unpleasantness & $-0.448$ & $0.123$ & $p{=}0.010$ & $0.60$ \\
\bottomrule
\end{tabular}

% Result 2 — verbalization controls (mechanism subset, n=1000)
\begin{tabular}{lrrr}
\toprule
Control for & image $\rho$ & caption $\rho$ & unique(image) \\
\midrule
Neutral caption       & $0.482$ & $0.379$ & $+0.310^{***}$ \\
Rich caption          & $0.482$ & $0.429$ & $+0.256^{***}$ \\
Neutral $+$ rich       & $0.482$ & ---     & $+0.201^{***}$ \\
\bottomrule
\end{tabular}
% *** p < 0.001. Unpleasantness mirrors: -0.279 / -0.198 / -0.153.
```
