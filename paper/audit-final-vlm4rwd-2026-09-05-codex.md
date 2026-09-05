# VLM4RWD submission review — 5 September 2026

Reviewed commit `e715429`, `paper/neurips_2026.tex` in the long configuration, and `paper-build/long.pdf`. This is a review, not an edited manuscript. No experiment outputs or submission artifacts were changed.

## Verdict

The paper has a coherent, credible workshop contribution and fits VLM4RWD. Its strongest story is a controlled, setup-dependent valence asymmetry, followed by a separately scoped localization study and a warning about evaluation readouts. The main quantitative findings checked below reproduce. Nevertheless, I would not give the current version an unconditional submission-ready or “all results sound” verdict: several scientific reporting and provenance issues remain, including an unsupported appendix pilot and inconsistent definitions of the text-only control.

This is plausibly competitive workshop work after correction, not a guarantee of acceptance or of NeurIPS main-conference novelty/generalization. The narrow dataset/task, six selected pairs, model-specific behavioral result, and unmatched mechanism analysis remain substantive limitations even when correctly disclosed.

## Submission compliance and verification

- The [official workshop instructions](https://vlm4rwd.github.io/) allow eight pages excluding references and appendices, require NeurIPS 2026 formatting and full anonymization, and explicitly include failure-mode evaluation and interpretability. The current PDF has eight main-text pages; references begin on page 9. It has 32 pages including references, appendices, and the checklist.
- The title block is anonymous; PDF metadata contains no author identity. This is not a certification of an unseen OpenReview form or code attachment.
- Fresh compilation succeeded. Extracted text from the fresh PDF is identical to the current `paper-build/long.pdf`. No undefined-reference or overfull-box warning was found in the fresh log. There are underfull-box warnings and a package UTF-8 warning. Main pages 1–8 were visually inspected; the layout is clean. Appendix bitmap figures have issues described below.
- `MPLCONFIGDIR=/private/tmp/appraisal-mpl python3 -m pytest -q`: **201 passed in 20.90 seconds**. These tests do not certify scientific validity or every manuscript number.
- The workshop website lists **September 5, 2026** as the submission deadline, without a timezone. Confirm the exact cutoff in the live [OpenReview portal](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/VLM4RWD); do not infer midnight local time from the website date.

## Highest-priority corrections

### 1. Remove or revalidate the 97% anger pilot

Location: `paper/neurips_2026.tex:681`, “Combining directions (supporting evidence).”

The paragraph presents a 30-image, 97%-anger result as supporting evidence for meaningful appraisal structure. Its available artifact, `results/stage_e/combo_pilot_metrics_valence.json`, is dated July 28 and names commit `7a4f9fa`. The runner `src/experiments/stage_e_combo.py:135` obtains image-conditioned logits with `bridge.run_with_hooks`; it boots the bridge at line 185. The repository's `docs/bridge-bug-2026-08-22.md` documents that this multimodal forward pass corrupts output logits and categorical decisions. I found no reference-HuggingFace replacement for this pilot.

This does not prove the effect is false, but it means the reported categorical rate is not validated. Being in an appendix or “not required” does not make it reliable supporting evidence. **Remove the paragraph, or rerun it on the validated stack before retaining it.** Removing it would not damage the central conflict story.

### 2. The text-only control is not the margin defined in the paper

Locations: `paper/neurips_2026.tex:140`, `:197`, `:739`; `src/experiments/shared/reporting.py:663`; `src/experiments/analyze_stage_f_unbounded.py:49`.

The paper defines the unbounded score as `logsumexp(positive label log-probabilities) - logsumexp(negative label log-probabilities)` and says the text-only control uses that score. The control helper instead subtracts the **maximum** negative-label log-probability from the maximum positive-label log-probability. These are different statistics.

Recomputing the same Qwen matched text-only parquet gives:

| Statistic | Mean paired magnitude difference | Pooled ratio | Two-sided signed-rank p |
| --- | ---: | ---: | ---: |
| Best-label margin, actually used | +0.328125 | 1.020761 | 0.6875 |
| Summed-category margin, described | −0.010659 | 0.999316 | 0.84375 |

The published numbers are reproducible, but under the wrong stated definition. **Use one consistent definition and regenerate all affected control numbers, or explicitly distinguish the best-label control from the category-mass margin.** This includes the image-present comparison and other models' text-only ratios; do not change only the displayed p-value. Both calculations remain nonsignificant, so this discrepancy does not overturn the main behavioral finding.

### 3. A nonsignificant six-pair test does not exclude a large imbalance

Locations: abstract `:70`, introduction `:103`, matched results `:197`, control discussion `:741–743`.

“This rejects a large wording imbalance” is not supported by a failure to reject zero with six pairs. Nor does the test establish that only small imbalances remain possible. A signed-rank test is also not directly a test of the arithmetic mean. The manuscript appropriately disclaims equivalence for the steering comparison; apply the same discipline here.

Suggested replacement: **“The six text-only pairs show no detectable asymmetry, but this small control does not establish equivalence in wording strength.”** To support an exclusion claim, specify a meaningful equivalence margin and report a suitable interval/test. To claim an image-dependent amplification, test the image-present minus text-only contrast directly on a consistent scale; significance in one condition and nonsignificance in another is not that interaction test.

### 4. Cross-image uncertainty ignores repeated photographs

Locations: `paper/neurips_2026.tex:812`; `src/experiments/shared/patching.py:303`.

The cross-image bootstrap resamples 60 rows as independent pairs. The surviving parquet contains 60 distinct pairs but only **51 unique donor photographs and 47 unique recipient photographs**. Different pairs therefore share inputs. This is a different dependence problem from the same-image duplicates, which are correctly collapsed.

**Reassess cross-image confidence intervals using resampling or a variance model that respects shared donor/recipient photographs.** Until then, do not treat the current non-overlapping intervals as conclusively calibrated evidence for the mid-band ordering. The point estimates are not disproved. Only the late-band per-row parquet is retained in this location; the interval helper reads earlier bands' summary JSONs rather than recomputing every interval from raw rows, contrary to the broad wording at `:806`. Recover the original rows if reanalysis of those bands is needed.

Separately, the matched behavioral bootstrap independently resamples positive and negative sentence columns, rather than resampling matched event pairs together (`analyze_stage_f_unbounded.py:95`). Align this with the paired design or explicitly describe the independent-polarity procedure. A read-only, pair-preserving sensitivity check with 10,000 draws and seed 0 gave approximately **[+0.18, +0.85]**, still excluding zero. Thus this check supports the headline's qualitative robustness; it is not a replacement published analysis.

## Other corrections before submission

### 5. Correct the connector descriptions

`paper/neurips_2026.tex:278` and `:307` repeatedly call both LLaVA connectors linear. LLaVA-1.5 uses an MLP projection; LLaVA-NeXT's implementation has two linear layers separated by a nonlinear activation, and the actual Mistral checkpoint specifies GELU. Use **“MLP projector”** or “two-layer MLP connector.” Sources: [official LLaVA documentation](https://huggingface.co/docs/transformers/model_doc/llava), [NeXT checkpoint configuration](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/main/config.json), [NeXT implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/modeling_llava_next.py).

### 6. Fix the remaining legacy figures and caption-control provenance

The appendix includes `paper/figures/stage_c_readout.png`, whose pleasantness plot says **rho = +0.507**, while the text and current HF summary say **+0.510**. This is the old bridge plot, not a rounding difference. Regenerate it from the validated run, or remove it; relabeling its title alone would misrepresent its data.

`paper/figures/stage_c_caption_baseline.png` has a visibly clipped title and internal-stage language. The caption-adjusted +0.310/+0.256/+0.201 values trace to the July `mechanism_summary.json` and bridge-derived image readouts in `stage_c_caption.py:187`. The HF transfer runner does not recompute these caption controls. This is weaker evidence of a problem than the categorical pilot: the image probe was subsequently shown to transfer on HF. Nevertheless, the exact caption-adjusted estimates have not thereby been revalidated. State their provenance and limitation, recompute on reference-stack readouts, or omit them. The paper's blanket reference-implementation description currently hides this distinction.

### 7. Identify the estimands in the label-balance paragraph

`paper/neurips_2026.tex:303` puts “35/35” graded and “20/35” categorical results together without identifying that `analyze_label_balance.py` uses the **matched, positive-image within-item contrast** for the former and the **varied-set, uncorrected override gap** for the latter. Both use photo-only intervals; this is not the crossed test emphasized in the surrounding section.

I reproduced 35/35, median +1.143729, and 20/35 with a minimum uncorrected gap of −0.713530. The numbers are sound for those definitions. Name the dataset, correction, and interval type. If instead reporting the corrected categorical measure, recompute its numbers: the worst balanced-subset gap is approximately **−76.8 percentage points**, not −71.4, although 20/35 still exclude zero on the positive side under the photo-only bootstrap.

### 8. Keep every headline properly scoped

- At `:109`, add **“on positive images”** to the more-than-four-times contribution. The six per-pair ratios of 4.3–5.5 compare negative and positive contexts on that group, not the two across-group conflict directions. The matched across-direction ratio is about 1.51.
- The 57% and 35% at `:99` and `:186` are **baseline-subtracted category rates**, not literal observed proportions of trials whose labels flipped relative to baseline. Say “baseline-corrected override rates” or describe the percentage-point excess. The correction is already explained elsewhere, but “flips on 57% of trials” invites the wrong interpretation.
- The abstract and deployment discussion equate context sensitivity with a demonstrated grounding failure. The task asks what a person is feeling, for which situational text can legitimately be informative; it does not establish a true emotion label under each counterfactual context. Frame this as **a potential grounding vulnerability when supplied context is misleading**, not proof that every change is an error.
- The introduction's claim that the effect “falls hardest” on people described badly is an unmeasured impact claim. Present it as a risk, not an observed distribution of harm.

## Story and language

Keep the current narrative order: controlled behavioral question → matched-pair result → separately scoped Gemma localization → limits across models and scoring rules. The explicit sentence that Gemma patching does **not** explain Qwen's asymmetry is important and should stay. The wording and person-grounding controls strengthen the story, as does reporting failed robustness rather than suppressing it.

The prose is generally specific and readable, not generic LLM boilerplate. Its weaknesses are repetition and rhetorical emphasis rather than technical terminology. Terms such as activation patching, probe, and log-odds are needed; removing them would hurt precision. Prefer these small changes:

- “clears all three crossed tests” → “has intervals excluding zero on all three readouts.”
- “manufactures a null” → “produces an apparent null result.”
- “the graded contrast carries the claim” → “we base this claim on the graded contrast.”
- The abstract is crowded with secondary results and caveats. Shorten it after correcting the scientific claims; avoid deleting the model and image-group qualifiers to save space.
- Fix “VEENA ... its cues agree instead of conflicting and does not test” at `:553` to “VEENA ... uses congruent cues and does not test ...”. Check singular-author “report” constructions.
- The checklist's broader-impact justification points to a dedicated main-text paragraph that is now in Appendix D (`paper/checklist.tex:143`). Update the cross-reference and remove stray spaces around commas. Verify whether an anonymous code archive is actually attached before changing its current “No” answer.
- The compute paragraph says its counts cover reported experiments, but omits the new question/frame and person-grounding controls. Update that claim/count scope; wall-clock totals remain undisclosed rather than experimentally verified.

## Results checked

| Finding | Check and outcome |
| --- | --- |
| Matched mirror contrast | Recomputed from saved Qwen rows: +0.496421; current crossed CI [+0.113875, +0.825339]. |
| Matched within-item contrast | Recomputed: +1.148007; photo CI [+0.943300, +1.344087]; all six positive-image ratios 4.328–5.454. |
| Four-model comparison | Recomputed bounded/unbounded contrasts and corrected point gaps from complete-label parquets; agrees with the current qualitative table conclusions. |
| Same-image patching | Recomputed over 51 unique photographs: pair 1 93.1667%, pair 2 87.8601%, consistent with 88–93%. |
| Label-balance sensitivity | Recomputed all 35 subsets; reproduces the reported values under the actual, differing definitions above. |
| Probe and steering summaries | Current HF artifact values support rho +0.509945, AUC +0.911784, and the reported steering slopes. This is artifact inspection, not a new GPU run. |

No GPU inference, new random-seed replication, full independent bibliography/novelty audit, or live submission-form audit was performed. Saved-output agreement is evidence for reproducibility, not proof that every experimental assumption is valid. The fixes above should be followed by a fresh eight-page build and an artifact-consistency check; the current 201 passing tests alone will not catch them.
