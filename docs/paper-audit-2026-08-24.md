# Paper Audit: When One Word Changes an Image Judgment: A Valence Asymmetry in Vision-Language Models

**Date:** 2026-08-24 · **Source:** `paper/neurips_2026.tex` (673 lines, incl. appendices), `paper/checklist.tex`, `paper/references.bib`, `paper/venue.tex`, built with tectonic 0.17.0

## The narrative as I read it

Vision-language models that see a photograph and read a conflicting sentence do not treat the two directions of conflict symmetrically. Holding the photograph and the described event fixed and flipping one valence word, negative wording moves Qwen3-VL-8B's emotion judgment four to five times farther than positive wording, on all six tested pairs; a no-image control argues the sentences themselves are not simply mismatched in strength. In Gemma-3-4B, replacement experiments locate downstream text-position states that restore 88–93% of the context effect, and a direction estimated only from text still moves the answer under conflict. Across four models the effect's size — and once, its sign — depends on which summary score is used and on whether multi-token emotion labels are scored whole, so the paper reports the cross-model comparison as a boundary on the claim rather than as a result.

That is a coherent, well-scoped story, and the paper tells it deliberately. The problems below are almost all about *delivery*, not substance.

## Summary

This is a careful draft with unusually honest calibration — negative results are reported as prominently as positive ones, the withdrawn measurements are disclosed, and the introduction explicitly disclaims novelty it cannot support. Three things block submission and are worth doing in this order. **First, length:** the body is 13 full pages plus a partial 14th, against limits of 8 (VLM4RWD) and 5 (Interp4Discovery). Nothing else matters as much, and the conditional-build machinery meant to produce the short version does not currently do anything (F3). **Second, a scope drop in the three most-read places:** the headline `+1.148` contrast is measured on positive images only, and the abstract, the introduction's finding 1, and the conclusion all state it without that restriction (F2). **Third, two mechanical defects a reviewer will see:** a stale numeric range that contradicts its own table (F4) and a red `[TODO: …]` that renders in the compiled PDF (F5).

The highest-leverage single edit is the abstract rewrite in F6, which fixes F2 at the same time.

## Findings

### F1. The body is 13 pages against limits of 8 and 5 — CRITICAL
**Where:** whole paper; limits confirmed against both live CFPs
**Issue:** Compiled with tectonic, the References heading lands on page 14, so the main text is 13 full pages plus a partial. VLM4RWD (Aug 30) allows 8 pages excluding references and appendices; Interp4Discovery (Aug 29) allows 5 (6 at camera-ready). This is a 40% cut for one venue and a 62% cut for the other.
**Why it matters:** A paper over the limit is desk-rejected without being read. With deadlines six and seven days out, this is the only finding with a hard clock on it.
**Suggested fix:** Both venues exclude appendices from the limit, so this is relocation, not deletion. Measured section costs, for planning:

| Section | pages | |
|---|---|---|
| 1 Introduction | 1.09 | |
| 2 Related work | 1.00 | best value-per-page to cut (F8) |
| 3 Experimental framework | 1.57 | scoring-rule derivation → appendix |
| 4 Reading/steering valence | 0.38 | already 90% in Appendix A (F8) |
| 5 Conflict resolution | 3.20 | 4 floats; `tab:factorial` → appendix |
| 6 Mechanism | 2.31 | 2 floats; `tab:crosspatch` → appendix |
| 7 How far it travels | 1.54 | keep `tab:models` |
| 8 Discussion | 0.79 | broader impacts → appendix |
| 9 Limitations | 0.95 | |
| 10 Conclusion | 0.46 | |

Use `scripts/build-paper.sh` to measure after each pass rather than estimating.

### F2. The headline contrast is stated without its scope restriction — CRITICAL
**Where:** abstract (~L53); §1 finding 1 (~L67); §10 Conclusion (~L411). Source of the restriction: §5.3 (~L214) and `tab:pairs` caption (~L217)
**Issue:** §5.3 states it precisely: *"On positive images it is $+1.148$ (bootstrap CI over 62 distinct photographs $[+0.943,+1.344]$)."* `tab:pairs` is captioned *"(\qwen{}, positive images, matched set)"*. But the abstract says only *"Across six pairs, negative wording moves \qwen{}'s judgment four to five times farther than positive wording; all six agree, and the contrast is $+1.148$"*, the introduction's finding 1 repeats the number with no restriction, and the conclusion opens with *"Flipping a single valence word … moves \qwen{}'s image-grounded emotion judgment four to five times farther when the word is negative."* A reader takes the claim to hold over the whole stimulus set.
**Why it matters:** This is the paper's single most-quoted number, sitting in its three most-read locations, stated more broadly than the experiment supports. It is also the easiest kind of overclaim for a reviewer to catch, and finding one costs the paper credit for the calibration it does elsewhere.
**Suggested fix:** Add the restriction in all three places — "on positive images" costs three words. The abstract rewrite in F6 already carries it.

### F3. The `\ifshort` build toggle does nothing — CRITICAL
**Where:** `paper/venue.tex`; `paper/neurips_2026.tex` L3–7
**Issue:** `venue.tex` documents that `\shorttrue` selects a 5-page Interp4Discovery build and `\shortfalse` an 8-page VLM4RWD build, and states that "every block guarded by `\ifshort` is RELOCATED to the appendix, never deleted." No `\ifshort` guard exists anywhere in the document — the only two occurrences of the string are the `\newif` declaration and a comment. Both targets compile to the same 13-page body; I built each and confirmed identical page counts.
**Why it matters:** The comments describe a capability that does not exist, so a build the author believes is available cannot be produced. Under deadline this is the kind of thing discovered at the wrong moment. (Flagging plainly: this scaffolding was added earlier in this session, ahead of the content work that would populate it — the comments describe the intended end state, not the current one.)
**Suggested fix:** Either land the guards as part of the F1 cut, or downgrade the comments in `venue.tex` to say the toggle is scaffolding and not yet wired, so the file stops asserting something false.

### F4. "Between 0.44 and 0.55" contradicts its own table — MAJOR
**Where:** §5.1 (~L151); source is the last row of `tab:models` (~L360)
**Issue:** The text reads *"the other three are image-led, between $0.44$ and $0.55$ (last row of Table~\ref{tab:models})."* That row gives $|\betatxt|/|\betaimg|$ as Qwen 1.14, LLaVA-NeXT 0.52, Gemma 0.44, LLaVA 0.68. "The other three" are 0.52, 0.44 and 0.68, so the range is 0.44 to **0.68**. The stated upper bound excludes the actual maximum.
**Why it matters:** The number is one line of prose away from the table that refutes it, in a paragraph justifying the choice of primary model. A reviewer who checks one number and finds it stale starts checking all of them — and this paper has already been through two rounds of late numeric correction, so it can least afford that impression.
**Suggested fix:** Replace with "between $0.44$ and $0.68$". Worth confirming the surrounding argument still reads correctly: 0.68 is still comfortably image-led, so the conclusion holds.

### F5. A red `[TODO: …]` renders in the compiled PDF — MAJOR
**Where:** Appendix H (Compute), end of the paragraph at L667
**Issue:** `\todo{Add wall-clock totals per run from the session logs for camera-ready; …}` sits unguarded in the appendix body. Confirmed present in the built PDF text, rendered by `\newcommand{\todo}` (L31) as red bracketed text. The *other* `\todo` in the file (L426, acknowledgements) is correctly suppressed by the `\if@anonymous` guard — so this one reads as an oversight rather than a convention.
**Why it matters:** Draft scaffolding visible in a submitted PDF signals the paper was uploaded without a final read.
**Suggested fix:** Comment the line out, or wrap it the way the acknowledgements block is wrapped. Better: define `\todo` to expand to nothing when `\if@anonymous` is true, so no future stray `\todo` can reach a submission.

### F6. The abstract is overloaded and switches models mid-argument — MAJOR
**Where:** abstract, ~L53 (~290 words)
**Issue:** The abstract carries four findings plus their caveats, and moves through three model scopes — Qwen for behavior, then *"For internal analysis, we use \gemma{}, where a text-trained valence readout is available,"* then "Across four models" — without giving the cold reader a reason for each switch. The Gemma sentence in particular explains a methodological choice before the reader knows why they should care. Combined with F2's missing scope restriction, the abstract needs a shape change, not sentence patches.
**Why it matters:** Most readers read only this. It currently asks them to hold three model scopes and four claims in mind across 290 words.
**Suggested fix:** Replace with the following (~235 words). Every claim is one the paper supports; the positive-image restriction from F2 is restored, and each model switch now states its own reason.

> When an image and an accompanying sentence disagree, does a vision-language model weigh both directions of disagreement equally? We pair EMOTIC photographs of people with one-sentence contexts and measure how a conflicting context shifts the model's emotion judgment away from a neutral-context baseline. Our sharpest test holds the photograph and the described event fixed and flips a single valence word (won$\leftrightarrow$lost, wonderful$\leftrightarrow$devastating). On positive images, negative wording moves \qwen{}'s judgment four to five times farther than positive wording; all six pairs agree, and the within-item contrast is $+1.148$ (95\% CI $[+0.94,+1.34]$). Run without an image, the same six pairs show no detectable difference, which argues against a large imbalance in sentence strength but cannot exclude a small one. A set of six unrelated events points the same way but does not survive resampling the events. In \gemma{}, where a text-trained valence probe is available, replacing downstream states at text positions restores $88$--$93\%$ of the context difference, and a direction estimated only from valenced text still moves the answer under conflict. Across four models, however, the conclusion depends on the summary score and on how multi-token labels are scored: scoring only a label's first piece manufactures a null in one model and reverses the categorical ordering in another. Negative text can therefore exert disproportionate influence on image judgments, but the measured size depends on model and measurement. Evaluations should test what each cue says, not only which modality carries it.

### F7. Superseded sources sit in the folder that gets uploaded — MAJOR
**Where:** `paper/neurips_2026old.tex`, `paper/neurips_2026.tex.pre-rescore-backup`, `paper/main.pdf`
**Issue:** Two superseded drafts and a stale PDF live alongside the live source. `neurips_2026old.tex` still contains the withdrawn `app:scale` appendix and the pre-reframe narrative; `main.pdf` was built before the last four commits and carries the old title. All three are untracked, so they are invisible to `git status` review but present on disk.
**Why it matters:** Overleaf projects and supplementary archives are routinely uploaded wholesale. A superseded draft containing retracted claims is the worst possible thing to ship beside the paper that retracts them — and for an anonymous submission, prior drafts can also carry identifying metadata.
**Suggested fix:** Move all three out of `paper/` — an `archive/` directory outside the upload root, or delete them since git holds the history. Rebuild `main.pdf` from current source (`scripts/build-paper.sh`) before submitting.

### F8. Two sections carry a page each without carrying the narrative — MAJOR
**Where:** §2 Related work (~L77–92, 1.00 pp); §4 Reading and steering valence (~L132–142, 0.38 pp)
**Issue:** Applying the "what breaks if it is cut?" test: §4 already delegates its substance to Appendix A and exists mainly to set up §6.3, so the body keeps a page of setup for one later paragraph. §2 is a full page of six paragraphs — it positions well rather than merely enumerating, which is a genuine strength, but it is 12.5% of an 8-page budget and 20% of a 5-page one.
**Why it matters:** These are the two places where F1's cut costs the least narrative. Identifying them now prevents the cut from reaching findings 1–3 later, which is where trimming does real damage.
**Suggested fix:** Collapse §4 to two sentences plus the Appendix A pointer, keeping the $\rho=0.510$ transfer number since §6.3 depends on it. Compress §2 to two paragraphs — modality conflict, and mechanistic VLM work — and relocate the rest, keeping the explicit "we do not claim" scoping paragraph in §1, which is worth more than any paragraph in §2.

### F9. Two statements of the same robustness check disagree — MINOR
**Where:** §3 (~L106) and §9 Limitations (~L402)
**Issue:** §3 says excluding the dual-polarity image changes the headline contrast *"($+0.496 \to +0.505$ on the matched set)"* — a change of 0.009. Limitations says *"removing it changes the matched-set mirror contrast by $0.008$."*
**Why it matters:** Trivial in magnitude, but it is the same quantity stated twice with two values, and it sits in the limitations section where a skeptical reader is looking hardest.
**Suggested fix:** Make Limitations read `$0.009$`, or restate it as "by less than $0.01$" so a future re-run does not desynchronize the two again.

### F10. The zero-by-construction caveat is explained three times — MINOR
**Where:** `tab:patching` caption (~L265), §6.1 body (~L286), §9 Limitations (~L402); plus a fourth pass in Appendix G (~L650)
**Issue:** That the image-token, BOS and prefix rows are zero by causal masking rather than by measurement is explained at length in all four locations, each time from scratch.
**Why it matters:** Correct and worth saying — but at roughly half a page of repetition, it is expensive under F1, and repetition reads as defensiveness about a point already made well.
**Suggested fix:** Keep the full explanation in the caption, reduce the §6.1 body mention to one sentence pointing at it, and delete the Limitations restatement.

### F11. Two appendix figures are never referenced — MINOR
**Where:** `fig:stagea` (~L490), `fig:stagec` (~L509)
**Issue:** Both are labelled but never `\ref`'d, so nothing in the text introduces them or states their takeaway.
**Suggested fix:** Add a `Figure~\ref{...}` pointer in the sentence each illustrates, or drop them.

### F12. Dead label on §7 — MINOR
**Where:** L328–329
**Issue:** The section carries both `\label{sec:robustness}` and `\label{sec:models}`. Only `sec:models` is referenced; `sec:robustness` has zero references.
**Suggested fix:** Delete `\label{sec:robustness}`. Two labels on one heading also make it easy to introduce a wrong cross-reference later.

## What is already working (keep it)

A revision under deadline will be tempted to strip these. They are why the paper is credible.

- **The explicit non-novelty paragraph** at the end of §1 ("We do not claim to discover that text can override vision … nor the first mechanistic study of multimodal conflict"), with citations for each disclaimer. Very few papers do this, and it buys trust for everything after it.
- **Reporting both the corrected and uncorrected override gaps everywhere**, and stating plainly that the categorical correction was applied after seeing the uncorrected results (§7). Disclosing the ordering of an analysis decision is exactly right.
- **Both bootstrap interval types, always** — photo-clustered and crossed — with the Clark (1973) items-as-fixed-effects argument for why the crossed one is the honest test, and the admission that Qwen clears none of the three crossed tests on the varied set. The paper's own primary model failing its own strictest test is reported in the abstract, the body, and the limitations.
- **The measurement-dependence finding kept as a finding**, not buried: that first-token scoring manufactured a null and reversed a model is stated in the abstract. It would be easy to demote this to an appendix note during a page cut. It should not be.
- **§6.2's refusal to resolve the mid-band disagreement** ("We report both and draw no conclusion about mid-band ordering") when two valid readouts give non-overlapping intervals in opposite directions.
- **Withdrawn measurements disclosed rather than deleted** — the superseded-stack results in Appendices D, F and G, and the two withdrawn subsections counted in the compute appendix.
- **The broader-impacts paragraph** treats affect inference as contested and disclaims the application, rather than performing compliance. If it moves to an appendix for length, it should move intact.

## Note outside the audit's scope

Not a writing finding, but it surfaced during verification and bears on submission: `paper/checklist.tex` answers **No** to items 5 and 11 (code archive, licenses). The handoff records this as deliberate pending the code release. Both venues are non-archival, so this may be acceptable — but it is worth a deliberate decision rather than a default, since item 5 is the reproducibility question and this paper's central methodological point is that a scoring choice reversed two models' conclusions.
