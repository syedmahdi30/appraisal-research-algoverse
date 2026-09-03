# Paper Audit: *When One Word Changes an Image Judgment: A Valence Asymmetry in Vision-Language Models*

**Date:** 2026-09-03 · **Source:** `paper/neurips_2026.tex` (long build, `\shortfalse`), `paper/tables/*.tex`, `paper/venue.tex`, `paper/references.bib`, compiled `paper-build/long.pdf` (8p main text, References p9, 31p total)

**Scope note:** Audited as a **VLM4RWD** submission (8pp, NeurIPS format, double-blind, non-archival, due Sep 5). Venue fit judged against the workshop's stated scope — grounded understanding, faithful reasoning, evidence localization, failure-mode analysis, evaluation methodology for grounding and faithfulness — and quality calibrated against two accepted VLM4RWD papers: *Do Vision-Language Models Understand Visual Persuasiveness?* (arXiv 2511.17036, VLM4RWD 2025, 8pp + appendix) and *AMVICC* (arXiv 2601.17037, 15pp, ICML template).

**Hard constraint governing every fix below:** the main text is at **exactly 8 pages with zero slack**, and prose compression has been measured not to buy page space (three passes removed 651 characters and recovered one rendered line). Any fix that adds main-text content must be funded by cutting main-text content, and must be verified against a real build via `./scripts/build-paper.sh long`.

---

## The narrative as I read it

Vision-language systems that read affect from photographs also receive whatever text arrives with the image, and the two can disagree. The paper's first claim is that the disagreement is not resolved evenhandedly: on matched counterfactual pairs that hold the photograph and the described event fixed and flip a single valence word, negative wording moves Qwen3-VL-8B's emotion judgment four to five times farther than positive wording, all six pairs agree, and the same sentences without an image show no detectable difference. The second claim localizes where a context effect lives in Gemma-3-4B: patching downstream text positions restores 88–93% of the context difference, the image's own valence becomes additionally readable at those same text positions with depth, and a valence direction estimated from text alone still moves the answer under conflict. The third claim is a caution: across four models the effect's size and sometimes its sign depend on the summary score and on whether a multi-token emotion label is scored in full, so no architectural boundary survives. The stated stake is that a symmetric evaluation — one that measures only accuracy, or only which modality tends to win — cannot see a directional error, and this one falls hardest on people the accompanying text already describes badly.

That narrative is coherent, calibrated, and unusually honest about its own limits. The problems below are almost entirely about **what the reader can see** and **which of the three claims the venue will care about**, not about whether the claims are supported.

## Summary

This paper is well above the quality bar set by the two accepted comparison papers. It reports photo-clustered *and* crossed bootstrap intervals where AMVICC reports no statistical test at all; it withdraws two experiments whose inference path was defective; it discloses that a categorical correction was applied after seeing uncorrected results; and its broader-impacts appendix is more specific than most conference papers manage. Acceptance is not the risk. Legibility is.

The three highest-leverage fixes, in order:

1. **F1 — Put evidence back in the main text.** Eight pages of main text contain **one figure and one table**, and hand the reader **15 cross-references spanning 11 appendices**. Two of the three contributions have no float in the main body at all. The 8pp accepted comparison paper carries five figures and seven tables. A workshop reviewer skimming for twenty minutes cannot verify claims 2 and 3 without leaving the paper.
2. **F3/F4 — Surface the deployment reading of results already in hand.** The paper's most venue-legible number (a 57% versus 35% override rate) first appears on page 7, and the cross-model result is framed only as a measurement caution when it is also a 75-point spread in failure rate between two models a practitioner might choose between.
3. **F2 — Flag the model switch in the abstract.** The behavioral headline is Qwen; the mechanism is Gemma. §4.3 says plainly that the mechanism is *not* an account of the Qwen asymmetry. That disclaimer never reaches the abstract or the introduction.

What is already working is listed at the end and is worth protecting: a revision pass under deadline pressure is exactly what destroys it.

---

## Findings

### F1. Eight pages of main text carry one figure and one table; two of three contributions have no visible evidence — CRITICAL

**Where:** `paper/neurips_2026.tex` main text §§3–5 (~L129–L343); floats at L74 (`fig:method`) and L195 (`\input{tables/minimal}`); everything else after `\appendix` (L374)

**Issue:** The compiled main text (pp. 1–8) contains exactly `Figure 1` (a design schematic) and `Table 1` (the matched-pairs comparison). Every other float lives in the appendix:

| Float | Supports | Where it is |
|---|---|---|
| `tab:models` — 4 models × 3 readouts | Contribution 3 (measurement caution) | Appendix (L489) |
| `tab:patching` — same-image patching, 88–93% | Contribution 2 (mechanism) | Appendix (L442) |
| `tab:crosspatch` — where image valence lives | Contribution 2, and the readout disagreement | Appendix (L463) |
| `tab:mechmodels` — cross-model replication | §4.4 mechanism replication | Appendix (L446/458) |
| `fig:conflict` — the asymmetry itself, plotted | Contribution 1 (the headline) | Appendix (L425) |
| `tab:factorial`, `tab:pairs`, `tab:resolution` | §4.2, §4.4 | Appendix |

The consequence compounds: §4.3 and §4.4 become dense walls of bracketed intervals in running prose, and the main text issues **15 appendix cross-references spanning Appendices A through K**. Even `Figure 1`'s own caption sources its "62–82%" range to `Table 5`, in the appendix.

**Why it matters:** This is the finding most likely to change the review outcome, and it is a *presentation* artifact of hitting 8 pages by relocation rather than a defect in the work. A reviewer skimming an 8-page workshop paper reads the abstract, the figures, the tables, and the section openers. Ours gives them a schematic and one table, so claims 2 and 3 arrive as unillustrated numeric prose. The accepted 8pp comparison paper (2511.17036) carries five figures and seven tables in the same budget; AMVICC carries four and eight. Against that, this paper reads as harder work for the same length — which reviewers experience as lower quality even when the underlying evidence is stronger. I am calling it CRITICAL rather than MAJOR because two of the three contributions the introduction promises have no verifiable evidence in the body.

**Suggested fix:** Promote **one** float, funded by a cut, and pick the one that buys the most:

- *Best value:* promote `tab:models` into §4.4. It displays the "readouts disagree / no architectural boundary" claim more compactly than the prose that currently describes it, so the promotion pays for part of itself. Fund it by cutting the §4.4 paragraphs it makes redundant — the text-only-control paragraph and the content-token sensitivity aside can compress to two sentences once the table is adjacent.
- *Second choice:* promote `fig:conflict` (or just its left Qwen panel) into §4.2 so the headline asymmetry is visible once. A single-panel version at `0.6\linewidth` is cheaper than the current two-panel appendix figure.
- Do **not** attempt both. Verify with `./scripts/build-paper.sh long` after each attempt and revert if References moves off page 9. Prior sessions established that `[h]`→`[tbp]` float placement changes nothing here, so do not spend time on it.

### F2. The abstract and introduction do not flag that the mechanism is a different model from the headline — MAJOR

**Where:** abstract (L69), introduction ¶4 (~L103); the disclaimer that is missing lives at §4.3 opening (~L228)

**Issue:** The abstract moves from "negative wording moves \qwen{}'s judgment four to five times farther" directly to "In \gemma{}, where a text-trained valence probe is available, replacing downstream states at text positions restores 88–93% of the context difference." The introduction does the same. §4.3 states the limit explicitly and well:

> "\gemma{} is not the model carrying the behavioral contrast of \S\ref{sec:minimal}, so what follows locates where a context effect lives in \gemma{}; it is not a mechanism for the \qwen{} asymmetry."

That sentence never reaches the abstract or the introduction.

**Why it matters:** The default reading of "here is an asymmetry, and here is where it lives in the network" is that the second explains the first. It does not, and the paper knows it does not. A reviewer who reaches §4.3 and discovers the disjunction after the abstract implied otherwise will discount the mechanism section — and possibly the paper's calibration, which is otherwise its strongest asset. It is MAJOR rather than CRITICAL only because the abstract does name both models explicitly, so the information is technically present.

**Suggested fix:** Six words in the abstract, roughly length-neutral against the clause it replaces. Change "In \gemma{}, where a text-trained valence probe is available, replacing downstream states…" to "In \gemma{}, a different model where a text-trained valence probe is available, replacing downstream states…", and in the introduction's mechanism sentence append ", which locates a context effect rather than explaining the \qwen{} asymmetry".

### F3. The paper's most deployment-legible number does not appear until page 7 — MAJOR

**Where:** §5 Discussion, "Implications" (~L323); absent from abstract (L69) and introduction (~L99–L107)

**Issue:** The abstract's only quantities are `+1.148` (a graded contrast on a bounded internal scale) and `88–93%` (a patching recovery percentage). The number a deployment reviewer can actually act on — *a single negative sentence flips the model's top emotion label on 57% of conflict trials against 35% in the mirror direction, after subtracting each image's own neutral-sentence error rate* — appears for the first time on page 7.

**Why it matters:** VLM4RWD's third pillar is "predictable failure modes" and "principled evaluation… across distribution shifts and counterfactual scenarios." An override rate is a failure rate; `+1.148` on a bounded valence scale is not something a reviewer can map onto a deployed system. The reader-attention budget says the abstract and introduction are read orders of magnitude more than §5, so the paper is currently spending its most persuasive venue-relevant fact in its least-read location.

**Suggested fix:** Add the rate to the introduction's motivating paragraph (¶1, ~L99), which already lists moderation queues and assistive agents but gives no magnitude. One sentence, with the honest caveat attached: *"On the varied set the model's top label crosses into negative on 57% of conflict trials against 35% in the mirror direction, once each image's own neutral-sentence error rate is subtracted; the graded contrast rather than this categorical one carries our claim (§4.2)."* Do **not** promote it to the abstract without the caveat — the corrected gap's crossed interval on the varied set is `[-8.6, +48.0]` and includes zero, which is precisely why §4.4 lets the graded contrast carry the claim. An uncaveated rate in the abstract would be the one genuine overclaim in this paper.

### F4. The cross-model result is framed only as a measurement caution, never as the deployment fact it also is — MAJOR

**Where:** §4.4 (~L276–L296), especially the second bolded result; §5 "Why do the models differ?" (~L318)

**Issue:** §4.4 is the longest subsection in the paper and its narrative is entirely about measurement: complete-label scoring overturns two LLaVA conclusions, the three readouts disagree, no architectural boundary survives. That is the thesis of the *sibling* Interp4Discovery submission ("The Readout Decides the Finding: Two Measurement Failures in Cross-Modal Interpretability"). The same table supports a fact this venue cares about more, and the paper never states it: **the corrected override gap runs from +63.9% (LLaVA-1.5-7B) to −11.2% (LLaVA-NeXT-7B)** — a 75-point spread in directional failure rate between two models that share a connector family. §5 currently answers "why do the models differ?" with "we do not know", which is honest but leaves the practitioner with nothing.

**Why it matters:** "We cannot explain the variation" and "the variation is large enough that you cannot infer one model's behavior from another's" are the same evidence with opposite usefulness. The second is directly actionable for anyone selecting a VLM for an affect-sensitive pipeline, it requires no new experiment, and it is squarely inside the workshop's evaluation-and-deployment-reliability track. Leaving it unstated makes the section read as a methods post-mortem at a deployment workshop.

**Suggested fix:** Two sentences, no new content. In §4.4, after the architectural-explanations paragraph: *"For a practitioner the unexplained variation is itself the result: the corrected gap spans +63.9% to −11.2% across the four models, so a system's directional behavior under conflicting text cannot be inferred from a sibling model, a connector family, or an image-token budget, and has to be measured per model."* Then retitle the §5 paragraph "Why do the models differ?" to "Why do the models differ, and what follows if we cannot say?" and close it with a pointer to that sentence. Note this competes with F1 for the same page; if you promote `tab:models` per F1, this sentence sits directly beneath it and reads better for it.

### F5. The paper's actionable recommendation is stated too abstractly to act on — MAJOR

**Where:** abstract final sentence (L69), §5 "What this means for deployed systems" (~L327)

**Issue:** The closing recommendation is *"Evaluations should test what each cue says, not only which modality carries it."* This is the paper's contribution to the workshop's "Benchmarks, datasets, and evaluation methodologies for grounding and faithfulness" topic, and it is the last sentence of the abstract — but it names no protocol. The paper has in fact demonstrated a specific four-part one: measure **both** conflict directions, against a **per-image neutral-context baseline**, on **matched counterfactual pairs** that hold the event fixed, with **intervals that resample sentences as well as images**. Every element is established in the body; none is named in the recommendation.

**Why it matters:** A recommendation a reader can implement is a contribution; a recommendation they must reverse-engineer from §3 is a slogan. The four elements are also the paper's clearest differentiator from the accepted comparison papers, neither of which reports uncertainty over stimulus sampling.

**Suggested fix:** Expand the §5 sentence (not the abstract, which has no room) into the concrete protocol: *"Concretely: measure both conflict directions on the same images, correct each against that image's own neutral-context response, use matched pairs that change only the contested word, and resample sentences as well as images when reporting intervals. Dropping any one of the four hides the asymmetry — the first three by construction, the fourth by understating how much of the effect is specific to six sentences."* That last clause is already demonstrated by the varied-set versus matched-set contrast and costs nothing to assert.

### F6. Abstract and introduction say "four to five times" where the body reports ratios of 4.3 to 5.5 — MINOR

**Where:** abstract (L69), introduction ¶4 (~L101), against §4.2.2 (~L195): "All six pairs agree, with ratios from $4.3$ to $5.5$"

**Issue:** "four to five times farther" understates an upper bound of 5.5. The direction of the error is conservative, so this is not an overclaim, but it is an avoidable mismatch between the paper's two most-read sentences and its table.

**Suggested fix:** "four to five times" → "between four and five and a half times", or simply "more than four times", in both locations. The second is shorter than the current text.

### F7. The headline statistic appears in the abstract without its interval — MINOR

**Where:** abstract (L69): "the within-item contrast is $+1.148$"

**Issue:** The paper's entire methodological stance is that intervals decide which claims are licensed — it is why the varied set is demoted and the matched set carries the claim. The abstract gives the headline number bare, while the body reports `+1.148` with `[+0.943, +1.344]` and Wilcoxon `p<0.001`.

**Suggested fix:** "the within-item contrast is $+1.148$ $[+0.94,+1.34]$" — about 14 characters, and it signals the paper's rigor in the one place reviewers form their impression. Build-verify; if it costs the page, drop F6's expansion to pay for it.

### F8. `vs.` and `versus` are both used — MINOR

**Where:** `vs.` 4× in §4.3 (~L245, L250); `versus` 3× elsewhere in the main text (§4.1 ~L156, §4.2.1 ~L182, §5 ~L323)

**Issue:** Two conventions for one word. The four `vs.` instances are all inside tight numeric comparisons (`49.1\% [45.0,53.4]$ vs.\ $45.2\%`), where the abbreviation is defensible; the `versus` instances are in running prose.

**Suggested fix:** Either make it `versus` everywhere, or state the rule implicitly by using `vs.` only between bracketed numeric pairs — which is what the current usage already almost does. The cheapest fix is to leave §4.3 alone and confirm no `vs.` appears outside a numeric comparison.

### F9. The crossed-versus-clustered interval distinction is presented as a statistical technicality, not as the uncertainty contribution it is — MINOR

**Where:** §4.2.1 (~L184), §4.4 (~L280)

**Issue:** The paper distinguishes photo-clustered intervals (sentences fixed) from crossed intervals (sentences resampled) and correctly treats the latter as the test that licenses claims about negative *language* rather than about six particular sentences. This is introduced as bootstrap bookkeeping with citations to Clark (1973) and Westfall (2014). Neither the abstract nor §5 identifies it as a methodological position, and the words "uncertainty" and "calibrated" appear nowhere in the main text — despite "Uncertainty" and "Robust Evaluation" being named workshop topics.

**Suggested fix:** One clause in §5's evaluation paragraph, folded into the F5 rewrite: name it as the paper's stance that *stimulus-sampling uncertainty, not just image-sampling uncertainty, is what determines whether a cross-modal conflict result generalizes*. It costs a phrase and claims a second workshop topic the paper already earns.

---

## Flagged notes outside the requested scope

- **`PAPER-CONTEXT.md` is stale and actively misleading.** Dated Aug 21, still framed for TAE (dropped), still lists the outstanding items as "verify fifteen unconfirmed citations / write the responsible-use section / cut 10pp → 8pp" — all three are done — and still says `results/stage_e` and `stage_f` are empty and `references_fixed.bib` needs repair. Anyone (or any future session) using it as context will make wrong decisions. Delete or rewrite it after the deadline.
- **Concurrent submission is undeclared.** This work is under review at Interp4Discovery. VLM4RWD's CFP permits previously *published* work (award-ineligible) and is silent on concurrent submission; both venues are non-archival, so this is very likely fine. Worth one line to the chairs if you want it on record rather than discovered.
- **Verification pass found no numeric mismatches.** Every figure quoted in the abstract, introduction, and §5 traces to its table and matches: `+1.148` and `[+0.943,+1.344]` (§4.2.2); `88–93%` against `tab:patching`'s 93.2%/87.9%; `57%/35%` and `76%/37%` and `+21.7%`/`+39.4%` against `tab:minimal` and §4.2.1; `62–82%` in Figure 1's caption against `tab:mechmodels`'s 62.3/66.0/81.6; `+63.9%`/`−11.2%` against `tab:models`. Build hygiene is clean: zero rendered `[TODO:]`, zero main-text em dashes, zero anonymity leaks (the single `grep` hit for "Syed" is the cited author Aaquib Syed), and no `.DS_Store` or stale `.bak` in `overleaf/vlm4rwd.zip`. The `\ifshort` toggle does what its comments claim, verified by building both targets.

---

## What is already working (keep it)

A revision under deadline pressure will be tempted to remove each of these. Do not.

- **The calibrated hedging is the paper's best feature, not padding.** "We therefore claim a controlled asymmetry in the tested setup, not a universal architectural law." "Ratios on a bounded scale are descriptive; the difference and its interval are primary." "We report both and draw no conclusion about mid-band ordering." Against AMVICC's "this can potentially be attributed to various factors", this reads as a different tier of care.
- **The disclosures no one would have caught.** That the categorical correction was applied *after* seeing uncorrected results; that two experiments used a defective wrapper path and are withdrawn; that `r^2=0.641` is a selection statistic and not a held-out estimate; that the zero rows in the patching tables are forced by causal masking and are alignment checks, not evidence that image tokens are inert. Each of these preempts a reviewer objection that would otherwise land.
- **The negative and null results kept in the main text.** The varied set failing crossed resampling, Qwen clearing none of the three crossed tests, the balanced-subset re-scoring breaking the categorical result in 15 of 35 cases. Reporting these next to the positive result is what makes the positive result believable.
- **§4.3's readout-disagreement paragraph.** Two non-overlapping intervals in opposite directions from the same model and intervention, reported with no conclusion drawn. Most authors would have picked the favorable one silently.
- **The broader-impacts appendix (`app:impacts`).** Three risks derived from *these results* rather than boilerplate, the refusal to endorse affect inference on images of people, the note that error may fall unevenly across groups that were not measured, and per-asset licence terms read from the source rather than assumed. This is more specific than most main-conference papers manage.
- **Related work positions rather than enumerates.** Each of the three paragraphs ends by naming how this work differs. Keep that shape if the section is compressed to fund F1.

---

## Disposition — applied 2026-09-03

Subset applied on request: **F1, F2, F4**. Both builds verified at cap afterwards (VLM4RWD 8p / References p9; Interp4Discovery 5p / References p6), flattened source verified identical to the toggled build, tests 159 passed.

| | Status | What was done |
|---|---|---|
| **F1** | applied, scaled down | Full `tab:models` and full `fig:conflict` were each promoted and each cost a page (measured, not estimated). Promoted a new compact `tab:gaps` instead — corrected override gap for four models with crossed intervals — now **Table 2** in the main text. Main text went from 1 float to 2. |
| **F2** | applied | Abstract: "In \gemma{}, **a different model** where a text-trained valence probe is available…". Intro: appended "; this locates a context effect rather than explaining the \qwen{} asymmetry". |
| **F4** | applied | Added the practitioner sentence to §4.4 ("the corrected gap spans $+63.9\%$ to $-11.2\%$… has to be measured per model", pointing at Table 2), and retitled the §5 paragraph to "Why do the models differ, and what follows if we cannot say?". |
| F3, F5–F9 | not applied | Deferred; no page budget remained. F3 in particular still stands and is cheap if a page ever frees up. |

**Funded by** (F1's page cost, in order of application):

1. §4.4 prose compression: readout enumeration already defined in §3, text-only-control paragraph, neutral-correction paragraph. Disclosure of the post-hoc correction kept verbatim.
2. §4.2.2's "One pair makes the effect concrete" paragraph **deleted** — the paper itself said it "illustrates one pair rather than adding a result". Its best fact survives: Figure 1's caption clause about the top label flipping from *joy* to *sadness* on 57 of 62 images was made unconditional, so it now appears in the long build too.
3. §5's "Where the cues meet" paragraph **deleted** — it proposed an experiment rather than reporting one.
4. §5's "Conflicts with stored model knowledge" paragraph **deleted as duplicated** — §2 ¶1 already makes the same positioning with the same two citations (`nooralahzadeh2026`, `lietzow2026`), so no reference was orphaned. Its one novel synthesis sentence was grafted onto "Why do the models differ".
5. §5 de-duplication: the uncorrected 76%/37% pair (already in §4.2.1 and Table 1), two adjacent sentences making the same caution, and a restatement of §4.3's localization result.

**Measured facts worth keeping for any future edit:**

- A full-size table (`tab:models`, `\scriptsize`, 12 data rows) and a full-size two-panel figure (`fig:conflict`, 1.6:1 aspect) each cost **one page** in the long build. A compact 3-row table costs about **4 lines**.
- Changing the new table's float placement `[h]`→`[t]` and its size `\small`→`\scriptsize` changed the page count by **nothing**, confirming the earlier finding that float placement does not buy space here.
- Prose compression again failed to shed lines: roughly 350 characters of §4.4 tightening recovered about 4 lines only because it removed whole sentences, not because of the character count. **Only deleting whole blocks reliably sheds lines.**
- **Dropping `app:impacts` would save zero main-text pages** — it is in the appendix, which the venue excludes from the limit. Only the ~4-line pointer paragraph in §5 is in budget, and it should stay.
- Compile log after the edits: no undefined references, no multiply-defined labels, no Overfull boxes. Several **Underfull** vbox/hbox warnings remain, which is loose inter-paragraph spacing from sitting exactly at the cap, not a defect.

## Disposition, second pass — 2026-09-03 (cold read)

| | Status | Notes |
|---|---|---|
| **F8** | **applied** | Abstract now reads "the within-item contrast is $+1.148$ $[+0.94,+1.34]$". |
| **F7** | **no change needed** | Verified rather than edited: `vs.` appears only where bracketed intervals are paired (§4.3, four instances), and `versus` everywhere else — running prose (§4.1 `+0.336` versus `-0.035`, §4.2.1 `57\%` versus `35\%`, §4.3 "at layer 18 versus after 16 further layers") and hyphenated compounds (`positive-versus-negative`). That is already a defensible document-wide rule, which was F7's own cheapest fix. |
| **F6** | **applied, then reverted** | "four to five times" → "more than four times" at five sites. The **cold read caught that the Figure 1 graphic itself reads "4–5× larger for LOST"** — it is a generated PDF (`scripts/generate_method_diagram.py`), so the prose change put the text in direct contradiction with the figure on the same page. That is worse than the original quibble, and regenerating the diagram risks changing its size at a zero-slack page budget. Reverted; prose and figure agree at "four to five" / "4–5×". **If this is ever revisited, the figure has to change with the prose.** |
| F3, F5, F9 | still open | Unchanged from the first pass: all need main-text page space. |

### Two things the cold read settled

- **The submission footer is correct as-is.** It reads "Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute." An accepted VLM4RWD paper (arXiv 2511.17036) instead shows "39th Conference … Workshop: VLM4RWD", which initially looked like a formatting miss on our side. It is not: `neurips_2026.sty` uses `\@trackname` (the workshop string) **only** under `\if@neuripsfinal`, so every non-final submission gets the generic "Submitted to …" line whatever track it targets. The comparison paper's footer is its camera-ready. **For camera-ready, switch to `\usepackage[final,dblblindworkshop]{neurips_2026}` and add `\workshoptitle{VLM4RWD}`.** Note `\if@workshop` is declared in the style file but never used, so the option has no other effect.
- **Figure 1's caption sources its "62–82%" to Table 6, in the appendix**, and the main text cites the appendix figure `fig:conflict` as Figure 2. Both are instances of the F1 pattern and both are acceptable; recorded so they are not mistaken for new defects.

## Disposition, third pass — 2026-09-03 (F5 and F9)

**F5 applied, merged with F9. F3 measured, drafted, and declined. F9 needs nothing further.**

§5's deployed-systems paragraph already carried two of the four protocol elements ("both conflict
directions … against a neutral baseline on the same images"), so only the two missing ones had to be
added. The recommendation now names the full protocol:

> it appears only when both conflict directions are measured against a per-image neutral baseline, on
> matched pairs that change only the contested word, with intervals resampling sentences as well as
> photographs.

**F5 and F9 were never two findings.** F9's substance — that stimulus-sampling uncertainty and not
just image-sampling uncertainty decides whether a cross-modal conflict result generalizes — is the
fourth clause above, and it is the element neither comparison paper reports at all. Naming it
additionally as a "stance" would only add the keyword, so standalone F9 is closed as redundant rather
than deferred.

**F3 was declined, not blocked.** It fits (measured below) and was drafted as: "In our tests it is
also large: a negative sentence flips the top label on 57% of conflict trials against 35% in the
mirror direction, though the graded contrast carries the claim (§4.2)." Declined because the varied
set's crossed interval for that gap includes zero, so the sentence has to qualify its own number in
the introduction, and the abstract already carries $+1.148$ as the single licensed headline. The
author chose to keep exactly one primary claim. **The rate is still stated in §5's Implications**, so
it is not absent from the paper — only from the introduction. Reinstate the sentence above verbatim if
that judgment changes.

### Correcting an earlier claim in this audit

The first pass asserted that F3, F5 and F9 each required funding by a cut. **That was wrong**, and it
was an over-generalization from F1, where the cost was a *float*, not prose. Measured on real builds:

| Edit | Where | Chars | Result |
|---|---|---|---|
| F5+F9, with an explanatory sentence | §5 | +285 | **9p — over** |
| F5+F9, minimal | §5 | +105 | 8p — fits |
| F5+F9 minimal **and** F3 together | §5 + §1 | +314 | 8p — fits |

A single +285 edit in §5 overflowed while +314 split across §5 and §1 did not. **The budget is a
page-break budget, not a character budget, and it is sensitive to where the text lands.** Never
predict fit from character counts; run `./scripts/build-paper.sh long`.

## Disposition, fourth pass — 2026-09-03 (duplicated broader-impacts statement)

Author observation, confirmed: the main text carried a standalone
`\paragraph{Broader impacts and responsible use}` whose **first two sentences were verbatim identical**
to the opening of `app:impacts`. Same text twice in one PDF.

**Not removed outright, because two things depended on it.** `app:impacts` is referenced exactly once
per build — the short build points at it from a clause inside its Limitations, the long build pointed
at it only from this paragraph. Deleting the paragraph would have left the appendix section with no
`\ref` in the long build, which is the orphaned-target defect `dab534b` already fixed once. It was also
the only responsible-use statement visible to a reviewer who reads just the eight pages, on a paper
that runs emotion inference over photographs of real people.

**Resolved by doing what the short build already does.** The standalone paragraph is gone; the stance
and the pointer now ride at the end of the long build's Limitations paragraph:

> Emotion inference from photographs is contested and we use it only as a diagnostic instrument, never
> as evidence that such judgments are valid; Appendix~\ref{app:impacts} gives the three risks that
> follow from these results, our data-release terms, and the responsible-use discussion.

Net $-225$ characters, both builds still at cap, `app:impacts` referenced once per build, no orphaned
label, no verbatim duplication. The responsible-use discussion itself is untouched in the appendix.

**Note for anyone reinstating F3:** this pass freed roughly the space F3 needs, and F3 was declined on
editorial grounds (one primary claim) rather than for want of room.

## Disposition, fifth pass — 2026-09-03 (F3 reinstated)

**F3 applied.** The introduction's motivating paragraph named three deployment settings — moderation
queues, assistive agents, intake tools — and gave no magnitude anywhere before page 7. It now reads:

> If negative phrasing moves those judgments further than positive phrasing does, the error is
> directional, and it falls hardest on people whom the accompanying text already describes badly. In our
> tests it is also large: after subtracting each image's neutral-sentence baseline, a negative sentence
> flips the top label on $57\%$ of conflict trials against $35\%$ in the mirror direction, though the
> graded contrast carries the claim (\S\ref{sec:conflict}).

**One change from the sentence recorded in the third pass.** That draft quoted $57\%$ with no indication
that it is the baseline-corrected rate; the uncorrected figure is $76\%$. An unqualified rate in the
paper's most-read paragraph is precisely the claim-scope slip this audit's verification pass exists to
catch, so the sentence now names the correction. Cost $+267$ characters rather than $+209$; both builds
remain at cap.

Cross-checked against §4.2.1, which reads "Correcting categorical overrides against each image's
neutral-context error rate gives $57\%$ versus $35\%$ (gap $+21.7\%$, $[+10.9,+31.9]$); the uncorrected
values are $76\%$, $37\%$, and $+39.4\%$." The introduction and §5's Implications now agree with it and
with each other.

The editorial tension noted when F3 was first declined still stands and was accepted deliberately: the
sentence qualifies its own number in the introduction, and the abstract continues to carry $+1.148$ as
the single licensed headline. That is the honest arrangement given the varied set's crossed interval for
this gap includes zero.

### Every finding is now closed

| | Outcome |
|---|---|
| F1 | applied, scaled to a compact `tab:gaps`; main text 1 float → 2 |
| F2 | applied |
| F3 | applied (fifth pass, with the correction qualifier) |
| F4 | applied |
| F5 | applied, subsuming F9 |
| F6 | applied then reverted — contradicted the Figure 1 graphic |
| F7 | no change needed; existing convention verified consistent |
| F8 | applied |
| F9 | subsumed by F5 |

Plus the author-spotted duplicated broader-impacts statement, resolved in the fourth pass.
