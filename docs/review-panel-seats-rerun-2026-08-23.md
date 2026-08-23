# Panel re-run — the two seats that died (2026-08-23, later session)

The 2026-08-23 panel completed 3 of 5 seats; Methodology (R1) and Devil's Advocate (DA) died on API
errors. Both were re-run here against the **current** draft, which has changed materially since
(§6.2 image-token demotion, baseline-corrected override rate, text-only stimulus control).

Protocol followed: v3.6.2 sprint contract, `reviewer/reviewer_full/v2`. Each seat ran a
paper-content-blind Phase 1 (contract paraphrase + scoring plan + `criteria_binding_unavailable`,
no tools, no manuscript) in a fresh context with no peer-output visibility, then a paper-visible
Phase 2 with its own Phase 1 fed back as data. Venue undecided, so no seat made a venue-fit claim.

**Role separation is not independence** — the skill says so itself. Two seats agreeing is weaker
evidence than two humans agreeing. What it does establish is that the finding is reachable from two
different starting prompts.

## No editorial decision was produced, and none can be

The contract pins `panel_size: 5`. §6 of `sprint_contract_protocol.md`: if
`len(usable_phase2_outputs) < panel_size`, abort with `[PANEL-SHRUNK]` and **do not** recompute the
`cross_reviewer_quantifier` thresholds against a smaller panel. The three surviving August seats
exist only as prose in `review-panel-2026-08-23.md`, not as conformant Phase 2 artifacts, and they
reviewed a different draft. So: two seat cards, no decision letter. Getting a decision requires
re-running all five against the current draft.

## Scores

| seat | D1 methodology_rigor | D3 argumentative_coherence |
|---|---|---|
| R1 (`methodology`) | **block** (repairable) | warn |
| DA (`da`) | not_assessed (ineligible) | **block** (repairable) |

D2/D4/D5/D6 are ineligible for both seats and returned `not_assessed`.

## Adjudication of DA CRITICALs (Iron Rule #4)

Every DA CRITICAL must be adjudicated visibly. I checked each against the manuscript rather than
deferring. **All three validate.** Three of them describe defects introduced in this session.

### C1 — VALIDATED. The connector rule-out rests on one confounded model.

Finding 4 and the Conclusion say two architectural explanations are "ruled out." The connector half
is carried solely by LLaVA-NeXT showing the asymmetry. Qwen3-VL and Gemma cannot corroborate it —
the proposition is specifically about the linear-projector family and neither model instantiates it.
LLaVA-NeXT is also the only model failing the stimulus control (1.69, unsaturated) and it swaps
Vicuna for Mistral. The mitigating sentence I added this session — "we do not rest the connector
argument on \llavanext{} alone," citing Qwen's 0.94/1.02 — is a non-sequitur: those figures show
Qwen's stimuli are balanced, not that a linear-projector model resists override. R1's W5 reaches the
same conclusion independently. Remedy: "undermined by a single confounded comparison," not "ruled
out." Does not touch the central thesis, which rests on the matched set.

### C2 — VALIDATED. My error, introduced this session. Three sentences now contradict Table 2.

| model | corrected override (crossed) | mirror contrast | unbounded margin | clears |
|---|---|---|---|---|
| Qwen3-VL | [−8.6, +48.0] | [−0.16, +0.91] | [−5.47, +1.80] | **none** |
| LLaVA-NeXT | [+1.0, +41.0] | [+0.23, +0.59] | [+0.50, +1.50] | **all three** |

False as written:
- line 63 — "*no model clears zero on all three readouts under the crossed bootstrap*"
- line 347 — "\qwen{}, \llavanext{} and \gemma{} **each clear zero on at least one readout**"
- line 353 — "**\qwen{} clears only on the override rate**"

All three were true before the baseline correction. I changed the table and the headline and never
propagated. The sentence at 347 makes the primary model an equal member of a three-model agreement
only by readmitting the uncorrected gap the paper disowns twice. R1's W6 found this independently.

### C3 — VALIDATED, with mitigation.

The correction is the sole reason Gemma's photo-clustered interval moves from [−0.4,+23.7] to
[+7.5,+28.7], and the sole reason the budget comparison collapses from 27 points to 3.6. Both check
out against the tables. Mitigating, and R1's W3 says so too: the rule pre-exists for the graded
readouts, is applied uniformly, both raw and corrected are tabled, the timing is disclosed, and it
moved the headline **down**. The defect is that the prose states convergence and the budget null as
established rather than conditional on the correction.

## R1's block, and what only R1 found

R1 blocked D1 on the trigger "A headline quantitative claim is reported without its denominator,
without an uncertainty interval, or without the control/baseline condition needed to interpret it."

- **W1 (Critical) — VERIFIED.** *Zero* uncertainty intervals on any patching result: both
  `tab:patching` and `tab:crosspatch` contain no interval at all. The abstract carries 88–93% and
  100%/63% from these tables. Worse, §6.2 line 301 argues from "(non-overlapping intervals)" that
  the paper never prints — establishing both that they were computed and that no reader can check
  the inference. The comparative claims ("distributed, not concentrated" from 47 vs 46; "close to
  additive"; "the ordering reverses") cannot survive without them.
- **W2 (Major).** After the §6.2 demotion, same-image patching has **no control that could have come
  out otherwise** — three measured rows and three forced to zero. My fix removed the only rows that
  looked like controls. Needs a size-matched non-affective post-context group to separate "text
  states carry valence" from "patching enough post-context positions recovers the effect."
- **W7 (Minor) — VERIFIED.** Bootstrap resample count and interval construction never stated
  anywhere. "resamples" only ever names the *unit*. (It is 2000, seed 0, percentile — recoverable
  from `analyze_stage_f_unbounded.py`; this is a one-line fix.)
- **W8 (Minor) — VERIFIED.** Checklist items 5 and 11 answer Yes to artifacts their own inline
  `% CONFIRM` comments say are not yet done. I audited items 4/7/8/9/10 earlier and missed these.
- **W9, W10, W11 (Minor).** Probe r² = 0.641 is the selection statistic with no test-split number;
  the "seventeen-fold range … every corrected gap between +18% and +26%" sentence silently drops
  LLaVA-1.5 (−10.0%, and its 579-token budget is never stated in the paper — including it would
  *strengthen* the argument); the 13–17 patching band and its 3× noise-floor multiplier are selected
  on the same data the results are read from, with no sensitivity check.

## Where the two seats converge

C2≡W6, C1≡W5, C3≡W3, and DA-M4≡R1-W4 (the "does not attenuate **at all**" equivalence claim from
+0.335 vs +0.336 with no interval on either — and worse than either seat could see, since the
no-conflict slope has no context sentence while the conflict slope does; that was already open as
handoff step 6).

## Action list, in the order I would take it

1. **C2 / W6** — repair the three sentences. Factual, not a judgment call.
2. **W1** — bootstrap the patching tables over the 60 images / 60 pairs and print the intervals, or
   drop the comparative claims and the "non-overlapping" citation. Blocks D1.
3. **C1 / W5** — downgrade "ruled out" to "undermined by a single confounded comparison"; delete the
   non-sequitur defense sentence.
4. **C3 / W3** — state convergence and the budget null as conditional on the correction.
5. **M4 / W4** — drop "at all"; re-run against neutral-context trials (handoff step 6).
6. **M2** — Appendix B claims LLaVA-NeXT carries "a stimulus confound the other models do not," but
   Gemma and LLaVA-1.5 were never measured on the current stack. Either run them (~30 forwards) or
   scope the sentence to the two models actually measured.
7. **W7, W8, W10** — one-line fixes each.
8. **W2, W9, W11** — controls and sensitivity checks; real work, lower priority.
