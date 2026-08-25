# Handoff — paper reframed, cleaned, and verified; the page cut is next

_Written 2026-08-23. Branch `main`, tip `4b5e589` pushed, plus one uncommitted one-line fix to
`paper/checklist.tex` (see Changes Made #8). Working tree otherwise clean apart from long-standing
untracked files (`PAPER-CONTEXT.md`, `PROJECT-INSTRUCTIONS.md`, `context6/7.md`, `session*.md`,
`graphify-out/`, PDFs, `paper/neurips_2026old.tex`, `paper/neurips_2026.tex.pre-rescore-backup`)._

## Goal

Get `paper/neurips_2026.tex` into a state Sneheel can review and that can actually be submitted:
honest claims, verified numbers, credited methods, plain language — then cut it to the venue page
limit. The reframe and clean-up are done. **The cut is not.**

## Current State

- **The paper claims what it can defend.** Story was deliberately changed this session (see #4):
  it now leads with the six matched minimal pairs on Qwen3-VL, treats the varied set as
  generalization evidence, keeps the Gemma mechanism work, and reports the cross-model comparison as
  a boundary rather than a finding. Title is now *"When One Word Changes an Image Judgment: A
  Valence Asymmetry in Vision-Language Models"*.
- **Every headline number re-derives from `results/`.** 13 were re-verified programmatically at the
  end of the clean-up pass. Six stale bridge-era figures were found and replaced.
- **All 53 bib entries are cited and every citation resolves.** No dangling `\ref` (after the fix in
  #8), environments balance, `$` parity holds, 47 tests pass.
- **It has been compiled once** (`paper/main.pdf`, Overleaf). The `ack` anonymity guard works — the
  built PDF goes straight from Conclusion to References with no acknowledgements section.
- **THE BLOCKER: the body is ~13.3 pages against an 8-page limit** (`docs/tae-experiment-plan.md:5`,
  "8 pp excl. refs + appendices"). That is a ~40% cut. Nothing else matters as much.
- **The 8-page figure is not confirmed against a live CFP.** That doc names TAE, which memory
  records as dropped in favour of VLM4RWD; it says VLM4RWD is "same trim". **Confirm before cutting.**
- Two panel seats (Methodology, Devil's Advocate) were re-run and adjudicated; three seats
  (Journal-Fit, Domain, Perspective) are stale — they reviewed a pre-correction, pre-rescore draft.
  No editorial decision exists and none can be synthesized (`panel_size: 5`).

## Active Files

- `paper/neurips_2026.tex` — the draft, 668 lines. Read this first.
- `paper/checklist.tex` — NeurIPS checklist. Items 5 and 11 are deliberately `\answerNo{}`; they
  flip back to Yes once the code archive is attached and licenses are listed.
- `paper/references.bib` — 53 entries, all cited. The 12 methodology entries were verified against
  arXiv/ACL/Crossref; three of them had been wrong.
- `docs/review-panel-seats-rerun-2026-08-23.md` — the two re-run seats, my adjudication of every DA
  CRITICAL, and the action list. Several items still open.
- `docs/paper-retraction-audit.md` — provenance for every number; says which stack produced what.
- `docs/rescore-runbook.md` — Colab commands, including step 7 (text-only control) and the LLaVA
  sequence re-score.
- `src/experiments/multitoken_scoring.py` + `stage_f_llava.py` — complete-label scoring. This is the
  code behind the result that reversed both LLaVA models.
- `src/experiments/patching_intervals.py` — recomputes every interval in Tables 4 and 5 from saved
  per-row outputs (CPU).
- `paper/main.pdf` — the Overleaf build. **Predates the last three commits.**

## Changes Made

1. **Re-ran the two dead panel seats** under the full two-phase sprint contract (paper-blind Phase 1,
   then paper-visible Phase 2, fresh contexts, no peer visibility). R1 blocked D1; DA blocked D3.
2. **Adjudicated all three DA CRITICALs — all validated.** The worst (C2) was self-inflicted: after
   the baseline correction, Qwen clears *none* of the three crossed intervals and LLaVA-NeXT cleared
   *all three*, while five sentences said otherwise. Fixed in all five places.
3. **Added bootstrap intervals to both patching tables.** The intervals already existed in the
   stored metrics and had simply never been printed. Printing them cost two claims: "contribute
   comparably" and "close to additive" hold on pair 1 and fail on pair 2.
4. **Reframed the paper** (user decision, after being offered four options): minimal pairs lead;
   five findings became four; §5 retitled from "Generality across models" (which it disproved) to
   "How far the result travels"; steering moved out of the caveat section into Mechanism; abstract
   and conclusion rewritten.
5. **Landed the complete-label re-score** (from the Codex session) and verified all of it against the
   parquets — LLaVA-1.5 moves from the paper's only null (−10.0%) to **+63.9%**, LLaVA-NeXT reverses
   to −11.2%. Ran the diagnostic the audit was missing: `sequence_sum`'s length bias is small
   (r = −0.23) and runs *against* the finding, while `content_mean`'s is fatal (r = +0.93,
   2,246/2,250 argmax on the two 3-token labels).
6. **Clean-up pass**: six stale bridge-era figures replaced (shared-image sensitivity ×2, the whole
   context-sentence table, the whole no-context table, layerwise raw separations, pilot appendix);
   a *fourth* instance of the stale matched-set claim found at line 171; `luo2025`,
   `conflictchallenges2025`, `unraveling2024` cited (all had been in the bib, uncited); the scoring
   rule grounded in `holtzman2021` + `brown2020`; jargon removed.
7. **Codex session after mine** (`44a8c2c`, `4b5e589`): retitled again, added `hewitt2019` for probe
   control tasks, reworded findings toward plainer language ("internal replacement experiments",
   "summary score", "split labels"), and reframed patching as *sufficiency* rather than necessity.
8. **Uncommitted:** `44a8c2c` deleted the withdrawn "Scale details" appendix but left
   `paper/checklist.tex` pointing at `\ref{app:scale}`, which would render as `??`. Rewrote that
   sentence. One line, needs committing.

## Failed Attempts

- **Do not cut before confirming the page limit.** The only recorded number (8 pp) is in a doc that
  names a venue that was dropped. Cutting to 8 when the real limit is 9 throws away a page.
- **Do not trust `results/` paths blindly.** The 0–12 and 13–17 cross-image patching parquets were
  destroyed by a fixed-path overwrite; only 18–28 survives, so those two bands' intervals come from
  stored metrics and cannot be independently re-bootstrapped.
- **`content_mean` is not a usable scoring rule** despite length normalization being the standard
  (GPT-3 uses it). On single-word labels it collapses onto the longest label. Do not "fix" the
  scoring by averaging.
- **A 5-seat editorial decision is not obtainable** from a partial panel — the contract forbids
  recomputing `cross_reviewer_quantifier` thresholds against a smaller panel. Don't try to synthesize
  one from the two re-run seats plus three stale prose summaries.
- **The measurement-validity framing is off the table.** `docs/tae-experiment-plan.md` describes a
  redraft that made measurement validity the contribution; memory records the mentor calling that
  overreach. The current framing deliberately keeps the phenomenon primary.
- **Do not revive**: the architecture-boundary claim (falsified twice now — first by LLaVA-NeXT, then
  reversed again by complete-label scoring), or the visual-token-budget hypothesis.
- **A `\label` after an unnumbered `\paragraph` captures the wrong counter** — it grabs whatever last
  stepped (an `itemize` item, in the case caught). Point at the enclosing section instead.

## Next Steps

1. **Commit the `app:scale` fix** in `paper/checklist.tex` (item #8 above). One line, already made.
2. **Confirm the page limit against the live CFP** for whichever of Interp4Discovery (Aug 29) /
   VLM4RWD (Aug 30) you are targeting. Everything below depends on the number.
3. **Send the current draft to Sneheel with an explicit note that the trim is pending**, so he
   reviews the argument and not the length. The reframe is the thing most worth his judgment and it
   gets expensive to reverse after a 40% cut.
4. **Do the cut.** Plan that survives the reframe: move §4 (probe foundations) almost entirely into
   Appendix A, keeping a paragraph in §6; compress §7 to its table plus half a page; relocate part
   of Limitations. That is roughly 5 pages without touching findings 1–3.
5. **Rebuild on Overleaf** — `paper/main.pdf` predates the last three commits and the title changed.
6. **Fix M4/W4**: §4.3 still says conflict "does not attenuate the intervention at all" from
   +0.335 vs +0.336 with no interval on either. Needs the neutral-context re-run (the no-conflict
   slope has no context sentence while the conflict slope does), not rewording. GPU.
7. **Open panel items** in `docs/review-panel-seats-rerun-2026-08-23.md`: W2 (same-image patching has
   no control that could have come out otherwise), W9 (probe r² is the selection statistic, no
   test-split number), W11 (patching band chosen on the same data, 3× noise-floor multiplier
   untested).
8. Optional: re-run the three stale panel seats against the current draft if a full editorial
   decision is wanted before submission.
