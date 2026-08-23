# Handoff — paper rewritten on raw HF; review panel found two verified defects

_Written 2026-08-23. Branch `main`, tip = the review-findings commit, pushed. Working tree clean
except pre-existing untracked files (`PAPER-CONTEXT.md`, `PROJECT-INSTRUCTIONS.md`, `session*.md`,
`graphify-out/`, PDFs, `paper/neurips_2026old.tex`, `paper/neurips_2026.tex.pre-rescore-backup`)._

**Read `docs/review-panel-2026-08-23.md` first, then `docs/paper-retraction-audit.md`.**

## Goal

Get `paper/neurips_2026.tex` into a clean, coherent, well-grounded draft suitable for circulating to
human peer readers. Per Sneheel: reduce jargon, move extra experiments to the appendix, ground the
methodology in prior literature, converge on a few key experiments, build a coherent story. **Venue
is deliberately undecided** — do not optimise for a page limit; Interp4Discovery (Aug 29) and
VLM4RWD (Aug 30) are both still open and the decision comes after the draft is good.

## Current State

- **The paper is fully rewritten on raw HuggingFace measurements.** Every number in it is
  re-derivable from `results/` on this machine. Qwen3-VL is the primary behavioural model (it is the
  only one of four that weights both cues comparably, 1.14 vs 0.44–0.55); Gemma-3-4B remains the
  mechanism model because the frozen probe was fit on it.
- **Compiles to ~19 pages: body ends ~p13, appendix to p19.** 11 tables, 4 figures, 2 remaining
  `\todo`s (both routine camera-ready: acknowledgements, wall-clock).
- **A simulated 5-seat review panel was run and returned only 3 seats.** Methodology and Devil's
  Advocate died on API errors and were never re-run. The panel is incomplete and the skill's own
  rule (every DA `[CRITICAL]` must be adjudicated) is unsatisfied.
- **Two defects are verified computationally and are NOT yet fixed in the paper:**
  1. §6.2's "image tokens recover 0%" is an **arithmetic identity** — those positions precede the
     context under causal masking, so patching them is a no-op (60/60 rows bitwise identical).
     Still stated as a finding in the abstract, contribution 5, Discussion and Conclusion.
  2. The override rate is **not baselined** while the graded readouts are. Correcting it against each
     cell's own neutral context takes Qwen's headline from **+39.8% to +21.1%** and collapses the
     spread across models to +21/+14/+18/−14.
- Everything is committed and pushed. Nothing is half-edited.

## Active Files

- `docs/review-panel-2026-08-23.md` — **the review findings, with what was verified vs. asserted.** New.
- `docs/paper-retraction-audit.md` — claim-by-claim triage from the bridge-bug re-scoring; all the
  raw-HF numbers and their provenance.
- `paper/neurips_2026.tex` — the draft. 641 lines. Overleaf-compiled; no local LaTeX toolchain.
- `paper/references.bib` — 50 entries. The 12 methodology entries added this session were **written
  from memory and are still unverified**; the file carries an in-line warning saying so.
- `docs/rescore-runbook.md` — Colab commands for every re-score, and what each outcome means.
- `src/experiments/stage_f_token_budget.py` — the raw-HF conflict runner; now takes `--bank minimal`.
  **Does not yet accept `--bank` for `--text-only`** — that blocks the top next step.
- `src/experiments/analyze_stage_f.py` / `analyze_stage_f_unbounded.py` — CPU analysers; both now
  cluster on `image_path` (photo), not person-annotation.

## Changes Made

1. Ported same-image and cross-image patching, layerwise localisation, and conflict steering to raw
   HF (`stage_f_patching_hf.py`, `stage_f_cross_patching_hf.py`, `stage_f_layerwise_hf.py`,
   `stage_f_arbitration_hf.py`). All import their aggregation from the bridge originals so the runs
   stay comparable.
2. Ran all of them. §6.2 carrier survives but the turn-boundary *concentration* does not (65/57 →
   46/40). §6.3 shape survives, the 9% late-band number becomes 31%. Steering **strengthens**:
   +0.335 vs a no-conflict +0.336, i.e. no attenuation, where the paper claimed 65%. Layerwise entry
   replicates (L13→L12) but the 7× amplification does not.
3. Fixed `analyze_stage_f_unbounded` to cluster on photograph, not person-annotation — EMOTIC
   annotates per person but the model sees the whole photo, so 29 rows were duplicates counted as
   independent. This made both analysers agree exactly and *improved* the crossed intervals.
4. Added `--bank minimal` to the raw-HF runner; ran the Qwen matched-pair set. It is now the paper's
   strongest result: within-item contrast **+1.148** [+0.94, +1.34], all six pairs concordant.
5. Rewrote the paper: Qwen primary, LLaVA boundary claim retracted, "1.8×" dropped, prompt sweep and
   scale comparison withdrawn, mechanism numbers updated, attention-share paragraph dropped.
6. Grounding pass: credited difference-of-means steering (it had **no citation at all**), grounded the
   crossed bootstrap in Clark 1973, credited logit lens and Alain & Bengio. 12 new bib entries.
7. Jargon pass: "full/minimal bank" → "varied/matched set"; plain section headings; removed internal
   "Stage A/C/D" vocabulary from the appendix.
8. Ran the review panel; verified its two most damaging claims against the parquets.

## Failed Attempts

- **Do not trust `results/figures/` to survive a Drive re-sync.** Two figures generated locally were
  wiped when `results/` was re-synced; `paper/figures/` and `results/figures/` are separate trees.
- **`colab_bootstrap.py --drive` `rmtree`s a non-symlink `results/`.** Its comment says "replace the
  empty local dir" but it never checks emptiness. Check whether `results/` is already a symlink
  before running it with `--drive`.
- **I gave the wrong minimal-pair indices once.** The published pairs are `(pos 0, neg 2)` and
  `(pos 4, neg 0)` — *not* `(1, 0)`. `stage_f_patching_hf.py` now keys published numbers by pair and
  prints `--` for an unpublished one.
- **`analyze_stage_f` attached Gemma's steering sweep to every model's analysis** from a fixed path,
  so a Qwen run reported Gemma's +0.215 as its own. Fixed, but be wary of other fixed-path reads.
- **The Perspective seat's label-count `[CRITICAL]` is wrong** — it claimed the 4-positive/7-negative
  split biases the override rate. Neutral base rates refute it (97% correct both directions in three
  models). The *asymmetric baseline* problem it gestures at is real; the mechanism it proposed is not.
  Do not re-litigate the label-count version.
- Do not revive the TAE measurement-validity framing (mentor called it overreach), the architecture-
  boundary claim (falsified by LLaVA-NeXT), or the visual-token-budget hypothesis (falsified twice).

## Next Steps

1. **Fix the §6.2 image-token claim.** Change "partly follows from prompt order" to "is determined
   by"; move the three forced rows (image, BOS, prefix) into a labelled sanity-check block or the
   appendix; strike the claim from the abstract, contribution 5, Discussion and Conclusion, replacing
   it with what the data do show (text recovers 88–93%). Apply the same demotion to the Qwen port.
   No new runs needed.
2. **Add a baseline-corrected override rate** everywhere the override rate appears, or say explicitly
   why it is not baselined when the graded readouts are. The numbers are in
   `docs/review-panel-2026-08-23.md`; the correction is CPU-only from existing parquets. Expect
   Qwen's headline to drop to +21.1%.
3. **Run the text-only matched-set control on Qwen.** This is the seats' shared `[CRITICAL]` and the
   paper's largest untested alternative explanation. Requires threading `--bank` through
   `run_text_only()` in `stage_f_token_budget.py` (it is already threaded through `run_base`), then
   ~15 forward passes. Report the text-only mirror contrast beside the conflict one.
4. **Re-run the two dead panel seats** (Methodology, Devil's Advocate). The DA especially — the
   panel's own rule requires its CRITICALs be adjudicated and none exist yet.
5. **State the multi-token scoring rule** (first-subtoken vs summed log-prob) and re-score LLaVA under
   it. The paper's only null is the result most exposed to a scoring artefact.
6. **Fix the steering baseline confound**: the no-conflict slope has no context sentence, the conflict
   slope does. Re-run against neutral-context trials, and drop "at all" from the no-attenuation claim.
7. **Verify the 12 methodology bib entries** against real records before circulating.
8. **Check the `ack` anonymity risk** on a fresh Overleaf build — `neurips_2026.sty` appears to print
   the acknowledgement unconditionally, and its `\todo` names the Algoverse program.
9. Add a responsible-use / broader-impacts paragraph; `checklist.tex` item 9 currently claims content
   that does not exist in §6.
