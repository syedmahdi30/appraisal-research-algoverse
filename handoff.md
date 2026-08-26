# Handoff — both venue builds hit their limits; Sneheel review and open panel items remain

_Written 2026-08-25. Branch `main`, tip `24a16ff`, **pushed and in sync with origin** (0 ahead, 0
behind). Working tree clean apart from long-standing untracked files (`PAPER-CONTEXT.md`,
`PROJECT-INSTRUCTIONS.md`, `context6/7/9.md`, `session*.md`, `graphify-out/`, root `main.pdf`,
`docs/robustness-plan.md`)._

## Goal

Get the valence-asymmetry paper submitted to two NeurIPS 2026 workshops with different page limits,
from one source of truth, and keep the experiment code trustworthy enough that the paper's numbers
re-derive from it.

## Current State

- **Both venue builds compile within their limits.** `./scripts/build-paper.sh both` →
  VLM4RWD 8pp (refs p9), Interp4Discovery 5pp (refs p6). The script *fails* rather than reports when
  a target is over, or when a float lands in the bibliography.
- **The two builds are genuinely different papers**, not one compressed. VLM4RWD leads with the
  phenomenon; Interp4Discovery leads with mechanism + measurement failure and has its own title,
  abstract, intro, and closing sections. Section order differs.
- **Deadlines: Interp4Discovery Aug 29 (11:59pm AoE), VLM4RWD Aug 30.** Both confirmed against live
  CFPs. Both non-archival, both exclude refs *and* appendices from the limit.
- **The 5pp build has no slack.** It took three passes to get from 6pp to 5pp. Any addition needs a
  matching cut.
- **Experiment code:** 134 tests pass. The Codex refactor (PR #3, merged) was independently verified
  as behaviour-preserving — see Changes Made #7.
- **Not done:** Sneheel has not reviewed the reframe. Open panel items from
  `docs/review-panel-seats-rerun-2026-08-23.md` (W2, W9, W11) and M4/W4 are untouched.
- `paper/checklist.tex` has **4** `\answerNo` (was 2). Compute-timing and new-assets flipped to No.
  Honest, but ratify it — compute-as-No may be over-conservative given hardware and pass counts are
  disclosed.

## Active Files

- `paper/neurips_2026.tex` — single source for both venues, branching on `\ifshort`. Read first.
- `paper/venue.tex` — one line, `\shortfalse` (VLM4RWD) / `\shorttrue` (Interp4Discovery). Scripts
  set it; do not hand-edit during a build.
- `paper/tables/{minimal,patching,models}.tex` — the three relocated tables, `\input` at exactly one
  place per build (body vs appendix differs by venue). Single-sourced deliberately.
- `scripts/build-paper.sh` — builds both, enforces page limits **and** the float-in-bibliography guard.
- `scripts/flatten-paper.py` — resolves `\ifshort` for one venue into standalone Overleaf source.
- `scripts/build-overleaf.sh` — regenerates both Overleaf projects and diffs each against its toggled
  build. **Run this after any paper edit before re-uploading.**
- `overleaf/{vlm4rwd,interp4discovery}/` + `.zip` — generated, git-ignored. Never hand-edit.
- `docs/paper-audit-2026-08-24.md` — the 12-finding audit. All 12 addressed.
- `src/experiments/shared/` — refactor's new modules (readouts, reporting, artifacts, patching,
  hf_runtime, sampling). `reporting.py` holds the paper-number functions.

## Changes Made

1. **Confirmed both page limits against live CFPs.** The only recorded figure (8pp) was in a doc
   naming a dropped venue. Interp4Discovery is **5**, not 8.
2. **Added the `\ifshort` toggle and `build-paper.sh`.** Initially inert — both targets emitted the
   same 13-page PDF. Wired for real later.
3. **Ran `/paper-audit`** → `docs/paper-audit-2026-08-24.md`, 12 findings.
4. **Codex applied the audit and cut 13pp → 8pp/5pp.** Verified independently; it fixed 11 of 12 but
   introduced four regressions (see Failed Attempts).
5. **Repaired those regressions:** restored 22 orphaned citations (Appendix B "Extended related work"
   + methods citations in Appendix D), restored the dual-polarity robustness check, fixed a truncated
   sentence and `log odds` hyphenation.
6. **Re-emphasized the 5pp build around mechanism and measurement** (`effd04d`) — new title, abstract,
   intro, phenomenon/mechanism/measurement sections, closing. Tables 1–2 moved into its body.
7. **Verified the Codex experiment refactor (PR #3) independently.** Diff-tested pre- vs
   post-refactor on the *real* Stage-F parquets: `flip_override`, `asymmetry_vs_floor`, `cell_means`,
   `minimal_pair_asymmetry` all identical, and they reproduce the paper's published numbers exactly
   (all six `tab:factorial` cells, 76.3%/36.9%, +39.4%, 62 photographs). Swept 126 artifact keys —
   identical. Negative-tested the AST boundary tests. All 9 CLIs run offline.
8. **Restored two incident-documenting docstrings** the refactor stripped (`24a16ff`).
9. **Rebased and pushed.** Local main was 2 ahead / 16 behind; clean rebase, now in sync.

## Failed Attempts

- **Do not treat "N tests pass" as evidence a refactor is behaviour-preserving.** It only proves
  self-consistency. The real check was diff-testing both trees on the real parquets.
- **A differential test on synthetic data was worthless** — my first pass reported "IDENTICAL" while
  actually comparing two identical `KeyError`s, because the synthetic frame lacked the real schema
  (`lp_*`, `condition`, `image_path`). Always confirm the probe *computed* something.
- **Codex's length cut deleted rather than relocated**, despite appendices being free at both venues.
  Cost: 22 of 53 bib entries orphaned, and the dual-polarity robustness check deleted instead of
  having its 0.008/0.009 inconsistency reconciled. Both repaired.
- **A naive `\ifshort` flattener would mis-nest** — `\if@anonymous` appears twice and a comment
  discusses it. `flatten-paper.py` masks comments and tracks nesting depth.
- **`\else` and `\fi` gobble the following space.** An inline `\ifshort\else …\fi We report` rendered
  as `list.We report` in *both* PDFs. Use `\ `. Found only by diffing flattened vs toggled builds.
- **Appendix floats land in the bibliography** without `\clearpage` before `\appendix`; `[t]`-only
  placement made it worse (own float page mid-references). Invisible in source and in the LaTeX log —
  it existed only in the rendered PDF. Guard added and negative-tested.
- **The 5pp build cannot be "mechanism-first" literally** — you cannot patch a difference you have
  not defined. The re-emphasis compresses the phenomenon to setup and expands mechanism/measurement.
- Still off the table from prior sessions: the measurement-validity framing (mentor called it
  overreach), the architecture-boundary claim (falsified twice), the visual-token-budget hypothesis.

## Next Steps

1. **Upload both zips to Overleaf as two projects** and share with Sneheel (Can View + comments is
   safer than Can Edit — anything he edits there must be hand-ported back to `paper/`).
2. **Get Sneheel's read on the two framings** before further edits, especially the Interp4Discovery
   re-emphasis. It is expensive to reverse.
3. **Ratify the checklist.** Decide whether compute-resources should stay `\answerNo`, and whether
   the anonymized code archive can be attached before Aug 29 (that flips item 11 too).
4. **Fix M4/W4**: §4.3 still argues conflict "does not attenuate the intervention" from +0.335 vs
   +0.336 with no interval on either. Needs the neutral-context re-run, not rewording. GPU.
5. **Open panel items** in `docs/review-panel-seats-rerun-2026-08-23.md`: W2 (same-image patching has
   no control that could have come out otherwise), W9 (probe r² is the selection statistic, no
   test-split number), W11 (patching band chosen on same data, 3× noise multiplier untested).
6. **Before any submission:** `./scripts/build-overleaf.sh` (rebuilds, re-verifies, and would catch a
   drifted Overleaf copy), then re-upload. Never upload a hand-edited `overleaf/` tree.
7. Optional: re-run the three stale panel seats against the current draft if a full editorial
   decision is wanted.
