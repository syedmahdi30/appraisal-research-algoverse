# Handoff — VLM4RWD submittable at `ec743be`; all five controls run, four usable

_Written 2026-09-04 evening. Branch **`main`**, tip `ec743be`, **3 commits ahead of origin
(unpushed)**. Both builds at cap. Tests **201 passed** (was 197). **Deadline 2026-09-05 ~21:00 PT**
(Sep 6 03:59 UTC)._

## Goal

Submit the valence-asymmetry paper to **VLM4RWD** (NeurIPS workshop, 8pp, double-blind,
non-archival). Interp4Discovery was submitted 2026-09-02 and is under review; its build is **frozen**
and every edit this session was guarded to keep it byte-identical. This session: analyse the five
reviewer controls that ran on Colab, write the usable ones into the paper, and fix what they exposed.

## Current State

- **`overleaf/vlm4rwd.zip` is current (17:27, after the last commit) and is the file to upload.**
  Flattened, verified identical to the toggled build, no `\ifshort` leak, carries the new table.
  VLM4RWD 8pp (References p9), Interp4Discovery 5pp (p6). Zero rendered `TODO`, zero main-text em
  dashes, no undefined / multiply-defined / overfull warnings.
- **The Interp4Discovery build is untouched**, verified by extracted-text hash at every step:
  `943f47456780bf3e`, unchanged from session start. All new prose is behind `\ifshort\else`.
- **Four of the five controls are in the paper. The fifth is void and is not cited anywhere.**
- 3 commits are **not pushed**.

### What the controls found

| control | result | in paper |
|---|---|---|
| `--person box` | paired Δ mirror **−0.045 [−0.170,+0.082]** — reproduces | yes |
| `--person crop` | paired Δ mirror **−0.043 [−0.228,+0.136]** — reproduces | yes |
| `--axis frame` | `original` reproduces exactly; see below | yes |
| `--axis question` | `original` reproduces exactly; see below | yes |
| `--generate` | **VOID** — 99.6% unparsed, truncation not result | **no** |

**Person grounding passed cleanly**, at full coverage (121 images, 62/60 groups, no
`--allow-missing`), with the paired ungrounded arm reproducing the published +1.148 / +0.496 exactly
in both runs. This answers the external reviewer's biggest objection.

**The wording sweeps are best read split by conflict direction**, which is how the paper reports them:

```
                 drop (neg text   rise (pos text   mirror
                  on POS images)   on NEG images)
original            -1.478           +0.982        +0.496
caption             -1.492           +0.461        +1.031
report              -1.407           +1.292        +0.115
message             -1.492           +0.671        +0.821
best_word           -1.448           +1.412        +0.036
state               -1.340           +0.580        +0.760
one_word            -1.444           +1.205        +0.240
```

**`drop` is invariant across all seven wordings (−1.34 to −1.49, a ~10% band); `rise` varies
threefold (+0.46 to +1.41).** The mirror contrast is the difference of the two magnitudes, so it
inherits that variance entirely: its sign holds everywhere, its magnitude ranges +0.036 to +1.031,
and under three of six alternatives its crossed interval includes zero. Within-item clears zero in
all seven (+0.86 to +1.43). Reported as a boundary on the magnitude, not the direction.

## Active Files

- `paper/neurips_2026.tex` — single source, branches on `\ifshort`. New: `app:wording` section
  (long-only), a long-only person-grounding paragraph in `app:behavior`, and swapped Limitations
  clauses. **Read first.**
- `paper/tables/wording.tex` — the seven-wording table. `\tabcolsep` is 4pt because 6pt overfills.
- `src/experiments/stage_f_controls.py` — `_keep_run_provenance`, `UNPARSED_CEILING`,
  `--max-new-tokens`.
- `docs/reviewer-controls-runbook.md` — Colab commands; now records why run 5 was void.
- `results/stage_f/controls_*` — all five parquets + metrics are on this machine.

## Changes Made

1. **`08e656c`** — person-grounding control reported instead of disclaimed. Replaced the appendix's
   "we did not run it" and the Limitations "the control this design still needs".
2. **`dff686e`** — wording sweeps, split by direction: new `app:wording` appendix section + table,
   and the two "untested" Limitations clauses replaced. Main text held at 8p by trimming two clauses
   the new evidence made redundant.
3. **`ec743be`** — two code defects the runs exposed, with 4 new tests.

## Failed Attempts

- **The main text is on a knife-edge at 8p.** A +349-char Limitations edit went to 9p; trimming to
  +20 chars **still** went to 9p. It only fit after deleting two now-redundant clauses (−128 net).
  Confirms the handoff's standing rule: **only deleting blocks buys lines.** Always run
  `./scripts/build-paper.sh` — never predict fit.
- **The appendix is free, the main text is not.** `\bibliography` is at line 370 and `\appendix` at
  379, so "main text" is everything before References. Appendix additions cost nothing.
- **A new 6-column table overfilled by 18.4pt.** `\setlength{\tabcolsep}{4pt}` fixed it. The baseline
  had zero overfull warnings; keep it that way.
- **The paper uses no prose em dashes** — the only `---` in the source are one comment and table
  placeholders. Two I added were removed.
- **`--reanalyze` clobbered four metrics files' provenance before I fixed it** (see below).

## Two things the results sync broke, one still open

**Fixed: the patching metrics were clobbered, and the paper was right.** The synced `results/` folder
restored **pre-dedupe** versions of `patching_qwen_metrics.json` and
`patching_llava_sequence_metrics.json`, dropping `n_unique_images` and reverting `text_all` to raw
60-row values. Two tests caught it. Recomputing from the intact parquets confirms the paper:
**51 unique images from 60 rows, Qwen 62.26% → the paper's 62.3%, LLaVA 66.03% → 66.0%.** Both files
restored with each script's own `--reanalyze`. Clobbered copies are in the session scratchpad.
**The paper's numbers were never wrong — the artifacts had regressed.**

**Open: four control metrics files carry my local library versions, not Colab's.** I ran
`--reanalyze` on the four non-generate controls *before* fixing the provenance bug, so their
`run`/`git`/`versions` are this machine's. The parquets are intact and every number is recomputed
from them, so no result is affected — only the provenance record. **The Colab environment is
recoverable** from `controls_generate_qwen_metrics.json`, which was never reanalyzed and still reads
`git 1667b43`, `transformers 4.57.6`, `torch 2.11.0+cu128`, `numpy 2.1.3`, `PIL 11.3.0` — all five
ran in that one session. Restoring it means writing those values back by hand; the per-run
timestamps are lost and should not be invented. Not done, because it is metadata and it was the night
before the deadline.

## Next Steps

1. **Upload `overleaf/vlm4rwd.zip` before 2026-09-05 ~21:00 PT.** Mandatory; the paper is ready.
2. **Decide on pushing the 3 commits.** Nothing is pushed.
3. **Restore the four clobbered provenance blocks** using the values above, or accept the loss and
   note it — a decision, not a task.
4. **Re-run `--generate` if it is ever wanted**, now that the budget is 32:
   `python -m src.experiments.stage_f_controls --generate --images-root <root>`. Priority 5 in the
   runbook, nobody asked for it, and it is cited nowhere. `UNPARSED_CEILING` will warn if it fails
   the same way.
5. **After the deadline:** rewrite or delete `PAPER-CONTEXT.md` (dated Aug 21, still TAE-framed), and
   publish the **real** repo at camera-ready — **not before the decision**, since a public repo
   matching the paper de-anonymizes the Interp4Discovery submission still under review. Note that
   `stage_f_controls.py` is already committed, which is part of why the wording sweep had to be
   disclosed rather than held back.
