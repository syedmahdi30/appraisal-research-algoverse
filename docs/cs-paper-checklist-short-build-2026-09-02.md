# CS paper checklist — Interp4Discovery (short build)

_Run 2026-09-02 against `1c40af3`, `\shorttrue`. Re-measures every item from
`cs-paper-checklist-2026-08-27.md`, which was run against `2149b44` on the **VLM4RWD (long)** build.
Measured against the built PDF, the flattened `overleaf/interp4discovery/main.tex`, and the compile
log — not eyeballed. ✅ pass · ⚠️ partial · ❌ fail · ➖ not applicable_

## Headline

**The short build passes the writing items more cleanly than the long build did** — zero long
unpunctuated runs (long build: 3 genuine), zero over-long main-text paragraphs excluding the
abstract (long build: 14 of 31), 13% passive against a 30% ceiling, and zero genuine LLM vocabulary.
Length is comfortable at 5pp with references on p6.

Three things need work, in order:

1. **Five orphaned targets, all short-build-specific** (§5.8, §8.3). `tab:models`, `tab:minimal`,
   `app:headroom`, `app:pilot`, `app:patching` are defined and never referenced **in this build**.
   Every one of them *is* referenced in the long build — the pointing sentences sit inside
   `\ifshort\else` branches, so the short build renders the float and drops the sentence that cites
   it. `tab:models` is the worst case: it is the four-model comparison, sitting in the **main text**
   with no textual pointer.
2. **Citations still unverified** (§10) — unchanged and still the highest-risk item in the document.
   One of 61 was verified this session (`emomm2026`, venue confirmed against the authors' own
   publication listing).
3. **No figure in the main text at all** (§2.6) — worse than the long build's finding, not better.
   All six figures are in the appendix.

---

## 1. Title and abstract

| | | |
|---|---|---|
| 1.1 | ✅ | **11 words** ("The Readout Decides the Finding: Two Measurement Failures in Cross-Modal Interpretability") |
| 1.2 | ✅ | Names the failure mode and the domain; carries "Cross-Modal Interpretability" and "Measurement Failures" |
| 1.3 | ✅ | No abbreviations in the title |
| 1.4 | ✅ | Problem, approach, results and significance all present |
| 1.5 | ✅ | Zero vague descriptors (probed novel / state-of-the-art / significant / important / groundbreaking / substantial) |
| 1.6 | ✅ | $\rho=+0.510$, $88$–$93\%$, four to five times |

⚠️ The short abstract is **255 words**, down from the long build's 291 and just above the 150–250
band. Rounds 12–13 removed two parenthetical layer ranges and split its densest sentence; further
cuts would now cost content.

## 2. Introduction

| | | |
|---|---|---|
| 2.1 | ✅ | Problem stated in the first paragraph |
| 2.2 | ✅ | Motivated, but **differently from the long build**: the motivation here is methodological (when localization evidence licenses a conclusion) rather than the long build's deployment settings |
| 2.3 | ➖ | No named method to preview |
| 2.4 | ❌ | **No itemised contributions list.** The long build has three `\item` contributions at its L111; the short build carries them in prose only. The one `itemize` in this build is the appendix minimal-pair bank |
| 2.5 | ⚠️ | The claims are specific and checkable, but a reviewer cannot skim them as a list |
| 2.6 | ❌ | **No first-page figure, and no figure anywhere in the main text.** All six `includegraphics` are in the appendix; `method_diagram.pdf` is long-build only. Page 1 carries no float at all |

## 3. Related work

| | | |
|---|---|---|
| 3.1 | ✅ | Every citation tied to task, comparison model, or method |
| 3.2 | ✅ | Main conflict papers present, in `app:related` |
| 3.3 | ➖ | **No related-work section in this build.** Positioning lives in the introduction's disclaimers plus `app:related` |
| 3.4 | ❌ | **Still not verified.** Unchanged from 2026-08-27 |
| 3.5 | ➖ | Not a method-comparison paper |

## 4. Method

| | | |
|---|---|---|
| 4.1 | ✅ | Symbols defined before use |
| 4.2 | ✅ | Few display equations |
| 4.3 | ✅ | All components described |
| 4.4 | ⚠️ | No overview figure for subsections to align with (see 2.6) |
| 4.5 | ✅ | No pseudocode in main text |
| 4.6 | ⚠️ | **Leans on appendices harder than the long build** — unavoidable at 5pp, but it is the main cost of the short framing |
| 4.7 | ✅ | Through several compression passes, most recently Rounds 12–13 |

## 5. Experiments

| | | |
|---|---|---|
| 5.1 | ❌ | Two datasets (EMOTIC, crowd-enVENT) against a target of three. Unchanged |
| 5.2 | ➖ | Four VLMs compared, not "baselines" in the method-comparison sense |
| 5.3 | ✅ | Text-only control, norm-matched random directions, caption controls, resolution sweep, label-balance re-scoring |
| 5.4 | ✅ | Bootstrap intervals throughout, in two forms, stricter one identified |
| 5.5 | ⚠️ | Hardware and per-experiment pass counts in the appendix; **wall-clock totals still missing** and not derivable — no timing data exists anywhere in the repo |
| 5.6 | ✅ | Nulls reported prominently, including in the abstract |
| 5.7 | ✅ | Three readouts, each defined and justified |
| 5.8 | ❌ | **`tab:models` and `tab:minimal` are never referenced in this build.** Both are referenced in the long build. `tab:models` is main-text |
| 5.9 | ✅ | §4 and §5 both go past the numbers |
| 5.10 | ✅ | **Supplementary archive now exists and self-verifies**: 98 files, 144 passed / 2 skipped running the command its README documents |

## 6. Writing quality

| | | |
|---|---|---|
| 6.1 | ✅ | Abbreviations defined at first use; "VLM" never used unexpanded in the main text |
| 6.2 | ✅ | **Zero** genuine runs over 25 words without internal punctuation, of 126 sentences. All 9 raw hits were `lineno.sty` gutter artefacts. Long build had 3 genuine |
| 6.3 | ✅ | **0 of 11** main-text paragraphs exceed ~10 rendered lines, excluding the abstract, which is one block at ~19. Long build: 14 of 31 |
| 6.4 | ✅ | **13% passive** (16 of 126) against a 30% ceiling. Higher than the long build's 8%, still comfortable |
| 6.5 | ✅ | **Zero genuine hits** across the 25-word probe. The single `leverage` match is the noun ("readout choices with comparable leverage"), not the LLM-flavoured verb |

## 7. Figures and tables

| | | |
|---|---|---|
| 7.1 | ⚠️ | **The same three thin captions**: `tab:steertext` (59 chars), `tab:staged` (59), `tab:contextbank` (54) |
| 7.2 | ➖ | The method diagram whose 5.8–7.6pt labels were flagged is **not in this build** |
| 7.3 | ⚠️ | Unchanged: blue/red close in luminance, identity carried by direct labels and position |
| 7.4 | ✅ | Every model named in results appears in a legend or column header |
| 7.5 | ⚠️ | **15 of 16 floats use `[h]`, one `[htbp]`, none `[t]`.** The long build's `[t]` float is its method diagram. The build's own guard confirms no float lands in the bibliography |
| 7.6 | ✅ | No redundant pairs |

## 8. Structure and formatting

| | | |
|---|---|---|
| 8.1 | ⚠️ | **Zero overfull boxes** — but **5 unique underfull sites**, a regression from the 2026-08-27 ✅. All 5 are in the appendix (`tex` L646/720/736/776 and `.bbl` L241); none is in the 5-page main text |
| 8.2 | ✅ | Standard structure |
| 8.3 | ❌ | **Three appendix sections orphaned in this build**: `app:headroom`, `app:pilot`, `app:patching`. All three are referenced in the long build |
| 8.4 | ⚠️ | Needs a visual pass, not mechanically checkable |
| 8.5 | ✅ | **Zero** consecutive float pairs without intervening text |

## 9. References

| | | |
|---|---|---|
| 9.1 | ✅ | NeurIPS format |
| 9.2 | ✅ | **Upgraded from ⚠️.** Licences added to `app:impacts`, read from each model card; checklist item 12 is now `\answerYes` |
| 9.3 | ✅ | NeurIPS-venue entries present |
| 9.4 | ✅ | Anonymous, no self-citations |
| 9.5 | ✅ | **61 entries** (was 53), no duplicates, none cited-but-missing. 58 cited in this build; `baayen2008`, `clip`, `siglip` are long-build-only and correctly omitted from the short bibliography by BibTeX |

## 10. Citation sanity — still the biggest open risk

| | | |
|---|---|---|
| 10.1 | ❌ | Not manually verified. **1 of 61 done** (`emomm2026`) |
| 10.2 | ❌ | Unconfirmed |
| 10.3 | ❌ | Not cross-checked |

The entry count has grown from 53 to 61 since the original run, so the exposure is larger, not
smaller. The eight added entries are the asset citations (models, vision towers, toolkits), which
were primary-source verified when added — leaving roughly 53 unverified. This remains the one item
no script can close.

## 11. Pre-submission

| | | |
|---|---|---|
| 11.1 | ✅ | Compiles clean, no errors, zero overfull boxes, no multiply-defined or undefined references |
| 11.2 | ⚠️ | Depends on the portal's naming rule |
| 11.3 | ✅ | No author identity in the PDF. The one "Syed" hit is the cited author Aaquib Syed. The anonymous snapshot is separately audited and clean |
| 11.4 | ✅ | **5pp main text, references p6**, inside the Interp4Discovery limit |
| 11.5 | ❌ | **Still not read start-to-finish by someone off the author list** |
| 11.6 | ✅ | Camera-ready `\todo`s exist and are correctly suppressed in anonymous builds; verified absent from the rendered PDF |

## What I would fix, in order

1. **The five orphaned targets.** Mechanical, no page cost, and the same class of fix the
   2026-08-27 run put first. The cause is structural and worth understanding: each pointing sentence
   lives in an `\ifshort\else` branch, so the short build inherits the float without its citation.
   `tab:models` matters most — a main-text table with no textual pointer is the version a reviewer
   sees first. A single clause in §5 ("Table~\ref{tab:models} collects the four-model comparison")
   fixes it, and §5 has no slack, so it must be paid for.
2. **Verify the remaining ~53 citations.** Highest risk, unchanged, and only you can do it.
3. **2.4, the missing contributions list.** A reviewer skims for it. At 5pp an itemised list is
   expensive; a single sentence naming the three contributions in order would recover most of the
   benefit.
4. **The three thin captions**, unchanged from the original run.

## What changed since 2026-08-27, and why

Most deltas are venue-structural rather than quality regressions: the short build drops the
related-work section, the contributions list, and every main-text figure to reach 5pp. Two are real
and worth acting on — the orphaned targets (§5.8, §8.3) and the underfull boxes (§8.1). The writing
items improved measurably. Nothing here blocks submission.
