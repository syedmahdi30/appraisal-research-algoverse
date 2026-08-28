# CS paper checklist — VLM4RWD build

_Run 2026-08-27 against `2149b44`. Every item below was measured against the built PDF or the
source, not eyeballed. ✅ pass · ⚠️ partial · ❌ fail · ➖ not applicable_

## Headline

The paper passes most of this checklist, and passes the writing items unusually well:
**zero bad boxes, 8% passive voice against a 30% ceiling, and zero LLM-flavoured vocabulary**
from a 25-word probe list. Three things need work, in order:

1. **Six orphaned labels** (§5.8, §8.3) — three appendix sections and three floats are defined
   and never referenced. This is a regression from the restructure and reviewers notice it.
2. **Citations are unverified** (§10) — the highest-risk open item in the whole document.
3. **No first-page figure** (§2.6) — the method diagram now exists but costs a page.

---

## 1. Title and abstract

| | | |
|---|---|---|
| 1.1 | ✅ | 13 words ("When One Word Changes an Image Judgment: A Valence Asymmetry in Vision-Language Models") |
| 1.2 | ✅ | Names the phenomenon and the finding; carries "Vision-Language Models" and "Valence" as keywords |
| 1.3 | ✅ | No abbreviations in the title at all |
| 1.4 | ✅ | Problem, approach, results and significance all present |
| 1.5 | ✅ | Zero vague descriptors ("novel", "important", "state-of-the-art", "significant" all absent) |
| 1.6 | ✅ | +1.148, four to five times, 88–93% |

⚠️ Not a checklist item, but the abstract is **291 words**, long for a workshop where 150–250 is
typical. It survived a compression pass already; further cuts would cost content you want.

## 2. Introduction

| | | |
|---|---|---|
| 2.1 | ✅ | Problem stated in the first paragraph |
| 2.2 | ✅ | Both: real deployment settings (moderation queues, assistive agents, intake tools) and four citations |
| 2.3 | ➖ | No named method to preview; the paper reports a phenomenon |
| 2.4 | ✅ | Three itemised contributions |
| 2.5 | ✅ | Each is specific and checkable |
| 2.6 | ❌ | **No first-page figure.** `paper/figures/method_diagram.pdf` now exists and would fill this, but at `\linewidth` it takes the build to 9pp |

## 3. Related work

| | | |
|---|---|---|
| 3.1 | ✅ | Every citation is tied to the task, a comparison model, or the method |
| 3.2 | ✅ | The main conflict papers are all present |
| 3.3 | ✅ | 664 words, far under 1.5 pages |
| 3.4 | ❌ | **Not verified.** Sneheel already flagged one claim with "verify if this is really true", and project notes record that some entries came from AI-assisted search |
| 3.5 | ➖ | No baseline table; this is not a method-comparison paper |

## 4. Method

| | | |
|---|---|---|
| 4.1 | ✅ | Symbols defined before use |
| 4.2 | ✅ | Few display equations; the rest are inline, as the item recommends |
| 4.3 | ✅ | All components described |
| 4.4 | ⚠️ | Subsections are clean, but there is no overview figure for them to align with (see 2.6) |
| 4.5 | ✅ | No pseudocode in the main text |
| 4.6 | ⚠️ | The main text leans on appendices for method detail in several places |
| 4.7 | ✅ | Already through several compression passes |

## 5. Experiments

| | | |
|---|---|---|
| 5.1 | ❌ | **Two datasets** (EMOTIC, crowd-enVENT) against a target of three. Defensible for a controlled-paradigm paper, but a reviewer may raise it |
| 5.2 | ➖ | Four VLMs compared, though not "baselines" in the method-comparison sense |
| 5.3 | ✅ | Several: text-only control, norm-matched random directions, caption controls, resolution sweep, and the new label-balance re-scoring |
| 5.4 | ✅ | Bootstrap intervals throughout, in two forms, with the stricter one identified |
| 5.5 | ⚠️ | Hardware and per-experiment pass counts are in the appendix; wall-clock totals are still missing |
| 5.6 | ✅ | Nulls are reported prominently, including in the abstract |
| 5.7 | ✅ | Three readouts, each defined and justified |
| 5.8 | ❌ | **`fig:conflict`, `tab:factorial`, `tab:pairs` are never referenced in the text** |
| 5.9 | ✅ | §4.4 and §5 both go past the numbers |
| 5.10 | ✅ | The supplementary archive self-verifies: 143 tests and three CPU analyses run inside it |

## 6. Writing quality

| | | |
|---|---|---|
| 6.1 | ✅ | Abbreviations defined at first use |
| 6.2 | ✅ | Of 215 sentences, 5 exceed 25 words without punctuation, and 2 of those are pdftotext artefacts |
| 6.3 | ⚠️ | **14 of 31 paragraphs exceed 10 rendered lines** |
| 6.4 | ✅ | **8% passive**, against a 30% ceiling |
| 6.5 | ✅ | **Zero hits** across a 25-word LLM-vocabulary probe (encompass, intricate, delve, leverage, pivotal, myriad, …) |

## 7. Figures and tables

| | | |
|---|---|---|
| 7.1 | ⚠️ | Three captions are under two lines: the two steering tables (74, 76 chars) and the sentence-set table (54) |
| 7.2 | ⚠️ | The new method diagram labels run 5.8–7.6pt, under the 8pt guideline |
| 7.3 | ⚠️ | Blue/red are close in luminance; identity is carried by direct labels and position, so grayscale still reads |
| 7.4 | ✅ | Every model named in results appears in a legend or column header |
| 7.5 | ✅ | Floats use `[t]` |
| 7.6 | ✅ | No redundant pairs |

## 8. Structure and formatting

| | | |
|---|---|---|
| 8.1 | ✅ | **Zero overfull or underfull boxes** |
| 8.2 | ✅ | Standard structure |
| 8.3 | ❌ | **Three appendix sections orphaned**: "Correcting for the scale's bounds" (`app:headroom`), "The no-context baseline confound" (`app:nocontext`), "Pilot vs. varied set" (`app:pilot`) |
| 8.4 | ⚠️ | Needs a visual pass, not mechanically checkable |
| 8.5 | ✅ | No consecutive floats without text |

## 9. References

| | | |
|---|---|---|
| 9.1 | ✅ | NeurIPS format |
| 9.2 | ⚠️ | Datasets and models are cited; explicit licences are still missing (checklist item 12) |
| 9.3 | ✅ | 8 NeurIPS-venue entries |
| 9.4 | ✅ | Anonymous submission, no self-citations |
| 9.5 | ✅ | **53 entries, no duplicates, all 53 cited, none cited-but-missing** |

## 10. Citation sanity — the biggest open risk

| | | |
|---|---|---|
| 10.1 | ❌ | Not manually verified |
| 10.2 | ❌ | Unconfirmed |
| 10.3 | ❌ | Not cross-checked |

Project notes state plainly that several references were surfaced by AI-assisted literature
search and remain unverified. With 53 entries and a related-work claim already flagged as
suspect, this is where a fabricated or misattributed citation would do the most damage. It is
also the one item on this list that no script can close for you.

## 11. Pre-submission

| | | |
|---|---|---|
| 11.1 | ✅ | Compiles clean, no errors, no bad boxes |
| 11.2 | ⚠️ | Depends on the submission portal's naming rule |
| 11.3 | ✅ | No author identity in the PDF or its metadata. The one "Syed" hit is the cited author Aaquib Syed, not you. The supplementary archive is separately guarded |
| 11.4 | ✅ | 8pp main text, references from p9 |
| 11.5 | ❌ | Not yet read start-to-finish by someone off the author list |
| 11.6 | ⚠️ | A camera-ready TODO for acknowledgements exists and is correctly suppressed in submission builds |

## What I would fix, in order

1. **The six orphaned labels.** Purely mechanical, no page cost, and the fastest credibility win
   available. Three appendix sections and three floats need a pointer from the main text.
2. **Verify the 53 citations.** Highest risk, and only you can do it.
3. **Decide on the first-page figure.** It exists and is ready; it needs a page paid for from §4.
4. **The three thin captions**, and split the longest paragraphs.
