# Paper Audit: "When One Word Changes an Image Judgment" / "The Readout Decides the Finding"

**Date:** 2026-09-01 · **Source:** `paper/neurips_2026.tex` (745 L), `paper/venue.tex`,
`paper/tables/{minimal,patching,models,mechmodels}.tex`, `paper/checklist.tex`,
`paper/references.bib`. Both `\ifshort` branches read and compiled (tectonic).
Numbers spot-checked against `results/stage_f/`.

## The narrative as I read it

A VLM shown a photograph of a person alongside a one-sentence context that contradicts it does
not weigh the two directions of contradiction equally: holding the photograph and the event fixed
and flipping one valence word, negative wording moves Qwen3-VL-8B's emotion judgment four to five
times farther than positive wording, on all six tested pairs and on positive images. The same
sentence pairs with no image show no detectable difference, so this is not a stimulus-strength
artifact. In Gemma-3-4B a text-trained valence probe transfers to image-conditioned states, and
activation patching finds downstream text positions sufficient to restore 88–93% of the context
effect. Two measurement results qualify all of it: in the one layer band where both readouts are
valid they give non-overlapping intervals in opposite directions, and scoring only a multi-token
emotion label's first piece manufactured a null in one model and reversed another — dissolving an
architectural boundary an earlier analysis had reported.

The short (Interp4Discovery) build promotes the two measurement failures to the headline; the long
(VLM4RWD) build leads with the behavioral asymmetry. Both narratives are legible and honestly
calibrated, and I could reconstruct each after one read.

## Summary

This is a careful paper. The line-level prose is clean — no contractions, no vague-judgment words,
no aspirational hedging, no citations in grammatical roles — and the calibration machinery
(withdrawn runs, superseded-stack provenance, a disclosed post-hoc correction, negative results at
equal prominence) is better than most submissions carry. Both builds compile inside their page
limits, no `\todo` leaks into either PDF, and there are no anonymity leaks. The findings below are
therefore mostly about *verifiability* rather than correctness.

The three highest-leverage fixes: **F1**, a scope word missing from the short build's abstract that
makes the headline claim read as general when the pattern reverses on the other half of the data;
**F2**, Table 4's six per-pair values do not average to the `+1.148` headline printed beside them,
because the same notation denotes two different estimators; and **F3**, the short build's
co-headline claim (readout-dependence) rests on four numbers that appear in no table in either
build. F1 and F2 are both reviewer-facing traps: each is the kind of thing a skeptical reader
checks first, and each currently reads as an error even though the underlying analysis is right.

## Findings

### F1. The short build's abstract drops the scope restriction on the headline claim — CRITICAL
**Where:** `neurips_2026.tex` L66 (short abstract)

**Issue:** The abstract states that "flipping a single valence word with the photograph and event
held fixed moves \qwen{}'s judgment four to five times farther when the word is negative, on all
six tested pairs." The restriction *on positive images* is absent. Every other statement of this
result carries it: the long abstract (L70, "On positive images, negative wording moves…"), the
short introduction (L89, "…with a within-item contrast of $+1.148$ on positive images"), and the
long conclusion (L344).

**Why it matters:** This is not a rounding of scope — the pattern reverses on the excluded half.
Table~6 (`tab:factorial`, L396–397) shows that on negative images a *positive* context moves
valence by $+0.762$ while a negative context moves it by $-0.017$: the negative image group is
already at the floor, so there the positive direction moves further. The within-item contrast is
only ever computed on the 62 positive images (`results/stage_f/…minimal_analysis.json` records
`image_group: positive`, `n_images: 62`). A reviewer who reads the abstract as written and then
reaches Table~6 finds an apparent contradiction in the paper's single most-read sentence.

**Suggested fix:** Insert the scope, matching the long abstract's phrasing:

> …where flipping a single valence word with the photograph and event held fixed moves \qwen{}'s
> judgment **on positive images** four to five times farther when the word is negative, on all six
> tested pairs.

### F2. Table 4 does not average to the +1.148 headline printed beside it — MAJOR
**Where:** `neurips_2026.tex` L195 and L402–419 (`tab:pairs`); short build L173

**Issue:** The text reports the within-item contrast as $+1.148$ and, two sentences later, points
at Table~4 for "every pair". Table~4's column is headed $|\Delta_{\rm neg}|-|\Delta_{\rm pos}|$ and
lists $1.169, 1.212, 1.121, 1.102, 1.134, 1.292$ — which average to **$1.172$**, not $1.148$. Every
pair covers the same 62 images, so a reader has no reason to expect the two to differ.

Both numbers are faithfully transcribed; the analysis is not wrong. They are different estimators
sharing one notation. Confirmed against the source: the headline takes the absolute value
per photograph and then averages over the 62 photographs
(`per-image (abs first) = 1.1480072716435483`, exactly the reported value), while the table takes
the absolute value of each pair's mean effects. $\overline{|x|} \neq |\bar{x}|$, and the gap is the
whole 0.024.

**Why it matters:** Adding up a six-row table is the cheapest check a reviewer performs, and it
currently fails. The paper has already retracted results once; an unexplained 2% discrepancy next
to the headline number invites the reader to start re-deriving everything else.

**Suggested fix:** Distinguish the two in the caption, e.g. append to the Table~4 caption:

> Each row averages the per-image effects within a pair before taking absolute values; the headline
> within-item contrast of \S\ref{sec:minimal} instead takes absolute values per photograph and
> averages over the 62 photographs, so these six values average to $+1.172$ rather than $+1.148$.

While in this table: the ratio column runs to $5.45$, so "four to five times" (L66, L70, L89, L105,
L344) and "ratios from $4.3$ to $5.5$" (L173, L195) sit slightly apart. Either is defensible;
saying "four to five" once and "4.3 to 5.5" elsewhere is what reads as sloppy.

### F3. Load-bearing statistics that appear in no table — MAJOR
**Where:** `neurips_2026.tex` L221 (short), L251 (long); L295

**Issue:** The readout-dependence result — one of the two failures the short build is *titled*
around — rests on four numbers given only in running prose: the layer-18 probe's image $74.9\%$
$[71.3,78.9]$ against text $87.4\%$ $[85.9,88.8]$. `tab:crosspatch` (L440–455) reports the
behavioral-valence row only, so a reader can verify half of the comparison and must take the other
half on trust. The same applies more mildly to the Qwen resolution sweep (L295: flat from 128 to
262 image tokens at AUC $0.983$–$0.986$), which is cited as evidence against a token-count
explanation and has no table anywhere.

**Why it matters:** In the short build this is a headline claim, and the paper's own argument is
that readouts disagree — which makes the probe-versus-behavioral comparison the evidence, not an
aside. A claim of this weight should be checkable from a float.

**Suggested fix:** Add a probe-readout row (or a parallel probe block) to `tab:crosspatch` for the
bands where the probe is valid, marking layers 18–28 as not-applicable with the reason already
given at L731 (the layer-18 activation is computed from layer 17). Both builds carry
`tab:crosspatch` in the appendix, so this costs no main-text space in either venue. For the
resolution sweep, a two-line table in `app:headroom` or a pointer to the metrics file would do.

### F4. The long build never says why its mechanism section belongs to its story — MAJOR
**Where:** `neurips_2026.tex` L231 (opening of §4.3)

**Issue:** The long build's title and headline claim are about Qwen3-VL-8B. Its mechanism section
opens by conceding that "\gemma{} is not the model carrying the behavioral contrast of
\S\ref{sec:minimal}, so what follows locates where a context effect lives in \gemma{}; it is not a
mechanism for the \qwen{} asymmetry." The honesty is right. But the paragraph answers "is this the
mechanism for my headline?" with *no* and then stops, leaving the reader to supply the reason the
section is in the paper at all.

**Why it matters:** By the narrative test — "what breaks if this section is cut?" — the long build
currently gives no answer, which makes its largest results section look like inherited material
from a different paper. That is a reviewer's cut suggestion, and cutting it would take the probe
transfer and the patching results with it.

**Suggested fix:** Add one sentence of positive motivation after the concession, e.g.: *We study it
in \gemma{} because it is the only model in our set where a text-trained valence probe gives an
independent internal readout, which is what makes the readout-dependence of \S\ref{sec:crosspatch}
measurable at all.* This also sets up F3's comparison.

### F5. The novelty claim is a universal negative — MAJOR
**Where:** `neurips_2026.tex` L122 (Related work, long build)

**Issue:** "…and none of this work asks whether matched positive and negative text pull with equal
force, which is what we test in both directions across four model designs." Appendix~D restates it
("do not isolate whether matched positive and negative text exert different pull", L470). A single
counterexample among `deng2025blindfaith`, `mixedsignals2025`, `contextvqa2026`, `signpost2026`,
`camel2025` refutes the sentence as written.

**Why it matters:** Reviewers of a conflict-in-VLMs paper are drawn from exactly the set of authors
being characterized. An absolute claim is also unnecessary: the paper's contribution survives intact
if it claims the comparison has not been made *under matched conditions in both directions*, which
is a much smaller target.

**Suggested fix:** Scope to what was checked: *We are not aware of prior work that measures both
conflict directions against a per-image neutral baseline with the event held fixed, which is the
comparison we run across four model designs.* The short build already handles this well at L95
("We do not claim to discover that text can override vision…") — mirror that register.

### F6. One quantity, two names — MAJOR
**Where:** "bounded valence" L142, L283, L285(caption), L658; "behavioral valence" L184, L218,
L224, L247, L249, L389, L431, and both `tab:crosspatch` and `tab:mechmodels` captions; defined at
L608

**Issue:** The Readouts paragraph (L142) introduces "*bounded valence*, the probability mass
difference $P(\text{positive})-P(\text{negative})$". Appendix~G (L608) defines "*Behavioral
valence* is the total probability of \{joy, pride, relief, trust\} minus the total probability of
\{anger, boredom, disgust, fear, guilt, sadness, shame\}". These are the same quantity, and the
paper never says so. The main text then uses "behavioral valence" as the readout label in two table
captions the reader meets before Appendix~G.

**Why it matters:** The scholarly-writing failure mode here is specific: a reader who has been told
the paper deliberately reports *three* readouts, and who then meets a fourth name, reasonably
concludes there is a fourth measure. Given that readout identity is the paper's own subject, this
is the worst possible place for a naming ambiguity.

**Suggested fix:** Use one term throughout — "bounded valence" is the more informative one, and it
is the term the three-readout list at L142/L283 depends on. If "behavioral valence" is kept for the
mechanism sections to signal "measured from the output rather than the probe", say so explicitly at
first use: *behavioral valence (the bounded score of \S\ref{sec:setup}, read from the output rather
than the probe)*.

### F7. `sec:arbitration` is multiply defined in the short build — MAJOR
**Where:** `neurips_2026.tex` L208 (inside the `\ifshort` mechanism section) and L593 (Appendix F.4,
unconditional)

**Issue:** Confirmed from the compile log:
`LaTeX Warning: Label 'sec:arbitration' multiply defined.` The long build is clean. The two
references to it (L485 and L492, both in unconditional appendix text) currently resolve to the
appendix subsection because the later definition wins, so the rendered PDF is correct today — by
accident of ordering, not by design.

**Why it matters:** The handoff records that the `\ifshort` branch trap "bit four times this
session, always silently." This is the same trap, still live: any reordering of the appendix, or a
future reference added from the short main text, flips the target with no error. It is also the
only warning either build emits.

**Suggested fix:** Rename the short-branch label at L208 (e.g. `sec:arbitration-short`) or delete
it, since nothing in the short build references it. Then re-run both builds and confirm the log is
warning-free.

### F8. Floats that no text references — MINOR
**Where:** `tab:minimal` (both builds); short build additionally `tab:models` (L267),
`tab:factorial`, `tab:pairs`, `fig:conflict`

**Issue:** `\ref{tab:minimal}` appears nowhere in the source. In the long build the table at least
sits inside the prose that discusses its contents (L191–195). In the short build it is emitted by a
bare `\ifshort \input{tables/minimal} \fi` at L381–383 — dropped into Appendix~A with no
introducing sentence at all. Also in the short build, `tables/models` is `\input` at L267 with no
`Table~\ref`, and `tab:factorial`, `tab:pairs` and `fig:conflict` are covered only by a collective
gesture at L173 ("Appendix~\ref{app:behavior} reports the full factorial matrix, every pair, and
the baseline-corrected override gaps").

**Why it matters:** Commit `fc8b002` referenced the orphaned floats for the long build; the short
branch was not swept. A table a reader arrives at with no idea what it is meant to show is a table
that will be skipped.

**Suggested fix:** Add `Table~\ref{tab:minimal}` to the sentence at L191 (long) and L173 (short),
and give the short build's Appendix~A a one-line lead-in before each `\input`. Add
`(Table~\ref{tab:models})` to the short build's L265.

### F9. "sixteen" is the only spelled-out number above ten — MINOR
**Where:** L221 and L251, "after sixteen further layers of mixing"

**Issue:** The paper's convention is words below ten and numerals at and above it (six pairs, four
models, three readouts, eight layers; 13 labels, 35 subsets, 51 photographs, 62 images, 150
annotations). "sixteen" breaks it, and does so in a sentence otherwise dense with layer numerals.

**Suggested fix:** "after 16 further layers of mixing."

### F10. Bare model names alternate with the macros — MINOR
**Where:** L184, L186 ("Qwen weights them most comparably", "Qwen's mirror contrast"); L287 ("That
is complete for Qwen and Gemma here, but not for LLaVA"); L145, L159 ("Gemma layer 18"); L646

**Issue:** `\qwen{}` renders "Qwen3-VL-8B", but L184 writes bare "Qwen" three lines after `\qwen{}`
appears in the same subsection. The reader sees two names for one model, and in a paper comparing
LLaVA-1.5-7B against LLaVA-NeXT-7B, bare "LLaVA" (L287, L289) is genuinely ambiguous.

**Suggested fix:** Use the macros everywhere a specific model is meant, and reserve bare "LLaVA"
for statements about both models jointly — where the paper should say "both LLaVA models", as it
already does at L295 and L323.

### F11. A robustness statistic carries no unit — MINOR
**Where:** L291, "…but not the categorical one ($20/35$, reversing to $-0.714$ once *anger* and
*sadness* are dropped)"

**Issue:** The categorical override gap is reported as a percentage everywhere else in the paper
($+21.7\%$, $-11.2\%$, $+63.9\%$, $+39.4\%$). Here the reversed value appears as $-0.714$ with no
unit, adjacent to a graded contrast quoted as $+1.144$, so the reader cannot tell which scale it is
on or whether $-0.714$ is a 0.7-point or a 71.4-point reversal.

**Suggested fix:** Give it the unit used for that readout, e.g. "reversing to $-71.4\%$", or name
the quantity if it is not the corrected gap.

### F12. "Headline contrast" points at the wrong statistic — MINOR
**Where:** L136 ("excluding it changes the headline contrast by less than $0.01$") against L386 and
the short build's L305

**Issue:** The appendix and the short limitations both report the overlap check on the *mirror*
contrast ($+0.496 \to +0.505$). The paper's headline contrast is the within-item contrast
($+1.148$). The claim at L136 is true of the mirror contrast; it is not stated for the headline one.

**Suggested fix:** Name the statistic: "excluding it changes the matched-set mirror contrast by less
than $0.01$ (Appendix~\ref{app:behavior})."

### F13. The long build asserts Qwen's suitability without the reason — MINOR
**Where:** L184, "making it the cleanest behavioral test"

**Issue:** The short build (L169) gives the actual argument — "so an asymmetry it shows cannot be
explained by it under-using the image" — which is what turns a regression-weight ratio into a
justification for model choice. The long build states the conclusion and drops the reasoning, in the
build that has more room for it.

**Suggested fix:** Port the short build's clause: "…$|\betatxt|/|\betaimg|=1.14$; the others range
from $0.44$ to $0.68$), so an asymmetry it shows cannot be explained by its under-using the image."

### F14. A correction described in the direction opposite to how it reads — MINOR
**Where:** L611, "The legacy first-content-token score changes \llava{} from a strong asymmetry to
a null and changes \llavanext{} from one crossed test clearing zero to all three."

**Issue:** The sentence is correct — legacy scoring is the subject, so it takes LLaVA-NeXT *to* all
three — but "changes X from one to all three" reads on first pass as the improvement the paper is
arguing for, which is the reverse of the point. The parallel clause about `\llava{}` primes exactly
that misreading.

**Suggested fix:** Recast with complete-label scoring as the subject, matching L287: "Complete-label
scoring moves \llava{} from a null to a strong asymmetry, and moves \llavanext{} from three crossed
tests clearing zero down to one."

## What is already working (keep it)

A revision pass under deadline pressure will be tempted to strip these. They are the reason the
paper is credible, and several would be very hard to rebuild.

- **The provenance and withdrawal machinery.** The superseded-inference-stack notes (L664, L694,
  L717), the withdrawn rephrasing sweep and scale comparison (L309, L339), and the compute appendix
  counting the withdrawn runs' forward passes (L739). Papers almost never do this.
- **The self-retraction in Appendix J** (L694): reporting that an earlier version of that appendix
  claimed a representational effect, and withdrawing it while noting the design argument does not
  depend on it.
- **The disclosed post-hoc decision** (L293): "We flag that we applied this categorical correction
  after seeing the uncorrected results." Keep the sentence and keep it where it is.
- **Reporting all three readouts including the ones that kill the result** — Qwen clearing none of
  the three crossed tests (L291), the varied set failing sentence resampling (L186), the
  balanced-subset check where the categorical readout reverses (L291). The temptation to report the
  favorable measure per model is explicitly named and declined.
- **The zero-by-construction framing.** Image-token rows are labelled alignment checks in the table
  caption, the main text (L215, L242, L297), and the figure panel. This was previously framed as a
  finding; the current treatment is right and should not drift back.
- **Calibrated closing claims:** "a controlled valence asymmetry in the tested setup, not a
  universal law" (L344), and the explicit non-claims at L226 and L599 (steering changes the outcome,
  not necessarily the cue weighting).
- **Clean line-level prose.** No contractions, no "interesting"/"important to note", no aspirational
  hedging, no parenthetical citations in grammatical roles. Do not let a hurried revision introduce
  them.

## Flagged notes (outside the narrative/style scope)

- **Submission viability is fine.** Both builds compile inside their limits (VLM4RWD 8pp, references
  p9; Interp4Discovery 5pp, references p6). No `\todo` reaches either PDF — `\todo` expands to
  nothing under `\if@anonymous`, which `neurips_2026.sty` sets true by default (L72), verified by
  extracting text from both PDFs. No anonymity leaks in the source. The Overleaf bundle is flattened
  to `main.tex` and does not sweep in `paper/overleaf.stale-snapshot.tex.bak`.
- **`paper/venue.tex` L2–3 still records the original deadlines (Aug 30 / Aug 29).** Comment-only, so
  it does not render, but it is now stale against the extended timeline and is the file a
  collaborator reads first to learn the venue constraints.
- **`paper/checklist.tex` carries four `\answerNo{}`:** item 5 (open access to code), item 8 (compute
  resources), item 12 (licences for existing assets), item 13 (new assets). Items 8 and 12 are both
  cheap to flip and both live in appendix space, which neither venue counts. The unresolved
  `\todo{}` at L739 requesting wall-clock totals is what holds item 8 at No.
