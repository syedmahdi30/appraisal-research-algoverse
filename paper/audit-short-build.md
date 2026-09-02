# Paper Audit: "The Readout Decides the Finding" — Interp4Discovery (short build)

**Date:** 2026-09-02 · **Source:** `paper/neurips_2026.tex` at `a815d7f`, compiled with
`\shorttrue`; `paper/tables/{patching,models,minimal}.tex`, `paper/references.bib`,
`paper/checklist.tex`. Numbers cross-checked against the tables the short build actually renders.

**Scope:** the `\ifshort` build only. The whole paper was read, including long-build-only branches,
because judging what the short abstract promises requires knowing what the short body delivers — but
every finding below is confined to text that reaches the 5-page Interp4Discovery PDF. Findings are
independent of `paper/audit.md`, whose 14 findings are all closed.

**Nothing here blocks submission.** The build is at 5pp with references on p6, emits no
multiply-defined / undefined-reference / overfull warnings, leaks no `\todo`, and is
anonymity-clean. These are quality findings, not gates.

## The narrative as I read it

Interpretability tools localize whatever the measurement puts in front of them, so a localization
result is only as trustworthy as the readout it is taken through. The paper demonstrates this on one
controlled cross-modal phenomenon — EMOTIC photographs against contradicting one-sentence contexts,
where flipping a single valence word moves Qwen3-VL-8B's judgment on positive images four to five
times farther when the word is negative. Three standard tools are applied to it. Two conclusions
hold (a text-trained probe transfers to image-conditioned states; patching localizes states
sufficient to restore each cue). Two failures qualify them: in the one layer band where both
readouts are valid they give non-overlapping intervals in *opposite* directions, and the cross-model
architectural story turned out to be an artifact of scoring only the first token of a multi-token
emotion label. The takeaway is methodological — check the readout before trusting the localization.

That narrative is clear, cohesive, and I could reconstruct it after one read. The claims are
specific, the confidence is calibrated to the evidence, and the paper consistently reports what
failed as prominently as what worked.

## Summary

This is a careful, unusually honest paper, and the short build is the sharper of the two framings:
leading with the measurement failures rather than the behavioral asymmetry gives it a thesis rather
than a result. Prose is clean — no contractions, no informal fillers, no aspirational hedging, no
vague-judgment words, no trendiness, no citations in grammatical roles. Every number I traced in the
main text matches its table.

The three highest-leverage fixes, in order: **F1**, the abstract and introduction both announce
*three* tools and then account for only two, leaving the paper's own organizing promise unclosed in
its two most-read passages; **F2**, the varied-set mirror contrast does not equal the difference of
the two cells the paper bolds for it, and unlike the matched-set version this discrepancy carries no
explanation; and **F3**, three mechanism-replication percentages in §5 have no table anywhere in
this build. F1 and F2 are both cheap — a clause and a caption sentence.

## Findings

### F1. The abstract and introduction promise three tools and deliver verdicts on two — MAJOR
**Where:** `neurips_2026.tex` L66 (short abstract), L91 (short introduction)

**Issue:** The abstract states "We apply three standard tools to one cross-modal phenomenon and
report where each does and does not license a conclusion." It then describes the ridge probe and
activation patching. The third tool never appears. The introduction is more explicit and has the
same gap: "Onto that phenomenon we apply a frozen text-trained linear probe, activation patching,
and a difference-of-means intervention. Two claims survive. The probe transfers across modalities,
and patching localizes states sufficient to restore each cue's effect…" — three named, two
adjudicated.

The missing verdict does exist, in §4 (L222, "The intervention still works under conflict"): adding
the text-derived pleasantness direction at layer 18 moves behavioral valence with slope $+0.335$
across 150 conflict trials, but "it does not show that the competition between the cues changed,
since a general output-valence direction could move the score without altering either cue's weight."

**Why it matters:** That is a *qualified* verdict — the tool works but licenses less than it
appears to — which is precisely the paper's thesis, arriving as a third instance of it. Burying it
costs the paper its best-fitting example. Meanwhile a reader who counts is left with an unclosed
promise in the abstract and introduction, the two passages that get read an order of magnitude more
than §4. A skeptical reviewer reads "three tools, two claims, two failures" and reasonably asks
whether the third tool was quietly dropped because it did not work.

**Suggested fix:** Cheapest version, one clause in the abstract after the patching clause:

> …and a difference-of-means direction fit on text alone still moves the answer under conflict,
> though it does not show that either cue's weight changed.

If the abstract cannot spare the words at zero page slack, the introduction is the more important of
the two to fix, since it has room and already names all three tools. Alternatively, change "three
standard tools" to "two localization tools" and demote the intervention — but that discards the
best-fitting instance of the paper's own argument, so I would not.

### F2. The varied-set mirror contrast is not the difference of the two cells the paper bolds — MAJOR
**Where:** `neurips_2026.tex` L171 (§3), against `tab:factorial` (L387–400, in this build)

**Issue:** §3 reads: "a negative context moves positive-image valence by $-1.156$ while a positive
context moves negative-image valence by $+0.762$; the per-image mirror contrast is $+0.409$."
$1.156 - 0.762 = 0.394$, not $0.409$. `tab:factorial` is rendered in this build, bolds exactly those
two cells, and its caption says "The mirror contrast compares the two conflict trials (bold)" — an
instruction to perform the subtraction that does not reproduce the number.

The paper already knows about this class of mismatch: `tab:pairs` (L404) carries a caption
explaining that its six rows average to $+1.172$ while the headline within-item contrast is
$+1.148$, because one estimator averages before taking absolute values and the other after. The
same two-estimator distinction explains $0.394$ against $0.409$. Nothing states it here. The single
word "per-image" in L171 is the only signal, and it appears in the main text rather than in the
caption that invites the subtraction.

**Why it matters:** This is the first quantitative claim in the paper's phenomenon section, and the
subtraction is one a skeptical reader performs automatically because the caption tells them to. A
reviewer who finds an arithmetic mismatch in the headline behavioral result starts re-checking
everything else — and the previous audit rated the identical issue on `tab:pairs` as MAJOR for the
same reason.

**Suggested fix:** Extend the `tab:factorial` caption with the note that already works elsewhere:

> The mirror contrast compares the two conflict trials (bold). It is computed per photograph and
> then averaged, so it is $+0.409$ rather than the $+0.394$ obtained by differencing these two cell
> means; see the note to Table~\ref{tab:pairs}.

Appendix caption only — no main-text page cost.

### F3. Three mechanism-replication percentages have no table in this build — MAJOR
**Where:** `neurips_2026.tex` L275 (§5, "What this costs the mechanism claims")

**Issue:** The short build states that replacement runs on Qwen3-VL-8B and LLaVA-1.5-7B "restore
$62\%$ and $66\%$ of the context difference from all text positions jointly, against $82\%$ for
Gemma-3-4B on the same context pair and readout." Those three numbers come from `tab:mechmodels`,
which is gated `\ifshort\else` at L438–446 and therefore never renders in this build. There is no
`\ref` to it in the short branch, so LaTeX raises nothing — the numbers are simply unverifiable
here, with no pointer to where they can be checked.

**Why it matters:** This paragraph carries the paper's cross-model mechanism claim, the one that
bounds how far the Gemma localization generalizes. It is also the claim most exposed to the paper's
own thesis, since the ports use wider layer bands and a coarser readout. Every other number in the
short main text traces to a rendered table; this paragraph is the exception, and it is the one a
reviewer is most likely to want to check.

**Suggested fix:** One sentence pointing at where the breakdown lives — "the per-group breakdown and
layer bands are in the extended version" — or, if `tab:mechmodels` fits the appendix at 5pp,
un-gate it. Given that the appendix has no page limit at this venue, un-gating is the better fix and
costs no main-text lines.

### F4. The abstract's central sentence carries four numbers and two layer ranges — MINOR
**Where:** `neurips_2026.tex` L66, the sentence beginning "A ridge probe trained only on text…"

**Issue:** That single sentence runs roughly sixty words and delivers $\rho=+0.510$, $88$–$93\%$,
$100\%$ with layers 0–12, and $63\%$ with layers 18–28, across two different experiments (same-image
context patching and cross-image patching). The abstract's other sentences are well-paced; this one
asks a cold reader to hold two experimental designs and four statistics at once.

**Why it matters:** Most readers stop at the abstract. This is the sentence carrying the evidence
for both surviving claims, and density is where a reader disengages.

**Suggested fix:** Split after the probe clause: "…transfers to image-conditioned states
($\rho=+0.510$). Activation patching localizes states sufficient to restore each cue: downstream
text positions restore $88$–$93\%$ of the context difference in Gemma-3-4B, while the image's own
valence is recovered from image positions early and from text positions late." Dropping the two
parenthetical layer ranges from the abstract costs nothing — both are in §4 and `tab:crosspatch` —
and buys back the words F1 needs.

### F5. The abstract opens a sentence with "And" — MINOR
**Where:** `neurips_2026.tex` L66: "…about which token group dominates. And the cross-model
conclusion was an artifact of a scoring choice…"

**Issue:** Formal academic prose does not open sentences with "And", "But", or "Or". This is the
only instance in the short build's main text, and it sits in the abstract.

**Suggested fix:** "The cross-model conclusion was likewise an artifact of a scoring choice…" — same
length, keeps the parallel with the preceding failure.

### F6. A contrast between the two sentence pairs holds for both pairs — MINOR
**Where:** `neurips_2026.tex` ~L215 (§4, "Where the sentence's effect can be restored"), against
`tab:patching`

**Issue:** The text says the pairs "disagree about how the recovery divides: on pair 1 the question
tokens and the tokens before the model's answer restore indistinguishable shares, while on pair 2
the question-token estimate is larger *and the two individual estimates sum to more than the joint
recovery*." The first half of the pair-2 clause is correct ($54.8\%$ against $38.2\%$, non-overlapping).
The second half is true of pair 1 as well: $49.1 + 45.2 = 94.3 > 93.2$, just as $54.8 + 38.2 = 93.0 > 87.9$.
Phrasing it as pair-2-specific implies pair 1 is additive, which it is not.

**Why it matters:** Super-additivity is the paper's stated reason for refusing to assign causal
shares — a conclusion that is in fact supported by *both* pairs. Attributing it to one weakens the
justification for no reason.

**Suggested fix:** "…while on pair 2 the question-token estimate is clearly larger. In both pairs
the two individual estimates sum to more than the joint recovery."

### F7. §3 and §4 each carry a silent stack of compatibility labels — MINOR
**Where:** `neurips_2026.tex` L163–166 (`sec:conflict`, `sec:integration`, `sec:asymmetry`,
`sec:minimal`) and L205–206 (`sec:patching`, `sec:crosspatch`)

**Issue:** These exist so that unconditional appendix cross-references resolve in both builds. That
is a legitimate device, but nothing in the source says so, and the pattern already misfired once:
a `sec:arbitration` label in the same L205 cluster collided with an unconditional appendix
definition and was removed at `ad482dd`. The two remaining stacks are correct today because each
twin sits in the mutually exclusive long branch, but that invariant is invisible.

**Suggested fix:** A one-line source comment above each stack recording why they exist and the rule
that keeps them safe — a compat label may duplicate a *long-branch* definition, never an
unconditional one.

### F8. The short build has no related-work section, and the intro carries all positioning — MINOR
**Where:** `neurips_2026.tex` L117 (`\ifshort\else` around §Related work)

**Issue:** The 5-page build omits related work entirely, deferring to `app:related`. The intro does
position — it disclaims discovering that text can override vision, and disclaims being the first
mechanistic study of multimodal conflict, each with citations — so this is not unpositioned work.
But a reviewer reading only the 5 main pages encounters no sustained comparison to prior work.

**Why it matters:** Novelty positioning is what a reviewer probes first on a short paper, and the
disclaimers, while admirably honest, tell the reader what the paper is *not* before the appendix
tells them what it is.

**Suggested fix:** No structural change at zero page slack. Consider one clause on what the work
adds, adjacent to the existing disclaimers, so the intro states the positive as well as the
negative — e.g. that the contribution is a case where three tools disagree on one controlled
phenomenon, not a new phenomenon.

### F9. One paragraph title opens with a conjunction — MINOR
**Where:** `neurips_2026.tex` ~L266, `\paragraph{Nor does averaging fix it.}`

**Issue:** Reads as a continuation of the previous heading rather than a standalone signpost, and
sits oddly against the other titles in §5, which are declarative sentences ("A scoring choice
manufactured a null and reversed an ordering.", "The readouts disagree, and no boundary replaces the
one that fell.").

**Suggested fix:** "Length-normalizing does not fix it either." — declarative, parallel with its
neighbors, and names the alternative it rejects.

## What is already working (keep it)

A revision pass under deadline strips exactly these things. They are why the paper reads as
trustworthy, and each is easy to delete by accident:

- **The scope restriction "on positive images"** in the abstract (L66), introduction (L89), §3 and
  the conclusion. The pattern reverses on negative images (`tab:factorial`: a *positive* context
  moves negative-image valence $+0.762$ against a negative context's $-0.017$). This qualifier is
  load-bearing in all four places, and in both language versions if the bilingual abstract is used.
- **The sufficiency/necessity discipline.** §4 closes: "these interventions locate states
  *sufficient* to restore each cue's effect. They do not establish necessity, and they do not
  explain why negative context moves behavior farther." Do not let a tightening pass turn this into
  a causal claim.
- **The refusal to resolve the mid-band disagreement.** "A paper reporting either number alone would
  state a confident and opposite localization result. We report both and draw no conclusion about
  mid-band ordering." This is the paper's thesis enacted rather than asserted.
- **The zeros framed as arithmetic, not measurement.** "under causal masking the patch copies a
  value onto itself and the zeros are arithmetic, not measured." A compression pass would render
  these as a finding about image tokens being inert, which would be false.
- **The negative result kept at full prominence.** The varied-set crossed interval
  $[-0.16,+0.91]$ includes zero and the text says so plainly, immediately after reporting the
  positive photo-clustered interval.
- **The disclosed post-hoc correction** and the withdrawn wrapper runs in §6.
- **The two-estimator note on `tab:pairs`.** It is the model for F2's fix; do not delete it while
  acting on F2.
- **"There is no neutral choice here, only a defensible one that must be stated."** The single best
  sentence in §5.

## Note outside the requested scope

The long (VLM4RWD) build's first contribution bullet, L111, states the four-to-five-times result
without the "on positive images" restriction that its own abstract, intro and conclusion all carry.
This is the same class as the previously closed F1 and is already logged as a deliberate
non-change in REVISION-LOG Round 10. Out of scope here; noted so it is not rediscovered.
