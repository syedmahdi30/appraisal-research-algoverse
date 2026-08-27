# Simulated review panel — VLM4RWD submission

_Run 2026-08-27 against `a033692` (8pp build). Target: VLM4RWD 2026, non-archival workshop._

**What this is.** Five reviewer lenses applied by one model. The seats are role-separated, not
independent: a blind spot in one seat is likely shared by the others, so treat agreement between
seats as weaker evidence than agreement between real reviewers. Every numeric claim below was
checked against the build or the source.

**Recommendation: Accept (weak).** The work is careful, the controls are real, and the paper
reports its own nulls more honestly than most workshop submissions. Everything below concerns
the calibration of claims, not their validity. Two findings (P1, P2) should be fixed before the
PI sees it; neither requires new experiments.

---

## Seat 1 — Venue fit (VLM4RWD)

Fit is good and has clearly been worked on. The abstract names the deployment setting, the intro
gives concrete pipelines, and §5 has a dedicated deployed-systems paragraph framing the asymmetry
as a faithfulness problem. That maps onto the workshop's grounding and evaluation themes.

The weakness is that the deployment claim is argued, not demonstrated. No pipeline is tested, no
downstream task is measured, and the monitoring proposal is explicitly untested as a mitigation.
The paper is honest about this, but a reviewer looking for "real-world" evidence will find one
paragraph of extrapolation from a 121-image lab task. That is normal for a workshop and I would
not reject on it, though it caps enthusiasm.

## Seat 2 — Methods and statistics

The bootstrap design is the strongest part of the paper: two interval types, with the crossed
version correctly identified as the one that licenses general claims about negative language.

**P1 (major).** The flagship number does not follow that standard. The within-item contrast
+1.148 is reported as "[+0.943, +1.344]; Wilcoxon p < 0.001 over 62 photograph averages", and
`reporting.py:331` confirms this is "a Wilcoxon signed-rank over images and a clustered bootstrap
over images" — sentences fixed. So the interval is photo-clustered, the weaker of the two
standards, and it is never labelled as such. Worse, it appears one sentence after the discussion
of which crossed intervals do and do not exclude zero, so a reader will naturally assume it is
crossed. Either label it, or report the crossed version alongside.

**P3 (major).** Qwen3-VL-8B, the headline model, clears **none** of the three crossed tests on the
varied set (§4.4). The behavioral claim therefore rests entirely on six matched sentence pairs, and
within those, only the mirror contrast clears zero under sentence resampling; the corrected override
gap does not ([−2.6, +45.4]). The paper says all of this. The issue is that the abstract and
conclusion do not carry the narrowness forward with equal weight.

**M4.** The correction supplying the headline 57%/35% was, by the paper's own admission, applied
after seeing the uncorrected numbers. The disclosure is commendable; the fact remains that the
primary figure comes from a post hoc analytic choice.

**M5.** n = 62 photograph averages for the headline, against 121 distinct images and 150
annotations, is never reconciled in the text.

## Seat 3 — Interpretability and mechanism

The patching design is sound and unusually well caveated: sufficiency not necessity, non-additivity
stated explicitly, and the layer band motivated by a separate layerwise scan.

**P2 (major).** The mechanism is Gemma-3-4B. The headline behavioral result is Qwen3-VL-8B. The
mechanism therefore cannot speak to the flagship phenomenon, and the paper never says so plainly.
§4.3 opens "downstream text positions are where the sentence's effect and the image's own valence
become jointly readable" without noting that this is a different model from the one carrying the
+1.148. One sentence fixes it.

**M6.** The mid-band readout disagreement (probe: text 87.4 vs image 74.9; behavior: image 96.0
vs text 74.1, non-overlapping in opposite directions) is reported with no conclusion drawn, which
is right. But it sits underneath a subsection summary asserting a "meeting point", which the
mid-band evidence cannot support.

**M7.** The probe's r² = 0.641 is a selection statistic with no held-out number anywhere, and the
13–17 patching band was chosen on the same data used to evaluate it. Both are disclosed; neither
has a sensitivity check.

**M8.** The zero-by-construction rows remain a table of results in which several entries could not
have come out otherwise. A size-matched non-affective control is the obvious missing condition.

## Seat 4 — Claims and measurement

**M9.** The paper states "Ratios on a bounded scale are descriptive; the difference and its interval
are primary." Yet "four to five times farther" leads the abstract, the introduction, contribution 1,
and the conclusion, while the difference it demotes to primary appears alongside it in only some of
those places. The paper leads with the measure it tells the reader not to lead with.

The §4.4 honesty about no architectural boundary surviving is a genuine strength and should not be
softened. Reporting three readouts and declining to pick the flattering one per model is exactly
right.

## Seat 5 — Devil's advocate

**P4 (major, and my strongest objection).** Every readout in this paper is built on a polarity
partition of **four positive labels against seven negative** — `POSITIVE_LABELS` is (joy, pride,
relief, trust); `NEGATIVE_LABELS` is (anger, boredom, disgust, fear, guilt, sadness, shame). The
categorical override gives "negative" nearly twice as many ways to win an argmax; bounded valence
sums four probabilities against seven; the log-odds margin does the same. A model with no valence
asymmetry whatsoever would still be easier to push toward the negative category than the positive
one under this scoring.

The design mitigates this. Effects are measured against each image's own neutral baseline, and the
within-item contrast holds photograph and event fixed. A constant label-space prior shifts the
baseline and partially cancels. But it need not cancel exactly, precisely because of the head-room
asymmetry the paper itself discusses elsewhere.

**The paper never mentions the label counts.** For a paper whose contribution is that measurement
choices manufacture and reverse conclusions, an unexamined 7-versus-4 asymmetry sitting under all
three readouts is the most exposed surface it has. A hostile reviewer will find it in the appendix
label list. This needs a paragraph, ideally with a balanced-subset re-run, but a reasoned argument
would do for a workshop.

---

## Ranked findings

| | Finding | Cost to fix |
|---|---|---|
| **P1** | Flagship +1.148 interval is photo-clustered and unlabeled, presented next to crossed-interval discussion | one clause |
| **P2** | Mechanism is Gemma, headline behavior is Qwen; never stated | one sentence |
| **P3** | Headline model clears no crossed test on the varied set; abstract does not carry the narrowness | rewording |
| **P4** | Label space is 4 positive vs 7 negative, unexamined, underneath all three readouts | a paragraph; ideally a balanced re-run |
| M4 | Neutral correction applied post hoc supplies the headline figure | already disclosed |
| M5 | n = 62 unexplained against 121 images | one clause |
| M6 | Mid-band disagreement undercuts the "meeting point" summary | rewording |
| M7 | Probe r² is a selection statistic; patching band chosen on evaluation data | needs a run |
| M8 | Zero-by-construction rows lack a control that could have come out otherwise | needs a run |
| M9 | Ratio leads the paper despite being demoted to descriptive | rewording |

## What would move this to a clear accept

P1 through P4. Three are wording; P4 is the only one that might want a re-run, and even a
reasoned paragraph would close most of the exposure. None of this requires new GPU time except
M7 and M8, which are already known open items.
