# Spec — judge / evaluator-choice robustness (TAE P1)

**22 Aug 2026.** The credibility-maker for the TAE reframe (see [tae-experiment-plan.md](tae-experiment-plan.md) P1).
The paper's constructive half rests on C1/C2/C3, but all three are computed with **one** scoring
function. This experiment shows the override conclusion is **evaluator-independent** — not an artefact
of the specific 13-label vocabulary or its pos/neg partition. It is the evaluator-robustness analogue
of the A2 36-redraw: same logic (vary an analyst degree of freedom, hold everything else fixed), a
different degree of freedom (the judge instead of the stimulus).

## Headline finding it's chasing
> The override-gap conclusion — positive on Gemma (+64%), positive on Qwen (+39%), null/reversed on
> LLaVA (−13%) — is **sign-stable across every reasonable choice of scoring evaluator.** The graded
> statistics may wobble; the categorical override conclusion does not.

If true, this is the missing empirical leg under **C3** (the calibration-free override rate is the
trustworthy summary). If it *fails* on some evaluator, that is also a publishable measurement result
and must be reported honestly.

## Why it's nearly free — the data already supports it
`stage_f_conflict.py:151` (Gemma) and `stage_f_qwen.py:162` (Qwen/LLaVA) both write
`{f"lp_{w}": lp[w] for w in EMOTION_LABELS}` to the base parquet: **per-label log-probabilities for all
13 labels, every row, every model.** `lp_w = log_softmax(logits[label_ids])_w`, i.e. log-prob over the
13-label set.

Consequence: for **any sub-partition of the 13 labels**, valence and argmax are exactly recoverable
from stored data. Softmax over a subset S equals the renormalised superset softmax restricted to S:
`P_S(w) = exp(lp_w) / Σ_{v∈S} exp(lp_v)`. So every evaluator below that uses only the existing 13
labels is a **CPU reanalysis of the stored parquets — no GPU, no forward passes, teammate-safe.** Only
an evaluator introducing *new* label words (outside the 13) needs a re-run.

## The current (anchor) evaluator — E0
- Labels: the 13 crowd-enVENT labels (`labels.py::EMOTION_LABELS`).
- `POS = {joy, pride, relief, trust}`, `NEG = {anger, boredom, disgust, fear, guilt, sadness, shame}`,
  sink `= {surprise, neutral}` (in the softmax denominator but in neither pos nor neg).
- Valence = ΣP(POS) − ΣP(NEG) over the 13-label softmax. Override = the argmax label's category
  follows text vs image (`analyze_stage_f._flip_override`).
- **First step: reproduce the published gaps from E0** (+64 / +39 / −13) as the anchor, exactly as the
  redraw method re-derives the full-bank values to 4 dp.

## Evaluator perturbations (Tier A — free, all 3 models, from stored `lp_*`)
Each is a defensible alternative "judge." All recomputed from stored logprobs.

| ID | Judge | Change from E0 | Tests |
|---|---|---|---|
| **J1** | Sparest antonym | `POS={joy}`, `NEG={sadness}` only | Does the effect need the full vocabulary, or survive the cleanest 1-vs-1 contrast? |
| **J2** | Drop weak positives | `POS={joy, pride}` (drop relief, trust) | Sensitivity to including arguably-weak positives. |
| **J3** | Re-file boredom | move `boredom` from NEG → sink | Boredom is low-arousal, not clearly negative; is the gap robust to this call? |
| **J4** | Denominator swap | renormalise over POS∪NEG only (drop surprise/neutral from the denominator) | Sensitivity to the sink-mass degree of freedom. |
| **J5** | Rank-only override | argmax category with no probability magnitudes | Pure categorical judge — the calibration-free extreme. |

**Aggregation (reuse existing machinery):** per evaluator, compute per-image override for both conflict
directions, gap = (neg>pos_img) − (pos>neg_img), bootstrap CI clustered over images — identical to
`analyze_stage_f._flip_override`. Also report the mirror contrast + CI per evaluator, so the "graded
wobbles, categorical holds" story is visible in one table.

## Deliverable — one table (the C3 backing)
Rows = evaluators J0(=E0)…J5; columns = Gemma / Qwen / LLaVA; cells = override gap [95% CI] + a
one-word verdict (positive / null). Expectation: every Gemma cell positive & CI clears 0, every LLaVA
cell null/reversed, Qwen positive. A companion column pair reports the mirror contrast to show it is
the *graded* statistic that moves, not the conclusion.

## Tier B — genuinely-new vocabulary (optional; needs a cheap re-run)
If a reviewer wants an evaluator sharing **no words** with the original 13 (e.g. positive/negative from
a sentiment lexicon, or a VAD-style axis), that needs new label logprobs → re-run the base pass storing
the new `lp_*`. Forward-pass only; default Colab GPU for Gemma-4B, Syed's A100 for all three. Lower
priority than Tier A.

## Tier C — external LLM-judge (P1b; strongest, API-gated)
The model **generates** an emotion word; an external LLM classifies that word's valence. This is the
most decisive evaluator-independence evidence — the judge never sees the logits — and it is the one
variant that is **not** re-scorable from stored data (needs generation + API). Gated by the API-budget
decision. Design: same stimuli, `generate` one emotion word per trial, judge → {positive, negative,
other}, recompute override. Report agreement with E0 and the override gap under the LLM-judge.

## How it enters the paper — two framings (author's call)
1. **As empirical backing for C3.** A subsection under §5 (the remedy): "C3 is not merely convenient —
   it is the summary that survives evaluator choice," with the Tier-A table. Lowest-friction.
2. **As a fifth link — A5.** Extend the assumption chain (Table 1): A1 statistic, A2 stimulus, A3
   intervention, A4 provenance, **A5 evaluator** — "the readout evaluator identifies the construct, not
   the analyst's label partition," diagnostic = re-score under alternative partitions and report the
   spread. This is the more ambitious framing and makes the experiment a first-class demonstration
   rather than a control. Recommended if the Tier-A table comes out clean.

## Implementation note
A single analysis script `src/experiments/analyze_judge_robustness.py`:
- reads the three base parquets (Gemma full-bank `conflict_pilot.parquet`, `conflict_qwen*.parquet`,
  `conflict_llava*.parquet`);
- defines the evaluators J0…J5 as (POS_set, NEG_set, denominator) tuples;
- for each, recomputes per-row valence/argmax from `lp_*`, then reuses `_flip_override` for the gap +
  CI and the existing mirror-contrast estimator;
- emits `results/stage_f/judge_robustness.json` + the LaTeX table.
No new forward passes for Tier A. Verify E0 reproduces the published gaps before trusting J1…J5.

## Open decisions
- **API budget for Tier C** (external LLM-judge)? If no, Tier A + B carry the claim; Tier A alone is
  already the evaluator-robustness table.
- **Framing 1 vs A5** — decide after seeing the Tier-A table.
