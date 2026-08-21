# TAE experiment plan — valence-asymmetry paper → Trust-AI-Eval

**Rev 2 — 22 Aug 2026.** Updated after the `tae_main.tex` redraft + REVISION-LOG.md + NUMBER-LEDGER.md.
Target: **TAE (Trust-AI-Eval)**, "Can We Trust AI Evaluation?", NeurIPS 2026. Deadline **Aug 29 (AoE)**,
8 pp excl. refs + appendices, non-archival, no reviewer duty. Fallback: **VLM4RWD** (Aug 30, same trim,
no exclusivity). InterpScience dropped.

## Status since Rev 1
The redraft executed the front half of the plan. The thesis is inverted (measurement validity is the
contribution; VLM valence asymmetry is the running case study), title is
*"When the Statistic Is Not the Construct."* Three failure demonstrations (A1/A2/A3) are the spine,
plus a self-referential provenance failure (A4). What this means for the experiment plan:

| Rev 1 item | State now | Note |
|---|---|---|
| #1 metric-flips-conclusion | **DONE** as **A1** | Went further: quantified head-room geometry 6.88:1 / 4.39:1 / 0.89:1 across models. |
| #8 ordering-artifact | **DONE** as **A3** | Done as **reanalysis of stored data — no A100 needed**. "Entailed by design?" column. |
| #4 sampling / #5 crossed bootstrap | **superseded** by **A2** (36-redraw) | A2 is stronger: 6/36 graded sign-flips, override gap positive 36/36. But A2 is a within-bank sensitivity analysis, **not** a crossed bootstrap (still open, below). |
| #6 token-order flip | **deprioritized** | A3 got the measurement point from existing data; the flip is no longer needed for the body. Park. |
| framing (retire "interchangeable"; provenance table) | **DONE** in redraft's S2/S7 | — |

**The redraft did not add:** any evaluator-independence test (#3), any unbounded readout (#2), a
crossed two-way bootstrap, seeds (#7), or a fix for the overwritten runs (A4 is disclosed, not fixed).
Those are the open surface.

## The conceptual gap the redraft exposes
The constructive half rests on three checks — **C1** head-room normalisation, **C2** mirror null,
**C3** override rate. Empirical support: C1/C2 ← A1, C3 ← A2's sign-stability. **But all three are
computed with a single scoring function.** The paper never tests whether its conclusions survive a
*different evaluator*. That is (i) a named TAE topic we otherwise don't touch, (ii) the exact next
question a reviewer asks a measurement-validity paper, and (iii) the missing leg under C3, the
load-bearing remedy. This makes judge-robustness the top open item.

**Compute:** Syed has personal A100 (Colab). Teammates default-GPU only → hand them forward-pass /
reanalysis / API work, not TransformerBridge mechanistic runs.

---

## Open experiments, re-ranked

### P1 — Judge / evaluator-choice robustness (#3). The credibility-maker. API / no A100.
- **What:** Re-score the conflict battery with evaluators independent of the closed-vocab
  P(pos)−P(neg) readout. Two variants:
  - **(a) second/disjoint label set** — a different emotion vocabulary partitioned into pos/neg,
    scored the same way. **Teammate-safe, no API budget needed.** Build this first.
  - **(b) external LLM-judge** — a judge model classifies the argmax emotion word's valence.
    Needs API access (gated by budget decision).
- **Claim it defends:** C3 (override rate) — show the override conclusion reproduces on Gemma/Qwen
  and stays null on LLaVA under evaluator (a), ideally (b) too.
- **TAE topic:** "Stress tests and judge reliability: model-judge reliability."
- **Why P1:** without it, "measurement validity" is argued but the paper's *own* readout is
  never shown evaluator-independent. Closes the one robustness axis C3 claims but doesn't test.
- **Owner:** Syed / teammate-safe for (a).

### P2 — Unbounded logit-margin readout (#2). Direct A1 companion. Default GPU.
- **What:** Re-score with an unbounded logit-margin readout alongside the bounded score. Show the
  surviving Gemma effect (and the LLaVA null) reproduces unbounded.
- **Claim it defends:** the whole A1 thesis — rebuts "your surviving result is *also* on a bounded
  score." Gives a bounded-normalised **and** an unbounded readout that agree and both reject LLaVA.
- **Converts:** the paper's own note that an unbounded replication "has not been run."
- **Owner:** Syed; feasible on default Colab GPU (Gemma-4B; +Qwen/LLaVA ideal).

### P3 — Re-run the two overwritten experiments with per-run output paths. A100. Thesis-consistency.
- **What:** Re-run same-image patching (restore the Pair-1 championship/funeral column) and
  cross-image patching (neutral-context band table 0-12 / 13-17 / 18-28), writing to **per-run
  output paths** so each survives. Re-derive the numbers.
- **Why:** A paper whose thesis is "claims must be re-derivable from stored artefacts" is stronger
  if it **fixes** its own broken provenance than if it only confesses it (A4). Keep the A4
  disclosure; fix the underlying loss. **Fix + disclose > disclose alone.**
- **Scope note:** the pilot rerun is **optional** — A2 already replaced that anecdote with the
  36-redraw, so the pilot numbers are no longer load-bearing. The patching reruns are the ones worth
  doing.
- **Owner:** Syed (A100 + TransformerBridge). Root cause: fixed output path invoked N times; fix is
  per-run paths + a re-derivation pass before submission.

### P4 — Crossed two-way bootstrap over (image × context). Reanalysis, free.
- **What:** Recompute headline CIs treating (image, context-sentence) as a crossed unit, not images
  alone. Upgrades A2 from sensitivity analysis to proper uncertainty.
- **Converts:** the paper's own Limitations line ("uncertainty propagated over images but not
  simultaneously over context sentences").
- **Owner:** any teammate (CPU reanalysis).

### P5 (ceiling-raiser) — One failure mode in an external published protocol. Analysis, careful.
- **What:** Show that A1's head-room confound (a within-condition bounded ratio computed against an
  off-centre baseline) appears in a *published* conflict protocol — e.g. a bounded within-condition
  ratio in Mixed Signals or SIGNPOST. Lifts the contribution from "we audited ourselves" to "here is
  a check the field needs," which is the strongest answer to the "why audit your own protocol?"
  objection.
- **Risk:** must not overclaim about others' work — frame as "this metric is *vulnerable to* the A1
  confound," verified against their reported numbers, not "their result is wrong."
- **Owner:** Syed (judgement call before drafting any such claim).

### P6 (stretch) — 3-seed repeat (#7). A100.
- Refit probe + Δμ at 3 seeds, re-score, re-run arbitration. The behavioural override is
  deterministic (say so); seed variance lives in the probe/Δμ. Cheap on A100 (cached activations).
  Lower priority: the robustness spine (36-redraw + 6 prompts + 2 scales) is already substantial.

---

## Recommended set for the 8 days
**P1(a) + P2 + P3 + P4**, with **P1(b)** if API budget clears and **P5** as the ceiling-raiser if a
defensible external example is found. P1(a)/P4 need no A100 (teammate-safe); P2 runs on default GPU;
P3/P6 are Syed's A100. If time compresses: **P1(a) + P2 + P4** are all doable without heavy compute
and cover judge, metric, and variance robustness — the three axes a TAE reviewer probes.

## Two judgment calls (from the redraft's "Open for author")
1. **A4 disclosure:** keep it **and** fix via P3. Costly honesty is the paper's most credible move at
   this venue; fixing the provenance makes the paper walk its own thesis.
2. **Self-audit generalizability:** P5 is the answer if a clean external example exists; otherwise the
   redraft's existing defense (corrected-next-to-erroneous, self-audit lets us report both) stands.

## Blocker (not an experiment)
Email TAE chairs (aiteval2026@gmail.com) — CFP is silent on concurrent NeurIPS-main / other-workshop
submission. Get dual-sub position in writing before building the VLM4RWD fallback on it.

## Mapping to TAE's call (for the meeting)
| TAE topic | Item |
|---|---|
| Measurement & causal validity (construct) | A1, A3 (done); P2 unbounded readout |
| Uncertainty & robustness (metrics, splits, variance) | A2 (done); P2, P4 crossed bootstrap, P6 seeds |
| Stress tests & judge reliability | **P1 judge/evaluator swap** |
| Auditing / provenance | A4 (done); P3 fix + P5 external audit |
