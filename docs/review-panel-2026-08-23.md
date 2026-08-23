# Simulated review panel — findings and verification (2026-08-23)

Ran the `academic-paper-reviewer` panel against `paper/neurips_2026.tex`. Venue binding declared
`criteria_binding_unavailable` (no venue chosen), so no seat made a venue-fit claim.

**Panel completed 3 of 5 seats.** Journal-Fit, Domain, and Perspective returned full reports.
Methodology (R1) and Devil's Advocate (DA) died on API/stream errors and were never re-run. The DA
gap matters procedurally: the skill's iron rule requires every DA `[CRITICAL]` to be visibly
adjudicated, and we have no DA report to adjudicate. **Treat this panel as incomplete.**

---

## Findings I verified computationally (these are established, not opinions)

### 1. The 0% image-token recovery in §6.2 is an arithmetic identity, not a measurement

Domain seat raised it; I confirmed it directly against the parquets:

```
pair1 (0,2)   image / bos / prefix_delim : max|patched − unpatched| = 0.000e+00, 60/60 rows EXACT
pair2 (4,0)   image / bos / prefix_delim : max ≈ 5e-02 (numerical), 0/60 exact
              question / suffix_delim    : max ≈ 1.1e+00  ← real intervention
```

**Why:** the prompt is `<bos><start_of_turn>user\n<start_of_image>{256 img tokens}Context: {ctx} …`.
Image, BOS and prefix-delimiter positions all *precede* the context, the image is held constant, and
the LM is causally masked. Their states are therefore identical across the positive- and
negative-context runs. Patching them copies a value onto itself. Three of Table 4's six rows are
forced to zero before any model runs.

The paper already concedes this *partly* in §6.2's second limit ("partly follows from prompt order …
we treat this as a consistency check") — but the abstract, contribution 5, the Discussion and the
Conclusion all still present it as a substantive causal finding. **"Partly" should be "entirely",
and the claim must be demoted everywhere else.**

**Bounded:** cross-image patching (§6.3) is unaffected — it varies the image and holds context fixed,
so image-token states genuinely differ. Verified: 0/60 no-op rows, max|Δ| = 1.6. §6.3 stands.

### 2. The override rate is not baselined, and baselining nearly halves the headline

Perspective seat flagged that the override rate has no baseline subtraction while the graded readouts
do — an internal inconsistency with §3's own stated design rule. I measured the consequence.

Neutral-context argmax base rate (is the label space biased negative?):

| model | pos images → pos | neg images → neg |
|---|---|---|
| Qwen3-VL | **80.0%** | 98.7% |
| Gemma-3-4B | 97.3% | 92.7% |
| LLaVA-NeXT | 97.3% | 96.0% |
| LLaVA-1.5 | 97.3% | 91.3% |

There is **no** global negative bias — so the 4-positive/7-negative label split is *not* driving the
effect, which defuses the seat's stated version of the concern. But Qwen's baselines are
**asymmetric**: a positive image already reads negative 20% of the time, while a negative image reads
positive only 1.3% of the time. That is a 20-point head start for the "negative overrides positive"
cell. Correcting each cell against its own neutral baseline:

| model | reported gap | baseline-corrected |
|---|---|---|
| **Qwen3-VL** | **+39.8%** | **+21.1%** |
| Gemma-3-4B | +9.6% | +14.2% |
| LLaVA-NeXT | +16.2% | +17.6% |
| LLaVA-1.5 | −17.6% | −14.2% |

**Qwen's headline nearly halves, and the four models converge** to roughly +21 / +14 / +18 / −14.
Qwen is no longer dramatically the largest effect. This is the single most consequential finding of
the review and it is not yet reflected anywhere in the paper.

### 3. Label partition confirmed

`POSITIVE = (joy, pride, relief, trust)` — 4. `NEGATIVE = (anger, boredom, disgust, fear, guilt,
sadness, shame)` — 7. `surprise` and `neutral` excluded. Real, but see above: the base rates show it
is not producing the effect.

---

## Consensus findings across seats (not independently verified)

- **Text-only baseline missing** — flagged `[CRITICAL]` by Journal-Fit *and* Perspective independently.
  Appendix B already reports the matched set is text-only imbalanced at ratio **1.25** ("a couple of
  positive members read weak in isolation and are flagged for revision"), which is *worse* than the
  varied set's 1.06. The paper never propagates this into the interpretation of its strongest result.
  Worse than the seats could see: that 1.25 figure comes from the **retracted Gemma bridge run**
  (`text_only_minimal.parquet`, Aug 18), and **no text-only matched-set measurement exists for Qwen at
  all**. `stage_f_token_budget --text-only` does not yet accept `--bank`.
- **"Three of four models" uses a disjunctive max-over-readouts criterion** the paper elsewhere
  disclaims — Journal-Fit + Perspective. Qwen's unbounded margin is −1.81, the *opposite sign*, and
  the abstract does not say so.
- **LLaVA null confounded with label tokenization** — all three seats. LLaVA is the only model whose
  labels are multi-token; the scoring rule (first-subtoken vs summed log-prob) is never stated.
- **Steering "no attenuation" is confounded with sentence presence** — Domain. The no-conflict slope
  (+0.336) is measured with **no context sentence**; the conflict slope (+0.335) has one. §3's own
  design rule mandates neutral-context baselining precisely because Appendix F shows any sentence
  shifts the readout. The paper breaks its own control in its final causal experiment.
- **Only one patching direction run** — Domain. All "carried by" claims are sufficiency, not necessity.
- **Missing citations, concrete** — Domain named: `luo2025` (in the .bib, **uncited**, and it is the
  closest prior art to the cross-modal transfer claim), Schwettmann et al. 2023 multimodal neurons,
  Heimersheim & Nanda 2024, Vig et al. 2020, Belinkov 2022, Hewitt & Liang 2019, Park et al. 2024.
  Also `unraveling2024` and `conflictchallenges2025` are in the .bib and uncited.
- **Anonymity risk** — Perspective checked `neurips_2026.sty` lines 104–109: the `ack` environment has
  no anonymity guard and always prints, and its `\todo` text names the Algoverse program. Verify on a
  fresh build.
- **Checklist mismatches** — Perspective: item 9 claims responsible-use content in §6 that does not
  exist; item 7 claims wall-clock numbers that are still a `\todo`; item 4 lists the withdrawn 12B
  model and omits both LLaVAs.
- **Bib entries unverified** — the methodology references added this session were written from memory
  and carry an in-file warning. Still unverified.

## Where a seat was wrong

The Perspective seat's `[CRITICAL]` on the label partition asserted the 4/7 split biases the override
rate. The neutral base rates refute that directly (97% correct in both directions for three models).
The *asymmetric baseline* problem it points at is real and serious; the *label-count* mechanism it
proposed is not.
