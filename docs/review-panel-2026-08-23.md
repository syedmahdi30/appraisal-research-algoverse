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

---

## Correction to finding 2, recomputed 2026-08-23 (later session)

I recomputed the baseline correction from the conflict parquets. **Two things in finding 2 above are
wrong.**

**(a) The wrong baseline was used.** The panel's base rates come from `condition == "none"` — no
context sentence at all. But the published mirror contrast baselines against `condition == "neutral"`
(the two neutral sentences `z0`/`z1`, "This photo was taken on a weekday / indoors"), precisely
because Appendix F shows that *any* sentence shifts the readout. Correcting the override rate against
the no-sentence condition would introduce the very confound §3 avoids. Everything below uses the
neutral-sentence baseline.

**(b) Gemma's reported gap was wrong.** The panel listed +9.6%. The published value is **+12.2%**
(`conflict_gemma-3-4b-it_metrics.json`, `dominance_gap` 0.1216). My raw recomputation reproduces every
published gap and crossed interval to within bootstrap noise, so the corrected column can be trusted.

Neutral-sentence argmax base rates (per image):

| model | pos image reads NEG | neg image reads POS |
|---|---|---|
| Qwen3-VL | 20.0% | 1.3% |
| Gemma-3-4B | 2.7% | 7.3% |
| LLaVA-NeXT | 2.7% | 4.0% |
| LLaVA-1.5 | 2.7% | 6.0% |

Override gap, raw vs. baseline-corrected (2000 resamples, seed 0 — the published settings):

| model | raw gap | photo-clustered | crossed | **corrected gap** | photo-clustered | crossed |
|---|---|---|---|---|---|---|
| Qwen3-VL | +39.4 | [+29,+49] | [+10,+66] | **+21.7** | [+10.9,+31.9] | [−8.6,+48.0] |
| Gemma-3-4B | +12.2 | [+0,+24] | [−12,+37] | **+18.1** | [+7.5,+28.7] | [−5.4,+41.2] |
| LLaVA-NeXT | +18.7 | [+7,+30] | [−2,+41] | **+20.5** | [+10.9,+30.8] | [+1.0,+41.0] |
| LLaVA-1.5 | −12.6 | [−22,−2] | [−43,+18] | **−10.0** | [−18.7,−1.3] | [−40.4,+20.9] |
| Qwen matched set | +40.8 | [+32,+49] | [+17,+63] | **+23.1** | [+12.6,+33.1] | [−2.6,+45.4] |
| Qwen 128 img tok | +41.8 | | | **+25.8** | [+15.5,+36.2] | [−5.7,+52.9] |
| Qwen 262 img tok (a) | +37.4 | | | **+20.6** | [+9.4,+30.8] | [−9.8,+46.7] |
| Qwen 262 img tok (b) | +38.6 | | | **+20.9** | [+9.9,+31.0] | [−9.6,+47.9] |

Corrected cell rates (neg-ctx override / pos-ctx override): Qwen 57.0/35.3, Gemma 40.3/22.2,
LLaVA-NeXT 34.7/14.2, LLaVA-1.5 24.5/34.4, Qwen matched 74.2/51.1.

### What this changes

1. **The models converge.** +21.7 / +18.1 / +20.5 / −10.0. Three designs land within 3.6 points.
   Qwen stops being dramatically the largest; Gemma's photo-clustered interval, which straddled zero
   raw ([+0,+24]), now clears it ([+7.5,+28.7]). Correction *rescues* Gemma rather than deflating it —
   it is image-led with a 7.3% neutral floor in the positive direction, which suppressed its raw gap.
2. **The override gap stops clearing zero under sentence resampling.** Raw, Qwen's crossed interval
   was [+10,+66] and the matched set's [+17,+63]; corrected they are [−8.6,+48.0] and [−2.6,+45.4].
   Only LLaVA-NeXT's corrected crossed interval clears zero, marginally ([+1.0,+41.0]). Contribution
   2's claim that on the matched set *both* the override gap and the mirror contrast clear zero under
   resampling is **false once the override gap is baselined the same way the mirror contrast is.**
   The mirror contrast still clears zero there ([+0.11,+0.83]); the override gap does not.
3. **The "27 points in the opposite direction" budget argument dies.** Budget-matched Gemma (256 img
   tokens) and Qwen (262) are +12.2 vs +39.4 raw, but +18.1 vs +21.7 corrected — a 3.6-point gap in
   the *same* direction. The token-budget hypothesis still fails, but for the opposite reason: across
   a 17-fold range in image tokens (128 → 2,147) and three model families, every corrected gap sits
   between +18 and +26. Budget predicts nothing because nothing varies.

### Why subtractive and not headroom-normalized

The mirror contrast is defined in §3 as "computed per image against that image's own neutral baseline
and then averaged" — subtractive, per image. The subtractive correction is the exact categorical
analogue. Headroom division is the correction Appendix H already calls "cruder" and unstable at small
denominators, and it is not what the graded readouts use. For the record, headroom-normalized gaps
against the no-sentence baseline were +35.2 / +18.9 / +20.5 / −14.0.
