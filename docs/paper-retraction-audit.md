# Paper retraction audit — every claim in `neurips_2026.tex` vs. the bridge bug

_Built 2026-08-22 from `docs/bridge-bug-2026-08-22.md` plus a source-level audit of which
runner booted which stack. Framing-independent: these statuses hold whichever venue we pick._

## Stack audit (source-level, not memory)

Every Gemma Stage F runner boots `src.bridge.boot.boot_gemma` → `TransformerBridge`:

| runner | stack |
|---|---|
| `stage_f_conflict`, `stage_f_prompts`, `stage_f_scaling` | **bridge** |
| `stage_f_patching`, `stage_f_layerwise`, `stage_f_cross_patching`, `stage_f_attribution` | **bridge** |
| `stage_f_text_only` | **bridge** (but text path verified exact, |Δ| = 0.0011) |
| `stage_f_llava`, `stage_f_qwen`, `stage_f_token_budget` | raw HF |

So: **no Gemma image-conditioned number in the paper avoided the bridge.** Qwen and both LLaVAs
never touched it.

## Triage by paper location

### Dead — bridge + absolute/categorical readout

| § | claim | action |
|---|---|---|
| Abstract | Gemma $84\%$ / $21\%$ override | retract; re-state at $44\%$ / $31\%$ (gap $+12\%$, CI $[-0,+24]$) |
| Contribution 1 | gap $+64\%$ CI $[+55,+72]$; cell means $-0.598$ / $+0.333$; mirror contrast $+0.265$ | retract all |
| §5.2 + Tab. factorial | every Gemma cell mean and the mirror contrast | retract table |
| §5.3 + Tab. minimal | minimal-pair gap $+65\%$, mirror contrast | retract |
| §7.1 Tab. models | entire **Gemma column** | retract column |
| §7.2 prompt robustness | $+46\%$ to $+74\%$, mean $+58\%$, 13,500 forwards | retract (all bridge override rates) |
| §7.3 scale | $84\%\to93\%$, $19\%\to42\%$, gap $+51\%$ | retract (**both** 4B and 12B are bridge) |
| App. prompt-variant, App. scale tables | as above | retract |
| Judge-robustness sweep (Gemma $+48$–$65\%$) | derived from bridge parquets | retract Gemma rows; Qwen/LLaVA rows stand |

### Likely intact — bridge, but differential / probe-scored. Re-confirm, don't assume.

| § | claim | why it should survive | confirmed? |
|---|---|---|---|
| §6.1 layerwise | onset ~L13, $7\times$ growth, peak ~L28 | probe-projected *difference* between two runs | **no** |
| §6.1 attention | $88\%$ / $3.5\%$ attention, $6\%$ ablation effect | attention weights are representational (cosine 0.980 at the probe site) | **no** |
| §6.2 same-image patching | turn boundary $65\%$ / $57\%$ | ratio of differences, probe-scored at L18 | **no** — this is Next Step 2 |
| §7.4 steering under conflict | slope $+0.215$, $65\%$ of no-conflict | a slope; Stage D slopes reproduced to 2–3% | **no** (denominator now $+0.3360$) |

**At-risk, and the session doc does not flag it:** §6.3 cross-image patching ($80\%$ / $9\%$ / $68\%$)
states *"Behavioral valence is the main readout"* — behavioural + bridge is exactly the failing
combination. The L18 probe is only a secondary check there. Re-score this on the probe, or on raw HF,
before relying on the image-tokens-to-text-states story.

### Intact

- Stage A (text path exact), appendix.
- Stage C read-out transfer — re-run raw HF: $\rho$ $+0.510$, AUC $0.912$, n=7,280.
- Stage D cross-modal causal steering — re-run raw HF: $+0.3360$ vs published $+0.3293$.
- Qwen: gap $+39\%$, mirror $+0.394$, patching (image tokens $0\%$, all text $65\%$).
- LLaVA-1.5, LLaVA-NeXT: raw HF throughout.
- §5.x text-only control (no-image symmetry) — bridge, but the text path is verified exact.

### Falsified independently of the bug

- **Contribution 3 / §7.1 closing para / Discussion:** "the asymmetry is architecture-dependent",
  "absent on the linear-projector design". LLaVA-NeXT is the same projector family and *shows* the
  asymmetry ($+19\%$, CI $[+8,+30]$). Delete the boundary claim.
- **"~$1.8\times$ larger" (Contribution 1, §5.2):** an artefact of the bounded scale; sign flips to
  $0.76$ on an unbounded log-odds readout. Delete.
- **LLaVA-1.5 "slightly reversed" ($-13\%$):** crossed (image × sentence) bootstrap gives
  $[-50\%,+13\%]$. Re-word as a **null**, not a reversal.
- **Gemma full-bank mirror contrast:** crossed CI $[-0.013,+0.527]$ stops clearing zero. (Moot —
  the underlying run is retracted anyway.)

## The consequence nobody has written down yet

With Gemma at $+12\%$ (CI touching zero), the cross-model picture is **not** the doc's
"3 of 4 with graded strength $+65/+39/+19/-13$" — that line still quotes the retracted Gemma number.
The actual standing is:

| model | gap | stack | verdict |
|---|---|---|---|
| \qwen{} | $+39\%$ $[+30,+49]$ | raw HF | clears zero |
| LLaVA-NeXT | $+19\%$ $[+8,+30]$ | raw HF | clears zero |
| \gemma{} | $+12\%$ $[-0,+24]$ | raw HF | **does not clear zero** |
| LLaVA-1.5 | $-13\%$, crossed $[-50,+13]$ | raw HF | null |

So it is **2 of 4 clearing zero**, with the paper's primary model demoted to a marginal
non-significant result. Two structural implications:

1. **Qwen must become the primary model** — it is the only strong, untouched effect.
2. **The entire §6 mechanism section is Gemma**, i.e. a mechanism for an effect that is marginal in
   that model. The Qwen patching replication (image tokens $0\%$, text $65\%$) becomes the
   load-bearing mechanism result and currently occupies three sentences.

Two other lines in `bridge-bug-2026-08-22.md` §5 still quote $+65\%$ for Gemma (the "matched-budget
pair" argument). The token-budget conclusion is unaffected — $+12\%$ at 256 tokens vs $+39\%$ at 262
tokens kills the hypothesis just as cleanly — but the stated evidence needs updating.

## Also unresolved

- Title and framing are Gemma-first (`Negative Words Defeat Positive Images`); both need to change.
- `results/` is git-ignored and this session's raw-HF outputs live **only on the Colab runtime**.

---

# Re-score results — §6.2 same-image patching (raw HF, 2026-08-22)

Tap gate passed first: mean probe gap (positive-context − negative-context) **+1.830** against a 0.05
threshold, segmentation OK 8/8. `post_attention_layernorm` reads the signal the bridge-fitted probe
was trained on.

Three runs, 60 positive images each, resid_post patched over layers 13–17, probe read-out at L18.

| group | Pair 1 (0,2) published | Pair 1 raw HF | Pair 2 (4,0) published | Pair 2 raw HF | (1,0) raw HF |
|---|---|---|---|---|---|
| image | −1% | **0%** | +1% | **−0%** | **−0%** |
| BOS | 0% | **0%** | −1% | **0%** | **0%** |
| prefix delimiters | 0% | **0%** | −1% | **0%** | **0%** |
| question | 22% | **47%** | 32% | **54%** | **46%** |
| assistant-turn boundary | **65%** | **46%** | **57%** | **40%** | **41%** |
| all aligned text | 85% | **93%** | 87% | **88%** | **82%** |

_Pair 1's published column is doc-only provenance (its run was overwritten). Pair 2 is
artifact-backed against `results/stage_f/patching_metrics.json`. (1,0) is an unpublished third pair,
run by mistake, kept because it replicates the pattern._

## Survives

1. **Image tokens are causally inert — 0%, −0%, −0%.** Three runs, two published pairs. This is the
   load-bearing mechanism claim and it reproduces exactly.
2. **BOS is inert (0%), and so are the user-turn prefix delimiters.** The "it is the turn scaffold,
   *not* BOS" split holds.
3. **The text stream carries essentially all of it: 82–93%**, against a published 85–87%.
4. **Additivity holds** — the sink parts sum to the structure total in every run.

## Does not survive: the turn-boundary *concentration*

The published result had the assistant-turn boundary dominating the question by 1.8–3× (65 vs 22,
57 vs 32). On raw HF the ordering flattens, and in Pair 2 it **reverses**:

| | question | turn boundary |
|---|---|---|
| Pair 1 shift | 22% → 47% (**+25**) | 65% → 46% (**−19**) |
| Pair 2 shift | 32% → 54% (**+22**) | 57% → 40% (**−17**) |

A systematic, reproducible re-attribution *within* the text stream: the bridge over-credited the turn
boundary and under-credited the question by a near-identical margin in both pairs, with the total
conserved. This is not noise and not the pair mixup — Pair 1 compares like with like.

### What has to change in the paper

- Table 3's bolded **65% / 57%** becomes **46% / 40%**; the question row becomes **47% / 54%**.
- Delete "especially the tokens immediately before the model's answer" from the §6.2 caption and the
  sentence after it. The carrier is the text stream; the *concentration* at the turn boundary was a
  bridge artefact.
- The attention-sink framing in Related Work (`xiao2024`, `activedormant2024`, `zhang2026anydepth`)
  loses its main support and should be demoted to a possibility, not an interpretation.
- This is, if anything, *more* consistent with §6.1's own attribution result — the last token sends
  88% of its attention to "prompt-template **and question** tokens".

### What it does not change

The Gemma-vs-Qwen mechanism contrast in §7.1 survives, though it needs re-wording. Raw-HF Gemma
remains **additive** (47 + 46 ≈ 93 recovered by all text), while Qwen is **superadditive** — its
parts sum to 18% against a 65% whole (`patching_qwen_metrics.json`: question 12%, turn markers 6%,
all text 65%), i.e. genuinely redundant across positions. Gemma moves toward Qwen but the kinds still
differ. Do not write that the contrast dissolves.

---

# Re-score results — §6.3 cross-image patching (raw HF, 2026-08-22)

60 donor/recipient pairs, neutral context, three bands. Published column is bridge + behavioural
valence (paper Table 4); raw HF is reported on both readouts where the probe is valid.

| band | image (pub) | image (raw HF val) | image (raw HF probe) | non-image text (pub) | text (raw HF val) | text (raw HF probe) | all (pub) | all (raw HF) |
|---|---|---|---|---|---|---|---|---|
| 0–12 | 80% | **100%** | 101% | 10% | **2%** | 8% | 100% | **100%** |
| 13–17 | 66% | **96%** | 75% | 65% | **74%** | 87% | 91% | **100%** |
| 18–28 | 9% | **31%** | *(invariant)* | 68% | **63%** | *(invariant)* | 79% | **90%** |

## The qualitative claim survives

Image-token recovery falls with depth (**100% → 96% → 31%**) while text-position recovery rises
(**2% → 74% → 63%**). That is the same shape as the published 80/66/9 and 10/65/68, and it supports
the paper's reading unchanged: visual valence starts in the image tokens and is progressively
readable from text-token states instead. The `all` sanity column is *better* than published
(100/100/90 vs 100/91/79).

## But the sharpest number in the table nearly quadruples

The paper leans on late-band image recovery falling to **9%** — near-total handoff out of the image
tokens. On raw HF it is **31% [24, 38]**. The module's own verdict downgrades from "image tokens
carry LITTLE" to "**a MODERATE share**". A third of the image-driven difference is still recoverable
from image tokens at layers 18–28.

- §6.3's "image-token recovery falls to $9\\%$" becomes $31\\%$, and the bolded $\\mathbf{9\\%}$ in
  Table 4 goes with it.
- "We interpret this as the visual valence signal **moving** from image tokens into text-token
  states over depth" should soften — the signal becomes *additionally* readable from text states
  while remaining partly readable from image tokens. It is a broadening, not a handoff.
- The "concrete site where the two cues can compete" argument is unaffected: text positions still
  recover 63% late, which is all that claim requires.

## A caveat the published table could not show

In the 13–17 band the two readouts disagree about which group dominates, with non-overlapping CIs:

| 13–17 | image | non-image text |
|---|---|---|
| probe (L18) | 75% [71, 79] | 87% [86, 89] |
| behavioural valence | 96% [92, 100] | 74% [66, 82] |

Both are raw HF, so this is not bridge damage — it is a real difference between reading at L18 and
reading at the output, with 16 layers of re-mixing in between. The published mid-band row (66 / 65)
presented the two groups as comparable; the honest statement is that their ordering depends on where
you read. Report the mid band on both readouts rather than picking the flattering one.

## Not re-run

The confirmatory fixed-negative-context run (image-driven gap shrinking 0.87 → 0.08, mid-band 85%
CI [51, 150]) is a separate doc-only number and was not part of this sweep. It stays flagged.

---

# Verdict on §6

**The mechanism section survives.** Both experiments reproduce their qualitative claims on raw HF,
and each loses one sharp quantitative claim:

- §6.2 keeps *image tokens are causally inert for the text-context effect* (0%, three runs) and loses
  *the carrier is concentrated at the assistant-turn boundary*.
- §6.3 keeps *visual valence is readable from image tokens early and from text states late* and loses
  *image-token recovery falls to 9%*.

So the rewrite is a retraction of §5 and §7.2/§7.3 plus a numbers pass over §6 — not the deletion of
§6. Under the outcome table above, this is the "both survive" branch.

---

# Provenance check — every number verified against artifacts (2026-08-22, post-Drive-sync)

`results/` now holds the full Drive tree at canonical paths (`results/stage_f/...`); the old nested
`results/results/` copy is gone. Bridge and raw-HF artifacts coexist, so every claim below is
re-derivable locally. All figures previously read off terminal output were re-checked against JSON
and match exactly.

## Conflict override gaps — all raw HF, all artifact-backed

Source: `results/stage_f/conflict_<model>_metrics.json`, key `flip_override`. n = 121 images
(62 positive, 60 negative) per model.

| model | neg overrides pos img | pos overrides neg img | gap | 95% CI | image AUC |
|---|---|---|---|---|---|
| \qwen{} | 76.3% | 36.9% | **+39.4%** | [+29.5, +49.0] | — |
| LLaVA-NeXT | 37.9% | 19.2% | **+18.7%** | [+7.8, +30.0] | 0.998 |
| \gemma{} | 43.5% | 31.4% | **+12.2%** | **[−0.4, +23.7]** | 0.982 |
| \gemma{} (legacy prompt) | 44.4% | 31.9% | +12.4% | [−0.5, +24.3] | 0.981 |
| LLaVA-1.5 | 27.7% | 40.3% | **−12.6%** | [−22.0, −2.8] | 0.989 |

Confirms **2 of 4 clear zero**. Gemma's CI touches zero on both prompt styles, and its image
discriminability (0.982) is healthy — so the weak gap is not a degraded-image artefact this time.

Qwen resolution sweep, same weights: 128 tok **+41.8%** [+32.1, +50.8]; 262 tok **+37.4%**
[+27.5, +47.0] and **+38.6%** [+28.6, +48.1]. Flat, AUC 0.983–0.986. Token budget stays falsified.

## Stage C and Stage D

`results/stage_c/metrics_hf.json`: pleasantness ρ **+0.50994**, AUC **0.91178**, unpleasantness
ρ **−0.44784**, retention 0.6633, 7,280 scored / 0 skipped. *Note for the write-up:* the polarity AUC
is computed on **440** single-label polar images (384 pos / 56 neg), not on all 7,280 — state both
denominators rather than letting "7,280" attach to the AUC.

`results/stage_d/steering_metrics_hf.json`: slopes pleasantness **+0.33602**, unpleasantness
**−0.31563**, suddenness **−0.07757**, random **−0.03495** (published +0.3293 / −0.3087 / −0.0726 /
−0.0270). `site_check.ok = true`, worst relative error **0.51%**. Base valence +0.0987.

## Patching — raw HF vs bridge, both on disk

Same-image (probe read-out), `patching_hf_pair1_metrics.json` / `patching_hf_metrics.json` vs
`patching_metrics.json`:

| | image | BOS | prefix | question | turn boundary | all text |
|---|---|---|---|---|---|---|
| Pair 1 (0,2) raw HF | 0.000 | 0.000 | 0.000 | **0.473** | **0.464** | 0.932 |
| Pair 2 (4,0) raw HF | −0.001 | 0.004 | 0.000 | **0.543** | **0.398** | 0.883 |
| Pair 2 (4,0) bridge | 0.007 | −0.008 | −0.011 | **0.320** | **0.566** | 0.867 |

Cross-image, `cross_patching_hf_{band}.json`: 0–12 image val 1.001 / probe 1.008, text 0.017 / 0.083;
13–17 image 0.960 / 0.749, text 0.741 / 0.874; 18–28 image **0.311**, text 0.634, all 0.899.

## Two published bridge numbers remain doc-only — and no longer matter

- **Same-image Pair 1 (championship/funeral, 22% / 65%).** The surviving
  `patching_metrics.json` is Pair 2 (pos 4, neg 0). Pair 1's bridge run is still lost.
- **Cross-image neutral-context three-band table (80/10/100 · 66/65/91 · 9/68/79).** The surviving
  `cross_patching_metrics.json` is the *confirmatory* run instead — band 18–28, **negative** context,
  n=60, image 0.076 / text 0.925 / all 1.068.

Neither gap blocks anything now: both tables are being replaced by the raw-HF sweeps above, which are
fully artifact-backed. The bridge numbers only appear in the paper as the values being retracted.

---

# Rebuilt §5 on raw HF — and a robustness finding that changes the claim (2026-08-22)

`analyze_stage_f.py` is CPU-only and writes stem-matched output, so §5 was rebuilt from the raw-HF
parquets locally. `analyze_stage_f_unbounded.py` was re-run on the raw-HF parquets too — the previous
`unbounded_crossed.json` had used the **bridge** Gemma parquet (`conflict_pilot.parquet`). The old
file is preserved as `unbounded_crossed_bridge-gemma.json`; the new one is `unbounded_crossed_rawhf.json`.

## Image-clustered results (the CIs the paper currently reports)

| model | $\beta_{img}$ | $\beta_{txt}$ | ratio | mirror contrast | 95% CI | override gap |
|---|---|---|---|---|---|---|
| \qwen{} | +0.510 | +0.584 | **1.14** | **+0.394** | [+0.249, +0.542] | +39% |
| LLaVA-NeXT | +0.828 | +0.364 | 0.44 | **+0.392** | [+0.337, +0.450] | +19% |
| \gemma{} (raw HF) | +0.723 | +0.321 | 0.44 | **+0.319** | [+0.181, +0.451] | +12% |
| LLaVA-1.5 | +0.749 | +0.414 | 0.55 | **−0.018** | [−0.081, +0.049] | −13% |

**Qwen is a strictly better primary model than Gemma ever was.** It is the only model with *balanced*
integration (1.14 — the property the paper claims for Gemma, which on raw HF is image-led at 0.44),
it has the largest mirror contrast, and its head-room normalisation is clean (drop 0.714 / rise 0.394).

**Do not report Gemma's head-room-normalised pull.** On raw HF it comes out as **62.716** vs a rise of
0.259 — degenerate, because the positive-image neutral baseline (+0.905) sits almost on the ceiling
and the denominator collapses. The published $0.63 / 0.26$ is not recoverable and the statistic is
not meaningful for this model.

## The crossed (image × sentence) bootstrap — the honest test, and it is unforgiving

Three readouts × four models. "Clears" means the crossed 95% CI excludes zero.

| model | override gap | crossed CI | bounded mirror | crossed CI | unbounded margin | crossed CI |
|---|---|---|---|---|---|---|
| \qwen{} | +39.8% | **[+7.3, +66.9]** ✓ | +0.394 | [−0.187, +0.891] ✗ | −1.567 | [−4.995, +1.811] ✗ |
| LLaVA-NeXT | +16.2% | [−2.4, +35.6] ✗ | +0.392 | **[+0.219, +0.577]** ✓ | +0.971 | **[+0.478, +1.471]** ✓ |
| \gemma{} (raw HF) | +9.6% | [−15.3, +31.3] ✗ | +0.319 | [−0.044, +0.666] ✗ | +4.742 | **[+3.076, +6.271]** ✓ |
| LLaVA-1.5 | −17.6% | [−49.6, +13.1] ✗ | −0.018 | [−0.447, +0.367] ✗ | +0.096 | [−1.090, +1.160] ✗ |

**No model clears zero on all three readouts.** LLaVA-NeXT clears two of three; Qwen and Gemma clear
one each, and on *different* readouts; LLaVA-1.5 clears none.

Two specific reversals worth noting:

- **Raw-HF Gemma's unbounded margin is +4.742 log-odds (ratio 3.07), crossed CI [+3.08, +6.27] —
  decisively positive.** On the *bridge* parquet the same statistic was **−0.958** (ratio 0.76, sign
  flipped). The earlier conclusion that "the ~1.8× magnitude claim is an artefact of the bounded
  scale" was itself computed on the corrupted stack. On raw HF the unbounded readout *strengthens*
  the asymmetry rather than reversing it. Drop "1.8×" from the abstract as planned, but for the
  original reason (bounded-scale dependence), not because the effect inverts.
- **Qwen's unbounded margin goes negative (−1.567, ratio 0.87)** because its bounded valence saturates
  at $\pm 1$; on log-odds its rise exceeds its drop.

### How to state this without overclaiming in either direction

With **6 sentences per polarity**, the crossed bootstrap resamples 6 items and is badly underpowered —
wide CIs are partly a design limit, not evidence of absence. The defensible claim:

> The asymmetry is robust to *image* sampling in 3 of 4 models, but with only six sentences per
> polarity the design cannot establish that it generalises to new *sentences*. Which readout shows it
> most strongly varies by model, tracking how each model uses the bounded valence scale.

**Qwen's override gap is the one statistic that clears zero under the crossed bootstrap
(+39.8% [+7.3, +66.9]).** That makes "Qwen primary, lead with the override rate" the defensible
structure — but the paper must report all three readouts per model rather than choosing per model.

Expanding the sentence banks is now the single highest-value remaining experiment: it is the only
thing that converts these into sentence-generalisable claims, and it needs no GPU beyond re-running
the conflict pass.

### Minor: two override definitions differ by a few points

`analyze_stage_f._flip_override` and `analyze_stage_f_unbounded` disagree slightly (Qwen +39.4 vs
+39.8; NeXT +18.7 vs +16.2; Gemma +12.2 vs +9.6; LLaVA-1.5 −12.6 vs −17.6) because they subset
conditions differently. Pick one definition for the paper and state it; do not mix them across tables.

---

# Correction: the crossed bootstrap was clustering on the wrong unit (2026-08-22)

`analyze_stage_f_unbounded._img` was `(condition == "none").cumsum()`, which counts
**person-annotations** — 150 units over 121 distinct photos. But EMOTIC annotates each person while
the model is shown the whole image with no bounding box, so two annotated people in one photo produce
**byte-identical forward passes**. Those 29 extra rows are duplicates, not independent observations,
and counting them as independent narrows every interval.

Fixed to cluster on `image_path`, the unit `analyze_stage_f._flip_override` already used. The two
tools now agree exactly on the override gap and its image-clustered CI (+39.4 / +18.7 / +12.2 / −12.6),
which confirms the diagnosis — the earlier few-point discrepancy was entirely this.

## Corrected cross-model table (raw HF, photo-clustered)

| model | override gap | image-only | crossed | mirror contrast | crossed | unbounded margin | crossed |
|---|---|---|---|---|---|---|---|
| \qwen{} | **+39.4%** | [+29.5, +49.0] | **[+6.4, +66.5]** ✓ | +0.409 | [−0.16, +0.91] | −1.81 | [−5.47, +1.80] |
| LLaVA-NeXT | +18.7% | [+7.8, +30.0] | [−1.9, +40.2] | +0.413 | **[+0.23, +0.59]** ✓ | +1.00 | **[+0.50, +1.50]** ✓ |
| \gemma{} | +12.2% | [−0.4, +23.7] | [−11.4, +36.9] | +0.405 | **[+0.06, +0.77]** ✓ | +4.64 | **[+3.05, +6.29]** ✓ |
| LLaVA-1.5 | −12.6% | [−22.0, −2.8] | [−42.6, +21.6] | **+0.016** | [−0.37, +0.42] | +0.05 | [−1.03, +1.13] |

**This is a better picture than the mis-clustered version gave.** Three of four models now clear zero
on at least one readout under the crossed bootstrap — Gemma and LLaVA-NeXT on two each — and
LLaVA-1.5 clears none.

## The headline should be the mirror contrast, not the override gap

The three positive models have **nearly identical** bounded mirror contrasts — $+0.409$, $+0.413$,
$+0.405$ — against LLaVA-1.5's $+0.016$. On this readout the effect looks like a shared, similarly
sized property of three architectures with one clean exception, whereas the override gap varies
three-fold across the same models ($+39\%$ / $+19\%$ / $+12\%$).

Two cautions before leaning on it. The three-way agreement to within $0.008$ is closer than the CIs
warrant and is probably partly coincidence — report the intervals, not the point agreement. And
Qwen's mirror contrast is the one that does *not* clear zero crossed, so the measure that looks most
stable across models is not the one that is most robust within Qwen. Report both readouts for all
four models and let the disagreement stand; it is a real property of the data.

Also corrected by the re-clustering: LLaVA-1.5's mirror contrast is $+0.016$, not $-0.018$. It was
never a reversal — it is a null sitting essentially on zero, which is the cleaner claim anyway.

---

# Minimal-pair bank on \qwen{} — the control that rescues the crossed intervals (2026-08-22)

Run with the new `--bank minimal` path on raw HF; analysed with the same
`analyze_stage_f` / `analyze_stage_f_unbounded` definitions as every other run.

| | full bank | minimal pairs |
|---|---|---|
| override gap | $+39.4\%$ | $+40.8\%$ |
| \quad photo-clustered | [+29.5, +49.0] | [+31.3, +49.0] |
| \quad **crossed** | [+6.4, +66.5] ✓ | **[+16.7, +63.2]** ✓ |
| mirror contrast | $+0.409$ | $+0.496$ |
| \quad photo-clustered | [+0.23, +0.59] | [+0.31, +0.67] |
| \quad **crossed** | [−0.16, +0.91] ✗ | **[+0.11, +0.83]** ✓ |
| unbounded margin | $-1.81$ | $-1.13$ (straddles, both) |
| $\beta_{txt}/\beta_{img}$ | 1.14 | **1.97** (text-led) |

**The minimal bank converts the mirror contrast from straddling zero to clearing it**, and tightens
the override gap's crossed interval from a 60-point span to 46. The mechanism is straightforward:
resampling six *unrelated events* perturbs the estimate far more than resampling six sentences that
differ only in one valence word. This is the strongest argument available that the effect is not an
artefact of which six sentences were chosen.

Within-item contrast (same photo, same event, one word flipped): **$+1.148$** [+0.94, +1.34],
Wilcoxon $p < 0.001$, $n = 62$ images $\times$ 6 pairs. All six pairs agree, at $+1.10$ to $+1.29$
(ratios 4.3–5.5). The superseded Gemma version of this test was $+0.166$, $p = 0.052$, with one pair
reversing — so this went from the paper's weakest claim to its strongest.

The unbounded margin stays negative on both banks ($-1.81$, $-1.13$). That is Qwen's saturation
property (documented in the four-model table), not a failure of this control.

## What this does and does not license

It licenses: "the asymmetry does not depend on which of these six events is used, and survives
resampling sentences." It does **not** license "generalizes to arbitrary context sentences" — every
sentence in both banks follows the frame *"This photo was taken \ldots"*. Widening across sentence
*forms*, not merely adding more of the same frame, is the remaining gap and is now stated that way in
Limitations.

## Two bugs found via this run

1. `analyze_stage_f` attached `arbitration_pilot.parquet` (a Gemma-only steering sweep at a fixed
   path) to **every** model's analysis, so the Qwen run reported Gemma's $+0.215$ slope as its own and
   `_verdict` repeated it in prose. Fixed to attach only to the runs it was measured alongside.
2. The two neutral contexts are **not interchangeable**: on positive images "taken on a weekday"
   gives $+0.914$ and "taken indoors" gives $+0.344$ — a $0.57$ spread in the baseline every effect is
   measured against. We average over both; recorded as a caveat in §5.3. Neutral framing is not inert.
