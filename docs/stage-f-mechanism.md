# Stage F mechanism — why negative text overrides a positive image (Gemma-3-4B)

**Team write-up, 2026-08-05.** This is the mechanistic follow-up to the Stage F modality-conflict run.
It answers the action item from the last meeting — *investigate why a positive image + negative context
causes a large valence drop* — and it **corrects** one conclusion from the earlier Stage F doc (the
"ceiling/floor, not a negativity asymmetry" reading; see the banner in `stage-f-results.md`).

## TL;DR
When a person image and a one-sentence text context carry conflicting emotional cues, **negative text
reliably overrides a positive image, but positive text does not override a negative image to the same
degree.** This is a genuine, controlled cross-modal effect — not a stimulus artifact and not a
ceiling/floor of the valence scale. Mechanistically:

- The asymmetry **only appears when an image is present** (text-only, the positive and negative context
  banks are equally strong).
- Adding a positive image **blunts the positive-text channel by ~76% but the negative-text channel by
  only ~20%** — so the image *suppresses agreeing text* far more than *conflicting negative text*.
- The divergence **enters the network mid-stack (~layer 13 of 34) and amplifies ~7× through the late
  layers (peak ~L28)**.
- The winning signal is carried **entirely by the text token stream — the image tokens are causally
  inert (~0% recovery)**. The model **broadcasts** the context's valence into the **structure/turn
  "sink" tokens** (BOS + turn delimiters) that the read-out attends to; the image's own valence never
  propagates out of the image tokens. **Replicated across two independent context pairs.**

**One-line takeaway for the group:** *negativity dominance in cross-modal conflict is a text-stream
phenomenon — the negative context hijacks the shared read-out via sink tokens, while the positive
image stays trapped in the image tokens.*

## Setup (shared across all four experiments)
- **Model / read-out:** `google/gemma-3-4b-it` via TransformerBridge (bf16). Internal read-out = the
  **frozen Stage A pleasantness probe** at `blocks.18.hook_attn_out`, last prompt token (never re-fit).
  Behavioral valence = closed-vocab P(positive) − P(negative) over the 13 emotion labels.
- **Images:** EMOTIC test **positive-valence group** (high-EMOTIC-valence persons) — the clean cell,
  because its neutral-context behavioral valence sits mid-scale (−0.058), so it has symmetric head-room
  and the effect is not a floor artifact.
- **Contexts:** the same 6 positive / 6 negative / 2 neutral one-sentence bank as the base run.
- **Baseline convention:** context effects are measured **vs the neutral context** (the no-context
  prompt is structurally non-comparable — established in the base run).

---

## 1. Is the drop real, or just ceiling/floor? → **Real.**
`analyze_stage_f.py::_asymmetry_vs_floor`. Per positive image, the negative-context "drop" vs the
mirror (negative image + positive context "rise"), with a bootstrap CI and a **head-room-normalized**
pull that divides each effect by the distance to the valence bound it moves toward (controls for the
floor).

| quantity | value |
|---|---:|
| drop (positive image, negative ctx) | **−0.598** |
| rise (negative image, positive ctx) | +0.333 |
| \|drop\| − \|rise\| | **+0.265**, 95% CI **[+0.095, +0.431]**, Mann–Whitney p = 0.005 |
| head-room-normalized pull (drop vs rise) | 0.632 vs 0.259 |
| within positive-image group: neg-ctx / pos-ctx | **−0.598 / +0.167 = 3.58×** |

The gap **survives head-room normalization** (0.63 vs 0.26), so it is not the floor. The sharpest cut
is *within the positive-image group*, where head-room is symmetric: a negative context moves the
read-out **3.6× more** than a positive one. → The earlier "ceiling/floor, symmetric" reading was
under-analyzed; there is a real residual asymmetry.

## 2. Cross-modal, or just stronger negative sentences? → **Cross-modal.**
`stage_f_text_only.py`. The obvious confound: maybe the negative sentences (funeral, accident) are just
more affectively extreme than the positive ones (birthday, championship). Control: run each context
sentence **with no image** (prompt = the base prompt minus `<start_of_image>`, token-identical
otherwise) through the same frozen probe.

| condition | positive-ctx effect | negative-ctx effect | \|neg\|/\|pos\| |
|---|---:|---:|---:|
| **text-only (no image)** | +0.706 | −0.752 | **1.06** (MW p = 0.155) |
| **+ positive image** | +0.167 | −0.598 | **3.58** |

Text-only the two banks are **statistically symmetric** (1.06) — the negative sentences are *not*
stronger stimuli (two are actually weak: "moments before the accident" reads +0.189, "worst week"
−0.350). The asymmetry appears **only when the image is added**, and it is not a ceiling effect
(positive-image + positive-context valence is only +0.109, far below the +1.0 the positive contexts
reach text-only — so the positive context *has* room to push and doesn't).

**Precise mechanism (report this, not "the image amplifies negative"):** adding a positive image
retains only **24%** of the positive-text effect (0.706 → 0.167) but **80%** of the negative-text
effect (0.752 → 0.598). The image **suppresses agreeing (positive) framing** far more than it
suppresses **conflicting (negative) framing**.

## 3. Where in the network does it enter? → **~Layer 13, amplifying to ~L28.**
`stage_f_layerwise.py` + `analyze_stage_f_layerwise.py`. Two probe-lenses on the last token using the
frozen probe weight as a fixed direction (logit-lens diagnostic): `resid_post · w` (running read-out)
per layer, paired across the same images and reported as a **scale-free effect size** `d` (so the
growing residual norm doesn't inflate late layers).

| | layer | paired effect size \|d\| |
|---|---:|---:|
| **entry** (first layer clearing the noise floor, read-out sign) | **~13** | 0.21 |
| read-out layer | 18 | 0.16 |
| **effect-size peak** | **~28** | **1.47** |

The divergence **enters mid-stack (~L13) and amplifies ~7× (scale-free) through the back half**,
peaking ~L28 — this is cross-modal integration compounding over depth, well upstream of and continuing
past the L18 read-out. (Caveat baked into the analyzer: raw projections grow with residual norm
±3→±57, and near-zero-variance early layers show spurious large `d`; both are gated out — do not read
the raw magnitudes or layer 0.)

An earlier last-token **attention knockout at L18** (`stage_f_attribution.py`) is consistent: at the
read-out layer the last token attends ~88% to template/question tokens, only ~3.5% to context, and the
*direct* context contribution is ~6% of the effect — i.e. the context reaches the read-out **indirectly**,
having been mixed into other positions upstream. Experiment 4 finds which positions.

## 4. Which tokens carry it? → **Text stream (sink tokens); image tokens inert.**
`stage_f_patching.py`. Donor = positive-context run, recipient = negative-context run (**same image**).
Overwrite the recipient's `resid_post` over layers [13,17] for one **position-aligned** token group at
a time, and measure recovery toward the positive read-out:
`recovery(G) = (patched − neg) / (pos − neg)`. Groups: **image** (256 tokens), **question** (the
identical question string), **structure** (BOS + turn delimiters, *excluding the read-out token*),
**text_all** (question ∪ structure). Context tokens differ in length and cannot be 1:1 patched — the
`1 − text_all` remainder is their share.

Probe-read-out recovery, **replicated across two independent context pairs**:

| group | pair 1 (championship / funeral) | pair 2 (wonderful news / devastating news) |
|---|---:|---:|
| **image** | **−1%** | **+1%** |
| question | 22% | 32% |
| **structure / turn (sink)** | **65%** | **56%** |
| text_all (all aligned text) | 85% | 87% |
| → remainder in context tokens | ~15% | ~13% |

Three replicated conclusions:
1. **Image tokens are causally inert (~0%).** Patching them does nothing → the negative context does
   **not** rewrite the image representation; the image's valence stays in the image tokens.
2. **The structure/turn tokens are the dominant carrier (~55–65%)** — and this is a *tiny* group
   (~10 tokens) vs the image's 256, so per-token they are extremely potent. The model **broadcasts** the
   context's valence into BOS/delimiter **sink** positions the read-out attends to (attention-sink
   behavior).
3. **~85–87% of the effect lives in the aligned text stream**; only ~13–15% remains in the literal
   context tokens. The effect is a text-stream phenomenon.

(Behavioral-valence recovery follows the same ordering — image ~0%, structure > question — but is
noisier: text_all 73–78%. The probe-level decomposition is the clean measurement.)

---

## Full mechanism (report-ready)
> **Controlled cross-modal negativity dominance:** negative text overrides a positive image more than
> positive text overrides a negative image. The asymmetry is real (survives head-room normalization),
> cross-modal (absent text-only, where the banks are symmetric at 1.06), and not a ceiling effect. It
> **enters the network ~L13 and amplifies ~7× to ~L28**, and is **carried by the text token stream —
> broadcast into structure/turn sink tokens — while the image tokens are causally inert**. The positive
> image loses because its valence never propagates out of the image tokens; the negative context wins
> by writing into the shared sink tokens the read-out reads.

## Threats to validity
- **Single seed, one layer's read-out, one model.** All runs seed 0, read-out at L18, Gemma-3-4B. A
  3-seed repeat and a second model (Qwen-VL / 12B Gemma) are the outstanding robustness steps.
- **Patching context-pair scope.** Two donor/recipient context pairs (championship/funeral,
  wonderful/devastating). The image-inert / sink-carrier pattern replicates across both, but more pairs
  would tighten it. (This directly guards the single-context pitfall that misled the Stage F pilot.)
- **Context-token ceiling.** The literal context sentences differ in length between donor and recipient
  and cannot be 1:1 patched, so aligned-group recovery caps below 100% (~13–15% remainder). We attribute
  *among the alignable positions*, not to the context tokens themselves.
- **Logit-lens is off-layer.** The layerwise projection applies the L18 probe direction at every layer
  as a diagnostic; onset/peak are gated for scale and sign, but absolute `d` values are lens-relative.
- **Sink interpretation is inferred.** "Broadcast into sink tokens" follows from the structure-group
  recovery; a per-token attribution within the structure group (BOS vs delimiters) would confirm which
  sink dominates.
- **Behavioral valence is negatively skewed** (even happy faces read slightly negative on the
  closed-vocab P[pos]−P[neg]); only *relative* effects are interpreted.

## Reproduce
```bash
# 1. is the drop real vs ceiling/floor?  (CPU, reuses the base-run parquet)
python -m src.experiments.analyze_stage_f

# 2. cross-modal vs stimulus confound?   (15 forwards, no images)
python -m src.experiments.stage_f_text_only

# 3. where does it enter?                (layer sweep, then scale-free CPU re-localization)
python -m src.experiments.stage_f_layerwise
python -m src.experiments.analyze_stage_f_layerwise

# 4. which tokens carry it?              (patching; second pair for robustness)
python -m src.experiments.stage_f_patching
python -m src.experiments.stage_f_patching --pos-idx 4 --neg-idx 0
```
Artifacts under `results/stage_f/`: `conflict_analysis.json` (asymmetry), `text_only_metrics.json`,
`layerwise_normalized.json`, `patching_metrics.json`; figures `stage_f_{conflict,attribution,layerwise}.png`.

## Relationship to the group's direction
This sharpens the group's shared finding (image and text write into one appraisal representation) into
a **causal, localized mechanism** for the conflict case. It also connects to the team's broader thread —
*emotional images shifting VLM behavior* — with a cautionary result: an image's emotional pull is
**real but fragile under conflicting text**, and mechanistically it is because the image's signal stays
in the image tokens while text commandeers the sink tokens the model actually reads out from. For
open-ended/ambiguous task contexts (where there is no strong overriding text), we'd expect the image to
matter *more* than it does here.
