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
  inert (~0% recovery)**. Specifically it lives in the **assistant-turn-boundary tokens**
  (`<end_of_turn>\n<start_of_turn>model`) right before the read-out (~55–65%); **BOS and the user-turn
  prefix carry 0%**, so this is *not* a global attention-sink effect. The image's own valence never
  propagates out of the image tokens. **Replicated across two independent context pairs.**
- **The behavioral effect replicates on a different-architecture VLM (Qwen3-VL-8B), near-quantitatively**
  (within-positive-image negativity ratio 3.64× vs Gemma's 3.58×; cross-modal on both) — see
  *Multi-model robustness* below.

**One-line takeaway for the group:** *negativity dominance in cross-modal conflict is a text-stream
phenomenon — the negative context routes into the assistant-turn preamble the read-out reads from,
while the positive image stays trapped in the image tokens.*

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

## 4. Which tokens carry it? → **The assistant-turn-boundary tokens; image inert; NOT BOS.**
`stage_f_patching.py`. Donor = positive-context run, recipient = negative-context run (**same image**).
Overwrite the recipient's `resid_post` over layers [13,17] for one **position-aligned** token group at
a time, and measure recovery toward the positive read-out:
`recovery(G) = (patched − neg) / (pos − neg)`. Groups: **image** (256 tokens), **question** (the
identical question string), and the structural/control tokens split into **bos** (first token),
**prefix_delim** (`<start_of_turn>user…` before the image), and **suffix_delim**
(`<end_of_turn>\n<start_of_turn>model`, the assistant-turn preamble *immediately before* the read-out
token, which is itself excluded); **structure** = their union, **text_all** = question ∪ structure.
Context tokens differ in length and cannot be 1:1 patched — the `1 − text_all` remainder is their share.

Probe-read-out recovery, **replicated across two independent context pairs**:

| group | pair 1 (championship / funeral) | pair 2 (wonderful / devastating) |
|---|---:|---:|
| **image** | **−1%** | **+1%** |
| question | 22% | 32% |
| bos | 0% | −1% |
| prefix delimiters | 0% | −1% |
| **suffix delimiters** (`<end_of_turn>…model`) | **65%** | **57%** |
| structure (= suffix, additive) | 65% | 56% |
| text_all (all aligned text) | 85% | 87% |
| → remainder in context tokens | ~15% | ~13% |

Three replicated conclusions:
1. **Image tokens are causally inert (~0%).** Patching them does nothing → the negative context does
   **not** rewrite the image representation; the image's valence stays in the image tokens.
2. **The carrier is the assistant-turn-boundary tokens (~55–65%), NOT a global BOS sink.** BOS and the
   user-turn prefix carry **0%**; the entire structure-group recovery is the `<end_of_turn>` /
   `<start_of_turn>model` preamble adjacent to the read-out. So this is **local aggregation into the
   turn scaffold where the model begins its answer** (recency + turn boundary), not the BOS
   attention-sink one might expect. The question tokens carry a secondary share (~22–32%).
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
> specifically the assistant-turn-boundary tokens (`<end_of_turn><start_of_turn>model`) adjacent to the
> read-out, not BOS — while the image tokens are causally inert**. The positive image loses because its
> valence never propagates out of the image tokens; the negative context wins by writing into the turn
> preamble the read-out reads.

## Multi-model robustness — Qwen3-VL-8B (different architecture)
The behavioral effect replicates on a second, unrelated VLM. `stage_f_qwen.py` runs the base pass +
text-only control on Qwen via raw HuggingFace (no TransformerBridge, no probe — behavioral valence
only), 150 EMOTIC images. It **replicates near-quantitatively:**

| metric | Gemma-3-4B | Qwen3-VL-8B |
|---|---:|---:|
| within-positive-image neg/pos ratio (graded valence) | **3.58×** | **3.64×** |
| text-only raw \|neg\|/\|pos\| (confound control) | 1.06 (symmetric) | 1.00 (symmetric) |
| override rate — neg-ctx overrides positive image | *(rerun pending)* | **77%** |
| override rate — pos-ctx overrides negative image | *(rerun pending)* | **35%** |

- **Same magnitude, different architecture.** The within-positive-image negativity ratio lands at
  3.64× vs Gemma's 3.58×, and text-only is symmetric on both (1.00 / 1.06) → **cross-modal on Qwen too**,
  not a stimulus artifact and not a Gemma quirk.
- **The image is integrated, not ignored** (positive-image neutral valence +0.63 — a happy face reads
  positive; Qwen weighs both modalities and negative text wins the conflict).
- **A read-out-regime change, handled.** Qwen is far more confident, so its closed-vocab valence
  **saturates to ±1** and floors negative images — the graded head-room metric is fragile at small n
  (a 20-image smoke looked symmetric; at n=150 it clears 0: |drop|−|rise| = +0.394, CI [+0.249,+0.542]).
  The robust primary metric is therefore a **calibration-free override rate** on the argmax emotion's
  valence category (shared across both models via `analyze_stage_f._flip_override`): on Qwen, negative
  context overrides a positive image **77%** of the time vs positive context overriding a negative image
  **35%** (gap +41%, CI [+32, +51]).

**Scope:** this replicates the *effect* (experiments 1–2), not yet the *mechanism* — the L13-entry /
turn-token-carrier findings need the patching port on Qwen (variable image-token counts + Qwen's own
turn scaffold), which is the next step. Reproduce (separate `requirements-qwen.txt` env):
```bash
python -m src.experiments.stage_f_qwen              # base pass (override rate + graded asymmetry)
python -m src.experiments.stage_f_qwen --text-only  # confound control (raw |neg|/|pos| ~ 1.0)
```

## Threats to validity
- **Single seed; mechanism on one model.** All runs seed 0. The *behavioral* effect now holds on two
  architectures (Gemma-3-4B + Qwen3-VL-8B), but the *mechanism* (L13 entry, turn-token carrier) is
  verified on Gemma only — the Qwen patching port is the outstanding step. A 3-seed repeat and a third
  model (12B / Gemma-4) would further tighten it.
- **Patching context-pair scope.** Two donor/recipient context pairs (championship/funeral,
  wonderful/devastating). The image-inert / turn-preamble-carrier pattern replicates across both, but more pairs
  would tighten it. (This directly guards the single-context pitfall that misled the Stage F pilot.)
- **Context-token ceiling.** The literal context sentences differ in length between donor and recipient
  and cannot be 1:1 patched, so aligned-group recovery caps below 100% (~13–15% remainder). We attribute
  *among the alignable positions*, not to the context tokens themselves.
- **Logit-lens is off-layer.** The layerwise projection applies the L18 probe direction at every layer
  as a diagnostic; onset/peak are gated for scale and sign, but absolute `d` values are lens-relative.
- **Carrier resolved to the suffix delimiters (not inferred).** The structure group was split and
  re-patched: BOS 0% / prefix-delims 0% / **suffix-delims 57–65%**, additive with structure. So the
  carrier is the assistant-turn preamble, not a BOS sink. (Open: this is one prompt template's turn
  scaffold; a different chat format could relocate it.)
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
in the image tokens while text commandeers the turn-preamble tokens the model actually reads out from. For
open-ended/ambiguous task contexts (where there is no strong overriding text), we'd expect the image to
matter *more* than it does here.
