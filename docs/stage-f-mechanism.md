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
- **Valence, not event semantics.** A token-matched minimal-pair control (won↔lost, best↔worst, …;
  only the valence word changes) reproduces the override gap (**+65% vs +64%**) — so the effect is
  about valence, not "funeral vs championship" event semantics. See *Minimal-pair valence control*.
- Adding a positive image **blunts the positive-text channel by ~76% but the negative-text channel by
  only ~20%** — so the image *suppresses agreeing text* far more than *conflicting negative text*.
- The divergence **enters the network mid-stack (~layer 13 of 34) and amplifies ~7× through the late
  layers (peak ~L28)**.
- The winning signal is carried **entirely by the text token stream — the image tokens are causally
  inert *for the text-context delta* (~0% recovery)**. Specifically it lives in the
  **assistant-turn-boundary tokens** (`<end_of_turn>\n<start_of_turn>model`) right before the read-out
  (~55–65%); **BOS and the user-turn prefix carry 0%**, so this is *not* a global attention-sink
  effect. **Replicated across two independent context pairs.**
- **The image's own valence is not trapped — it migrates into the same text-stream channel.** A
  cross-image patching experiment (vary the image, hold context fixed) shows visual valence
  *originates* in the image tokens (early layers) and *broadcasts into the text-stream positions* over
  depth (by the read-out, the image tokens no longer carry it; the turn scaffold does). So both cues
  are integrated into one shared text-stream read-out channel, and negative text wins the competition
  *within* that channel — not because the image is trapped. See *Cross-image patching* below.
- **Both the effect AND the mechanism replicate on a different-architecture VLM (Qwen3-VL-8B):** the
  negativity ratio lands at 3.64× (vs Gemma's 3.58×, cross-modal on both), and the image tokens are
  causally inert on Qwen too (0% patch recovery) — the carrier is distributed across Qwen's text stream
  rather than concentrated in the turn scaffold. See *Multi-model robustness* below.
- **Prompt-robust:** the behavioral override holds across 6 prompt phrasings (dominance gap +46% to
  +74%, mean +58%, every CI clears 0), and the reported base prompt is mid-pack — not the best case.
  See *Prompt robustness* below.
- **Holds at scale:** dominance persists from Gemma-3-4B to 12B (gap +51%, CI clears 0); with scale
  the model becomes *more* text-driven in BOTH directions (neg-override 84%→93%, pos-override
  19%→42%) — the image loses influence, negativity dominance does not. See *Model-scale robustness*.

**One-line takeaway for the group:** *negativity dominance in cross-modal conflict is a shared-channel
phenomenon — both cues are read out from the text-stream positions (the assistant-turn preamble); the
image's valence reaches that channel by broadcasting out of the image tokens over depth, and the
negative context wins the competition there.*

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

This control matches the banks on *strength* but not *content*. The stronger version — matching
*content* too, so only the valence word differs — is the token-matched minimal-pair control below
(*Minimal-pair valence control*); it reproduces the effect (override gap +65% vs +64%), ruling out
the "different event semantics" alternative.

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
1. **Image tokens are causally inert *for the text-context delta* (~0%).** Patching them does nothing →
   the negative context does **not** rewrite the image representation. (This is specific to the context
   delta; the image's *own* valence does live in the image tokens and later migrates to the text stream
   — see *Cross-image patching*. The two are not in tension: this experiment varies the text, so image
   tokens hold nothing text-specific to recover.)
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
> **enters the network ~L13 and amplifies ~7× to ~L28**, and the text-context signal is **carried by the
> text token stream — specifically the assistant-turn-boundary tokens (`<end_of_turn><start_of_turn>model`)
> adjacent to the read-out, not BOS; the image tokens are causally inert *for that context delta***. Both
> cues are ultimately read out from a **shared text-stream channel**: the image's own valence originates
> in the image tokens and **broadcasts into those same text-stream positions over depth** (cross-image
> patching), so the positive image loses not because its valence is trapped but because the negative
> context **dominates the competition in the shared channel** the read-out reads.

## Cross-image patching — where does the image's own valence live? (T1.2, Gemma-3-4B)
The same-image experiment (§4) shows the image tokens are inert *for the text-context delta*. The
mirror question — where the **image's own valence** lives — needs the complementary design
(`stage_f_cross_patching.py`): **donor = a positive-valence image, recipient = a negative-valence
image, holding the context fixed (neutral)**. Because the context and prompt are identical, the two
runs have byte-identical `input_ids` (only `pixel_values` differ), so *every* position — including the
context tokens — is 1:1 patchable. We overwrite the recipient's `resid_post` over a layer band for one
token group and measure recovery toward the donor image's read-out:
`recovery(G) = (patched − neg_img) / (pos_img − neg_img)`, 60 donor/recipient pairs.

Behavioral-valence recovery across three depth bands (the metric valid at every band — see the probe
caveat below):

| patch band | `image` tokens | `text_all` (non-image) | `all` (sanity) |
|---|---:|---:|---:|
| **0–12** (early) | **80%** | 10% | 100% |
| **13–17** (mid) | 66% | 65% | 91% |
| **18–28** (late) | **9%** | **68%** | 79% |

- **Visual valence migrates image → text over depth.** Early, it is recoverable *only* from the image
  tokens (80% vs 10%); by the late band the image tokens no longer carry it (9%) and the text-stream
  positions do (68%). The probe read-out corroborates for the bands where it is valid: patching image
  tokens fully restores the L18 read-out early/mid (115% / 125%).
- **So the image's valence is not trapped** — it broadcasts into the same text-stream channel that
  carries the text context. This *corrects* the earlier inference ("valence never propagates out of the
  image tokens") and is consistent with the balanced/integrated dominance regression (β_img/β_txt ≈
  1.19) and the L18 attention pattern (the last token reads mostly text/template positions).
- **Probe caveat (baked into the tool).** The probe tap is `attn_out` L18, computed from L17 — upstream
  of any `resid_post` at L≥18 — so probe recovery is invariant-by-construction for the 18–28 band
  (identically 0 for every group, including `all`). Only behavioral valence is meaningful there;
  `stage_f_cross_patching._probe_valid` gates this and the runner prints a warning.

**Confirmatory run (`--context negative`) — inconclusive by design, but informative.** Repeating with
the negative context held constant restates negativity dominance *behaviorally*: a negative caption
collapses the image-driven valence gap from **0.87 (neutral) to 0.08** — a positive image (−0.85) and
a negative image (−0.93) read nearly identically once the caption is negative, i.e. the text has all
but erased the image's behavioral contribution. The probe read-out at the mid band still shows the
image tokens encode visual valence under competition (**85%, CI [51,150]**). But *because* the
behavioral gap is ~0.08, the late-band valence-recovery denominator is near-zero and its CIs blow up
(text_all 93% **[15, 212]**), so this run **cannot localize whether the image still broadcasts to the
text stream under competition**. The neutral-context migration result stands as the primary finding;
broadcast-under-competition remains open — and may be intrinsically hard to measure behaviorally,
precisely because dominance collapses the signal being localized. Reproduce:
```bash
python -m src.experiments.stage_f_cross_patching                     # neutral ctx, band 13-17
python -m src.experiments.stage_f_cross_patching --layers 0-12       # early band
python -m src.experiments.stage_f_cross_patching --layers 18-28      # late band (valence only)
python -m src.experiments.stage_f_cross_patching --context negative  # confirmatory: broadcast under conflict?
python -m src.experiments.stage_f_cross_patching --reanalyze         # CPU recompute + CIs
```

## Minimal-pair valence control — valence, not event semantics (Gemma-3-4B)
The §2 text-only control matches the banks on *strength* (|neg|/|pos| ≈ 1.06) but not *content*:
"funeral" and "championship" engage different event semantics, so a reviewer can argue the asymmetry
is about semantics, not valence — the strongest surviving alternative. `stage_f_conflict.py
--bank minimal` closes it with 6 minimal pairs that hold sentence structure and wording constant and
flip ONLY the valence-bearing token — verified **token-identical** on the Gemma tokenizer (per-pair
`token_delta = 0`, so the read-out position is byte-identical within each pair): won↔lost, best↔worst,
got↔lost, wonderful↔devastating, celebration↔memorial, joyful↔heartbreaking. Same 150 images, same
metric (`analyze_stage_f --parquet conflict_minimal.parquet`).

| metric | full bank | minimal pairs |
|---|---:|---:|
| override gap (neg>pos img − pos>neg img) | +64% [+55, +72] (84/21) | **+65% [+56, +73] (85/20)** |
| asymmetry vs floor \|drop\|−\|rise\| | +0.265 [+0.095, +0.431] | **+0.330 [+0.151, +0.501]**, p=0.003 |
| head-room-normalized pull (drop vs rise) | 0.63 vs 0.26 | **0.92 vs 0.23** |
| text-only \|neg\|/\|pos\| (no image) | 1.06 (MW p=0.155) | 1.25 (MW p=0.120) |
| image-conditioned \|neg\|/\|pos\| | 3.58 | 3.58 |

- **The effect is unchanged** — override gap **+65% vs +64%**, statistically indistinguishable. When the
  only difference between conditions is a single token-matched valence word, negativity dominance holds.
  ⇒ the asymmetry is **valence, not event semantics** — the strongest surviving alternative is ruled out.
  This is now the headline stimulus control; the original-bank comparison moves to the appendix.
- The residual asymmetry is if anything **sharper** under the cleaner control (|drop|−|rise| +0.330,
  head-room pull 0.92 vs 0.23).
- **Cross-modal amplification replicates.** Text-only the minimal banks are ~symmetric (1.25, MW
  p=0.120 — not a significant stimulus asymmetry at n=6 pairs; a couple of positive members read weak
  text-only, e.g. "celebration for a close friend" −0.09), and the image inflates the ratio to **3.58**
  — the asymmetry is created by the image, not the sentences. Honest caveat: the minimal bank's
  text-only ratio (1.25) is slightly above the original bank's (1.06), so we lean on the override /
  asymmetry-vs-floor metrics (which control for it) as primary, not the raw text-only symmetry.

**Strictest test — within-item, within-event paired contrast.** Because a minimal pair holds the
event constant, we can compare the negative and positive member *on the same photo*
(`analyze_stage_f._minimal_pair_asymmetry`): per image and per pair, `|Δneg| − |Δpos|` vs that image's
neutral, on the positive-image group. This is the tightest possible test — and it is **positive but
marginal**: overall **+0.166, bootstrap CI [+0.016, +0.324], Wilcoxon p = 0.052** (n = 62 images × 6
pairs), weaker than the cross-cell asymmetry-vs-floor (+0.330). The per-pair breakdown explains why —
and is the more informative result:

| pair | swap | Δneg | Δpos | \|Δneg\|−\|Δpos\| |
|---|---|---:|---:|---:|
| mp0 | won ↔ lost | −0.547 | **+0.469** | +0.078 |
| mp1 | best ↔ worst | −0.534 | −0.055 | +0.479 |
| mp2 | got ↔ lost | −0.658 | −0.264 | +0.394 |
| mp3 | wonderful ↔ devastating | −0.742 | +0.285 | +0.457 |
| mp4 | celebration ↔ memorial | −0.348 | **+0.453** | −0.105 |
| mp5 | joyful ↔ heartbreaking | −0.413 | +0.215 | +0.198 |

- **Negative framing is reliable; positive framing is fragile.** Every negative member drops the
  read-out hard and consistently (Δneg −0.35 to −0.74). Positive framing's lift is event-dependent:
  strong concrete positives break through even on a positive image (won a championship +0.47, a
  celebration +0.45), while abstract or oddly-parsed positives do not ("best day" −0.06; "got the job
  they *wanted*" −0.26 — the model appears to read "wanted" as unfulfilled longing). The two strong
  positives (mp0, mp4) drag the paired asymmetry down and mp4 even reverses.
- **Refinement, not contradiction.** Within matched events, negativity dominance is carried by the
  **consistency** of negative framing, not by negatives being uniformly larger in magnitude. The
  category-level override gap (+65%) and cross-cell asymmetry (+0.330) remain the primary robust
  results; the within-item paired test is reported as a stricter, honest check (marginal) that sharpens
  the mechanism. (Stimulus note: revise mp1/mp2's positive members in any follow-up — they read weak.)

Reproduce:
```bash
python -m src.experiments.stage_f_conflict  --bank minimal   # base pass → conflict_minimal.parquet
python -m src.experiments.stage_f_text_only --bank minimal   # matched text-only control
python -m src.experiments.analyze_stage_f   --parquet conflict_minimal.parquet   # incl. per-pair table
```

## Multi-model robustness — Qwen3-VL-8B (different architecture)
The behavioral effect replicates on a second, unrelated VLM. `stage_f_qwen.py` runs the base pass +
text-only control on Qwen via raw HuggingFace (no TransformerBridge, no probe — behavioral valence
only), 150 EMOTIC images. It **replicates near-quantitatively:**

| metric | Gemma-3-4B | Qwen3-VL-8B |
|---|---:|---:|
| within-positive-image neg/pos ratio (graded valence) | **3.58×** | **3.64×** |
| text-only raw \|neg\|/\|pos\| (confound control) | 1.06 (symmetric) | 1.00 (symmetric) |
| override rate — neg-ctx overrides positive image | **84%** | **76%** |
| override rate — pos-ctx overrides negative image | 21% | 37% |
| override gap (95% CI, clustered over images) | **+64% [+55, +72]** | **+39% [+30, +49]** |

- **Same magnitude, different architecture.** The within-positive-image negativity ratio lands at
  3.64× vs Gemma's 3.58×, and text-only is symmetric on both (1.00 / 1.06) → **cross-modal on Qwen too**,
  not a stimulus artifact and not a Gemma quirk.
- **The image is integrated, not ignored** (positive-image neutral valence +0.63 — a happy face reads
  positive; Qwen weighs both modalities and negative text wins the conflict).
- **A read-out-regime change, handled.** Qwen is far more confident, so its closed-vocab valence
  **saturates to ±1** and floors negative images — the graded head-room metric is fragile at small n
  (a 20-image smoke looked symmetric; at n=150 it clears 0: |drop|−|rise| = +0.394, CI [+0.249,+0.542]).
  The robust primary metric is therefore a **calibration-free override rate** on the argmax emotion's
  valence category (shared across both models via `analyze_stage_f._flip_override`): negative context
  overrides a positive image **76%** vs positive context overriding a negative image **37%** on Qwen
  (gap +39%, CI [+30, +49]), and **84% vs 21%** on Gemma (gap +64%, CI [+55, +72]) — the same direction
  and both strongly significant, if anything *more* pronounced on the smaller Gemma model.

**The mechanism replicates too — with a twist.** Porting the activation-patching experiment to Qwen
(`stage_f_qwen_patching.py`, raw HF hooks, recovery on behavioral valence, 60 positive images):

| patched group | Gemma-3-4B | Qwen3-VL-8B |
|---|---:|---:|
| **image tokens** | **−1%** | **0%** |
| question | 22% | 12% |
| suffix / turn scaffold (alone) | **65%** | 6% |
| **all aligned text** | 85% | **65%** |

- **Image tokens are causally inert on both** (~0%) — the headline mechanism (*the image tokens do not
  carry the text-context delta; the text stream carries the conflict*) **generalizes across
  architectures.** (The cross-image migration finding is Gemma-only so far; the Qwen ports test the
  context-delta carrier, not where visual valence lives.)
- **The fine-grained locus differs.** Gemma *concentrates* the carrier in the assistant-turn scaffold
  (that ~4-token group alone recovers 65%); Qwen *distributes* it — no single text group recovers much
  (question 12%, suffix 6%), yet all-text recovers 65% (super-additive → a redundancy signature: the
  negative signal is copied across the text positions, so you must patch them all). Same high-level
  mechanism, model-specific concentration.

**Scope note:** the layer-*entry* localization (Gemma's ~L13) is not yet run on Qwen (needs a
probe-free layer lens); the Qwen band was a broad mid+late default. Reproduce (separate
`requirements-qwen.txt` env):
```bash
python -m src.experiments.stage_f_qwen               # base pass (override rate + graded asymmetry)
python -m src.experiments.stage_f_qwen --text-only   # confound control (raw |neg|/|pos| ~ 1.0)
python -m src.experiments.stage_f_qwen_patching      # carrier (image inert; distributed vs Gemma)
```

## Prompt robustness — the effect is not an artifact of one phrasing (Gemma-3-4B)
The most common objection to a behavioral VLM result is *did you cherry-pick the prompt?*
`stage_f_prompts.py` answers it by holding the SCORING fixed (the same calibration-free
argmax-emotion override rate) and the turn scaffold byte-identical, then varying only the user-turn
text — question wording, context placement, and the `Context:` label — across 6 natural rephrasings.
Full context bank (6 pos / 6 neg / 2 neutral), 150 EMOTIC test extremes, **13,500 forwards**:

| variant | phrasing / framing | neg-ctx > pos img | pos-ctx > neg img | dominance gap | 95% CI |
|---|---|---:|---:|---:|---:|
| `v0_original` | *What single emotion is this person feeling?* (base run) | 84% | 19% | **+65%** | [+0.57, +0.73] |
| `v1_howfeel` | *How is this person feeling?* | 82% | 26% | +57% | [+0.47, +0.66] |
| `v2_oneword` | *In one word, what emotion is this person experiencing?* | 76% | 29% | **+46%** | [+0.37, +0.56] |
| `v3_mostlikely` | *What is the most likely emotion of the person in this photo?* | 76% | 27% | +49% | [+0.38, +0.60] |
| `v4_ctx_after` | context placed *after* the question | 84% | 29% | +54% | [+0.45, +0.63] |
| `v5_no_label` | bare context sentence, no `Context:` label | 89% | 16% | **+74%** | [+0.65, +0.81] |

- **The direction never flips and every CI clears 0** — dominance gap **+46% to +74%, mean +58%**.
  The negativity-dominance effect is a property of the conflict, not of one sentence.
- **Anchor check:** `v0_original` reproduces the published base run (84% / 19% here vs 84% / 21%
  reported) → the sweep faithfully re-implements the original pipeline, so the other five rows are
  trustworthy deltas.
- **We did not maximize the effect with our phrasing.** The base prompt (`v0`, +65%) sits mid-pack;
  the *unlabelled* variant (`v5`, +74%) is actually stronger. So the reported number is not the
  best-case cherry-pick.
- **Weakest variant is interpretable, not a threat.** `v2_oneword` (+46%) is the softest — asking for
  "one word" nudges the model toward a terse, image-literal descriptor that slightly blunts text
  override; it still clears 0 comfortably.

Reproduce (Gemma venv):
```bash
python -m src.experiments.stage_f_prompts             # 6 variants x full bank (A100)
python -m src.experiments.stage_f_prompts --reanalyze # per-variant gap + CI + verdict (CPU)
```

## Model-scale robustness — Gemma-3 4B → 12B
`stage_f_scaling.py` (behavioral-only; the 4B probe does not transfer to 12B, so scoring is the
shared calibration-free override rate — comparable to the 4B/Qwen numbers). Same EMOTIC test
extremes, canonical prompt, and full context bank across both sizes; only `--model` changes. 150
images, ran on the A100:

| size | params | neg-ctx overrides pos img | pos-ctx overrides neg img | dominance gap | 95% CI |
|---|---:|---:|---:|---:|---:|
| `gemma-3-4b-it`  | 4.3B  | 84% | 19% | **+65%** | [+0.57, +0.73] |
| `gemma-3-12b-it` | 12.2B | **93%** | **42%** | **+51%** | [+0.42, +0.59] |

- **Anchor check:** 4B reproduces the base run (84% / 19% ≈ published 84% / 21%) → the behavioral-only
  scaling path matches the probe-era pipeline.
- **Dominance PERSISTS at scale.** The 12B gap is **+51%, CI [+0.42, +0.59]** — well clear of 0. A 3×
  parameter increase does not remove the effect.
- **Read the components, not just the gap.** The headline is NOT "the effect weakens": negative
  override *rises* with scale (84% → **93%**). The gap narrows only because *positive* override rises
  faster (19% → **42%**, more than double). ⇒ **the larger model is more text-driven in BOTH
  directions — the image loses influence overall**, and negativity override is pushing into its
  ceiling (93%).
- **Two honest caveats.** (i) The two gap CIs OVERLAP ([0.57,0.73] vs [0.42,0.59] share [0.57,0.59]),
  so the −14% narrowing is **not** clearly significant — do not claim the asymmetry significantly
  decreased; the clearly-significant, large change is pos-override doubling. (ii) With neg-override
  near 100%, the gap metric is mechanically compressed from above, so part of the "narrowing" is a
  ceiling artifact of the metric, not a weakening of the phenomenon.
- **Reportable trend:** negativity dominance holds across 4B→12B, and *overall* text-dominance
  increases with scale (both override rates up; image influence down) — a scaling trend, not a null.

Reproduce (Gemma venv; one model per invocation to avoid OOM):
```bash
python -m src.experiments.stage_f_scaling --model google/gemma-3-4b-it
python -m src.experiments.stage_f_scaling --model google/gemma-3-12b-it
python -m src.experiments.stage_f_scaling --reanalyze   # gap-vs-size table + component deltas + figure
```

## Threats to validity
- **Single seed; layer-entry localization on one model.** All runs seed 0. The behavioral effect now
  holds on two architectures (Gemma-3-4B + Qwen3-VL-8B) AND across scale (4B→12B, see *Model-scale
  robustness*); the image-inert / text-carried mechanism holds on both architectures. What is
  Gemma-4B-only is the *layer-entry* localization (~L13) and the carrier's concentration (Gemma
  concentrates, Qwen distributes) — a probe-free layer lens on Qwen would close it. A 3-seed repeat
  remains the main open robustness item.
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
  carrier is the assistant-turn preamble, not a BOS sink. Scope: the *behavioral* effect is now shown
  prompt-robust across 6 phrasings (see *Prompt robustness*); what remains format-specific is the
  *carrier locus* — this is one prompt template's turn scaffold, and a different chat format could
  relocate where the signal aggregates without changing the behavioral outcome.
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
**real but fragile under conflicting text**, and mechanistically it is because both cues are read out
from a shared text-stream channel (the turn-preamble tokens) — the image's valence reaches it by
broadcasting out of the image tokens, but a conflicting negative context dominates that channel. For
open-ended/ambiguous task contexts (where there is no strong overriding text), we'd expect the image to
matter *more* than it does here.
