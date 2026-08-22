# The TransformerBridge multimodal bug, and what survives it

_Session of 2026-08-21/22. Every number below was measured this session; commit hashes are on `main`._

## TL;DR

TransformerBridge computes a **different forward pass from raw HuggingFace for Gemma-3 on
byte-identical inputs**. Internal representations are sound (residual-stream cosine ≥ 0.978 through
layer 32); the **output logits are corrupted** (6.15 nats, argmax flips, image separation AUC 0.788
vs 0.982). Gemma is the only model whose published numbers used the bridge.

Consequence: results scored on the **L18 probe** survive; results scored on **behaviour** do not.
The paper's headline Gemma override gap falls from **+65% to +12% (CI touching zero)**. Stage A and
Stage C both reproduce exactly. Qwen and both LLaVAs were always raw HF and are untouched.

---

## 1. How it was found

Not by looking for it. The chain was:

1. A new runner reproduced **LLaVA-1.5 exactly** (−0.125896, identical CI) but gave Gemma
   **44%/31% (gap +12%)** against a published **84%/19% (gap +65%)**.
2. Prompt ruled out: `--prompt-style legacy` reproduces the published scaffold byte-for-byte
   (only difference: one duplicated `<bos>`) and still returns +12%.
3. The tell was a control nobody had run before — **no-context image separation**:

   | run | stack | separation gap | AUC |
   |---|---|---|---|
   | Gemma published | **TransformerBridge** | +0.374 | **0.788** |
   | Gemma re-run | raw HF | +1.674 | 0.982 |
   | Qwen published | raw HF | +1.558 | 0.987 |
   | LLaVA-1.5 published | raw HF | +1.471 | 0.989 |

   Every raw-HF run across three models sits at ~0.98. The one bridge run sits at 0.79.

## 2. What the bug is — and is not

`diagnose_image_pathway` (commits `62c44a8`, `366fef6`, `30e8684`), Gemma-3-4B:

- `input_ids` **identical** (289 tokens)
- `pixel_values` **identical**, `max|Δ| = 0.000000`
- emotion token ids **identical**
- **text-only**: same argmax, behavioural valence `|Δ| = 0.0011` → the text path is **exact**
- **image-conditioned**: **0/5 argmax agreement**, valence `|Δ|` up to **0.80**, 6.15 nats
- image influence: bridge **17.6** vs HF **14.5** → it does not under-weight the image, it applies
  it *incorrectly*

So: not preprocessing, not the prompt, not tokenisation. The bridge's multimodal **forward** differs.
`transformer_lens.__version__` does not resolve, so the ≥3.2.1 Gemma-3 multimodal hotfix required by
`.claude/rules/bridge-rules.md` cannot be confirmed installed — a plausible proximate cause.

### Where it goes wrong (`--layer-scan`, commits `915726e`, `d57ade0`, `f8c43d1`)

| site | cosine (bridge vs HF) |
|---|---|
| `resid_post` L0–L32 | ≥ 0.978 (0.999+ across L8–L28) |
| `attn_out` L18 (**probe site**) | **0.980** |
| `resid_post` L33 | norm ratio 95.7 — **comparison artefact**, not a bug¹ |

¹ TransformerLens `hook_resid_post` on the last block is *pre* final-layernorm; HF's last
`hidden_states` is *post*-norm. Different tensors by construction.

At 2,560 dimensions, cosine 0.980 implies a probe read-out correlation of ~0.98 between stacks —
exactly why Stage C reproduces. **Representations are sound; the corruption is at the output.**

## 3. Triage

| status | experiments |
|---|---|
| **Intact — text path exact** | Stage A: probes, layer selection, text steering |
| **Intact — verified this session** | Stage C read-out transfer (re-run, see §4) |
| **Intact — never used the bridge** | Qwen, LLaVA-1.5, LLaVA-NeXT (all raw HF) |
| **Intact — verified this session** | Stage D causal steering (re-run, see §4) |
| **Likely intact — DIFFERENTIAL measures** | same-image patching recovery (a ratio of differences, probe-scored at L18); layerwise onset; arbitration slope. *Not yet re-confirmed.* |
| **Dead — ABSOLUTE / categorical measures** | Gemma conflict override rates; cell means; the mirror contrast |

### The organising principle

Stage D reproduced its slopes to within 2–3% **even though the base valence differs between stacks**
(+0.0987 raw HF vs +0.1799 published). That is the key:

* **Differential measurements survive.** Correlations (Stage C), slopes (Stage D), and ratios of
  differences (patching recovery) subtract off the systematic offset the bug introduces.
* **Absolute / categorical measurements die.** An override rate asks "did the argmax cross a
  category boundary", which depends on exactly where the model sits — precisely what the bug moves.

This narrows the damage considerably: it is the conflict override rates specifically, not the
mechanism work.

## 4. Re-runs completed

### Stage C — SURVIVES (`01c8778`, `c98d07a`)

Full test split, 7,280 images, 0 skipped, tap `post_attention_layernorm`:

| metric | published (bridge) | raw HF |
|---|---|---|
| pleasantness ρ vs valence | +0.507 | **+0.510** |
| polarity AUC | 0.898 | **0.912** |
| unpleasantness ρ | −0.448 | **−0.448** |
| p vs random null | 0.010 | 0.010 |
| retention | 0.660 | 0.663 |

**The shared-channel foundation holds.** Cross-modal read-out transfer is not a bridge artefact.

### Stage D — SURVIVES (`a0c93b2`, `ba979d3`)

150 EMOTIC test images, resid_post L18, 1,200 direction examples. Site verified first: Δμ norms
352.314 / 358.223 / 195.938 vs published 352.819 / 358.377 / 194.937 (worst rel err **0.51%**).

| direction | raw HF slope | published |
|---|---|---|
| pleasantness | **+0.3360** | +0.3293 |
| unpleasantness | **−0.3156** | −0.3087 |
| suddenness (specificity) | −0.0776 | −0.0726 |
| random (null) | −0.0350 | −0.0270 |

Ratios: 9.6× the random null (published ~12×), 4.3× the specificity control (published ~4.5×).
**The causal capstone survives** — a text-derived direction injected under image input still shifts
the model's output, which no captioning account explains.

### Tap verification — a trap avoided

The probe was fit on the bridge's `blocks.18.hook_attn_out`. Raw HF equivalent candidates:

| module | probe r² (ref: Stage A 0.641) | spearman |
|---|---|---|
| `self_attn` | **−6.2637** | +0.041 |
| `self_attn.o_proj` | −6.2637 (same tensor) | +0.041 |
| **`post_attention_layernorm`** | **+0.6345** | **+0.775** |

Gemma post-norms the attention output before the residual add, and TransformerLens folds that into
its attn-block output. **Running Stage C on the plausible-looking `self_attn` would have scored
ρ ≈ 0.04 on images and read as "cross-modal transfer failed"** — destroying the paper on a false
negative. Always verify the tap when porting a bridge-fitted probe.

## 5. Other findings this session (independent of the bug)

### Two hypotheses falsified

**The architecture-boundary claim is dead.** LLaVA-NeXT (`llava-v1.6-mistral-7b`, 2147 image tokens,
AUC 0.998) **shows** the asymmetry: 38%/19%, gap **+19% CI [+8,+30]**. Same linear-projector family
as LLaVA-1.5, so "linear-projector design ⇒ no asymmetry" is false. The effect is present in 3 of 4
models; LLaVA-1.5 is a lone outlier, not a design boundary.
*Caveat: NeXT-mistral changes the language backbone; `llava-v1.6-vicuna-7b` is the clean comparison
and has not been run.*

**The visual-token-budget hypothesis is dead.** Qwen resolution sweep, identical weights:

| budget | image tokens | gap | discriminability AUC |
|---|---|---|---|
| 448px | 128 | +42% [+32,+51] | 0.983 |
| 896px | 262 | +37% [+28,+47] | 0.986 |
| 1344px | 262 | +39% [+29,+48] | 0.986 |

Flat across a 2× budget change with image quality held (a clean null, not a masked effect). Killed
independently by a matched-budget cross-model pair: **Gemma 256 tokens → +65%** vs **Qwen 262 tokens
→ +39%**. *Gotcha: `_prep_image` only downscales and EMOTIC photos are ~640px, so `--max-side 896`
and `1344` were both no-ops giving identical runs.*

**Net:** the asymmetry is present in 3 of 4 models with graded strength (+65/+39/+19/−13) and **no
validated explanation for the variation.** Report it that way rather than inventing a third
mechanism. Two ruled-out architectural explanations fit Interp4Discovery's negative-results track.

### Robustness re-analysis (CPU, no GPU — `a8d6b3d`)

**Judge robustness** — 6 pos/neg label partitions, direction never flips:
Gemma +48% to +65% · Qwen +34% to +39% · LLaVA-1.5 −11% to −31%.
*(Fixed a real bug first: `_published_gap_beside` hardcoded `conflict_analysis.json`, so with all
parquets in one directory Qwen and LLaVA were anchored to Gemma's gap.)*

**Crossed (image × sentence) bootstrap vs the published image-only CI:**

| | image-only | crossed |
|---|---|---|
| Gemma mirror contrast | [+0.095, +0.431] | **[−0.013, +0.527]** |
| Minimal-pair mirror contrast | [+0.151, +0.501] | [+0.095, +0.564] |
| LLaVA-1.5 override gap | [−26%, −9%] | **[−50%, +13%]** |

The full-bank graded contrast **stops clearing zero** once sentence variance is propagated; the
minimal-pair bank still clears. LLaVA-1.5's "reversal" becomes a **null**, not a reversal.

**Unbounded log-odds readout:** |drop|/|rise| = **1.80 bounded → 0.76 unbounded** (sign flips).
Negative images sit at valence −0.806 = log-odds −6.34, where probability space compresses the rise.
**The "~1.8× larger" magnitude claim is an artefact of the bounded scale.** The override rate is
categorical and scale-free, and survives all three robustness axes — lead with it, drop "1.8×".

## 6. Methodological cautions

Three of the assistant's own summary statistics were wrong this session, each initially convincing:

1. **`pearson = −0.993` "supports the token budget"** — pooled a flat within-model sweep with one
   distant cross-model point, and silently excluded the two models that break the pattern.
2. **`max|Δ|/RMS` "catastrophic divergence at every layer"** — Gemma's outlier dimensions; a 5%
   change in ONE coordinate of 2,560 gives max/RMS ≈ 2.5 while cosine stays 1.000000.
3. **A 0.99 cosine cut calling 0.980 "disagreement"** — against a 7,280-image direct measurement.

**Pattern: when a summary statistic contradicts a large direct measurement, the statistic is wrong.**
Also: EMOTIC annotates per *person*, so `image_path` recurs — a head slice returns the same photo
repeatedly (this bit the pathway diagnostic's sampling).

## 7. Tooling added

| file | purpose |
|---|---|
| `analyze_stage_f_unbounded.py` | unbounded log-odds readout + crossed bootstrap |
| `stage_f_token_budget.py` | token-budget experiment; per-run output paths; text-only control; `--prompt-style`, `--show-prompt` |
| `diagnose_image_pathway.py` | bridge vs raw HF parity; `--layer-scan` |
| `stage_c_transfer_hf.py` | Stage C on raw HF; `--verify-tap` |
| `stage_d_steering_hf.py` | Stage D on raw HF; `--verify-dirs` |

All new runners use **per-run output paths** keyed by (model, budget, style), fixing the fixed-path
clobbering that previously destroyed three published numbers.

## 8. Next steps

1. ~~Stage D re-run~~ — **done, survives** (+0.336 vs +0.329). See §4.
2. **Re-score same-image patching on raw HF** to confirm the 57–65% turn-boundary result. Expected to
   survive (probe-scored), but confirm rather than assume.
3. **Paper**: retract the +65%, promote Qwen to primary, drop "1.8×" from the abstract, rewrite §7.1
   and the Discussion (the LLaVA boundary claim is falsified), re-word LLaVA-1.5 as a null.
4. `llava-v1.6-vicuna-7b` to close the backbone confound.

> **Artifact risk:** `results/` is git-ignored, so this session's outputs (`metrics_hf.json`,
> `tap_verification_hf.json`, all token-budget runs) exist **only on Colab**. Download or copy them
> somewhere durable before that runtime is recycled.
