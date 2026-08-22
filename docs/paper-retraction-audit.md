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
