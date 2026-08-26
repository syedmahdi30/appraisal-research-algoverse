# Stage F robustness plan — next experiments before submission

Planning doc, 2026-08-17. Maps every candidate robustness test to (a) the paper claim it protects, (b) the reviewer objection it kills, (c) compute cost, and (d) what changes in the paper if it fails. Ordered by priority within tiers; tiers ordered by value per GPU-hour. Current state: 6-prompt sweep, Qwen3-VL replication, and 4B→12B scaling are DONE and in the draft; the items below are what remains.

**Compute context.** Colab A100-40GB, credit-constrained (Arnav already ran out once). Tier 0 is mostly CPU re-analysis, run it first while credits recover. Every behavioral pass reuses the existing `stage_f_*` harness; cost estimates below assume the base-pass benchmark of 2,250 forwards ≈ one short A100 session.

---

## Tier 0 — statistical hygiene (CPU + <1 GPU-hour, do this week)

### T0.1 Cluster all headline uncertainty by distinct image
- **What:** Recompute every headline CI and test (asymmetry index, MW test, override-rate CIs) treating the 121 distinct images, not the 150 person annotations, as the independent unit. The override-rate CIs already cluster; the bootstrap on the graded asymmetry may not.
- **Protects:** every number in §5. **Objection killed:** "repeated people from the same photo are not independent" — the exact kind of detail a NeurIPS statistics reviewer leads with.
- **Cost:** CPU only (`analyze_stage_f` re-run on the existing parquet).
- **If it fails:** CIs widen slightly; the asymmetry CI has enough margin (lower bound +0.095) that this is very unlikely to flip anything. If it does, that is essential to know now, not in rebuttal.

### T0.2 Multi-seed repeat of the direction/probe estimation
- **What:** The forwards are deterministic; what varies with seed is upstream, the probe fit and the Δμ direction estimation (and the crowd-enVENT split). Refit probe + Δμ at 3 seeds, re-score the existing Stage F activations, re-run arbitration with each seed's Δμ.
- **Protects:** the frozen-readout results (§4, §5.1) and the arbitration slope (§7.4). **Objection killed:** "single seed" — currently the first item in the paper's own limitations.
- **Cost:** probe refits are cheap (cached activations); arbitration re-runs are 1,050 forwards per seed ≈ 2 extra short sessions. The behavioral asymmetry (§5.2) is seed-independent and needs no repeat, say so explicitly in the paper.
- **If it fails:** a seed-sensitive arbitration slope would downgrade §7.4 to "steering arbitrates for the reference direction"; the behavioral asymmetry stands regardless.

### T0.3 More patching context pairs
- **What:** Extend the patching experiment from 2 donor/recipient pairs to all 6 diagonal pairs (p0/n0 … p5/n5), same 60 images, same [13,17] band.
- **Protects:** the carrier decomposition (§6.2, Table 3). **Objection killed:** "two context pairs is anecdotal" — and it directly guards against the single-context pitfall that already burned the Stage F pilot once.
- **Cost:** 4 additional pairs × the existing per-pair budget ≈ 1 A100 session.
- **If it fails:** if some pairs show a different carrier split, report the range honestly; the image-inert result (the load-bearing part) is very unlikely to move.

---

## Tier 1 — the two experiments reviewers will demand (1 GPU-day each, highest priority after T0)

### T1.1 Matched lexical minimal pairs
- **What:** New 12-sentence bank of near-minimal pairs holding event structure and wording constant, flipping only valence: won/lost the championship; best/worst day of their life; got/lost the job they wanted; wonderful/devastating news; celebration/funeral held for a close friend; moments after a joyful/painful goodbye. Rerun the base pass (150 images × ~13 conditions) + the text-only control with the new bank.
- **Protects:** the core cross-modal claim (§5.3). **Objection killed:** the strongest surviving alternative explanation: "funeral and championship engage different semantic machinery, so the asymmetry is about event semantics, not valence." The current text-only control shows matched *strength*, not matched *content*. This was flagged as the #1 addition by every literature review; two of the current sentences (p4/n0) are already near-minimal, which is encouraging.
- **Cost:** ~2,000 base forwards + 13 text-only forwards ≈ 1 A100 session. Analysis reuses `analyze_stage_f` unchanged.
- **If it succeeds:** promote to the main text as the headline control, demote the current bank comparison to the appendix. **If the asymmetry shrinks substantially:** that is itself a publishable finding (part of the effect is semantic, part valence); the paper's framing shifts from "valence asymmetry" to "affective-content asymmetry" but survives.

### T1.2 Cross-image patching (where does visual valence travel?)
- **What:** Complement the same-image design: donor = positive-image run, recipient = negative-image run, same (neutral or positive) context; patch image tokens, then text groups, same recovery metric. Optionally sweep the band ([13,17] vs [18,28]) since visual valence may transfer earlier (cf. information-flow literature).
- **Protects:** the carefully-worded image-token claim (§6.2 conclusion 1) and the Discussion's conflict-type taxonomy. **Objection killed:** "you never actually located visual valence, only the context delta" — currently pre-registered in Limitations. This also turns the taxonomy paragraph (parametric conflicts localize to image tokens, external-text conflicts to text stream) from citation-reconciliation into something partially tested in our own setup.
- **Cost:** ~1 A100 session (mirrors the existing patching budget).
- **Expected result and payoff:** image-token patching should now recover a large share (visual valence lives in image-token states before transfer). If it does, the paper gains a clean two-sided mechanism figure: context delta rides the text stream; image valence rides the image tokens and is read out from them, which is exactly why conflicting text can commandeer the answer-preparation states without touching the image representation. If image tokens recover little even here, the current wording already survives, but the discussion of where image valence lives must weaken.

---

## Tier 2 — the architecture axis (Sneheel's priority; 1–2 GPU-days total)

### T2.1 LLaVA-family model (third unification architecture) — the important one
- **What:** Behavioral pass + text-only control + coarse patching on a LLaVA-style VLM (LLaVA-1.5-7B for the classic linear-projector design, or LLaVA-OneVision/NeXT for a maintained variant). Port of the raw-HF-hooks harness (`stage_f_qwen.py` is the template; swap processor and token-span logic for LLaVA's unpooled patch tokens).
- **Why this over newer models:** Sneheel's point, and it is the right one: Gemma 3 (SigLIP pooled to 256 soft tokens) and Qwen3-VL (native dynamic-resolution ViT) are two unification designs; LLaVA (frozen ViT features through a linear/MLP projector, unpooled) is the third major family. Three families with the same behavioral asymmetry and image-inert patching turns "two architectures" into "the dominant VLM unification designs," which is a categorically stronger generality claim than a third checkpoint from a family already covered.
- **Cost:** LLaVA-7B fits the A100 in bf16. Base pass + text-only ≈ 1 session; patching ≈ 1 more.
- **If it fails:** a LLaVA model that does NOT show the asymmetry would be a genuinely interesting boundary condition (pooling/projector design modulates the effect) and gets reported as such, not hidden. The claim scopes to "chat-tuned VLMs with X property" and the paper still stands on Gemma + Qwen.

### T2.2 Newer Gemma generation (behavioral-only)
- **What:** Run the override-rate pass on a Gemma 4 multimodal checkpoint (raw HF; no TransformerBridge adapter exists yet, but behavioral scoring does not need one). Note: the meeting notes say "Gemma 3.5 or 3.8 ... artifact of Gemma 4," which looks like a transcription garble; the sensible reading is "a newer Gemma generation, to check the results are not an artifact of the Gemma 3 family." Confirm intent with Sneheel before burning credits.
- **Protects:** generality across model generations. **Objection killed:** "Gemma 3 training-data quirk."
- **Cost:** 1 session (2,250 forwards, behavioral only).
- **Priority:** below T2.1, per Sneheel's own ranking. If credits force a choice, LLaVA wins.

### T2.3 (Skip unless free) Second scaling curve on Qwen (2B/4B/8B)
- Nice symmetry with the Gemma 4B→12B curve, but low marginal value; the scaling story is already supported. Skip under credit constraints.

---

## Tier 3 — mechanism depth (stretch, only if T0–T2 are done)

- **T3.1 Probe-free layer lens on Qwen:** closes the one mechanism gap that is currently Gemma-only (the ~L13 entry localization). Moderate effort (needs a probe-free divergence metric per layer, e.g. paired logit-lens on the valence labels).
- **T3.2 Steering arbitration on a second model:** would extend the causal §7.4 across architectures, but requires building the whole probe/Δμ pipeline for Qwen. Expensive; explicitly out of scope for this deadline.
- **T3.3 Head-level localization at the Gemma turn boundary:** which attention heads write the context signal into the suffix tokens. Scientifically attractive, not needed for any current claim.

---

## What is already done (do not respend)
| Meeting item | Status |
|---|---|
| Prompt rephrasing robustness | Done, 6 variants, gap +46% to +74%, in draft §7.2 |
| Scaling test | Done, 4B→12B, in draft §7.3 |
| Different unification architecture #2 | Done, Qwen3-VL behavioral + coarse patching, in draft §7.1 |
| Lit search, paper skeleton | Done (three deep-research artifacts; full draft in `paper/`) |

## Suggested sequencing against the deadline
1. **Now (CPU, zero credits):** T0.1 clustering re-analysis; draft the minimal-pair sentence bank (writing sentences costs nothing) and sanity-check them text-only next time a GPU is up (13 forwards).
2. **First GPU session back:** T1.1 minimal-pair base pass (biggest claim-protection per credit).
3. **Second session:** T1.2 cross-image patching, then T0.3 extra pairs in the same session if time remains.
4. **Third/fourth sessions:** T2.1 LLaVA (base + text-only, then patching).
5. **If credits and calendar allow:** T0.2 seed repeat, T2.2 Gemma 4.
6. **Paper integration as results land:** each finished tier updates a specific pre-registered limitation, which reviewers read as a paper that anticipated and closed its own gaps.

## Decision points to confirm with Sneheel
- Target venue: the meeting mentions workshop targeting (Interp Science, InterpAgent, VLM workshops). A workshop deadline compresses this plan to T0 + T1.1 only; a main-conference cycle fits everything through T2.1. The tier order is chosen so cutting from the bottom is always safe.
- The "Gemma 3.5/3.8 vs artifact of Gemma 4" garble (T2.2): confirm which direction he means before running.
- Whether the LLaVA variant should be classic 1.5 (cleanest architectural contrast, older) or OneVision (maintained, but architecturally closer to modern designs).
