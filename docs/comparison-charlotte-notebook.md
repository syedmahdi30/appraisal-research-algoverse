# Comparison — Charlotte's prelim notebook vs this pipeline

Compares `emotic_tlens_prelim_07122026_1.ipynb` (Charlotte / `charlotte9`) against this repo's
Stage A→D pipeline. Same hypothesis (cross-modal appraisal transfer, Gemma-3-4B via
TransformerBridge, EMOTIC + crowd-enVENT), but different methodology and — critically —
**opposite headline results on the central transfer test**. This doc records why, because the
team is reconciling it (C. Li is updating the notebook to match the Q&A prompt format here).

## Head-to-head

| | Charlotte's notebook | This pipeline (A→C→D) |
|---|---|---|
| Scope | one broad prelim sweep | staged: text gate → read-out → causal |
| Read-out tap | `hook_resid_post` only | `hook_attn_out` (MHSA) L18, after sweeping all 3 taps |
| Probe direction | fits probes **on images** (VAD) + a fresh text probe for H1 | **frozen** text probes (Stage A), never re-fit on images |
| Transfer metric | Spearman (text-probe→image) | Spearman + polarity AUC (scale-invariant) + 100-draw null |
| Prompts | text = **bare sentence**; image = "What emotion…" | **parallel emotion-question** prompt on BOTH sides |
| Verbalization control | none | neutral + rich caption baseline + semipartial |
| Causation | listed as "next step" | Stage D steering done |
| Data | EMOTIC **train**, 2,000 persons, 1 seed, Tandon `mat2py` (224²) | EMOTIC **test**, up to 7,280, original-res |
| Text side | not verified in-domain | Stage A gate (r²=0.64) + causal steering |

## What Charlotte did that this pipeline does NOT (her breadth)
- All three VAD dims: valence R²=0.198 (L17), **arousal 0.241 (L33)**, dominance 0.062 (hard).
- 26-category emotion probe, macro-AUROC 0.659 @ L20.
- Position ablation: last-token (0.198) > mean-pool (0.176) — we assumed last-token, she showed it.
- **In-domain image valence probe** (fit on image acts) — a complementary result this repo lacks.

## What this pipeline does that Charlotte's does NOT (depth/rigor)
- Text-side replication gate first (verifies the probe works in-domain + causally before transfer).
- Frozen-probe discipline (text→image only; never re-fit on images).
- Scale-invariant transfer metric + a real random-direction null with a p-value.
- Verbalization ruled out (caption baseline + semipartial).
- Causal steering (Stage D). Prompt alignment across modalities. Full test split.

## The results contrast (the important part)
**Agreement:** image activations encode valence, mid-layer. Her image-fit valence probe
R²≈0.198 @ **L17** (≈ρ 0.44); this repo's read-out ρ 0.507 @ **L18**. Both find the signal in
the middle of the network, essentially same layer.

**Disagreement on H1 (does the TEXT-trained direction transfer to images?):**
- Charlotte: **peak ρ = 0.041 @ L5 → NULL.**
- This repo: **ρ = 0.507 → strong**, survives caption controls, confirmed causally (Stage D).

Sharp framing: **her own image-fit probe proves the image acts contain valence (R²≈0.2), yet her
text→image transfer is ~0.04.** So the valence is present; her text *direction* fails to reach
it, while this repo's reaches it. **The disagreement is methodology, not data** — the shared
subspace exists (both show images encode valence); the question is only whether it's reachable
from text, which depends on how the text probe is built/applied.

## Root cause (leading hypothesis): prompt/position mismatch
A linear probe reads one token's residual, and what's in it depends on the task framing.

- **Charlotte's text side has NO emotion question** (bare event sentence → "ready to continue"
  summary), but her **image side has the question** ("what emotion…" → the model actively
  computes the appraisal). So her text probe learns direction **A** (appraisal in an unquestioned
  summary) and applies it at the image's questioned position where valence lives along a
  *different* direction **B**. A·B misaligned → ρ≈0.04.
- **This repo asks the emotion question on BOTH sides** (`TEXT_EMOTION_PROMPT` /
  `IMAGE_EMOTION_PROMPT`), so the text probe learns **B** and applies **B** → transfer works.

The probe direction is NOT a modality-invariant "pleasantness vector"; it's "the direction
pleasantness occupies at this token under this task framing." Cross-modal transfer requires the
two sides to put the feature in the same place — i.e. matched framing. Compounding factor: tap
(`resid_post` vs `attn_out` L18).

## Reconciliation test (to confirm)
Re-run Charlotte's transfer step on her exact images but with (a) the emotion-question prompt on
the text side and (b) `hook_attn_out` L18. If her 0.04 jumps toward ~0.5, the discrepancy is
pinned to prompt/tap — turning a contradiction into a paper-worthy methodological point ("how the
probe must be framed for appraisals to transfer"). Also double-check a likely typo in her
`build_inputs` image branch (`proc(text=[prompt], image,s=[image], …)` looks like it should be
`images=[image]`); the behavioral gate produced sensible emotions, so the effective path worked,
but confirm pixels aren't dropped on the probe path.
