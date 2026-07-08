# Experiment 1 — Cross-Modal Appraisal Transfer (design)

## Question
Do **appraisal directions** learned from a language model's *text* activations transfer to
the same model's *image-conditioned* activations — as read-out probes and as causal steering
vectors? Positive transfer would be evidence for a shared, modality-agnostic appraisal geometry
rather than verbalization-mediated emotion inference.

## Models
- **Primary (bridge-native): `google/gemma-3-4b-it`.** SigLIP vision encoder → multi-modal
  projector → Gemma3 text backbone → lm_head. Everything (read-out + steering) runs through
  `TransformerBridge`. See `docs/models-gemma3.md`.
- **Verification (fallback): a Qwen VLM via raw HF hooks.** Qwen2.5-VL / Qwen3-VL are NOT in
  TransformerBridge; use raw PyTorch forward hooks in a separate env. Verification, not a dependency.

## Datasets
- **crowd-enVENT** (text) — 6,600 event descriptions, 21 appraisal dims (1–5 Likert), 13 emotion labels.
- **EMOTIC** (images) — 23,571 images / ~34,320 people, 26 emotion categories + continuous VAD (1–10).
See `docs/datasets.md`.

## Appraisal targets
Six primary appraisal dimensions (crowd-enVENT names): **Pleasantness, Unpleasantness,
Suddenness, Event Predictability, Own Responsibility (self-agency), Others' Responsibility
(other-agency)**. Probe method and metrics in `docs/probes.md`.

## Four-stage plan

### Stage A — Text replication gate (crowd-enVENT, LM backbone via bridge)
Reproduce Tak et al. inside Gemma's LM: closed-vocab emotion accuracy, correctness filtering
(prediction == human label), layer-wise probe r² for each appraisal at `hook_resid_post`,
`hook_attn_out`, `hook_mlp_out`. Learn unique-effect steering vectors and test steering at
β ∈ {±1,±2,±4}.
- **GO/NO-GO gate:** appraisal probes must recover the Tak-style **mid-layer, MHSA-dominant**
  localization, AND text steering must shift emotion outputs in the theory-predicted direction.
  If this fails, STOP — cross-modal transfer would not be interpretable.

### Stage B — Image task setup (EMOTIC)
Map EMOTIC's 26 categories onto a shared ~7-emotion space aligned with crowd-enVENT; single-label
filtering; build closed-vocab image prompts; confirm Gemma answers sensibly. (Setup, not a claim.)

### Stage C — Cross-modal read-out
Apply the **frozen text-trained** probes to image-conditioned last-token activations. Validate the
pleasantness probe against EMOTIC's continuous **valence** (1–10) as a ground-truth proxy. Report the
**transfer gap** = text-probe r² on text vs. on image activations.

### Stage D — Cross-modal steering
Inject unit steering vectors `z_a` under image input at the Stage-A critical layers; measure
emotion-label distribution shift vs. appraisal-theory predictions; sweep layers and β ∈ {±1,±2,±4}.

## Controls (all stages)
- Norm-matched **random** direction vectors.
- **Caption baseline** — describe the image, then run the text pipeline on the caption.
- **Non-emotional isomorphic** task (object naming).
- **Shuffled-label** probes.

## Decision rules
- If cross-modal read-out r² collapses to chance while the caption baseline succeeds → transfer is
  **verbalization-mediated**; pivot to the caption-mediated hypothesis.
- If the pleasantness probe fails to track EMOTIC valence but other appraisals transfer → treat
  valence as a domain-shift artifact and re-anchor on a different appraisal.

## Timeline
Weeks 1–2: Stage A as a hard gate. In parallel: submit EMOTIC access form immediately; download
crowd-enVENT now. Only after Stage A passes: Stage B/C/D.
