# Next experiments — paper-strengthening roadmap

Ideation pass after the A→D arc completed (2026-07). Context: team is ~3–4 weeks from writing;
lead (Sneheel) is explicitly pushing for a **novel question**, not just more models. This doc
records the recommendation so it survives a context clear.

## The core opinion
The A→C→D arc is already a complete, strong result. **More models add credibility, not novelty
— which is exactly what the lead is warning against.** Split effort ~20/80: a thin credibility
layer (cheap, overnight) + one sharp new question as the real contribution.

## Models axis (do overnight; NOT the contribution)
- **Different (Qwen-VL) > bigger (Gemma-12B/27B).** A distinct architecture replicating transfer
  + steering is the "not a Gemma quirk" generalization result. A bigger Gemma is expected-if-works.
- **Best framing = a scaling trend** (4B → 12B → Qwen), plotting transfer ρ / steering slope vs
  scale — "it works AND changes with scale" beats one more data point.
- Feasibility: Gemma-3-12B fits A100-40GB bf16 (~24 GB); **27B does not** without quant/multi-GPU.
  Qwen needs the raw-HF-hooks path (repo scaffolds `qwen_verify`, separate env, transformers ≥4.57).
- Thin credibility layer: **≥3 seeds** (the caveat in every stage write-up — cheapest reviewer win)
  + **Qwen verify** (read-out + steering). Skip 27B and a big model zoo.

## Novel questions (ranked) — all reuse existing frozen-direction + Δμ-steering infra
1. **Appraisal-SPECIFIC emotion steering (TOP PICK).** Stage D only moved *valence*. Steer the
   OTHER appraisals to produce theory-predicted *specific emotions*, cross-modally, on one image:
   `+other-responsibility + unpleasant → anger`, `+self-responsibility + unpleasant → guilt/shame`,
   `+suddenness + unpleasant → fear`. If the emotion shifts follow the appraisal→emotion mapping,
   that **causally validates appraisal *theory*** in a VLM (not just "we can move valence"). Most
   distinctive, appraisal-theory-grounded (matches the project framing), ~a day on existing infra.
   This would be "Stage E".
2. **Behavioral consequence of steering (steering-to-safety).** Steering currently moves the
   emotion *word*; does it move *downstream behavior*? Steer "threat/unpleasantness" under an image
   and measure shift in risk assessment / advice / refusal. Ties to the Anthropic desperation paper
   and the team's safety angle; upgrades the claim to "appraisal reps causally shape behavior."
3. **Modality conflict.** Pleasant image + unpleasant text (and vice versa): which modality wins in
   the shared subspace, and can steering arbitrate? Cleanest test of the "*shared* representation"
   claim (maps to the team's Experiment 2 "recognition vs adoption" and Sneheel's conflicting-inputs
   idea).

## Recommendation
Headline stays A→C→D. Add thin robustness (seeds + Qwen) for credibility, and **#1
appraisal-specific emotion steering** as the novel centerpiece. Keep #2 as the safety-flavored
second novel result. Do the model runs overnight because they're free — but don't let them BE the
contribution.

## Other team directions surfaced (context, mostly others' threads)
- Arnav: VLM emotion **source attribution** — 4/6 emotions read from face/body, 2/6 from scenery;
  wants to validate broadly (Azure run crashed, needs rerun). Could be made mechanistic with this
  repo's probes (patch/ablate face-region vs background image tokens). Maps to Experiment 2.
- Charlotte: video **emotional theory-of-mind** via the **MOMENTS** dataset (multi-person scenes,
  track/attribute emotions); newer models (Gemma 4) may encode full video vs frame-by-frame.
- Sneheel: adversarial **image optimization to maximize an emotional state** (e.g. desperation/
  calmness) and measure behavior change — adversarial-image-jailbreak style but targeting emotion.
- Appendix-worthy (Sneheel agreed): **diff-of-means beats probe-direction for steering** (this
  repo's Stage A v2 finding).

## Open Syed TODOs from the 2026-07 meeting
- Expand cross-modal steering to bigger/more models (A100 overnight ok).
- Ideation pass on next directions → share in Slack (this doc).
- Add the **cross-modal steering (Stage D) write-up to the shared team doc** (repo docs done;
  the shared doc still needs it).
- Monday check-in: **Mon Jul 20, 3:00 PM ET** (Sneheel to confirm).
