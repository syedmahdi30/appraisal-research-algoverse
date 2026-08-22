# Runbook — re-score §6 on raw HF (Colab / A100)

Everything below runs on the Colab runtime with `HF_TOKEN` set and EMOTIC mounted. Nothing here needs
TransformerBridge; that is the point. Order matters only for step 0 and the `--verify-tap` gate.

## 0. Back up the existing artifacts FIRST — before anything else touches `results/`

`results/` is git-ignored, so last session's raw-HF outputs exist only on that runtime. This machine
has nothing newer than **Aug 18** — `results/stage_c/` and `results/stage_d/` are empty locally.
If the runtime is recycled, the Stage C and Stage D re-verification is gone and has to be re-run.

```bash
DEST=/content/drive/MyDrive/algoverse-results-2026-08-22
mkdir -p "$DEST"
cp -r results/stage_a results/stage_c results/stage_d results/stage_e results/stage_f "$DEST"/
ls -R "$DEST" | head -50
```

## 1. Verify the read-out tap before spending a sweep on it

A wrong tap does not raise. It silently reports that no token group carries anything, which reads
exactly like "the mechanism result did not replicate". The Stage C port measured the difference:
`self_attn` scores r² −6.26 where `post_attention_layernorm` scores +0.634 against Stage A's 0.641.

```bash
python -m src.experiments.stage_f_patching_hf --verify-tap
```

Expect `verdict: OK` — a positive mean probe gap (positive-context minus negative-context) and
segmentation OK on every image. If it says `SUSPECT`, stop and try the other candidates before
running anything else.

## 2. Same-image patching — the carrier result (paper Table 3)

Published (bridge): assistant-turn boundary **65% / 57%**, image tokens ~0%, all-aligned-text 85%/87%.
Expected to survive: a ratio of differences, probe-scored at L18 where bridge-vs-HF cosine is 0.980.

```bash
python -m src.experiments.stage_f_patching_hf                        # pair 1 (pos 0 / neg 2)
python -m src.experiments.stage_f_patching_hf --pos-idx 1 --neg-idx 0 # pair 2
```

The run prints raw HF beside the published bridge number per token group. Outputs:
`results/stage_f/patching_hf.parquet`, `patching_hf_metrics.json`.
**Pair 2 overwrites pair 1's files** — copy them aside between runs, the same fixed-path clobbering
that destroyed three published numbers before:

```bash
cd results/stage_f && cp patching_hf.parquet patching_hf_pair1.parquet \
  && cp patching_hf_metrics.json patching_hf_pair1_metrics.json && cd -
```

## 3. Cross-image patching — the riskier one (paper Table 4)

Published (bridge, behavioural valence): image tokens **80% / 66% / 9%**, non-image text
**10% / 65% / 68%** across the three bands.

This is the re-score that could genuinely fail. The 18–28 band has no valid probe column — the probe
tap is upstream of the injection, so probe recovery is invariant-by-construction there and the number
is scored on behavioural valence, which is precisely what the bridge corrupts. That band carries the
paper's "visual valence moves from image tokens into text states over depth" claim.

```bash
python -m src.experiments.stage_f_cross_patching_hf --layers 0-12
cp results/stage_f/cross_patching_hf_metrics.json results/stage_f/cross_patching_hf_0-12.json

python -m src.experiments.stage_f_cross_patching_hf --layers 13-17
cp results/stage_f/cross_patching_hf_metrics.json results/stage_f/cross_patching_hf_13-17.json

python -m src.experiments.stage_f_cross_patching_hf --layers 18-28
cp results/stage_f/cross_patching_hf_metrics.json results/stage_f/cross_patching_hf_18-28.json
```

Read the **probe and valence columns side by side**. For 0–12 and 13–17 both are valid, so:

- probe and valence agree → the behavioural readout is trustworthy in this design, and the 18–28
  band's valence-only number can be believed too.
- probe and valence diverge → the behavioural readout is carrying bridge damage, and §6.3 cannot be
  re-stated from the 18–28 band at all.

That comparison is the actual deliverable of step 3, not any single percentage.

## 4. Back up again

```bash
cp -r results/stage_f "$DEST"/stage_f-after-rescore
```

## What each outcome means for the paper

| outcome | consequence |
|---|---|
| both survive | §6 stands as written; it becomes the paper's strongest surviving Gemma content, and the rewrite is a retraction of §5/§7 only |
| same-image survives, cross-image does not | keep the carrier result, delete §6.3 and the image-to-text-over-depth claim; the "concrete site where the two cues compete" argument goes with it |
| same-image does not survive | there is no mechanism section, and the paper is a measurement-validity report |

Not yet ported, and needed only if §6 survives: §6.1 layerwise onset and the attention analysis
(`stage_f_layerwise`, `stage_f_attribution`). Both are probe-projected differences, so they sit in the
safe category — but so did these, and they were still worth measuring.
