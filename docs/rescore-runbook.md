# Runbook — re-score §6 on raw HF (Colab / A100)

Everything below runs on the Colab runtime with `HF_TOKEN` set and EMOTIC mounted. Nothing here needs
TransformerBridge; that is the point. Order matters only for step 0 and the `--verify-tap` gate.

## 0. Back up the existing artifacts FIRST — and do NOT run `--drive` blind

`scripts/colab_bootstrap.py --drive` **symlinks** `results/` to
`/content/drive/MyDrive/algoverse-appraisal/results`. Two cases, and they differ a lot:

- If last session ran with `--drive`, `results/` is already a symlink and every output went straight
  to Drive. Nothing is at risk.
- If it did not, `results/` is a real directory on the runtime — and `link_drive()` does
  `shutil.rmtree(link)` on any non-symlink `results/` before creating the link. Its comment says
  "replace the empty local dir", but it does not check that the dir is empty.
  **Running `--drive` now would delete last session's Stage C/D re-verification.**

So check first, in a notebook cell:

```python
import os, pathlib
p = pathlib.Path("results")
print("symlink:", p.is_symlink(), "->", os.readlink(p) if p.is_symlink() else "(real dir)")
print(sorted(x.name for x in p.iterdir()) if p.exists() else "MISSING")
```

If it is **already a symlink**: you are fine, skip to step 1 (and use `--skip-deps --drive` freely).

If it is a **real directory**: copy it out before running the bootstrap with `--drive` at all.

```bash
DEST=/content/drive/MyDrive/algoverse-results-2026-08-22
mkdir -p "$DEST"
cp -r results/* "$DEST"/
ls -R "$DEST" | head -50
```

This machine has nothing newer than **Aug 18** — `results/stage_c/` and `results/stage_d/` are empty
locally — so if that runtime is recycled, the Stage C and Stage D re-verification is gone.

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

The two published pairs are **(pos 0, neg 2)** championship/funeral and **(pos 4, neg 0)**
wonderful/devastating — *not* (1, 0). Only Pair 2 is artifact-backed; Pair 1's run was overwritten by
the old fixed-path clobbering and survives only in `docs/stage-f-mechanism.md`.

```bash
python -m src.experiments.stage_f_patching_hf                         # Pair 1 (pos 0 / neg 2)
python -m src.experiments.stage_f_patching_hf --pos-idx 4 --neg-idx 0 # Pair 2 (pos 4 / neg 0)
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

Only needed if `results/` is a real directory rather than a Drive symlink:

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

---

# Round 2 — closing the last bridge-measured numbers in §6.1 and §7.4

After the §6.2/§6.3 re-scores, three numbers in the paper still came from the superseded stack: the
layerwise onset, the attention analysis, and the conflict steering slope. The paper discloses the
stack problem, so each of these is something a reviewer can pull on.

## 5. Layerwise localization (§6.1)

60 positive-group images × 14 contexts = 840 forwards. The raw projections grow with residual-stream
norm, so the paper's numbers come from the scale-free re-analysis, not from the runner.

```bash
python -m src.experiments.stage_f_layerwise_hf
python -m src.experiments.analyze_stage_f_layerwise --parquet layerwise_hf.parquet
```

The analyzer now takes `--parquet` and writes a stem-matched `layerwise_hf_normalized.json`, so it
cannot clobber the original analysis.

Published reference (bridge): entry near **L13** (|d| = 0.21 against a 0.125 noise floor), amplifying
**7.0×** to a peak |d| = **1.47** at **L28**. Watch the entry layer, the peak layer, and the ratio —
the interpretive claim is "enters mid, amplifies late", so the ratio matters more than either endpoint.

## 6. Conflict steering slope (§7.4)

150 incongruent cells × 7 betas = 1,050 forwards. Verify the direction first — a wrong residual site
does not error, it just steers with a vector that means nothing.

```bash
python -m src.experiments.stage_f_arbitration_hf --verify-dir   # expect rel err < 5%
python -m src.experiments.stage_f_arbitration_hf
```

Published reference (bridge): behavioral-valence slope **+0.215**, reported as 65% of the no-conflict
slope. Note the denominator has already moved — Stage D's no-conflict slope is **+0.336** on raw HF,
not +0.329 — so the reported *fraction* changes even if the slope itself reproduces. The runner prints
the fraction against the corrected denominator.

The probe slope should come out at ~0. That is not a bug: the probe tap (`post_attention_layernorm`,
inside layer 18) is upstream of the `resid_post` injection at the same layer, so it is invariant to
steering by construction. It is recorded to demonstrate that, not to score.

## 7. Text-only matched-set control on Qwen (the panel's shared CRITICAL)

The largest untested alternative explanation in the paper: the matched-set result may reflect the
sentences being imbalanced in isolation rather than anything cross-modal. Appendix B currently
reports a text-only imbalance of **1.25** for the matched set — worse than the varied set's 1.06 —
but that figure comes from the **retracted Gemma bridge run** (`text_only_minimal.parquet`, Aug 18).
No text-only matched-set measurement exists for Qwen at all.

15 forwards, no images, seconds of GPU. `--bank` now reaches `run_text_only`, so this writes to its
own key and cannot overwrite the full-bank control.

```bash
# the varied-set control, if it has not been run on this checkpoint
python -m src.experiments.stage_f_token_budget --model Qwen/Qwen3-VL-8B-Instruct --text-only

# the one that is actually missing
python -m src.experiments.stage_f_token_budget --model Qwen/Qwen3-VL-8B-Instruct \
    --text-only --bank minimal

python -m src.experiments.stage_f_token_budget --aggregate   # CPU; pairs each run with its own bank
```

Writes `results/stage_f/text_only_qwen3-vl-8b-instruct_minimal{.parquet,_metrics.json}`. The runner
compares against `conflict_qwen3-vl-8b-instruct_minimal_metrics.json` — the matched-set base run,
not the varied-set one. Read `reference_ratio` (the raw |neg|/|pos| ratio) and the `per_base_run`
verdict.

What each outcome means:

- **`STIMULUS confound (ratios match)`** — the text-only ratio lands near the image-conditioned
  ratio. The matched-set asymmetry is a property of the six sentences, not of the conflict. This is
  the outcome that costs the paper its strongest result, and it must be reported as such.
- **`CROSS-MODAL amplification`** — the image inflates the ratio well past the text-only reference.
  The control passes and the within-item contrast survives.
- **`image dampens (reversed)`** — report it; do not fold it into either verdict above.

Report the text-only mirror contrast beside the conflict one in §5 either way. Note that a ratio is
not the mirror contrast, and the paper already warns (§3, last paragraph) that ratios can mislead on
their own — LLaVA reaches 3.05× purely through the score's bounds. Read this control as a check on
the stimulus set, not as a second effect size.

## Not ported: the attention analysis (§6.1, second paragraph)

The $88\%$ / $3.5\%$ attention shares and the $6\%$ knockout are the one piece still on the old stack.
Reading attention patterns on raw HF needs `attn_implementation="eager"` plus `output_attentions=True`,
and the knockout needs the layer-18 attention mask rewritten through a forward pre-hook rather than a
pattern edit. That is a real port, not a translation. Options, in order of cost: leave the paragraph
with its current numbers and a stated caveat; drop the paragraph (the patching result in §6.2
supersedes its conclusion anyway); or port it.
