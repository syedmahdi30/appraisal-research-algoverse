# MEMORY — operational notes

Validated, durable operational findings for this repo. Keep entries short and actionable.
Broad/stable onboarding rules live in `CLAUDE.md`; this file is for evolving lessons.
Use the `/update-memory` skill to add entries.

## Environment / versions
- Verified: Gemma is **gated** — set `HF_TOKEN` and accept the license on the Hub before boot,
  or loading errors link to the agreement page.
- Verified: **bf16 required** for Gemma 3 — boot with `dtype=torch.bfloat16`.
- Known issue: TransformerBridge version metadata is inconsistent (PyPI 3.4.0 vs GitHub tag 3.3.0).
  The Gemma3 multimodal hotfix is **v3.2.1** (PR #1295) — pin `>=3.2.1`, ideally newest 3.x.
- Next-time rule: after any TransformerBridge/transformers upgrade, re-run `scripts/smoke_test.py`
  and confirm `bridge.cfg.is_multimodal is True`.
- Known issue: Qwen3-VL needs `transformers>=4.57.0`, which conflicts with the Gemma 5.x line —
  keep Qwen in a SEPARATE venv (`requirements-qwen.txt`).

## TransformerBridge quirks
- Avoid: `start_at_layer` — raises `NotImplementedError` in the bridge. Only `stop_at_layer` works.
  Plan layer sweeps around full forward passes.
- Known issue: the SAME alias hook strings (`hook_resid_post`, `hook_attn_out`, `hook_mlp_out`)
  exist on BOTH vision-encoder layers and LM blocks. Always qualify by path: `blocks.{i}...` = LM.
- Verified: default bridge numerics = raw HF (no LayerNorm folding). Call
  `bridge.enable_compatibility_mode()` only if porting HookedTransformer-era folded-weight code.
- Avoid: full-sequence caching of all hooks — it blows memory. Use `names_filter` + last-token only.

## Tokenization / labels
- Avoid: assuming emotion labels are single-token. Verify per model with `bridge.tokenizer.encode(" "+w)`
  (keep the leading space for the SentencePiece ▁ prefix).
- Verified (Gemma-3-4b-it, 2026-07): ALL 13 crowd-enVENT emotion labels are SINGLE-token — closed-vocab
  scoring needs no first-subtoken/summed-logprob workaround. Re-verify if the model changes.

## Stage A localization RESULT (verified 2026-07, A100, seed 0)
- Verified: Tak-style localization REPLICATES on Gemma-3-4b-it. All 6 appraisals peak mid-network
  (MHSA `hook_attn_out`, layers 17-21) with val r2 far above the shuffled baseline (~0):
  pleasantness L18 r2=0.64, unpleasantness L18 0.60, self_responsblt L21 0.44, suddenness L17 0.32,
  other_responsblt L19 0.31, predict_event L17 0.24. critical_layer=18. n_train=4320 n_val=1080.
- Frozen probes saved to results/stage_a/probes.npz (unique-effect steering vectors ready for Stage C).
- This clears the LOCALIZATION half of the go/no-go gate.
- Analyzer: `python -m src.experiments.analyze_stage_a` -> table + results/figures/stage_a_localization.png.

## Stage A steering RESULT: SUPPORTS (via diff-of-means, verified 2026-07)
- KEY METHOD FINDING: the read-out direction is NOT the steering direction. v1 (probe/ridge
  unique-effect vector, scaled as a fraction of residual norm) NEVER beat a random control at any
  beta (swamped <0.02, control moves as much 0.02-0.08, chaos >0.1). v2 (difference-of-means
  Δμ_a = mean(act|rating high) − mean(act|rating low), steered by beta*Δμ in natural units) gives a
  CLEAN causal effect at a SINGLE layer (L18 resid_post): pleasantness slope +0.084 (monotonic, +sign),
  unpleasantness -0.074 (monotonic, -sign), random control ~10x smaller and flat (slope -0.007).
- Verdict: BOTH halves of the Stage A gate pass — read-out replicates AND diff-of-means steering is
  causal (theory-predicted signs, control flat). Run: `python -m src.experiments.stage_a_steering_v2`.
- Avoid: steering with probe weights or brute-forcing beta as a fraction of residual norm (v1 dead end).
  Use diff-of-means at natural scale; optional multi-layer band strengthens but single layer already works.
- Stage C: PRIMARY test is read-out transfer (frozen probes on image acts). Cross-modal steering (Stage D)
  should reuse the diff-of-means recipe, NOT the probe-direction/residual-fraction approach.

## Stage C read-out RESULT: SUPPORTS transfer (verified 2026-07, EMOTIC test n=1000, seed 0)
- Frozen text pleasantness probe (L18 hook_attn_out) applied UNCHANGED to image-conditioned last-token
  activations tracks EMOTIC continuous valence: Spearman +0.482 (Pearson +0.499, n=1000); unpleasantness
  MIRRORS at -0.442. Polarity AUC (shared-7 pos vs neg) 0.936 / 0.080 — but only n=58 single-label images.
  Retention = |image rho| / |text rho| ~0.60-0.63 (keeps ~60% of the text effect crossing to images).
- Beats a 100-draw random-direction null at p=0.010 (empirical floor). BUT the null is WIDE: mean|rho|=0.131,
  p95=0.319, max=0.372 — Gemma activations are strongly ANISOTROPIC, so random dirs correlate ~0.37 with
  valence. Probe (0.48) clears all 100 but the margin over the best random dir is modest. Report the null
  distribution, never a single control number.
- METRIC LESSON: probe is 1-5, EMOTIC valence is 1-10 → raw r2 is scale-confounded and reads as false null.
  Use SCALE-INVARIANT metrics (Spearman/Pearson + polarity AUC). data.emotic has NO appraisal columns;
  only pleasantness/unpleasantness have an image-side anchor (valence) — the other 4 appraisals can only be
  tested via Stage D steering.
- CODE LESSON: image forwards MUST be wrapped in torch.no_grad() — SigLIP does 4096-patch eager attention
  ([1,16,4096,4096] fp32 ~4GB/layer); without no_grad the retained graph OOMs a 40GB A100 on image 1.
- OPEN QUESTION (mechanism): read-out transfer alone cannot separate SHARED-GEOMETRY from VERBALIZATION-
  MEDIATED (model internally captions the image). NEXT = caption baseline (neutral caption -> text pipeline
  -> pleasantness read-out vs valence); if caption rho >= image rho the signal is plausibly verbal.
- Run: `python -m src.experiments.stage_c_transfer` (config/stage_c.yaml, n_images=1000, n_random=100).

## Stage C caption baseline RESULT: transfer is LARGELY VERBALIZATION-MEDIATED (verified 2026-07, n=1000)
- Test: generate a NEUTRAL caption per image ("Describe this image in one sentence.", greedy, meta
  preamble stripped, 64 new tokens), run it through the TEXT pipeline, apply the SAME frozen probe,
  correlate with valence. `python -m src.experiments.stage_c_caption` (--preview N first).
- Result: pleasantness caption rho=+0.379 vs image rho=+0.482 (unpleasantness -0.356 vs -0.442). A
  neutral caption alone reproduces ~79% of the correlation / ~62% of the explained variance of the
  direct image read-out. Both appraisals mirror correctly in the caption pathway too.
- INTERPRETATION (honest, per analysis-rules): the cross-modal read-out transfer is SUBSTANTIALLY
  verbalization-mediated — most of the signal is available in a plain verbal description (Gemma
  volunteers affect words like "stressed"/"gloomy" even from a neutral prompt). The image>caption gap
  (~0.10 rho) is an UPPER BOUND on any non-verbal contribution and is CONFOUNDED by caption lossiness
  (greedy one-sentence bottleneck) — so it is NOT clean evidence of shared non-verbal appraisal geometry.
  Do NOT claim shared multimodal geometry from this.
- LESSON: caption run cost 1h43m for 1000 images (~6 s/img: generate + text forward). PERSIST the
  captions + per-image predictions (parquet) so refinements (semipartial correlation, richer-caption
  robustness) don't require re-generating. Gemma greedy caption = "Here's a one-sentence description
  of the image:\n\n<caption>" — strip the preamble (split first blank line).
- NEXT options: (a) accept "largely verbalization-mediated" and write up Stage C; (b) semipartial corr
  (image read-out vs valence controlling for caption read-out) to quantify the unique non-verbal residual
  — needs persisted captions; (c) richer-caption robustness (does caption rho rise to meet image rho?);
  (d) scale image read-out to full 7280 test split for the reported number.

## Smoke test (verified 2026-07 on A100)
- Verified: `scripts/smoke_test.py` passes — Gemma 3 boots via bridge, `cfg.is_multimodal=True`,
  `n_layers=34`, one forward pass caches 102 tensors (3 taps × 34 layers), all three LM taps fire.
  Stack: transformers 5.12.1, transformer-lens 3.x, torch 2.11+cu128, Python 3.12.
- Known issue (harmless): boot logs many `Hook alias ... on SiglipVisionEncoderLayerBridge did not
  resolve` warnings — the bridge registers LM-style aliases on the SigLIP vision tower. We only probe
  LM `blocks.{i}...`, so these are expected; `boot_gemma` now filters them.

## Compute / Colab workflow
- Decision: runs on a Colab **A100** (40 GB) via the VS Code Colab extension; local files
  sync to the runtime, so NO GitHub clone is needed. Gemma-3-4B uses ~8 GB — ample headroom.
- Known issue: Colab runtimes are **ephemeral** — deps, weights, and outputs wiped each session.
  Canonical command: `!python scripts/colab_bootstrap.py --drive` at session start (installs
  deps, loads HF_TOKEN from Colab Secrets, symlinks data/ + results/ to Drive). See docs/colab.md.
- Next-time rule: keep Colab's CUDA-matched torch — requirements pin only `torch>=2.2` (lower
  bound) so pip won't swap it and break CUDA.
- Verified: HF_TOKEN comes from Colab Secrets (🔑 icon, name `HF_TOKEN`, notebook access ON) —
  never paste tokens into code or chat.
- Avoid: calling `google.colab.drive.mount` or `userdata.get` inside a `!python` subprocess —
  they need the live kernel and crash with `'NoneType' object has no attribute 'kernel'`. Do both
  in a NOTEBOOK CELL first; `os.environ['HF_TOKEN']` and the `/content/drive` mount are then
  inherited by every later `!python`. `colab_bootstrap.py` only does deps + symlinks + env check.
- Verified: EMOTIC loader is correct — conversion yields train 23706 / val 3334 / test 7280 =
  34,320 persons (the canonical EMOTIC total). Repo lives on GitHub; clone + `%cd` on Colab so
  `import src` resolves (running a lone script from /content fails: No module named 'src').

## Data access
- Verified: crowd-enVENT is a free direct download: `romanklinger.de/data-sets/crowd-enVent2023.zip`.
- Known issue: EMOTIC requires a signed non-commercial access form
  (`s3.sunai.uoc.edu/emotic/download.html`) — submit early, approval latency is critical-path.

## EMOTIC Annotations.mat structure (verified against the real file)
- Images (`emotic.zip`) and annotations (`Annotations.mat`) are SEPARATE downloads. mat is the
  classic MATLAB format — read with `scipy.io.loadmat(squeeze_me=True, struct_as_record=False)`.
- Top-level keys: `train` (17077 imgs), `val` (2088), `test`. Each = array of image structs
  (`filename`, `folder`, `image_size`, `original_database`, `person[]`).
- Image `folder` is RELATIVE, e.g. `'mscoco/images'`, `'emodb_small/images'`, `'framesdb/images'`.
  emotic.zip nests everything under a top `emotic/`, so images_root = `data/raw/emotic/emotic`.
- Avoid: assuming one category convention. **train**: `annotations_categories` is a STRUCT with
  `.categories`. **val/test**: `combined_categories` is a BARE string array (no `.categories`),
  and `annotations_*` are per-annotator arrays. `src/data/emotic.py::_person_categories` handles both.
- Continuous VAD: prefer `combined_continuous` (struct .valence/.arousal/.dominance, val/test);
  fall back to `annotations_continuous` (train). Scale 1-10. See `_person_continuous`.
- Canonical command: `python scripts/download_data.py --dataset emotic --archive emotic.zip
  --annotations Annotations.mat` → writes `data/processed/emotic_{split}.parquet`.
