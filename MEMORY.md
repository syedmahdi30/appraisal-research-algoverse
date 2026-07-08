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
