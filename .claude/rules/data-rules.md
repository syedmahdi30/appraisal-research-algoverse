# Rules — data preprocessing and splits

- **Use predefined splits.** crowd-enVENT ships train/val/test (4,320 / 1,080 / 1,200) — do not
  re-split. EMOTIC has official partitions — use them. Never mix data across the Stage A/C boundary.
- **Verify tokenization before logit scoring.** Emotion/appraisal labels are NOT assumed single-token.
  Run `src/data/labels.py::verify_label_tokenization(tokenizer)` per model; keep the leading space
  (SentencePiece ▁). For multi-token labels use first-subtoken OR summed log-prob — identically across
  Gemma and Qwen.
- **Correctness filtering.** For probe training, keep only examples where the model's closed-vocab
  prediction equals the human emotion label. Record how many examples survive filtering.
- **Label mapping is lossy and must be surfaced.** The EMOTIC-26 → shared-7 mapping
  (`EMOTIC_TO_SHARED`) loses information; single-label filtering discards multilabel rows. Report both
  as first-class caveats in Stage C, never silently.
- **Gated data: never fabricate access.** EMOTIC requires a signed form. If the archive isn't present,
  report the exact blocker and the manual step — do not invent a download.
- **Raw vs processed.** Downloads land in `data/raw/`; loaders emit normalized tables to
  `data/processed/` (parquet). Both trees are git-ignored. Loaders must be deterministic and
  re-runnable without re-downloading.
- **Ratings scale.** crowd-enVENT appraisals are 1–5; EMOTIC VAD is 1–10. Keep scales explicit in
  column names/metadata; do not silently rescale when comparing (e.g. pleasantness probe vs valence).
