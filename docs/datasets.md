# Datasets

## crowd-enVENT (text) — PRIMARY for Stage A
- Troiano, Oberländer & Klinger 2023, *Computational Linguistics* 49(1):1–72, DOI 10.1162/coli_a_00461.
- **Free direct download:** `https://www.romanklinger.de/data-sets/crowd-enVent2023.zip`
- Reference code: GitHub `sarnthil/crowd-enVent-modeling`.
- **6,600** experiencer event descriptions, each annotated with appraisal dimensions on a
  **1–5 Likert** scale. The six experiment targets map to these exact `generation.tsv` columns:
  `pleasantness`, `unpleasantness`, `suddenness`, `predict_event` (Event Predictability),
  `self_responsblt` (Own Responsibility), `other_responsblt` (Others' Responsibility).
- **13 emotion labels** (`emotion` column): anger, boredom, disgust, fear, guilt, joy, pride,
  relief, sadness, shame, surprise, trust, "no emotion" (neutral). Text is `generated_text`.
- Tak et al. sampling: 500 examples per emotion, EXCEPT guilt and shame (250 each).
- **Actual layout (verified):** the zip extracts to `data/raw/corpus/crowd-enVent_generation.tsv`
  (6600×61; the self-annotated experiencer data we use) and `crowd-enVent_validation.tsv` (reader
  re-annotations, unused). `data/raw/predictions/` holds paper model outputs (unused).
- **SPLIT CAVEAT:** the download does NOT ship separate train/val/test files. `src/data/crowd_envent.py`
  derives a deterministic emotion-stratified split at the canonical sizes (4,320 / 1,080 / 1,200) with
  a fixed seed. Reproducible, but not guaranteed identical to the paper's partition — report as a caveat.

## EMOTIC (images) — Stage B/C
- Kosti, Alvarez, Recasens, Lapedriza; CVPR-W 2017 / TPAMI 2019.
- **GATED:** signed non-commercial research/education form at `https://s3.sunai.uoc.edu/emotic/download.html`
  (~3.2 GB). Some images belong to MSCOCO / ADE20K. **Submit early** — approval latency is critical-path.
- Reference code: `rkosti/emotic`, `Tandon-A/emotic`. Annotations ship as `Annotations.mat`,
  converted with `mat2py.py --data_dir ... --generate_npy`.
- **26 discrete multilabel emotion categories** + continuous **Valence/Arousal/Dominance (1–10)**;
  per-person bounding boxes.
- Image count is reported inconsistently: 18,313 images / 23,788 people (2017 CVPR-W) vs
  23,571 images / 34,320 people (later works). **Cite the exact release you download.**
- Expected local path: `data/raw/emotic/` → processed to `data/processed/emotic_{split}.parquet`.

## CAER-S (backup)
- 70,000 stills from 79 TV shows, annotated for Ekman's 6 + neutral. Fallback image source if EMOTIC
  access is delayed.

## Shared emotion label space (Stage B mapping)
EMOTIC's 26 categories are mapped onto a shared ~7-emotion space aligned with crowd-enVENT's 13
labels; single-label filtering is applied. The mapping is **lossy** and must be a first-class caveat
in Stage C results. The canonical mapping lives in `src/data/labels.py` (`EMOTIC_TO_SHARED`).

## Splits
- crowd-enVENT: use the predefined train/val/test splits — do not re-split.
- EMOTIC: use the dataset's official train/val/test partitions; never mix across the Stage A/C boundary.
