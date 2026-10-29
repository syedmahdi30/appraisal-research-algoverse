# Does Negative Text Pull Harder? A Cross-Modal Valence Asymmetry Depends on How It Is Read

Code, stimuli, and released results for the VLM4RWD 2026 workshop paper (camera-ready source:
`paper/neurips_2026.tex`).

## What the paper finds

We pair EMOTIC photographs with one-sentence contexts that flip one valence word
(*won ↔ lost*) and measure how each sentence shifts a vision-language model's emotion judgment
against that photograph's own neutral-context response.

1. **On a bounded score, negative text looks much stronger.** In Qwen3-VL-8B, with
   P(positive) − P(negative) as the readout, the mirror contrast between the two conflict directions
   is +0.496, with an interval excluding zero when word pairs are resampled.
2. **Most of that is the measurement.** On the log-odds of the same probabilities the contrast
   reverses sign without clearing zero. It ranges from +0.24 to +0.76 depending on which neutral
   sentence defines the baseline. A larger within-photograph contrast reverses on negative
   photographs, so it tracks conflict versus agreement, not valence (Table 2 of the paper).
3. **Across four models, the readout and the label scoring decide size and sometimes sign.**
   Scoring only the first token of a multi-token emotion label manufactures a null in LLaVA-1.5-7B
   and reverses the categorical ordering in LLaVA-NeXT-7B.
4. **In Gemma-3-4B**, downstream text positions restore 88–93% of a context difference under
   activation patching; a frozen text-trained valence probe transfers to image-conditioned states
   (ρ = +0.510).

## Layout

```
config/            experiment configs; stage_f_matched_photos.csv pins the published photo set
src/data/          crowd-enVENT and EMOTIC loaders, context and neutral sentence banks
src/experiments/   runners (GPU) and analyses (CPU) for every stage
src/probes/        ridge probes
release/results/   the released results bundle (model outputs; see "Results bundle")
release/MANIFEST.json   every released file: sha256, whether and where the paper cites it
scripts/           data download, results packaging/rehydration, paper build
paper/             LaTeX source, tables, figures
tests/             pytest suite
```

## Install and test

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # Qwen3-VL runs use requirements-qwen.txt
pytest tests/
```

## Results bundle

`release/results/` holds every forward-pass output and metrics file behind the paper, plus
superseded runs the paper mentions (marked `"cited_in_paper": false` in `release/MANIFEST.json`).
EMOTIC's own annotations are **not** redistributed: wherever a file carried an EMOTIC rating, that
column is replaced by `<column>_emotic_row`, an index into `data/processed/emotic_test.parquet`.

```bash
# With EMOTIC access (build data/processed/ first, see "Data"): restores the original columns
python scripts/rehydrate_results.py

# Without EMOTIC: copies the bundle as-is. Everything below except the probe-transfer numbers runs.
python scripts/rehydrate_results.py --no-emotic
```

## Reproducing the paper's numbers (CPU, no images needed)

Each command reads `results/` and was checked against a fresh rehydrated copy of the bundle.

| Paper | Command |
|---|---|
| Table 2 (every readout × contrast, matched set) and the corrected categorical gaps of Tables 3, 4, 9 | `python -m src.experiments.analyze_matched_readouts` |
| Table 9 (four models: mirror contrast, unbounded margin, uncorrected gap) | `python -m src.experiments.analyze_stage_f_unbounded` |
| Balanced label subsets (Sec. 4.3) | `python -m src.experiments.analyze_label_balance` |
| Tables 7–8 (patching intervals) | `python -m src.experiments.patching_intervals` |
| Wording robustness (Table 6) | `python -m src.experiments.stage_f_controls --reanalyze controls_frame_qwen` and `... controls_question_qwen` |
| Generated answers (appendix): flip rates, override gap, strict and synonym parses | `python -m src.experiments.stage_f_controls --reanalyze controls_generate_qwen` |
| Person grounding (App. A) | `python -m src.experiments.stage_f_controls --reanalyze controls_person_box_qwen` and `... controls_person_crop_qwen` |
| Probe localization (Table 11) | `python -m src.experiments.analyze_stage_a` |

`analyze_matched_readouts` also writes `paper/tables/readouts.tex`, so the main-text table cannot
drift from its source.

## Data

* **crowd-enVENT** (text, public): `python scripts/download_data.py --dataset crowd-envent`
* **EMOTIC** (images, gated non-commercial agreement): request access from the EMOTIC authors, then
  `python scripts/download_data.py --dataset emotic --archive emotic.zip --annotations Annotations.mat`.
  MSCOCO and ADE20K images are distributed by their own projects.

## Re-running forward passes (GPU)

Image-conditioned results use reference HuggingFace implementations in bf16 on one 40 GB A100.
Runbooks: `docs/reviewer-controls-runbook.md` (Qwen3-VL controls, including the camera-ready
neutral-bank, equal-room and generation runs). Principal entry points:

```bash
python -m src.experiments.stage_f_qwen                         # Qwen3-VL-8B varied set
python -m src.experiments.stage_f_controls --equal-room        # camera-ready runs; see runbook
python -m src.experiments.stage_f_controls --neutral-bank
python -m src.experiments.stage_f_controls --generate
python -m src.experiments.stage_f_patching_hf                  # Gemma-3-4B same-image patching
python -m src.experiments.stage_f_cross_patching_hf            # Gemma-3-4B cross-image patching
```

The published 150 annotations are pinned by `config/stage_f_matched_photos.csv`. Re-deriving them
with `select_extreme_rows` is not reproducible across library versions, because EMOTIC's valence
ties at the cut are broken by an unstable sort.

## License

Code: MIT. Released results inherit the terms of the models and datasets they derive from; EMOTIC
is for non-commercial research use under its own agreement.
