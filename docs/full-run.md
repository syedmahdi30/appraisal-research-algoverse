# Full-run runbook — Stages E & F

How to reproduce the full (non-pilot) Stage E and Stage F runs on an A100, what to watch in the
output, and the remaining robustness steps. Pilot results and their verdicts live in
`docs/stage-e-results.md` and `docs/stage-f-results.md`.

## 0. Session bootstrap (Colab, per docs/colab.md)
In a notebook cell, mount Drive and set the token, then:
```bash
python scripts/colab_bootstrap.py --drive     # mounts Drive; stages data/ + results/ from it
python scripts/smoke_test.py                   # bf16 boot + is_multimodal assert + one forward
git fetch && git checkout stage-ef-pilot       # (or main once merged)
```
Preconditions the stages assume (all from prior stages, mirrored via Drive):
- `results/stage_a/probes.npz` + `metrics.json` (frozen probes, critical layer 18).
- `data/processed/emotic_{train,val,test}.parquet` (image paths must match the clone path — see the
  caveat at the bottom).
- crowd-enVENT raw text under `data/raw/corpus/` (Stage E direction building only).

## 1. Stage E — appraisal-specific & compositional emotion steering
Config: `config/stage_e.yaml` (`n_images: 150`, `matched_norm_control: true`). Directions/profiles are
text-only and cheap; the two combo runs are the cost (~2 h each at ~0.33 s/forward).

```bash
# once (text-only; skip if results/stage_e/directions.npz already present from the pilot)
python -m src.experiments.stage_e_directions
python -m src.experiments.analyze_appraisal_profiles

# raw (entangled) directions — 150 img, 25 arms × 6 β  (~2 h)
python -m src.experiments.stage_e_combo
python -m src.experiments.analyze_stage_e

# valence-decorrelated directions — the compositionality test  (~2 h)
python -m src.experiments.stage_e_combo   --decorrelate valence
python -m src.experiments.analyze_stage_e --decorrelate valence
```
Outputs (raw unsuffixed, decorrelated `_valence`): `results/stage_e/{combo_pilot*.parquet,
combo_pilot_metrics*.json, combo_analysis*.json}`, `results/figures/stage_e_combo_pilot*.png`.

**What to check:**
- Startup line `decorrelate=valence: |cos with valence axis| … 0.79 → 0.00` — de-correlation took effect.
- `COMPOSITIONAL synthesis` row and the `*` markers — arms where the target is the combo argmax **and**
  beats both matched-norm singles. The pilot headline is **A1 anger** composing only in the
  decorrelated run (`combo beats matched: True`, other^⊥→joy / unpleasant→fear).
- `fr` column (lexical-frequency control) — a genuine target stays rank ≤2 after removing the
  baseline-frequency trend. Cross-check the `R` random arm as the perturbation null.
- Verdict tiers: SIGNAL (≥3 compositional) / PARTIAL COMPOSITIONAL (1–2) / SINGLE-APPRAISAL / NULL.

## 2. Stage F — modality conflict & steering arbitration
Config: `config/stage_f.yaml` (`n_images: 150`, `full_context_bank: true`).

```bash
python -m src.experiments.stage_f_conflict              # base: 150 img × 15 conditions = 2,250 fwd (~13 min)
python -m src.experiments.stage_f_conflict --arbitrate  # 150 incongruent cells × 7 β = 1,050 fwd (~6 min)
python -m src.experiments.analyze_stage_f               # OLS dominance + context effect + breakdown + arbitration
```
Outputs: `results/stage_f/{conflict_pilot.parquet, conflict_metrics.json, arbitration_pilot.parquet,
arbitration_metrics.json, conflict_analysis.json}`, `results/figures/stage_f_conflict.png`.

**What to check:**
- Dominance β_img / β_txt and the ratio (0.8–1.25 ⇒ balanced/integrated). Both must be sign-correct.
- Context effect **vs the neutral context** (no-context is excluded as non-comparable). Expect the
  ceiling/floor pattern: context pulls hardest *against* the image's saturated valence.
- Arbitration `behavioral valence slope` within ±50 % of Stage D's ~0.33; the probe slope is ~0 by
  construction (upstream of the injection).
- The `RAW per-condition means` breakdown is the diagnostic for any baseline oddity (is `none` low, or
  a single context an outlier, and is it in the probe or only behavioral valence?).

## 3. Remaining robustness (not yet automated)
The full runs above are **single-seed** (seed 0) — the one caveat flagged in both result docs.
- **3-seed repeat.** Re-run each stage at seeds 0/1/2 and report cross-seed mean ± spread on the
  headline metrics (E: A1 compositional win-rate & slope; F: dominance ratios & arbitration slope).
  *Note:* the runners currently read a single `seed` from config and write fixed output filenames, so
  seeds would clobber each other — add seed-suffixed outputs (or archive between runs) before looping.
  This is a small code change, deferred until the single-seed result is accepted.
- **Positive-emotion composition.** Pride/surprise fail (joy attractor); a different handle (not the
  valence-anchored combo) is needed — stated future work, not a full-run parameter.
- **Cross-model / layer.** Out of scope for this branch (scope guard: layer 18, Gemma-3-4B only).

## Known caveat — EMOTIC image paths
`data/processed/emotic_*.parquet` stores **absolute Colab paths**
(`/content/appraisal-research-algoverse/…`). If the repo is cloned to a different path, `Image.open`
silently skips every image (watch the `n_skipped` count). Either clone to that exact path or
regenerate the parquet with `emotic.convert_mat_to_parquet` after mounting.
