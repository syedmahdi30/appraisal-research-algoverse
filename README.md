# Cross-Modal Appraisal Transfer & Multimodal Valence Conflict in VLMs

This repository contains the replication code, dataset loaders, probe training pipelines, activation patching harnesses, and analysis scripts for the paper:
**"When One Word Changes an Image Judgment: A Valence Asymmetry in Vision-Language Models"** (anonymous double-blind submission).

---

## 1. Overview & Findings

1. **Valence Asymmetry in Multimodal Conflict**: When an image and a one-sentence text context carry conflicting emotional cues, negative text overrides positive image judgments significantly more than positive text overrides negative image judgments.
2. **Valence, Not Event Semantics**: 6 token-matched minimal pairs flipping only the valence word (*won ↔ lost*, *wonderful ↔ devastating*, *celebration ↔ memorial*) reproduce the override gap (+65% vs +64%), confirming the effect is driven by valence rather than event semantics.
3. **Shared Text-Stream Carrier**: Same-image activation patching reveals that image tokens are causally inert for the context difference (~0%), with downstream text positions recovering 88–93% of the delta. Cross-image patching shows visual valence migrates from image tokens (early layers, 100%) to text-token positions (late layers, 63%).
4. **Cross-Modal Appraisal Probe Transfer**: Frozen linear probes fit on text appraisal ratings transfer to image activations ($\rho = +0.510$, Polarity AUC $0.912$), and text difference-of-means directions causally steer multimodal outputs under image input (slope $+0.336$).
5. **Multi-Model Evaluation & Scoring Rules**: Evaluated across Gemma-3-4B, Gemma-3-12B, Qwen3-VL-8B, LLaVA-1.5-7B, and LLaVA-NeXT-7B. Teacher-forced complete sequence-sum scoring is used for multi-token emotion words.

---

## 2. Repository Layout

```
├── config/                  # Experiment YAML configs (stages A, C, D, E, F)
├── data/                    # Datasets (raw/ and processed/ parquets)
├── docs/                    # Detailed experimental logs, audits, and runbooks
├── paper/                   # LaTeX manuscript, figures, and bibliography
├── results/                 # Pre-computed evaluation metrics, probe weights, and figures
│   ├── stage_a/             # Text probe weights (probes.npz) & localization metrics
│   ├── stage_c/             # Readout transfer metrics & caption control parquets
│   ├── stage_d/             # Cross-modal steering slopes & evaluation JSONs
│   └── stage_f/             # Conflict parquets, patching JSONs, crossed bootstrap outputs
├── scripts/                 # Utility scripts (packaging, environment check, download)
├── src/
│   ├── bridge/              # TransformerBridge boot and hook utilities
│   ├── data/                # Loaders for crowd-enVENT, EMOTIC, and minimal pairs
│   ├── experiments/         # Stage A-F experiment runners and analysis modules
│   │   └── shared/          # Raw-HF hooks, sequence scoring, and bootstrap reporting
│   └── probes/              # Ridge probe training, evaluation, and unique-effect vectors
└── tests/                   # 134 unit and regression tests
```

---

## 3. Quickstart & Verification

### Installation

```bash
# Clone or extract archive
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run Unit Tests
```bash
pytest tests/  # 134 tests covering labels, probes, sequence scoring, and hooks
```

---

## 4. Reproducing Paper Results & Figures (CPU-Only, Seconds)

The repository includes pre-computed evaluation parquets and metrics in `results/`, allowing full reproduction of all paper tables and figures on a laptop in seconds without needing GPUs or raw image downloads.

```bash
# 1. Cross-model conflict tables, override gaps, and minimal-pair contrasts (§5, §7, Table 1 & Table 5)
python -m src.experiments.analyze_stage_f

# 2. Crossed (image × sentence) bootstrap and unbounded log-odds margins (§5.2, Table 2)
python -m src.experiments.analyze_stage_f_unbounded

# 3. Stage C semipartial caption controls (§4, Table 3)
python -m src.experiments.analyze_stage_c_mechanism

# 4. Stage A text localization summary (§4, Appendix A)
python -m src.experiments.analyze_stage_a
```

---

## 5. End-to-End GPU Reproduction Pipeline

To rerun model forward passes and activations from scratch (GPU required, e.g. NVIDIA A100 40GB or 80GB):

### Step 0: Data Acquisition
* **crowd-enVENT** (Text, public direct download):
  ```bash
  python scripts/download_data.py --dataset crowd_envent
  ```
* **EMOTIC** (Images, gated non-commercial academic agreement):
  1. Submit the academic access request at `s3.sunai.uoc.edu/emotic/download.html`.
  2. Once approved and downloaded (`emotic.zip` and `Annotations.mat`), run:
  ```bash
  python scripts/download_data.py --dataset emotic --archive /path/to/emotic.zip --annotations /path/to/Annotations.mat
  ```

### Step 1: Stage A — Text Probe Training & Steering
```bash
python -m src.experiments.stage_a_text           # Fits Ridge probes -> results/stage_a/probes.npz
python -m src.experiments.stage_a_steering_v2   # Causal diff-of-means text steering
```

### Step 2: Stage C & D — Cross-Modal Transfer & Steering
```bash
python -m src.experiments.stage_c_transfer_hf   # Evaluates frozen text probe on EMOTIC images
python -m src.experiments.stage_c_caption       # Neutral and rich caption controls
python -m src.experiments.stage_d_steering_hf   # Injects text directions under image input
```

### Step 3: Stage F — Modality Conflict & Activation Patching
```bash
# Multimodal conflict evaluation (Gemma, Qwen, LLaVA)
python -m src.experiments.stage_f_conflict --bank minimal    # Minimal-pair valence control
python -m src.experiments.stage_f_text_only --bank minimal   # Unimodal text strength control

# Activation patching
python -m src.experiments.stage_f_patching_hf               # Same-image position patching (Table 3)
python -m src.experiments.stage_f_cross_patching_hf         # Cross-image depth patching (Table 4)
```

---

## 6. License
This codebase is licensed under the [MIT License](LICENSE).
