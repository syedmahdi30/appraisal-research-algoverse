# Cross-Modal Appraisal Transfer

Do appraisal directions learned inside a language model's text activations transfer to
the same model's **image-conditioned** activations? This repo replicates text-side
appraisal probing/steering (Tak et al. 2025 style) on **Gemma-3-4B** via
**TransformerBridge**, then tests whether the *frozen* text-derived directions read out
and steer emotion behaviour on images.

- **Primary model:** `google/gemma-3-4b-it` (bridge-native, SigLIP + Gemma 3 backbone).
- **Verification model:** a Qwen VLM via raw HF hooks (NOT bridge-supported — separate env).
- **Datasets:** crowd-enVENT (text, free download) and EMOTIC (images, gated form).

See `docs/experiment-1.md` for the full design and the four-stage plan (A→D).

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="hf_..."          # Gemma is gated — accept the license on the Hub first

python scripts/check_environment.py   # versions, CUDA, HF_TOKEN
python scripts/smoke_test.py          # boots Gemma, one forward pass, hooks fire
```

## Layout

```
config/     experiment configs (stage_a.yaml, stage_c.yaml)
docs/       experiment design, model notes, dataset notes, probe method
src/
  bridge/     bridge boot, hook/cache utilities, multimodal input construction
  data/       crowd-enVENT + EMOTIC loaders, shared emotion label space
  probes/     ridge probe training, unique-effect vectors, evaluation
  experiments/ Stage A (text), Stage C (transfer), Qwen verification
scripts/    check_environment.py, smoke_test.py, download_data.py
tests/      unit tests for the GPU-free logic (labels, probe math, hooks)
results/    per-stage metrics, probes, figures (git-ignored payloads)
data/       raw/ and processed/ (git-ignored)
```

## Stage gate

Stage A (text replication) is a **hard go/no-go gate**: appraisal probes must recover
the Tak-style mid-layer / MHSA-dominant localization AND text steering at β ∈ {±1,±2,±4}
must shift emotion outputs in the theory-predicted direction. Do not start Stage C
(cross-modal) until Stage A passes. See `docs/experiment-1.md`.

## Workflow skills

`.claude/skills/` provides `/experiment-setup`, `/run-stage-a`, `/run-stage-c`,
`/analyze-results`, and `/update-memory`. Operational lessons live in `MEMORY.md`.
