# Setup

## Primary environment (Gemma 3 + TransformerBridge)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="hf_..."     # Gemma is gated — accept the license on the Hub first
```

Validate:
```bash
python scripts/check_environment.py   # Python/torch/CUDA/transformers/bridge versions + HF_TOKEN
python scripts/smoke_test.py          # boots Gemma, one forward pass, confirms hooks fire
```
`smoke_test.py` requires GPU + HF access; `check_environment.py` runs anywhere.

## Qwen verification environment (separate!)
Qwen3-VL needs `transformers>=4.57.0`, which conflicts with the 5.x line the Gemma bridge uses.
Use a distinct virtualenv:
```bash
python -m venv venv-qwen && source venv-qwen/bin/activate
pip install -r requirements-qwen.txt
```

## Data
```bash
python scripts/download_data.py --dataset crowd-envent   # free direct download
# EMOTIC is gated: submit the form at https://s3.sunai.uoc.edu/emotic/download.html,
# then place the archive under data/raw/emotic/ and re-run:
python scripts/download_data.py --dataset emotic --archive /path/to/emotic.zip
```

## Compute budget
- Gemma 3 4B in bf16 ≈ 8 GB weights; with `names_filter`-limited, last-token-only caching it fits a
  single 24 GB GPU (RTX 3090/4090/A5000). 3 hooks × 34 layers × last-token ≈ a few MB/example.
- **Avoid full-sequence caching of all hooks** — that is what blows memory.
- Qwen3-VL-8B at 4-bit fits a single H100 / A100-40GB.

## Version verdict discipline
Classify the environment as **Ready / Ready-with-warnings / Blocked**. Never mark Ready unless a real
forward pass succeeds and `bridge.cfg.is_multimodal is True`. Re-run the smoke test after any upgrade.
