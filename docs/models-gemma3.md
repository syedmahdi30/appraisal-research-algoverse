# Gemma 3 4B via TransformerBridge — model notes

## Architecture (verified)
- **34 transformer layers**; hidden size **2560**; head_dim 256; SentencePiece tokenizer
  (shared with Gemma 2), ~262K vocab, byte-level fallback + whitespace preservation + number splitting.
- Adapter structure: `model.vision_tower` (SigLIP) → `model.multi_modal_projector` →
  `model.language_model` (Gemma3TextModel) → `lm_head`. The language model follows the same
  patterns as `Gemma3ArchitectureAdapter`.
- **Attention:** 5:1 local:global interleave (5 local sliding-window layers per global layer).
  Because 4B has 34 layers, its **last layer is a local (not global) attention layer** — a known
  quirk. Local window shortened to 512 tokens for KV-cache efficiency.
- **bf16 required** (`dtype=torch.bfloat16`).

## Vision
- SigLIP encoder at fixed **896×896**, patch size 14 → 64×64 grid = 4,096 patches, then pooled
  ("condensed") into a **fixed 256 soft image tokens per image** (`mm_tokens_per_image=256`).
- Image placeholder token `<start_of_image>` (`boi_token_index=255999`); optional Pan-and-Scan for
  non-square/high-res images. Gemma uses **bidirectional** attention over image tokens.
- The 256 image tokens are spliced in at the `<start_of_image>` placeholder — **never hardcode token
  positions**. Use `input_ids.shape[-1] - 1` for the final prompt token.

## Boot (canonical loader)
```python
import torch
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "google/gemma-3-4b-it", device="cuda", dtype=torch.bfloat16,
)
assert bridge.cfg.is_multimodal   # confirm the multimodal adapter was picked
```
`boot_transformers(model_name, hf_config_overrides=None, device=None, dtype=torch.float32,
tokenizer=None, load_weights=True, trust_remote_code=False, model_class=None, hf_model=None,
n_ctx=None, device_map=None, n_devices=None, max_memory=None)`.

## Hook names (use the legacy aliases)
Three taps Tak et al. probe, on the **language-model** blocks:
- `blocks.{i}.hook_resid_post` — residual/hidden state
- `blocks.{i}.hook_attn_out`   — MHSA output
- `blocks.{i}.hook_mlp_out`    — FFN output

Canonical names are component-path based ending in `.hook_out` (e.g. `blocks.0.attn.q.hook_out`).
Vision-tower hooks live under `model.vision_tower` and reuse the **same alias strings**
(`hook_resid_post`, etc.), disambiguated only by parent path. For this experiment, always cache
**`blocks.{i}...`** (language model) keys. See `src/bridge/hooks.py`.

## Read-out and steering
- Read-out: `run_with_cache(input_ids, pixel_values=..., names_filter=keep)`; `pixel_values` flows
  through `**kwargs → forward` (only accepted when `cfg.is_multimodal is True`).
- Steering: `run_with_hooks(input_ids, pixel_values=..., fwd_hooks=[(name, hook_fn)])`.
- During **generation**: `generate(..., pixel_values=...)` passes `pixel_values` only on the first
  step (vision encoder runs once). Use the hook-context manager so hooks persist across decode steps;
  steer the last position each step once the KV cache advances.

## Pitfalls (verified)
- `start_at_layer` raises `NotImplementedError` in the bridge — only `stop_at_layer` works.
- Same alias hook strings exist on vision + LM layers — qualify by path.
- Default numerics = raw HF (no LN folding). `enable_compatibility_mode()` only if porting
  folded-weight HookedTransformer code.
- Gemma3 multimodal hotfix landed in TransformerBridge **v3.2.1** (PR #1295) — pin `>=3.2.1`.
- SigLIP has a `.vision_model` wrapping shim for transformers <5.6.0 vs ≥5.6.0 — pin + smoke-test.

## Unverified (smoke-test before scaling)
- Single-token status of emotion labels in Gemma's tokenizer (check empirically, see `src/data/labels.py`).
- An official multimodal `run_with_cache` example — the `pixel_values`-through-`**kwargs` path is
  documented at `forward` level and inferred for `run_with_cache`/`run_with_hooks`.
