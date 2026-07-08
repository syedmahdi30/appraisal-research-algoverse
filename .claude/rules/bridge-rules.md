# Rules — TransformerBridge usage

- **Boot in bf16 and assert multimodal.** Always `dtype=torch.bfloat16`; immediately
  `assert bridge.cfg.is_multimodal` after boot. If it's False, the wrong adapter was picked — stop.
- **Never `start_at_layer`.** It raises `NotImplementedError`. Only `stop_at_layer` exists. Design
  layer sweeps as full forward passes (optionally truncated with `stop_at_layer`).
- **Qualify hook names by path.** `hook_resid_post` / `hook_attn_out` / `hook_mlp_out` exist on BOTH
  the vision tower and the LM blocks. For appraisal read-out always use `blocks.{i}...` (language model).
  Never cache vision-tower keys for the appraisal probes.
- **Cache selectively.** Pass a `names_filter` and index the last token (`[:, -1]`). Never full-sequence
  cache all hooks — it exhausts memory. See `src/bridge/hooks.py::keep_language_taps`.
- **Don't hardcode token positions.** Image tokens are spliced at `<start_of_image>`; the last prompt
  token is `input_ids.shape[-1] - 1`. Compute it, never assume a fixed index.
- **pixel_values only for multimodal.** `forward`/`run_with_cache`/`run_with_hooks` accept
  `pixel_values` only when `cfg.is_multimodal`. During `generate`, it's consumed on the first step only.
- **Numerics = raw HF by default.** Only call `enable_compatibility_mode()` when deliberately porting
  folded/centered HookedTransformer-era code, and note it in the run config.
- **Pin and re-smoke-test.** Pin `transformer-lens>=3.2.1` (Gemma3 multimodal hotfix). After any
  upgrade of transformer-lens/transformers, re-run `scripts/smoke_test.py` before trusting numbers.
- **Steering hook contract.** Steering hooks add `beta * z_unit` at a single position and return the
  modified activation; keep them pure and dtype-cast `z_unit` to the activation dtype.
