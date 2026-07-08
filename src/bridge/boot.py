"""Boot Gemma 3 4B through TransformerBridge, with the project's safety checks.

See docs/models-gemma3.md and .claude/rules/bridge-rules.md. torch / transformer_lens
are imported lazily so that importing this module (e.g. in tests) does not require them.
"""
from __future__ import annotations

import os

DEFAULT_MODEL = "google/gemma-3-4b-it"


def boot_gemma(
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    require_multimodal: bool = True,
):
    """Boot Gemma 3 in bf16 and assert the multimodal adapter was selected.

    Returns the `TransformerBridge`. Raises RuntimeError with an actionable message if
    HF_TOKEN is missing (Gemma is gated) or the multimodal adapter was not picked.
    """
    import warnings

    import torch
    from transformer_lens.model_bridge import TransformerBridge

    # The bridge tries to register LM-style hook aliases on the SigLIP vision tower, which
    # don't resolve there. We only probe language-model blocks (blocks.{i}...), so these are
    # expected and harmless — silence them to keep boot output readable.
    warnings.filterwarnings(
        "ignore",
        message=r"Hook alias .* on SiglipVisionEncoderLayerBridge.* did not resolve",
    )

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is not set. Gemma is gated: accept the license on the Hub and "
            "`export HF_TOKEN=hf_...` before booting."
        )

    bridge = TransformerBridge.boot_transformers(
        model_name,
        device=device,
        dtype=torch.bfloat16,  # bf16 required for Gemma 3
    )

    if require_multimodal and not getattr(bridge.cfg, "is_multimodal", False):
        raise RuntimeError(
            f"{model_name} booted but cfg.is_multimodal is False — the multimodal adapter "
            "was not selected. Check the transformer-lens version (>=3.2.1 for the Gemma3 "
            "multimodal hotfix)."
        )
    return bridge
