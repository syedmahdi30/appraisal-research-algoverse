"""Multimodal input construction and last-token activation read-out for Gemma 3.

See docs/models-gemma3.md. The 256 image soft-tokens are spliced at the
`<start_of_image>` placeholder, so the last prompt token is always
`input_ids.shape[-1] - 1` — never a hardcoded index.
"""
from __future__ import annotations

from .hooks import keep_language_taps

# Gemma 3 chat template with an image slot. `prepare_multimodal_inputs` expands
# <start_of_image> into 256 soft image tokens via the HF processor.
IMAGE_EMOTION_PROMPT = (
    "<start_of_turn>user\n<start_of_image>"
    "What single emotion is this person feeling?<end_of_turn>\n"
    "<start_of_turn>model\n"
)

TEXT_EMOTION_PROMPT = (
    "<start_of_turn>user\n{text}\n"
    "What single emotion is the writer feeling?<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def build_image_inputs(bridge, image, prompt: str = IMAGE_EMOTION_PROMPT):
    """Run the HF processor for one PIL image; returns the bridge input dict."""
    return bridge.prepare_multimodal_inputs(text=prompt, images=[image])


def last_token_activations(bridge, inputs, taps=None):
    """Cache the LM taps and return `{tap_name: [d_model]}` at the final prompt token.

    `inputs` is the dict from `prepare_multimodal_inputs` (or a text-only tokenization);
    `pixel_values` is forwarded only when present. `taps` defaults to all three LM taps.
    """
    keep = keep_language_taps() if taps is None else keep_language_taps(taps)
    kwargs = {}
    if inputs.get("pixel_values") is not None:
        kwargs["pixel_values"] = inputs["pixel_values"]

    _, cache = bridge.run_with_cache(inputs["input_ids"], names_filter=keep, **kwargs)

    last = inputs["input_ids"].shape[-1] - 1
    return {name: cache[name][0, last] for name in cache}
