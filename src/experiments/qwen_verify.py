"""Qwen VLM verification path — raw HF hooks, NO TransformerBridge.

Qwen2.5-VL / Qwen3-VL are not in TransformerBridge. Run this in the SEPARATE
`requirements-qwen.txt` env (transformers>=4.57 for Qwen3-VL). This is verification, not
a dependency: it re-checks that a Stage-A-style appraisal signal exists in another VLM.

Qwen-VL uses VARIABLE-count image tokens (dynamic resolution, <|vision_start|> /
<|image_pad|> / <|vision_end|>), not Gemma's fixed 256 — recompute the last-token span
from the processor output each time; never hardcode positions.
"""
from __future__ import annotations

import argparse

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_qwen(model_name: str = DEFAULT_MODEL):
    """Load a Qwen3-VL model + processor via raw transformers (needs transformers>=4.57)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def register_readout_hooks(model, store: dict):
    """Register forward hooks on each decoder layer; capture last-token hidden states.

    The attribute path to the decoder layers varies by transformers version — verify
    `model.model.language_model.layers` for your install and adjust if needed.
    """
    layers = model.model.language_model.layers  # verify per HF version

    def make_hook(i: int):
        def hook(_module, _inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    return handles


def make_steer_pre_hook(z_unit, beta: float, pos: int = -1):
    """Forward PRE-hook adding beta*z_unit to the residual at position `pos`.

    Register on a decoder layer to steer its input residual. Recompute `pos` from the
    processor output when image tokens are present (variable count).
    """
    def pre_hook(_module, args):
        hidden = args[0]
        hidden[:, pos, :] = hidden[:, pos, :] + beta * z_unit.to(hidden.dtype)
        return (hidden, *args[1:])

    return pre_hook


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen-VL verification (raw HF hooks)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.parse_args()
    raise SystemExit(
        "qwen_verify is a template for the fallback verification path. Wire it to the "
        "Stage A/C protocol once the Gemma gate passes; run in the requirements-qwen.txt env."
    )


if __name__ == "__main__":
    main()
