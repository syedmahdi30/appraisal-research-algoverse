"""Lightweight runtime helpers shared only by raw-HuggingFace experiment runners."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image


_LANGUAGE_LAYER_CACHE: dict[int, torch.nn.ModuleList] = {}
VISION_PATH_MARKERS = ("vision", "siglip", "vision_tower")


def find_language_layers(model, verbose: bool = True) -> torch.nn.ModuleList:
    """Locate and cache the model's language-decoder layers, excluding vision paths."""
    key = id(model)
    if key in _LANGUAGE_LAYER_CACHE:
        return _LANGUAGE_LAYER_CACHE[key]
    best = None
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not name.endswith("layers"):
            continue
        if any(marker in name for marker in VISION_PATH_MARKERS):
            continue
        if "language_model" in name or best is None:
            best = (name, module)
            if "language_model" in name:
                break
    if best is None:
        raise RuntimeError("could not locate the language-model decoder layers on this model")
    if verbose:
        print(f"  language-model layers: {best[0]}  ({len(best[1])} layers)")
    _LANGUAGE_LAYER_CACHE[key] = best[1]
    return best[1]


def _submodule(layer_module, tap: str):
    module = layer_module
    for part in tap.split("."):
        module = getattr(module, part)
    return module


@contextmanager
def last_token_tap(model, layer: int, tap: str):
    """Capture the most recent last-token output from a decoder-layer submodule."""
    layers = find_language_layers(model)
    target = _submodule(layers[layer], tap)
    store: list = [None]

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[0] = tensor[0, -1].detach().float().cpu().numpy()

    handle = target.register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


@contextmanager
def capture_residuals(model, layers):
    """Capture full-sequence decoder outputs for the requested layer indices."""
    language_layers = find_language_layers(model, verbose=False)
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            store[layer_index] = tensor[0].detach()

        return hook

    for layer_index in layers:
        handles.append(language_layers[layer_index].register_forward_hook(make_hook(layer_index)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_residuals(model, donor: dict[int, torch.Tensor], donor_indices, recipient_indices):
    """Patch mapped residual positions at every decoder layer present in ``donor``."""
    language_layers = find_language_layers(model, verbose=False)
    donor_index = torch.as_tensor(np.asarray(donor_indices), dtype=torch.long)
    recipient_index = torch.as_tensor(np.asarray(recipient_indices), dtype=torch.long)
    handles = []

    def make_hook(layer_index):
        source = donor[layer_index]
        values = source[donor_index.to(source.device)]

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = (output[0] if is_tuple else output).clone()
            hidden[0, recipient_index.to(hidden.device), :] = values.to(
                device=hidden.device, dtype=hidden.dtype
            )
            return (hidden,) + tuple(output[1:]) if is_tuple else hidden

        return hook

    for layer_index in donor:
        handles.append(language_layers[layer_index].register_forward_hook(make_hook(layer_index)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_gemma_hf(model_name: str):
    """Load the Gemma raw-HF model and processor with the established runtime options."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    return model, AutoProcessor.from_pretrained(model_name)


def load_vlm(model_name: str, max_side: int | None = None):
    """Load a Qwen-family or general image-to-text model, processor, and family label."""
    from transformers import AutoProcessor

    lower_name = model_name.lower()
    if "qwen" in lower_name:
        if "qwen3" in lower_name:
            from transformers import Qwen3VLForConditionalGeneration as model_class
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as model_class
        family = "qwen"
        processor_kwargs = {"max_pixels": max_side * max_side} if max_side else {}
    else:
        try:
            from transformers import AutoModelForImageTextToText as model_class
        except ImportError:
            from transformers import LlavaForConditionalGeneration as model_class
        family = "llava"
        processor_kwargs = {}
    model = model_class.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    ).eval()
    try:
        processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    except (TypeError, ValueError):
        print(
            f"  [warn] processor rejected {list(processor_kwargs)}; "
            "relying on image resize alone"
        )
        processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, family


def resize_long_side(image: Image.Image, max_side: int | None) -> Image.Image:
    """Downscale an image's long side with the runner's established bicubic arithmetic."""
    if not max_side or max(image.size) <= max_side:
        return image
    width, height = image.size
    scale = max_side / float(max(width, height))
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC
    )


def encode_image_prompt(processor, image, prompt: str, device) -> dict:
    """Encode one image/prompt pair and move each returned tensor to ``device``."""
    encoded = processor(text=prompt, images=[image], return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}
