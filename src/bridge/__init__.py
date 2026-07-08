"""TransformerBridge boot + hook/cache utilities for Gemma 3.

Pure-logic helpers (hook filters, steering-hook factory) live in `hooks` and are
importable without torch/GPU. Model-touching code lives in `boot` and `multimodal`.
"""
from .hooks import keep_language_taps, language_tap, make_steer_hook, TAP_SUFFIXES

__all__ = ["keep_language_taps", "language_tap", "make_steer_hook", "TAP_SUFFIXES"]
