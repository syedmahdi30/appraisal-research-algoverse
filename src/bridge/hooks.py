"""Hook-name filters and steering-hook factory.

These are deliberately free of torch/model imports so they can be unit-tested and
reused anywhere. The three taps Tak et al. probe, on the *language-model* blocks:

    blocks.{i}.hook_resid_post   residual / hidden state
    blocks.{i}.hook_attn_out     MHSA output
    blocks.{i}.hook_mlp_out      FFN output

The SAME alias strings exist on the vision tower, disambiguated only by parent path,
so we additionally require the name to start with "blocks." (the LM blocks).
"""
from __future__ import annotations

from typing import Callable

# Tap suffixes, in the order Tak et al. report them.
TAP_SUFFIXES: tuple[str, ...] = ("hook_resid_post", "hook_attn_out", "hook_mlp_out")


def language_tap(name: str) -> bool:
    """True iff `name` is one of the three LM-block taps (not a vision-tower hook)."""
    return name.startswith("blocks.") and name.endswith(TAP_SUFFIXES)


def keep_language_taps(suffixes: tuple[str, ...] = TAP_SUFFIXES) -> Callable[[str], bool]:
    """Build a `names_filter` for `run_with_cache` restricted to given LM taps.

    Pass e.g. `keep_language_taps(("hook_attn_out",))` to cache only MHSA outputs.
    """
    suffixes = tuple(suffixes)
    for s in suffixes:
        if s not in TAP_SUFFIXES:
            raise ValueError(f"unknown tap suffix {s!r}; expected one of {TAP_SUFFIXES}")

    def keep(name: str) -> bool:
        return name.startswith("blocks.") and name.endswith(suffixes)

    return keep


def resid_post_name(layer: int) -> str:
    """Legacy alias for the residual-stream tap at a given LM layer."""
    return f"blocks.{layer}.hook_resid_post"


def make_steer_hook(z_unit, beta: float, pos: int = -1):
    """Return a forward hook that adds `beta * z_unit` at token position `pos`.

    Signature matches TransformerBridge / TransformerLens hooks: `hook(act, hook)`,
    where `act` is `[batch, pos, d_model]`. `z_unit` is a unit-norm steering vector
    (torch tensor); it is cast to the activation dtype so bf16 activations stay bf16.
    """
    def hook(act, hook):  # noqa: ARG001 - `hook` arg is part of the TL contract
        act[:, pos, :] = act[:, pos, :] + beta * z_unit.to(act.dtype)
        return act

    return hook
