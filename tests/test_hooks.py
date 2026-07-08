import numpy as np
import pytest

from src.bridge import hooks


def test_language_tap_accepts_lm_blocks():
    assert hooks.language_tap("blocks.20.hook_resid_post")
    assert hooks.language_tap("blocks.0.hook_attn_out")


def test_language_tap_rejects_vision_tower():
    # Same alias suffix, but on the vision tower — must be excluded.
    assert not hooks.language_tap("model.vision_tower.encoder.layers.3.hook_attn_out")
    assert not hooks.language_tap("blocks.5.hook_q")  # not one of the three taps


def test_keep_language_taps_subset():
    keep = hooks.keep_language_taps(("hook_attn_out",))
    assert keep("blocks.10.hook_attn_out")
    assert not keep("blocks.10.hook_mlp_out")


def test_keep_language_taps_rejects_unknown_suffix():
    with pytest.raises(ValueError):
        hooks.keep_language_taps(("hook_bogus",))


def test_resid_post_name():
    assert hooks.resid_post_name(12) == "blocks.12.hook_resid_post"


def test_make_steer_hook_adds_scaled_vector():
    # Minimal numpy stand-in for a torch activation tensor supporting .to(dtype).
    class Arr(np.ndarray):
        def to(self, dtype):
            return self.astype(dtype)

    d = 4
    z = np.ones(d).view(Arr)
    act = np.zeros((1, 3, d)).view(Arr)
    hook = hooks.make_steer_hook(z, beta=2.0, pos=-1)
    out = hook(act, hook=None)
    assert np.allclose(out[0, -1], 2.0)   # last position shifted by beta * z
    assert np.allclose(out[0, 0], 0.0)    # other positions untouched
