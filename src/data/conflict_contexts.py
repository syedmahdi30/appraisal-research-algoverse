"""Stage F — modality-conflict context bank and prompt builder (pilot-plan-stage-e-f.md).

Generic one-sentence contexts that fit any person photo, so we can pit a TEXT appraisal cue
against the IMAGE's own valence. `text_code` is the appraisal polarity used in the Stage F
regression: negative context = -1, neutral = 0, positive = +1 (no-context is also 0).

The prompt mirrors Stage C/D's IMAGE_EMOTION_PROMPT with the context inserted between the image
and the question; the last-token read-out position is unchanged (`input_ids.shape[-1] - 1`).
"""
from __future__ import annotations

import numpy as np

NEGATIVE_CONTEXTS: tuple[str, ...] = (
    "This photo was taken moments after they received devastating news.",
    "This photo was taken moments before the accident.",
    "This photo was taken at the funeral of a close friend.",
    "This photo was taken just after they lost their job.",
    "This photo was taken during the worst week of their life.",
    "This photo was taken right after a painful goodbye.",
)

POSITIVE_CONTEXTS: tuple[str, ...] = (
    "This photo was taken moments after they won the championship.",
    "This photo was taken at their surprise birthday party.",
    "This photo was taken just after they got the job they wanted.",
    "This photo was taken on the best day of their life.",
    "This photo was taken moments after hearing wonderful news.",
    "This photo was taken at a celebration held in their honor.",
)

NEUTRAL_CONTEXTS: tuple[str, ...] = (
    "This photo was taken on a weekday.",
    "This photo was taken indoors.",
)

# A stronger neutral->negative fallback that names "this person" inside the context, used by the
# NULL-diagnosis retry in analyze_stage_f if plain contexts are not attended at all (F4).
STRONG_CONTEXT = ("This person has just been told that someone they love has died, "
                  "and they are devastated.")

TEXT_CODE = {"none": 0, "neutral": 0, "positive": +1, "negative": -1}


def context_prompt(ctx: str | None) -> str:
    """Gemma chat prompt with an image slot and an optional context sentence.

    `ctx=None` reproduces Stage C/D's IMAGE_EMOTION_PROMPT exactly (no-context condition).
    """
    ctx_part = "" if not ctx else f"Context: {ctx} "
    return (f"<start_of_turn>user\n<start_of_image>{ctx_part}"
            f"What single emotion is this person feeling?<end_of_turn>\n"
            f"<start_of_turn>model\n")


def sample_contexts(seed: int) -> dict[str, str]:
    """Deterministically pick one context per polarity for the base pass (F2)."""
    rng = np.random.default_rng(seed)
    return {
        "positive": str(rng.choice(POSITIVE_CONTEXTS)),
        "negative": str(rng.choice(NEGATIVE_CONTEXTS)),
        "neutral": str(rng.choice(NEUTRAL_CONTEXTS)),
    }
