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

# --------------------------------------------------------------------------- minimal-pair bank
# Robustness control (T1.1): the base bank matches negative/positive STRENGTH (text-only |neg|/|pos|
# ≈ 1.0) but NOT content — "funeral" and "championship" engage different event semantics, so a
# reviewer can argue the cross-modal asymmetry is about event semantics, not valence. These pairs
# hold the sentence structure and wording constant and flip ONLY the valence-bearing token(s), so the
# within-pair contrast isolates valence. Each entry is (positive, negative, swapped-token(s)); the
# pair index is preserved end-to-end (context_id `mp{i}`) so downstream analysis can recover the
# within-pair flip. Pre-registered here so the pairing is fixed before the GPU run.
MINIMAL_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("This photo was taken moments after they won the championship game.",
     "This photo was taken moments after they lost the championship game.", "won ↔ lost"),
    ("This photo was taken on the best day of their life.",
     "This photo was taken on the worst day of their life.", "best ↔ worst"),
    ("This photo was taken moments after they got the job they wanted.",
     "This photo was taken moments after they lost the job they wanted.", "got ↔ lost"),
    ("This photo was taken moments after they heard wonderful news.",
     "This photo was taken moments after they heard devastating news.", "wonderful ↔ devastating"),
    ("This photo was taken at a celebration for a close friend.",
     "This photo was taken at a memorial for a close friend.", "celebration ↔ memorial"),
    ("This photo was taken during a joyful moment.",
     "This photo was taken during a heartbreaking moment.", "joyful ↔ heartbreaking"),
)
MINIMAL_POSITIVE: tuple[str, ...] = tuple(p for p, _, _ in MINIMAL_PAIRS)
MINIMAL_NEGATIVE: tuple[str, ...] = tuple(n for _, n, _ in MINIMAL_PAIRS)


def build_conditions(bank: str = "full") -> list[tuple[str, str, str | None]]:
    """`(condition, context_id, sentence)` list for the base pass / text-only, for a named bank.

    Shared by `stage_f_conflict` and `stage_f_text_only` so both run the exact same conditions.
      * `full`    — the original 6 pos / 6 neg / 2 neutral bank (ids `p{i}`/`n{i}`/`z{i}`).
      * `minimal` — the matched minimal-pair bank; the SAME id `mp{i}` tags the positive AND negative
                    member of pair `i`, so a per-pair (within-item) valence flip is recoverable from
                    the parquet by grouping on `context_id`.
    Both banks share the same 2 neutral contexts as the within-structure baseline, and both lead with
    the structurally non-comparable no-context row (`none`).
    """
    conds: list[tuple[str, str, str | None]] = [("none", "none", None)]
    if bank == "minimal":
        for i, (pos, neg, _swap) in enumerate(MINIMAL_PAIRS):
            conds.append(("positive", f"mp{i}", pos))
            conds.append(("negative", f"mp{i}", neg))
    elif bank == "full":
        conds += [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
        conds += [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
    else:
        raise ValueError(f"unknown bank {bank!r} — expected 'full' or 'minimal'")
    conds += [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]
    return conds


def verify_minimal_pairs(tokenizer) -> list[dict]:
    """Per-pair token-length parity for the minimal bank — run on the model's tokenizer (Colab).

    A minimal-pair valence control is only clean if the paired sentences are near-identical in
    tokens; a large length gap reintroduces a positional/length confound at the read-out. Reports
    each pair's token counts and delta so any non-minimal pair is caught BEFORE the GPU run.
    """
    out = []
    for i, (pos, neg, swap) in enumerate(MINIMAL_PAIRS):
        pt, nt = len(tokenizer.encode(pos)), len(tokenizer.encode(neg))
        out.append({"pair": f"mp{i}", "swap": swap, "pos_tokens": pt, "neg_tokens": nt,
                    "token_delta": pt - nt, "pos": pos, "neg": neg})
    return out


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
