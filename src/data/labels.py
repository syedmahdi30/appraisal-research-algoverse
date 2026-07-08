"""Shared emotion labels, appraisal targets, and the EMOTIC->shared mapping.

These constants are the single source of truth for the label space (docs/datasets.md).
`verify_label_tokenization` must be run per model before any logit-based closed-vocab
scoring — labels are NOT assumed single-token (docs/probes.md, data-rules.md).
"""
from __future__ import annotations

from typing import Iterable

# crowd-enVENT's 13 emotion labels (closed vocabulary for Stage A).
EMOTION_LABELS: tuple[str, ...] = (
    "anger", "boredom", "disgust", "fear", "guilt", "joy", "pride",
    "relief", "sadness", "shame", "surprise", "trust", "neutral",
)

# Six primary appraisal dimensions this experiment targets (crowd-enVENT names).
APPRAISAL_TARGETS: tuple[str, ...] = (
    "pleasantness",
    "unpleasantness",
    "suddenness",
    "event_predictability",
    "own_responsibility",      # self-agency
    "others_responsibility",   # other-agency
)

# Shared ~7-emotion space for cross-modal alignment (Stage B).
SHARED_EMOTIONS: tuple[str, ...] = (
    "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral",
)

# EMOTIC's 26 discrete categories.
EMOTIC_CATEGORIES: tuple[str, ...] = (
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion", "Confidence",
    "Disapproval", "Disconnection", "Disquietment", "Doubt/Confusion",
    "Embarrassment", "Engagement", "Esteem", "Excitement", "Fatigue", "Fear",
    "Happiness", "Pain", "Peace", "Pleasure", "Sadness", "Sensitivity",
    "Suffering", "Surprise", "Sympathy", "Yearning",
)

# EMOTIC-26 -> shared-7. Lossy by construction; unmapped categories -> None and are
# dropped by single-label filtering. Surface this as a caveat in Stage C (data-rules.md).
EMOTIC_TO_SHARED: dict[str, str | None] = {
    "Affection": "joy",
    "Anger": "anger",
    "Annoyance": "anger",
    "Anticipation": "surprise",
    "Aversion": "disgust",
    "Confidence": "joy",
    "Disapproval": "anger",
    "Disconnection": "sadness",
    "Disquietment": "fear",
    "Doubt/Confusion": "surprise",
    "Embarrassment": "sadness",
    "Engagement": "neutral",
    "Esteem": "joy",
    "Excitement": "joy",
    "Fatigue": "sadness",
    "Fear": "fear",
    "Happiness": "joy",
    "Pain": "sadness",
    "Peace": "neutral",
    "Pleasure": "joy",
    "Sadness": "sadness",
    "Sensitivity": "sadness",
    "Suffering": "sadness",
    "Surprise": "surprise",
    "Sympathy": "sadness",
    "Yearning": "surprise",
}


def map_emotic_label(category: str) -> str | None:
    """Map an EMOTIC category to the shared space; None if intentionally unmapped."""
    if category not in EMOTIC_TO_SHARED:
        raise KeyError(f"unknown EMOTIC category: {category!r}")
    return EMOTIC_TO_SHARED[category]


def verify_label_tokenization(tokenizer, labels: Iterable[str] = EMOTION_LABELS) -> dict[str, dict]:
    """Report per-label token ids and whether each is single-token for this tokenizer.

    Keeps the leading space so the SentencePiece ▁ word-boundary marker is included, as
    it is at generation time. Returns `{label: {"ids": [...], "n_tokens": k,
    "single_token": bool}}`. Multi-token labels require first-subtoken or summed-logprob
    scoring — kept identical across Gemma and Qwen for comparability.
    """
    report: dict[str, dict] = {}
    for w in labels:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        report[w] = {"ids": ids, "n_tokens": len(ids), "single_token": len(ids) == 1}
    return report
