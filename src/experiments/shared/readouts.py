"""Closed-vocabulary behavioral readouts for raw-HuggingFace runners."""
from __future__ import annotations

import torch

from ...data.labels import EMOTION_LABELS

QUESTION = "What single emotion is this person feeling?"
POSITIVE = ("joy", "pride", "relief", "trust")
NEGATIVE = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")


def user_text(context_sentence: str | None) -> str:
    """Build the stable user-visible prompt text for an optional context."""
    context = "" if not context_sentence else f"Context: {context_sentence} "
    return f"{context}{QUESTION}"


def first_content_token_ids(processor) -> dict[str, int]:
    """Return each label's first non-whitespace token, rejecting collapsed vocabularies."""
    tokenizer = processor.tokenizer
    token_ids = {}
    for label in EMOTION_LABELS:
        encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        token_ids[label] = next(
            (token for token in encoded if tokenizer.decode([token]).strip()),
            encoded[0] if encoded else -1,
        )
    distinct = len(set(token_ids.values()))
    if distinct < len(EMOTION_LABELS):
        raise ValueError(
            f"emotion label token ids collapsed ({distinct}/{len(EMOTION_LABELS)} distinct) — "
            f"the read-out would be degenerate. Tokenizer {type(tokenizer).__name__}; "
            "inspect encode(' joy')."
        )
    return token_ids


def closed_vocab_logprobs(logits_last, token_ids) -> dict[str, float]:
    """Return log probabilities normalized over the project's 13 emotion labels."""
    index = torch.tensor([token_ids[label] for label in EMOTION_LABELS], device=logits_last.device)
    values = torch.log_softmax(logits_last[index].float(), dim=-1)
    return {label: float(values[i]) for i, label in enumerate(EMOTION_LABELS)}


def closed_vocab_valence(logits_last, token_ids) -> float:
    """Return P(positive labels) minus P(negative labels) within the closed vocabulary."""
    index = torch.tensor([token_ids[label] for label in EMOTION_LABELS], device=logits_last.device)
    probabilities = torch.softmax(logits_last[index].float(), dim=-1)
    by_label = {label: probabilities[i].item() for i, label in enumerate(EMOTION_LABELS)}
    return sum(by_label[label] for label in POSITIVE) - sum(by_label[label] for label in NEGATIVE)


def model_readout(model, inputs, token_ids) -> tuple[float, dict[str, float]]:
    """Score the final prompt position of one raw-HuggingFace forward pass."""
    moved = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**moved)
    logits_last = output.logits[0, -1].float()
    return closed_vocab_valence(logits_last, token_ids), closed_vocab_logprobs(logits_last, token_ids)
