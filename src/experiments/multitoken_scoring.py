"""Teacher-forced closed-vocabulary scoring for labels that span multiple tokens."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..data.labels import EMOTION_LABELS

POSITIVE = ("joy", "pride", "relief", "trust")
NEGATIVE = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")


def _common_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
    if not sequences:
        return 0
    limit = min(len(seq) for seq in sequences)
    for index in range(limit):
        if len({seq[index] for seq in sequences}) != 1:
            return index
    return limit


def append_label_tokens(
    inputs: Mapping[str, torch.Tensor],
    label_token_ids: Sequence[Sequence[int]],
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Duplicate one processed prompt and append a right-padded label sequence to each copy."""
    if not label_token_ids:
        raise ValueError("at least one label sequence is required")
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("expected one unbatched prompt with input_ids shape [1, sequence]")

    batch_size = len(label_token_ids)
    prompt_length = input_ids.shape[1]
    max_label_length = max(len(ids) for ids in label_token_ids)
    if max_label_length == 0:
        raise ValueError("label sequences must not be empty")

    labels = torch.full(
        (batch_size, max_label_length),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    label_mask = torch.zeros_like(labels)
    for row, ids in enumerate(label_token_ids):
        if not ids:
            raise ValueError("label sequences must not be empty")
        labels[row, :len(ids)] = torch.as_tensor(ids, dtype=input_ids.dtype, device=input_ids.device)
        label_mask[row, :len(ids)] = 1

    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.cat((input_ids.repeat(batch_size, 1), labels), dim=1),
    }
    prompt_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    batch["attention_mask"] = torch.cat((prompt_mask.repeat(batch_size, 1), label_mask), dim=1)

    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(f"unsupported processed-input field for label batching: {key}")
        if value.ndim == 2 and value.shape[1] == prompt_length and key != "image_sizes":
            raise ValueError(
                f"sequence-aligned processed-input field is not supported for label batching: {key}"
            )
        repeats = (batch_size,) + (1,) * (value.ndim - 1)
        batch[key] = value.repeat(repeats)
    return batch, prompt_length


def gather_label_sequence_scores(
    logits: torch.Tensor,
    prompt_length: int,
    label_token_ids: Sequence[Sequence[int]],
    *,
    common_prefix_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total and content-token-mean log probability for each teacher-forced label."""
    if logits.ndim != 3 or logits.shape[0] != len(label_token_ids):
        raise ValueError("logits batch must match the number of label sequences")
    prefix = (_common_prefix_length(label_token_ids)
              if common_prefix_length is None else common_prefix_length)
    if any(len(ids) <= prefix for ids in label_token_ids):
        raise ValueError("every label must contain at least one token after the shared prefix")

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    summed, means = [], []
    for row, ids in enumerate(label_token_ids):
        token_scores = torch.stack([
            log_probs[row, prompt_length + offset - 1, token_id]
            for offset, token_id in enumerate(ids)
        ])
        summed.append(token_scores.sum())
        means.append(token_scores[prefix:].mean())
    return torch.stack(summed), torch.stack(means)


def closed_vocab_readout(scores: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    """Normalize 13 label scores and return behavioral valence plus label log probabilities."""
    if set(scores) != set(EMOTION_LABELS):
        missing = sorted(set(EMOTION_LABELS) - set(scores))
        extra = sorted(set(scores) - set(EMOTION_LABELS))
        raise ValueError(f"scores must cover exactly the emotion labels; missing={missing}, extra={extra}")
    values = torch.tensor([scores[label] for label in EMOTION_LABELS], dtype=torch.float64)
    normalized = torch.log_softmax(values, dim=0)
    logprobs = {label: float(normalized[i]) for i, label in enumerate(EMOTION_LABELS)}
    probs = normalized.exp()
    by_label = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
    valence = sum(by_label[label] for label in POSITIVE) - sum(by_label[label] for label in NEGATIVE)
    return valence, logprobs


def score_label_sequences(
    model,
    inputs: Mapping[str, torch.Tensor],
    label_token_ids: Mapping[str, Sequence[int]],
    *,
    pad_token_id: int,
    label_batch_size: int = 4,
) -> dict[str, dict]:
    """Teacher-force all labels in bounded batches and compute sum and mean readouts."""
    if label_batch_size < 1:
        raise ValueError("label_batch_size must be positive")
    if set(label_token_ids) != set(EMOTION_LABELS):
        raise ValueError("label_token_ids must cover exactly the emotion labels")

    device = model.device
    moved = {key: value.to(device) for key, value in inputs.items()}
    ordered = [(label, list(label_token_ids[label])) for label in EMOTION_LABELS]
    common_prefix = _common_prefix_length([ids for _, ids in ordered])
    sum_scores: dict[str, float] = {}
    mean_scores: dict[str, float] = {}

    for start in range(0, len(ordered), label_batch_size):
        chunk = ordered[start:start + label_batch_size]
        ids = [token_ids for _, token_ids in chunk]
        batch, prompt_length = append_label_tokens(moved, ids, pad_token_id)
        with torch.no_grad():
            logits = model(**batch).logits
        if logits.shape[:2] != batch["input_ids"].shape:
            raise RuntimeError(
                "model output sequence length does not match processed input_ids; the model may be "
                "expanding image placeholders inside forward, so teacher-forced token positions "
                "cannot be gathered safely with this processor/model pairing"
            )
        summed, content_mean = gather_label_sequence_scores(
            logits,
            prompt_length,
            ids,
            common_prefix_length=common_prefix,
        )
        for (label, _), sum_score, mean_score in zip(chunk, summed, content_mean):
            sum_scores[label] = float(sum_score.detach().cpu())
            mean_scores[label] = float(mean_score.detach().cpu())

    sum_valence, sum_logprobs = closed_vocab_readout(sum_scores)
    mean_valence, mean_logprobs = closed_vocab_readout(mean_scores)
    return {
        "sequence_sum": {
            "valence": sum_valence,
            "logprobs": sum_logprobs,
            "scores": sum_scores,
        },
        "content_mean": {
            "valence": mean_valence,
            "logprobs": mean_logprobs,
            "scores": mean_scores,
        },
    }
