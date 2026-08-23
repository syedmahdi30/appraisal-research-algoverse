import math

import pytest
import torch

from src.data.labels import EMOTION_LABELS
from src.experiments.multitoken_scoring import (
    append_label_tokens,
    closed_vocab_readout,
    gather_label_sequence_scores,
    score_label_sequences,
)


def _target_logprob(target_logit: float, vocab_size: int) -> float:
    return target_logit - math.log(math.exp(target_logit) + vocab_size - 1)


def test_gather_label_sequence_scores_uses_predecessor_positions():
    """Moving the gather index off prompt_len+j-1 must change this result."""
    vocab_size = 8
    prompt_len = 3
    label_ids = [[1, 2], [3, 4, 5]]
    logits = torch.zeros((2, prompt_len + 3, vocab_size))

    target_logits = ((1.0, 2.0), (0.5, 1.5, 2.5))
    for row, (ids, values) in enumerate(zip(label_ids, target_logits)):
        for offset, (token_id, value) in enumerate(zip(ids, values)):
            logits[row, prompt_len + offset - 1, token_id] = value

    summed, content_mean = gather_label_sequence_scores(logits, prompt_len, label_ids)

    expected = [
        sum(_target_logprob(v, vocab_size) for v in target_logits[0]),
        sum(_target_logprob(v, vocab_size) for v in target_logits[1]),
    ]
    assert summed.tolist() == pytest.approx(expected)
    assert content_mean.tolist() == pytest.approx([expected[0] / 2, expected[1] / 3])


def test_content_mean_excludes_the_shared_whitespace_prefix():
    """Including LLaVA's shared leading-space token in the mean would length-bias the sensitivity."""
    vocab_size = 8
    prompt_len = 2
    label_ids = [[7, 1], [7, 2, 3]]
    logits = torch.zeros((2, prompt_len + 3, vocab_size))

    for row, ids in enumerate(label_ids):
        logits[row, prompt_len - 1, ids[0]] = -3.0  # shared prefix: deliberately low probability
        for offset, token_id in enumerate(ids[1:], start=1):
            logits[row, prompt_len + offset - 1, token_id] = 2.0

    summed, content_mean = gather_label_sequence_scores(logits, prompt_len, label_ids)

    content_lp = _target_logprob(2.0, vocab_size)
    assert content_mean.tolist() == pytest.approx([content_lp, content_lp])
    assert summed[0].item() != pytest.approx(summed[1].item())


def test_append_label_tokens_right_pads_and_repeats_multimodal_inputs():
    """Padding as attended content or failing to repeat image tensors would corrupt a batched score."""
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.tensor([[[[1.0]]]]),
        "image_sizes": torch.tensor([[20, 30]]),
    }

    batch, prompt_len = append_label_tokens(inputs, [[4, 5], [6]], pad_token_id=0)

    assert prompt_len == 3
    assert batch["input_ids"].tolist() == [[10, 11, 12, 4, 5], [10, 11, 12, 6, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert batch["pixel_values"].shape == (2, 1, 1, 1)
    assert batch["pixel_values"].flatten().tolist() == [1.0, 1.0]
    assert batch["image_sizes"].tolist() == [[20, 30], [20, 30]]


def test_append_label_tokens_rejects_unextended_sequence_aligned_fields():
    """Silently leaving position-like fields at prompt length would misalign appended labels."""
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "token_type_ids": torch.tensor([[0, 0, 0]]),
    }

    with pytest.raises(ValueError, match="sequence-aligned"):
        append_label_tokens(inputs, [[4, 5], [6]], pad_token_id=0)


def test_closed_vocab_readout_normalizes_sequence_scores_before_valence():
    """Using raw sequence likelihoods instead of a 13-way softmax would not yield the defined score."""
    scores = {label: 0.0 for label in EMOTION_LABELS}

    valence, logprobs = closed_vocab_readout(scores)

    assert valence == pytest.approx((4 - 7) / 13)
    assert sum(math.exp(v) for v in logprobs.values()) == pytest.approx(1.0)
    assert set(logprobs) == set(EMOTION_LABELS)


def test_score_label_sequences_respects_microbatch_and_returns_both_rules():
    """Ignoring label_batch_size could turn a valid score into an A100 out-of-memory failure."""
    class NextTokenModel:
        device = torch.device("cpu")

        def __init__(self):
            self.batch_sizes = []

        def __call__(self, input_ids, attention_mask, **unused):
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros((*input_ids.shape, 40))
            for row in range(input_ids.shape[0]):
                for pos in range(input_ids.shape[1] - 1):
                    if attention_mask[row, pos + 1]:
                        logits[row, pos, input_ids[row, pos + 1]] = 2.0
            return type("Output", (), {"logits": logits})()

    label_ids = {
        label: [30, index + 1] + ([20] if index in {1, 2, 4, 8} else [])
        for index, label in enumerate(EMOTION_LABELS)
    }
    model = NextTokenModel()
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    result = score_label_sequences(
        model,
        inputs,
        label_ids,
        pad_token_id=0,
        label_batch_size=5,
    )

    assert model.batch_sizes == [5, 5, 3]
    assert set(result) == {"sequence_sum", "content_mean"}
    assert result["content_mean"]["valence"] == pytest.approx((4 - 7) / 13)
    assert set(result["sequence_sum"]["logprobs"]) == set(EMOTION_LABELS)
    assert set(result["content_mean"]["logprobs"]) == set(EMOTION_LABELS)


def test_score_label_sequences_rejects_model_side_sequence_expansion():
    """Gathering by input position is invalid if a legacy VLM expands image tokens inside forward."""
    class ExpandedSequenceModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1] + 2, 40))
            return type("Output", (), {"logits": logits})()

    label_ids = {
        label: [30, index + 1]
        for index, label in enumerate(EMOTION_LABELS)
    }
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    with pytest.raises(RuntimeError, match="sequence length"):
        score_label_sequences(
            ExpandedSequenceModel(),
            inputs,
            label_ids,
            pad_token_id=0,
            label_batch_size=13,
        )
