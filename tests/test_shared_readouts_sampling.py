import math

import pandas as pd
import pytest
import torch

from src.data.labels import EMOTION_LABELS
from src.experiments.shared.readouts import (
    QUESTION,
    closed_vocab_logprobs,
    closed_vocab_valence,
    first_content_token_ids,
    model_readout,
    user_text,
)
from src.experiments.shared.sampling import select_extreme_rows, select_ranked_pairs


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        label = text.strip()
        index = EMOTION_LABELS.index(label)
        return [900, 100 + index]

    def decode(self, ids):
        return " " if ids == [900] else EMOTION_LABELS[ids[0] - 100]


class FakeProcessor:
    tokenizer = FakeTokenizer()


def test_readout_contract_and_prompt_text():
    ids = first_content_token_ids(FakeProcessor())
    assert ids == {label: 100 + i for i, label in enumerate(EMOTION_LABELS)}
    assert user_text(None) == "What single emotion is this person feeling?"
    assert user_text("They won.") == "Context: They won. What single emotion is this person feeling?"

    logits = torch.zeros(256)
    logits[ids["joy"]] = 2.0
    logprobs = closed_vocab_logprobs(logits, ids)
    assert sum(math.exp(value) for value in logprobs.values()) == pytest.approx(1.0)
    assert closed_vocab_valence(logits, ids) > 0


def test_extreme_selection_preserves_published_order_and_pairing():
    df = pd.DataFrame({"image_path": [f"p{i}" for i in range(6)],
                       "valence": [-3, -2, -1, 1, 2, 3]})
    selected = select_extreme_rows(df, 4)
    assert selected["image_path"].tolist() == ["p4", "p5", "p0", "p1"]
    assert selected["image_group"].tolist() == ["positive", "positive", "negative", "negative"]

    positive, negative = select_ranked_pairs(df, 2)
    assert positive["image_path"].tolist() == ["p5", "p4"]
    assert negative["image_path"].tolist() == ["p0", "p1"]


def test_tied_valence_selection_matches_the_former_default_pandas_sort_order():
    df = pd.DataFrame({"image_path": [f"p{i}" for i in range(17)],
                       "valence": [-1] + [0] * 15 + [1]})
    former_order = df.sort_values("valence")

    expected_extremes = pd.concat([
        former_order.tail(4).assign(image_group="positive"),
        former_order.head(4).assign(image_group="negative"),
    ]).reset_index(drop=True)
    expected_positive = former_order.tail(4).sort_values("valence", ascending=False).reset_index(drop=True)
    expected_negative = former_order.head(4).sort_values("valence").reset_index(drop=True)

    selected = select_extreme_rows(df, 8)
    positive, negative = select_ranked_pairs(df, 4)

    assert selected["image_path"].tolist() == expected_extremes["image_path"].tolist()
    assert positive["image_path"].tolist() == expected_positive["image_path"].tolist()
    assert negative["image_path"].tolist() == expected_negative["image_path"].tolist()


def test_model_readout_scores_the_final_prompt_token():
    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, input_ids):
            logits = torch.zeros((1, input_ids.shape[-1], 256))
            logits[0, -1, 100 + EMOTION_LABELS.index("joy")] = 2.0
            return type("Output", (), {"logits": logits})()

    token_ids = {label: 100 + i for i, label in enumerate(EMOTION_LABELS)}
    valence, logprobs = model_readout(FakeModel(), {"input_ids": torch.tensor([[1, 2]])}, token_ids)

    assert valence > 0
    assert max(logprobs, key=logprobs.get) == "joy"


def test_qwen_runner_keeps_shared_readout_compatibility_aliases():
    from src.experiments import stage_f_qwen

    assert stage_f_qwen.QUESTION == QUESTION
    assert stage_f_qwen.emotion_token_ids is first_content_token_ids
    assert stage_f_qwen.valence_score is closed_vocab_valence
    assert stage_f_qwen.emotion_logprobs is closed_vocab_logprobs
    assert stage_f_qwen.readout is model_readout
    assert stage_f_qwen._user_text is user_text


def test_qwen_readout_aliases_accept_legacy_tok_ids_keyword():
    from src.experiments import stage_f_qwen

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, input_ids):
            logits = torch.zeros((1, input_ids.shape[-1], 256))
            logits[0, -1, 100 + EMOTION_LABELS.index("joy")] = 2.0
            return type("Output", (), {"logits": logits})()

    tok_ids = {label: 100 + i for i, label in enumerate(EMOTION_LABELS)}
    logits = torch.zeros(256)
    logits[tok_ids["joy"]] = 2.0

    assert stage_f_qwen.valence_score(logits, tok_ids=tok_ids) > 0
    emotion_logprobs = stage_f_qwen.emotion_logprobs(logits, tok_ids=tok_ids)
    assert max(emotion_logprobs, key=emotion_logprobs.get) == "joy"
    valence, logprobs = stage_f_qwen.readout(
        FakeModel(), {"input_ids": torch.tensor([[1, 2]])}, tok_ids=tok_ids)
    assert valence > 0
    assert max(logprobs, key=logprobs.get) == "joy"
