import pytest

from src.data import labels


def test_emotion_labels_count():
    assert len(labels.EMOTION_LABELS) == 13
    assert "neutral" in labels.EMOTION_LABELS


def test_appraisal_targets_are_the_six():
    assert len(labels.APPRAISAL_TARGETS) == 6
    assert "pleasantness" in labels.APPRAISAL_TARGETS
    assert "others_responsibility" in labels.APPRAISAL_TARGETS


def test_emotic_mapping_covers_all_26_categories():
    assert len(labels.EMOTIC_CATEGORIES) == 26
    for cat in labels.EMOTIC_CATEGORIES:
        assert cat in labels.EMOTIC_TO_SHARED  # every category has an explicit mapping


def test_mapped_targets_are_in_shared_space():
    for target in labels.EMOTIC_TO_SHARED.values():
        if target is not None:
            assert target in labels.SHARED_EMOTIONS


def test_map_emotic_label_unknown_raises():
    with pytest.raises(KeyError):
        labels.map_emotic_label("NotAnEmotion")


def test_verify_label_tokenization_flags_multitoken():
    # Fake tokenizer: splits on characters, so multi-char labels are multi-token.
    class FakeTok:
        def encode(self, s, add_special_tokens=False):
            return list(range(len(s.strip())))

    report = labels.verify_label_tokenization(FakeTok(), ["joy", "ab"])
    assert report["joy"]["n_tokens"] == 3 and not report["joy"]["single_token"]
    # keeps leading space then strips in fake -> "ab" has 2 chars
    assert report["ab"]["n_tokens"] == 2
