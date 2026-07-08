"""EMOTIC parsing logic — covers both Annotations.mat category conventions.

Uses lightweight stand-ins for scipy mat_struct objects (attribute access + a
`_fieldnames` list), so these run without scipy or the real .mat file.
"""
import types

import numpy as np
import pandas as pd

from src.data import emotic


def _struct(**fields):
    ns = types.SimpleNamespace(**fields)
    ns._fieldnames = list(fields)
    return ns


def test_person_categories_train_convention():
    # train: annotations_categories is a single struct with .categories
    person = _struct(
        annotations_categories=_struct(categories=np.array(["Disconnection", "Doubt/Confusion"], dtype=object)),
    )
    assert emotic._person_categories(person) == ["Disconnection", "Doubt/Confusion"]


def test_person_categories_valtest_convention():
    # val/test: combined_categories is a BARE array of strings (no .categories field)
    person = _struct(
        combined_categories=np.array(["Anger", "Annoyance", "Engagement"], dtype=object),
        annotations_categories=np.array([_struct(categories=np.array(["Anger"], dtype=object))], dtype=object),
    )
    assert emotic._person_categories(person) == ["Anger", "Annoyance", "Engagement"]


def test_person_continuous_prefers_combined():
    person = _struct(
        combined_continuous=_struct(valence=3, arousal=6, dominance=6),
        annotations_continuous=np.array([_struct(valence=1, arousal=1, dominance=1)], dtype=object),
    )
    cont = emotic._person_continuous(person)
    assert (cont.valence, cont.arousal, cont.dominance) == (3, 6, 6)


def test_scalar_reduces_annotator_array():
    assert emotic._scalar(np.array([2.0, 4.0])) == 3.0
    assert emotic._scalar(5) == 5.0
    assert emotic._scalar(None) is None


def test_to_shared_single_label_filters_and_reports():
    df = pd.DataFrame({
        "categories": [
            ["Anger"],                 # -> {anger}            keep
            ["Engagement"],            # -> {neutral}          keep
            ["Anger", "Happiness"],    # -> {anger, joy}       drop (multilabel)
            ["Bogus"],                 # unknown -> {}         drop
        ],
        "valence": [2, 5, 8, 4],
    })
    kept = emotic.to_shared_single_label(df)
    assert list(kept["shared_label"]) == ["anger", "neutral"]
    assert kept.attrs["n_dropped_multilabel"] == 2
    assert kept.attrs["unknown_categories"] == ["Bogus"]
