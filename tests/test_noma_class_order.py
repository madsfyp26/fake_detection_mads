"""NOMA sklearn class order: p(fake) column and Fake/Real strings must match training labels."""

import os

import numpy as np
import pytest


class _Est:
    def __init__(self, classes: list[int]) -> None:
        self.classes_ = np.asarray(classes, dtype=int)


class _Pipe:
    def __init__(self, classes: list[int]) -> None:
        self.steps = [("svm", _Est(classes))]


def test_fake_proba_column_respects_noma_fake_class_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOMA_FAKE_CLASS_LABEL", "1")
    from detectors.noma import noma_fake_proba_column_index

    pipe = _Pipe([0, 1])
    assert noma_fake_proba_column_index(pipe) == 1


def test_p_fake_raw_and_preds_fake_label_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Fake is sklearn label 1, column 1 is P(fake) and argmax labels use classes_."""
    monkeypatch.setenv("NOMA_FAKE_CLASS_LABEL", "1")
    from detectors.noma import noma_p_fake_raw_confidence_and_preds_from_probas

    pipe = _Pipe([0, 1])
    probas = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=float)
    raw, _conf, preds = noma_p_fake_raw_confidence_and_preds_from_probas(pipe, probas)
    assert np.allclose(raw, [0.8, 0.1])
    assert preds == ["Fake", "Real"]


def test_p_fake_raw_default_fake_label_0() -> None:
    """Mozilla convention: label 0 = Fake, column 0 = P(fake)."""
    if "NOMA_FAKE_CLASS_LABEL" in os.environ:
        pytest.skip("NOMA_FAKE_CLASS_LABEL is set in environment")
    from detectors.noma import noma_p_fake_raw_confidence_and_preds_from_probas

    pipe = _Pipe([0, 1])
    probas = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=float)
    raw, _conf, preds = noma_p_fake_raw_confidence_and_preds_from_probas(pipe, probas)
    assert np.allclose(raw, [0.7, 0.1])
    assert preds == ["Fake", "Real"]
