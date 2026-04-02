"""Unit tests for StackedEnsemble classifier."""

import numpy as np
import pytest
from src.models.ensemble import StackedEnsemble


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 30))
    y = rng.integers(0, 5, size=200)
    return X, y


def test_fit_predict(sample_data):
    X, y = sample_data
    model = StackedEnsemble()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert set(preds).issubset(set(range(5)))


def test_predict_proba_shape(sample_data):
    X, y = sample_data
    model = StackedEnsemble()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 5)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(200), atol=1e-5)


def test_confidence_threshold(sample_data):
    X, y = sample_data
    model = StackedEnsemble(confidence_threshold=0.99)
    model.fit(X, y)
    preds, confs, mask = model.predict_with_confidence(X)
    assert confs[mask].min() >= 0.99 or mask.sum() == 0


def test_unfitted_raises(sample_data):
    X, _ = sample_data
    model = StackedEnsemble()
    with pytest.raises(RuntimeError):
        model.predict(X)
