"""Unit tests for StackedEnsemble."""

import numpy as np
import pytest

from src.models.ensemble import StackedEnsemble

SMALL = dict(n_estimators_rf=20, max_depth_rf=5, n_estimators_xgb=20, n_folds=3, n_jobs=1)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 5, size=300)
    X = rng.standard_normal((300, 8)) + y[:, None]  # separable-ish
    return X, y


def test_fit_predict(data):
    X, y = data
    m = StackedEnsemble(**SMALL).fit(X, y)
    assert m.predict(X).shape == (300,)
    assert (m.predict(X) == y).mean() > 0.6


def test_predict_proba_shape_and_sum(data):
    X, y = data
    p = StackedEnsemble(**SMALL).fit(X, y).predict_proba(X)
    assert p.shape == (300, 5)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_meta_learner_is_trained_on_out_of_fold_predictions(data, monkeypatch):
    """k fold-models + 1 final model per base learner; meta input has 2 * n_classes columns."""
    X, y = data
    m = StackedEnsemble(**SMALL)
    built = []
    orig = m._make_rf
    monkeypatch.setattr(m, "_make_rf", lambda: built.append(1) or orig())
    m.fit(X, y)
    assert len(built) == SMALL["n_folds"] + 1
    assert m.meta.coef_.shape == (5, 10)


def test_smote_inside_folds_runs_on_imbalanced_data():
    rng = np.random.default_rng(1)
    y = np.array([0] * 200 + [1] * 100 + [2] * 12)
    X = rng.standard_normal((len(y), 6)) + y[:, None]
    m = StackedEnsemble(use_smote=True, **SMALL).fit(X, y)
    assert set(m.predict(X)).issubset({0, 1, 2})


def test_confidence_threshold(data):
    X, y = data
    m = StackedEnsemble(confidence_threshold=0.99, **SMALL).fit(X, y)
    _, conf, mask = m.predict_with_confidence(X)
    assert (conf[mask] >= 0.99).all()


def test_unfitted_raises(data):
    with pytest.raises(RuntimeError):
        StackedEnsemble(**SMALL).predict(data[0])


def test_save_load_roundtrip(tmp_path, data):
    X, y = data
    m = StackedEnsemble(**SMALL).fit(X, y)
    m.save(str(tmp_path / "m.pkl"))
    np.testing.assert_array_equal(StackedEnsemble.load(str(tmp_path / "m.pkl")).predict(X), m.predict(X))
