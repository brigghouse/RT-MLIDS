"""Stacked ensemble: Random Forest + XGBoost with a logistic-regression meta-learner.

The meta-learner is trained on *out-of-fold* base-learner probabilities, and
SMOTE (optional) is applied inside each training fold only, so neither the
meta-features nor the validation folds see synthetic or in-sample data.
"""

from __future__ import annotations

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


class StackedEnsemble:
    def __init__(
        self,
        n_estimators_rf: int = 500,
        max_depth_rf: int = 15,
        n_estimators_xgb: int = 300,
        learning_rate_xgb: float = 0.05,
        max_depth_xgb: int = 6,
        confidence_threshold: float = 0.85,
        n_folds: int = 5,
        use_smote: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        class_names=None,
    ):
        self.n_estimators_rf = n_estimators_rf
        self.max_depth_rf = max_depth_rf
        self.n_estimators_xgb = n_estimators_xgb
        self.learning_rate_xgb = learning_rate_xgb
        self.max_depth_xgb = max_depth_xgb
        self.threshold = confidence_threshold
        self.n_folds = n_folds
        self.use_smote = use_smote
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.class_names = list(class_names) if class_names is not None else None
        self._fitted = False

    # -- construction helpers -------------------------------------------------
    def _make_rf(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators_rf, max_depth=self.max_depth_rf,
            n_jobs=self.n_jobs, random_state=self.random_state)

    def _make_xgb(self):
        return XGBClassifier(
            n_estimators=self.n_estimators_xgb, learning_rate=self.learning_rate_xgb,
            max_depth=self.max_depth_xgb, eval_metric="mlogloss",
            n_jobs=self.n_jobs if self.n_jobs > 0 else 0,
            random_state=self.random_state)

    def _resample(self, X, y):
        if not self.use_smote:
            return X, y
        k = max(1, min(5, int(np.bincount(y)[np.bincount(y) > 0].min()) - 1))
        return SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(X, y)

    # -- training -------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedEnsemble":
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        oof = np.zeros((len(X), 2 * n_cls))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for tr, va in skf.split(X, y):
            Xt, yt = self._resample(X[tr], y[tr])
            rf, xgb = self._make_rf().fit(Xt, yt), self._make_xgb().fit(Xt, yt)
            oof[va] = np.hstack([rf.predict_proba(X[va]), xgb.predict_proba(X[va])])
        self.meta = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.meta.fit(oof, y)
        Xf, yf = self._resample(X, y)
        self.rf = self._make_rf().fit(Xf, yf)
        self.xgb = self._make_xgb().fit(Xf, yf)
        self._fitted = True
        return self

    # -- inference ------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be trained before inference.")
        meta_X = np.hstack([self.rf.predict_proba(X), self.xgb.predict_proba(X)])
        return self.meta.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_with_confidence(self, X: np.ndarray):
        """Return (predictions, max_probability, mask of predictions >= threshold)."""
        proba = self.predict_proba(X)
        conf = proba.max(axis=1)
        return self.classes_[np.argmax(proba, axis=1)], conf, conf >= self.threshold

    # -- persistence ----------------------------------------------------------
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "StackedEnsemble":
        return joblib.load(path)
