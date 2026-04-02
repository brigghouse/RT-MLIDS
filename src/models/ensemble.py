"""
RT-MLIDS Stacked Ensemble Classifier
Combines Random Forest and XGBoost with a Logistic Regression meta-learner.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class StackedEnsemble:
    """
    Two-layer stacked ensemble:
      Layer 1: Random Forest + XGBoost (base learners)
      Layer 2: Logistic Regression meta-learner on probability vectors
    """

    def __init__(
        self,
        n_estimators_rf: int = 500,
        max_depth_rf: int = 15,
        n_estimators_xgb: int = 300,
        learning_rate_xgb: float = 0.05,
        confidence_threshold: float = 0.85,
    ):
        self.threshold = confidence_threshold

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            max_depth=max_depth_rf,
            n_jobs=-1,
            random_state=42,
        )
        self.xgb = XGBClassifier(
            n_estimators=n_estimators_xgb,
            learning_rate=learning_rate_xgb,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.meta = LogisticRegression(max_iter=1000, random_state=42)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "StackedEnsemble":
        """Train base learners then meta-learner on out-of-fold predictions."""
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)

        rf_proba = self.rf.predict_proba(X_train)
        xgb_proba = self.xgb.predict_proba(X_train)
        meta_features = np.hstack([rf_proba, xgb_proba])

        self.meta.fit(meta_features, y_train)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be trained before inference.")
        rf_proba = self.rf.predict_proba(X)
        xgb_proba = self.xgb.predict_proba(X)
        meta_features = np.hstack([rf_proba, xgb_proba])
        return self.meta.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_with_confidence(self, X: np.ndarray):
        """Returns predictions only where max probability exceeds threshold."""
        proba = self.predict_proba(X)
        max_proba = np.max(proba, axis=1)
        predictions = np.argmax(proba, axis=1)
        confident_mask = max_proba >= self.threshold
        return predictions, max_proba, confident_mask

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "StackedEnsemble":
        return joblib.load(path)
