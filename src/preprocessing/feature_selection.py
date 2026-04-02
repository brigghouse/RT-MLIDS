"""
Mutual Information Gain feature selector for RT-MLIDS.
Selects top-k features maximising class separability.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class MIGFeatureSelector:
    """
    Selects the top-k features using Mutual Information Gain (MIG).
    Model-agnostic and captures non-linear feature-class dependencies.
    Reducing from 53 to 30 features cuts inference latency ~18%.
    """

    def __init__(self, k: int = 30):
        self.k = k
        self.selected_features_: list[str] | None = None
        self.scores_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIGFeatureSelector":
        scores = mutual_info_classif(X, y, random_state=42)
        self.scores_ = scores
        top_k_idx = np.argsort(scores)[::-1][: self.k]
        self.selected_features_ = list(X.columns[top_k_idx])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted before transform.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def feature_importance_report(self) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Selector must be fitted first.")
        return pd.DataFrame(
            {"feature": self.selected_features_,
             "mi_score": self.scores_[
                 np.argsort(self.scores_)[::-1][: self.k]
             ]}
        ).sort_values("mi_score", ascending=False)
