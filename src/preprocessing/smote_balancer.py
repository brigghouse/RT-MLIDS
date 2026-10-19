"""Thin SMOTE wrapper (use inside cross-validation folds, never before splitting)."""

import numpy as np
from imblearn.over_sampling import SMOTE


class SMOTEBalancer:
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        counts = np.bincount(y)
        k = max(1, min(self.k_neighbors, int(counts[counts > 0].min()) - 1))
        return SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(X, y)
