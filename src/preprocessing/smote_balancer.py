"""
SMOTE-based class balancer for RT-MLIDS.
Addresses severe class imbalance in intrusion detection datasets
(e.g. U2R: 52 samples vs Normal: 67,343 in NSL-KDD training set).
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter


class SMOTEBalancer:
    """
    Wraps imblearn SMOTE with dataset-aware logging.
    """

    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._smote = SMOTE(
            k_neighbors=k_neighbors,
            random_state=random_state
        )

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        before = Counter(y)
        X_res, y_res = self._smote.fit_resample(X, y)
        after = Counter(y_res)
        print("[SMOTEBalancer] Class distribution before:", dict(before))
        print("[SMOTEBalancer] Class distribution after: ", dict(after))
        return X_res, y_res
