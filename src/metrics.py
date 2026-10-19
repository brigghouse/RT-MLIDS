"""Evaluation helpers: binary + multi-class metrics, Wilson CI, latency."""

from __future__ import annotations

import time
from math import sqrt

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import NORMAL_INDEX


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def multiclass_metrics(y_true, y_pred) -> dict:
    """Weighted-average metrics over the five classes (weighted recall == accuracy)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def binary_metrics(y_true, y_pred, normal_index: int = NORMAL_INDEX) -> dict:
    """Attack (positive) vs normal, after collapsing the four attack classes."""
    t = (np.asarray(y_true) != normal_index).astype(int)
    p = (np.asarray(y_pred) != normal_index).astype(int)
    return {
        "accuracy": accuracy_score(t, p),
        "precision": precision_score(t, p, zero_division=0),
        "recall": recall_score(t, p, zero_division=0),
        "f1": f1_score(t, p, zero_division=0),
    }


def latency_us_per_flow(predict_fn, X, batch: int = 512, repeats: int = 20) -> float:
    """Mean batch-inference latency in microseconds per flow (model only)."""
    xb = X[:batch]
    predict_fn(xb)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(xb)
        times.append((time.perf_counter() - t0) / len(xb) * 1e6)
    return float(np.mean(times))
