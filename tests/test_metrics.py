import numpy as np

from src.data import NORMAL_INDEX
from src.metrics import binary_metrics, multiclass_metrics, wilson_ci


def test_wilson_interval_contains_estimate_and_is_wide_for_small_n():
    lo, hi = wilson_ci(16, 20)
    assert lo < 0.8 < hi and (hi - lo) > 0.25
    lo2, hi2 = wilson_ci(8000, 10000)
    assert (hi2 - lo2) < 0.02


def test_binary_collapses_attack_classes():
    y = np.array([NORMAL_INDEX, 0, 2, 3, 4])
    p = np.array([NORMAL_INDEX, 2, 0, NORMAL_INDEX, 4])  # dos<->probe confusion is still a correct "attack"
    m = binary_metrics(y, p)
    assert m["recall"] == 3 / 4 and m["precision"] == 1.0


def test_weighted_recall_equals_accuracy():
    y = np.array([0, 1, 1, 2, 3, 4, 1]); p = np.array([0, 1, 2, 2, 3, 1, 1])
    m = multiclass_metrics(y, p)
    assert abs(m["recall"] - m["accuracy"]) < 1e-12
