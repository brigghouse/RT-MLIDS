"""Evaluate a saved model on the NSL-KDD test set.

    python -m src.evaluate --data-path data/NSL_KDD --model models/saved/rt_mlids.pkl
"""

import argparse

import joblib

from src.data import CLASS_NAMES, load_nsl_kdd, map_labels
from src.metrics import binary_metrics, multiclass_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/NSL_KDD")
    ap.add_argument("--model", default="models/saved/rt_mlids.pkl")
    a = ap.parse_args()
    bundle = joblib.load(a.model)  # {"model": StackedEnsemble, "preprocessor": NSLKDDPreprocessor}
    _, test = load_nsl_kdd(a.data_path)
    y = map_labels(test["label"])
    pred = bundle["model"].predict(bundle["preprocessor"].transform(test))
    print("classes:", CLASS_NAMES)
    print("multi-class (weighted):", {k: round(v, 4) for k, v in multiclass_metrics(y, pred).items()})
    print("binary (attack vs normal):", {k: round(v, 4) for k, v in binary_metrics(y, pred).items()})


if __name__ == "__main__":
    main()
