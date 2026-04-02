"""
Offline evaluation script for RT-MLIDS on NSL-KDD or CIC-IDS-2018.

Usage:
    python src/evaluate.py --dataset nsl-kdd --data-path data/NSL_KDD/
    python src/evaluate.py --dataset cic-ids-2018 --data-path data/CIC_IDS_2018/
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.models.ensemble import StackedEnsemble
from src.preprocessing.feature_selection import MIGFeatureSelector
from src.preprocessing.smote_balancer import SMOTEBalancer


def load_nsl_kdd(data_path: str):
    train = pd.read_csv(f"{data_path}/KDDTrain+.txt", header=None)
    test  = pd.read_csv(f"{data_path}/KDDTest+.txt",  header=None)
    train = train.iloc[:, :-1]
    test  = test.iloc[:, :-1]
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test,  y_test  = test.iloc[:, :-1],  test.iloc[:, -1]
    return X_train, y_train, X_test, y_test


def evaluate(args):
    print(f"[RT-MLIDS] Loading dataset: {args.dataset}")
    if args.dataset == "nsl-kdd":
        X_train, y_train, X_test, y_test = load_nsl_kdd(args.data_path)
    else:
        raise NotImplementedError(f"Dataset loader for {args.dataset} not implemented.")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    selector = MIGFeatureSelector(k=30)
    X_train = selector.fit_transform(pd.DataFrame(X_train), pd.Series(y_train))
    X_test  = selector.transform(pd.DataFrame(X_test))

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    balancer = SMOTEBalancer()
    X_train, y_train = balancer.fit_resample(X_train, y_train)

    print("[RT-MLIDS] Training ensemble...")
    model = StackedEnsemble()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== RT-MLIDS Evaluation Results ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if args.save_model:
        model.save("models/saved/rt_mlids.pkl")
        print("[RT-MLIDS] Model saved to models/saved/rt_mlids.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-MLIDS Offline Evaluation")
    parser.add_argument("--dataset",    required=True, choices=["nsl-kdd", "cic-ids-2018"])
    parser.add_argument("--data-path",  required=True)
    parser.add_argument("--save-model", action="store_true", default=False)
    evaluate(parser.parse_args())
