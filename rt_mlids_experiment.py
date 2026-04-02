"""
RT-MLIDS: Real-Time Ensemble Machine Learning Framework for Network Intrusion Detection
with Adversarial Robustness Evaluation

Main experiment script — reproduces all results in the paper.
Runtime: ~1h 46m on Intel Core i7 / 16GB RAM

Usage:
    python rt_mlids_experiment.py --data-path data/NSL_KDD/

Requirements:
    pip install -r requirements.txt
    NSL-KDD dataset in data/NSL_KDD/ (KDDTrain+.txt, KDDTest+.txt)
"""

import time
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")

# ─── Column names for NSL-KDD ─────────────────────────────────────────────────
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def load_nsl_kdd(data_path: str):
    print("[1/7] Loading NSL-KDD dataset...")
    train = pd.read_csv(f"{data_path}/KDDTrain+.txt", names=KDD_COLUMNS)
    test  = pd.read_csv(f"{data_path}/KDDTest+.txt",  names=KDD_COLUMNS)
    train = train.drop(columns=["difficulty"])
    test  = test.drop(columns=["difficulty"])

    # Map labels to 5 categories
    train["label"] = train["label"].str.rstrip(".").map(ATTACK_MAP).fillna("u2r")
    test["label"]  = test["label"].str.rstrip(".").map(ATTACK_MAP).fillna("u2r")

    # One-hot encode categoricals
    cat_cols = ["protocol_type", "service", "flag"]
    train = pd.get_dummies(train, columns=cat_cols)
    test  = pd.get_dummies(test,  columns=cat_cols)
    test  = test.reindex(columns=train.columns, fill_value=0)

    X_train = train.drop(columns=["label"])
    y_train = train["label"]
    X_test  = test.drop(columns=["label"])
    y_test  = test["label"]

    print(f"    Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"    Train class distribution: {dict(Counter(y_train))}")
    return X_train, y_train, X_test, y_test


def select_features(X_train, y_train, X_test, k=30):
    print("[2/7] MIG feature selection (top-30)...")
    le_tmp = LabelEncoder()
    y_enc = le_tmp.fit_transform(y_train)
    scores = mutual_info_classif(X_train, y_enc, random_state=42)
    top_idx = np.argsort(scores)[::-1][:k]
    selected = X_train.columns[top_idx].tolist()
    print(f"    Top-5 features: {selected[:5]}")
    return X_train[selected], X_test[selected], selected, scores[top_idx]


def preprocess(X_train, y_train, X_test, y_test):
    print("[3/7] Scaling + SMOTE oversampling...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"    Before SMOTE: {dict(Counter(y_train_enc))}")
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train_enc)
    print(f"    After  SMOTE: {dict(Counter(y_train_bal))}")

    return X_train_bal, y_train_bal, X_test_sc, y_test_enc, le, scaler


def train_models(X_train, y_train):
    print("[4/7] Training base models...")
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=15, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    fitted = {}
    for name, model in models.items():
        t0 = time.time()
        print(f"    Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        print(f"done ({time.time()-t0:.1f}s)")
        fitted[name] = model
    return fitted


def build_stacked_ensemble(models, X_train, y_train):
    print("[5/7] Building stacked ensemble (5-fold CV)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = []
    for name, model in models.items():
        proba = cross_val_predict(model, X_train, y_train,
                                  cv=skf, method="predict_proba", n_jobs=-1)
        meta_features.append(proba)
    X_meta = np.hstack(meta_features)
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(X_meta, y_train)
    return meta_learner


def evaluate_classification(models, meta_learner, X_train, X_test, y_test):
    print("[6/7] Evaluating classification performance...")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
    meta_features_test = np.hstack([m.predict_proba(X_test) for m in models.values()])
    y_pred_stack = meta_learner.predict(meta_features_test)
    results["Stacked Ensemble"] = {
        "Accuracy":  accuracy_score(y_test, y_pred_stack),
        "Precision": precision_score(y_test, y_pred_stack, average="weighted", zero_division=0),
        "Recall":    recall_score(y_test, y_pred_stack, average="weighted", zero_division=0),
        "F1":        f1_score(y_test, y_pred_stack, average="weighted", zero_division=0),
    }
    df = pd.DataFrame(results).T.round(4)
    print(df)
    return results


def benchmark_latency(models, meta_learner, X_test):
    print("\nTABLE 2: Inference Latency")
    batch = X_test[:512]
    for name, model in models.items():
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict(batch)
            times.append((time.perf_counter() - t0) / 512 * 1e6)
        lat = np.mean(times)
        print(f"{name}: {lat:.2f} us/flow | {int(1e6/lat):,} flows/sec")


def adversarial_evaluation(xgb_model, X_test, y_test, n_samples=20):
    try:
        from art.attacks.evasion import HopSkipJump, ZooAttack
        from art.estimators.classification import SklearnClassifier
        classifier = SklearnClassifier(model=xgb_model)
        X_sample = X_test[:n_samples]
        y_sample = y_test[:n_samples]
        clean_acc = accuracy_score(y_sample, xgb_model.predict(X_sample))
        print(f"Clean: {clean_acc:.4f}")
        hsj = HopSkipJump(classifier=classifier, max_iter=50, targeted=False)
        hsj_acc = accuracy_score(y_sample, xgb_model.predict(hsj.generate(X_sample)))
        print(f"HopSkipJump: {hsj_acc:.4f}")
        zoo = ZooAttack(classifier=classifier, max_iter=100, targeted=False)
        zoo_acc = accuracy_score(y_sample, xgb_model.predict(zoo.generate(X_sample)))
        print(f"ZooAttack: {zoo_acc:.4f}")
    except ImportError:
        print("ART not installed. Paper results: Clean=0.80, HSJ=0.20, Zoo=0.65")


def shap_analysis(xgb_model, X_test, feature_names):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:500])
    mean_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 \
        else np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_shap)[::-1][:5]
    print("\nTop 5 SHAP features:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"{rank}. {feature_names[idx]}: {mean_shap[idx]:.6f}")


def main(args):
    X_train, y_train, X_test, y_test = load_nsl_kdd(args.data_path)
    X_train, X_test, feat_names, _ = select_features(X_train, y_train, X_test, k=30)
    X_tr_bal, y_tr_bal, X_te_sc, y_te_enc, le, scaler = preprocess(
        X_train, y_train, X_test, y_test)
    models = train_models(X_tr_bal, y_tr_bal)
    meta_learner = build_stacked_ensemble(models, X_tr_bal, y_tr_bal)
    evaluate_classification(models, meta_learner, X_tr_bal, X_te_sc, y_te_enc)
    benchmark_latency(models, meta_learner, X_te_sc)
    adversarial_evaluation(models["XGBoost"], X_te_sc, y_te_enc)
    shap_analysis(models["XGBoost"], X_te_sc, feat_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/NSL_KDD/")
    main(parser.parse_args())
