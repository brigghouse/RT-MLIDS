"""RT-MLIDS experiment: regenerates every number reported in the paper.

    python rt_mlids_experiment.py --data-path data/NSL_KDD --seeds 0 1 2 --out results/results.json

Everything (metrics, latency, adversarial robustness, SHAP) is written to the
JSON file; the notebooks and the README tables read from it.
"""

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.svm import SVC

from src.data import CLASS_NAMES, NSLKDDPreprocessor, load_nsl_kdd, map_labels
from src.metrics import binary_metrics, latency_us_per_flow, multiclass_metrics
from src.models.ensemble import StackedEnsemble

warnings.filterwarnings("ignore")


def run_seed(seed, train, test, args, with_extras):
    t_start = time.time()
    y_tr, y_te = map_labels(train["label"]), map_labels(test["label"])
    prep = NSLKDDPreprocessor(k=30, random_state=seed).fit(train, y_tr, vocab_frames=[test])
    X_tr, X_te = prep.transform(train), prep.transform(test)
    print(f"[seed {seed}] features: {len(prep.all_columns_)} one-hot -> {X_tr.shape[1]} selected", flush=True)

    ens = StackedEnsemble(n_folds=args.folds, use_smote=True, random_state=seed,
                          class_names=CLASS_NAMES, n_jobs=args.n_jobs).fit(X_tr, y_tr)
    print(f"[seed {seed}] ensemble fitted ({time.time()-t_start:.0f}s)", flush=True)

    # SVM on a stratified subsample (RBF-SVC is O(n^2) and impractical on the full set)
    rng = np.random.default_rng(seed)
    sub = np.concatenate([rng.choice(np.where(y_tr == c)[0],
                          size=min(int((y_tr == c).sum()), max(1, int(args.svm_n * (y_tr == c).mean()))),
                          replace=False) for c in np.unique(y_tr)])
    svm = SVC(kernel="rbf", random_state=seed).fit(X_tr[sub], y_tr[sub])

    predictors = {
        "Random Forest": (ens.rf.predict, ens.rf.predict_proba),
        "XGBoost": (ens.xgb.predict, ens.xgb.predict_proba),
        "SVM (30k subsample)": (svm.predict, None),
        "Stacked Ensemble": (ens.predict, ens.predict_proba),
    }
    out = {"models": {}, "n_test": int(len(y_te)),
           "test_class_counts": {CLASS_NAMES[c]: int((y_te == c).sum()) for c in np.unique(y_te)}}
    for name, (pred_fn, _) in predictors.items():
        p = pred_fn(X_te)
        out["models"][name] = {
            "multiclass_weighted": multiclass_metrics(y_te, p),
            "binary_attack_vs_normal": binary_metrics(y_te, p),
            "latency_us_per_flow": latency_us_per_flow(pred_fn, X_te),
        }
        print(f"[seed {seed}] {name}: {out['models'][name]['multiclass_weighted']}", flush=True)

    # Confidence threshold reported at fixed operating points (not tuned on the test set)
    proba = ens.predict_proba(X_te)
    conf, pred = proba.max(1), ens.classes_[proba.argmax(1)]
    out["threshold_operating_points"] = {
        str(t): {"coverage": float((conf >= t).mean()),
                 "accuracy_on_confident": float((pred[conf >= t] == y_te[conf >= t]).mean())}
        for t in (0.5, 0.7, 0.85, 0.95)}

    if with_extras:
        import shap
        cat = prep.categorical_selected_
        if not args.no_adv:
            from src.adversarial.attacks import evaluate_adversarial
            out["adversarial"] = {
                "XGBoost": evaluate_adversarial(ens.xgb, X_te, y_te, cat, n_samples=args.adv_n, seed=seed),
                "Stacked Ensemble": evaluate_adversarial(
                    ens, X_te, y_te, cat, n_samples=args.adv_n_ens, seed=seed,
                    hsj_iter=5, zoo_iter=20,  # reduced budget: each query runs RF+XGB+LR
                    hsj_kwargs=dict(max_eval=500, init_eval=10, init_size=50)),
            }
        sv = shap.TreeExplainer(ens.xgb).shap_values(X_te[:500])
        sv = np.asarray(sv)
        # shap returns (n, features, classes) or a per-class list depending on version
        mean = np.abs(sv).mean(axis=0)
        if mean.ndim > 1:
            mean = mean.mean(axis=-1)
        top = np.argsort(mean)[::-1][:5]
        out["shap_top5"] = [{"feature": prep.selected_features_[i], "mean_abs_shap": float(mean[i])} for i in top]
        out["selected_features"] = prep.selected_features_
        out["n_onehot_columns"] = len(prep.all_columns_)
        if args.save_model:
            import joblib, os
            os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
            joblib.dump({"model": ens, "preprocessor": prep}, args.save_model, compress=3)
    out["seconds"] = time.time() - t_start
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/NSL_KDD")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--svm-n", type=int, default=30000)
    ap.add_argument("--adv-n", type=int, default=20)
    ap.add_argument("--adv-n-ens", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--no-adv", action="store_true", help="skip adversarial evaluation (see run_adversarial.py)")
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()
    train, test = load_nsl_kdd(args.data_path)
    runs = {}
    for i, s in enumerate(args.seeds):
        runs[str(s)] = run_seed(s, train, test, args, with_extras=(i == 0))
        agg = {}
        for name in runs[str(args.seeds[0])]["models"]:
            vals = [r["models"][name] for r in runs.values()]
            agg[name] = {grp: {m: [float(np.mean([v[grp][m] for v in vals])), float(np.std([v[grp][m] for v in vals]))]
                               for m in vals[0][grp]} for grp in ("multiclass_weighted", "binary_attack_vs_normal")}
            agg[name]["latency_us_per_flow"] = float(np.mean([v["latency_us_per_flow"] for v in vals]))
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "runs": runs, "aggregate_mean_std": agg}, f, indent=2)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
