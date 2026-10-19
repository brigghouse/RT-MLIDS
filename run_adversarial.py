"""Adversarial evaluation of a saved model; writes results/adversarial.json.

    python rt_mlids_experiment.py --no-adv --save-model models/saved/rt_mlids.pkl
    python run_adversarial.py --model models/saved/rt_mlids.pkl
"""
import argparse, json, warnings
import joblib
from src.adversarial.attacks import evaluate_adversarial
from src.data import load_nsl_kdd, map_labels

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/saved/rt_mlids.pkl")
ap.add_argument("--data-path", default="data/NSL_KDD")
ap.add_argument("--results", default="results/adversarial.json")
ap.add_argument("--adv-n", type=int, default=20)
ap.add_argument("--adv-n-ens", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

b = joblib.load(a.model); ens, prep = b["model"], b["preprocessor"]
_, test = load_nsl_kdd(a.data_path)
X, y, cat = prep.transform(test), map_labels(test["label"]), prep.categorical_selected_
adv = {}

def save():
    R = {"results": adv, "config": {"seed": a.seed, "n_xgboost": a.adv_n, "n_ensemble": a.adv_n_ens,
                               "xgboost_budget": "HSJ max_iter=20, ZOO max_iter=100",
                               "ensemble_budget": "HSJ max_iter=5 (max_eval=500), ZOO max_iter=20"}}
    json.dump(R, open(a.results, "w"), indent=2)

adv["XGBoost"] = evaluate_adversarial(ens.xgb, X, y, cat, n_samples=a.adv_n, seed=a.seed, hsj_iter=20); save(); print("xgb done", flush=True)
adv["Stacked Ensemble"] = evaluate_adversarial(ens, X, y, cat, n_samples=a.adv_n_ens, seed=a.seed, hsj_iter=5, zoo_iter=20,
                                               hsj_kwargs=dict(max_eval=500, init_eval=10, init_size=50)); save(); print("ens done", flush=True)
