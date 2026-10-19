"""Black-box adversarial evaluation with the Adversarial Robustness Toolbox (ART).

Install: ``pip install adversarial-robustness-toolbox`` (the PyPI name is *not* ``art``).

Both attacks (HopSkipJump, ZOO) only query predicted probabilities, so any
model exposing ``predict_proba`` can be attacked through ``BlackBoxClassifier``.
Raw ART output can be an invalid NSL-KDD record (values outside [0, 1],
fractional one-hot columns), so we report accuracy both on the raw adversarial
points and after *projection* onto the valid feature space.
"""

from __future__ import annotations

import numpy as np

from src.metrics import wilson_ci


def project_to_valid(X_adv, categorical_idx):
    """Clip to the [0, 1] MinMax range and round one-hot columns to {0, 1}."""
    out = np.clip(X_adv, 0.0, 1.0)
    if len(categorical_idx):
        out[:, categorical_idx] = np.round(out[:, categorical_idx])
    return out


def _accuracy_with_ci(y_true, y_pred):
    k, n = int((y_true == y_pred).sum()), len(y_true)
    lo, hi = wilson_ci(k, n)
    return {"accuracy": k / n, "ci95": [lo, hi], "n": n}


def evaluate_adversarial(model, X, y, categorical_idx=(), n_samples=100, seed=0,
                         hsj_iter=50, zoo_iter=100, attacks=("hsj", "zoo"), hsj_kwargs=None):
    """Attack a random test sample and return clean / adversarial / projected accuracy.

    ``model`` needs ``predict`` and ``predict_proba``. Samples are drawn at random
    (seeded), not taken from the head of the file.
    """
    from art.attacks.evasion import HopSkipJump, ZooAttack
    from art.estimators.classification import BlackBoxClassifier

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    Xs, ys = X[idx].astype(np.float32), np.asarray(y)[idx]
    n_feat, n_cls = Xs.shape[1], model.predict_proba(Xs[:1]).shape[1]
    clf = BlackBoxClassifier(lambda x: model.predict_proba(np.asarray(x, dtype=np.float32)),
                             (n_feat,), n_cls, clip_values=None)
    res = {"clean": _accuracy_with_ci(ys, model.predict(Xs))}
    builders = {
        "hsj": lambda: HopSkipJump(classifier=clf, max_iter=hsj_iter, targeted=False, verbose=False,
                                   **(hsj_kwargs or {})),
        "zoo": lambda: ZooAttack(classifier=clf, max_iter=zoo_iter, targeted=False,
                                 nb_parallel=min(n_feat, 10), verbose=False),
    }
    cat = list(categorical_idx)
    for name in attacks:
        adv = builders[name]().generate(Xs)
        res[name] = {
            "raw": _accuracy_with_ci(ys, model.predict(adv)),
            "projected": _accuracy_with_ci(ys, model.predict(project_to_valid(adv, cat))),
            "median_l2": float(np.median(np.linalg.norm(adv - Xs, axis=1))),
        }
    return res
