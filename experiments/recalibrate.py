"""Post-hoc recalibration of the selected global models.

Imbalance corrections (class weights, SMOTE-Tomek) shift predicted risks
upwards without improving discrimination (van den Goorbergh et al., 2022).
For every saved global model (fl_models_seed*.pt) this script maps its
validation-set probabilities to the outcome with

  - Platt scaling: logistic regression of y on logit(p);
  - isotonic regression of y on p;

and reports, on the untouched test set, for the raw and the two
recalibrated predictions: Brier score, calibration-in-the-large (the
intercept of a logistic model of y with logit(p) as offset; 0 is ideal),
calibration slope (slope of y on logit(p); 1 is ideal), expected
calibration error over 10 equal-frequency bins, and AUC-ROC / AUC-PR.
Platt scaling is monotone, so it leaves both AUCs unchanged; isotonic
regression can create ties and move them slightly.

Recalibration needs only the validation predictions, which the server can
obtain the same way it selects the best round, so it is applicable in the
federated setting as a final step.

    python experiments/recalibrate.py --dataset eicu --experiment main

Writes recalibration.csv into results/runs/<dataset>/<experiment>/.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import _threading_limits  # noqa: E402,F401  (must be imported before numpy/sklearn/torch)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

import data as data_mod  # noqa: E402
import federated as fed  # noqa: E402
from models import get_model  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs")
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_in_the_large(y, p, iters=50):
    """Intercept a of logit P(y=1) = a + logit(p), by Newton's method."""
    off = logit(p)
    a = 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a + off)))
        grad = np.sum(y - q)
        hess = np.sum(q * (1 - q))
        step = grad / max(hess, 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return a


def calibration_slope(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def ece(y, p, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    err = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return err


def metrics(y, p):
    return dict(brier=brier_score_loss(y, p), citl=calibration_in_the_large(y, p),
                slope=calibration_slope(y, p), ece=ece(y, p),
                auc_roc=roc_auc_score(y, p), auc_pr=average_precision_score(y, p),
                mean_p=float(p.mean()), prevalence=float(y.mean()))


def recalibrate_seed(dataset, d, seed):
    run = pd.read_csv(os.path.join(d, f"fl_final_results_seed{seed}.csv")).iloc[0]
    bundle = data_mod.build_clients_cached(data_mod.load_cohort(dataset), seed=seed,
                                           config=run["config"], imbalance=run["imbalance"],
                                           dataset=dataset)
    weights = torch.load(os.path.join(d, f"fl_models_seed{seed}.pt"))
    model = get_model(run["model"], bundle["n_features"])
    yv, yt = bundle["y_val_global"].astype(int), bundle["y_test_global"].astype(int)

    rows = []
    for strategy, flat in weights.items():
        pv = fed.predict_proba(model, bundle["X_val_global"], flat)
        pt = fed.predict_proba(model, bundle["X_test_global"], flat)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv).reshape(-1, 1), yv)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(pv, yv)
        for method, p in (("raw", pt),
                          ("platt", platt.predict_proba(logit(pt).reshape(-1, 1))[:, 1]),
                          ("isotonic", iso.predict(pt))):
            rows.append(dict(seed=seed, strategy=strategy, method=method,
                             imbalance=run["imbalance"], **metrics(yt, p)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic", choices=data_mod.DATASETS)
    ap.add_argument("--experiment", default="main")
    args = ap.parse_args()
    d = os.path.join(RUNS, args.dataset, args.experiment)

    rows = []
    for path in sorted(glob.glob(os.path.join(glob.escape(d), "fl_models_seed*.pt"))):
        seed = int(path.rsplit("seed", 1)[1][:-3])
        rows.extend(recalibrate_seed(args.dataset, d, seed))
        print(f"seed {seed} done", flush=True)
    if not rows:
        raise SystemExit(f"no saved models in {d}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "recalibration.csv"), index=False)
    cols = ["brier", "citl", "slope", "ece", "auc_roc", "auc_pr", "mean_p"]
    summary = df.groupby(["method", "strategy"], sort=False)[cols].mean()
    print(summary.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
