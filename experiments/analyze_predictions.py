"""Threshold and calibration analysis of the saved test-set probabilities
(results/runs/<dataset>/<experiment>/fl_test_preds_seed*.npz), to explain how a strategy
can rank patients well (high AUC) and still score a low F1.

Per strategy and seed: the confusion matrix at the fixed 0.5 threshold and
at the validation-calibrated threshold (read from fl_final_results), the
Brier score, and quantiles of the predicted probability for survivors and
non-survivors. Also plots the predicted-probability distributions by class
for one seed.

Writes prediction_analysis.csv and prob_distributions.png into the same
directory.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, confusion_matrix  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs")
STRATEGIES = ["FedAvg", "FedProx", "SCAFFOLD", "FedAdagrad", "FedAdam", "FedYogi", "FedCluster"]


def cm_row(y, p, t, prefix):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return {f"{prefix}_tp": tp, f"{prefix}_fp": fp, f"{prefix}_fn": fn, f"{prefix}_tn": tn,
            f"{prefix}_precision": tp / (tp + fp) if tp + fp else 0.0,
            f"{prefix}_recall": tp / (tp + fn) if tp + fn else 0.0,
            f"{prefix}_pred_pos_rate": (tp + fp) / len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic", choices=["mimic", "eicu", "eicu_aps"])
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--plot_seed", type=int, default=42)
    args = ap.parse_args()
    d = os.path.join(RUNS, args.dataset, args.experiment)

    rows = []
    for path in sorted(glob.glob(os.path.join(glob.escape(d), "fl_test_preds_seed*.npz"))):
        seed = int(path.rsplit("seed", 1)[1][:-4])
        final = pd.read_csv(os.path.join(d, f"fl_final_results_seed{seed}.csv")).set_index("strategy")
        z = np.load(path)
        y = z["y_test"].astype(int)
        for s in STRATEGIES:
            if s not in z:
                continue
            p = z[s]
            t = float(final.loc[s, "threshold"])
            row = dict(strategy=s, seed=seed, threshold=t, brier=brier_score_loss(y, p),
                       mean_p_pos=p[y == 1].mean(), mean_p_neg=p[y == 0].mean())
            for q in (0.1, 0.5, 0.9):
                row[f"p_pos_q{int(q * 100)}"] = np.quantile(p[y == 1], q)
                row[f"p_neg_q{int(q * 100)}"] = np.quantile(p[y == 0], q)
            row.update(cm_row(y, p, 0.5, "at05"))
            row.update(cm_row(y, p, t, "calib"))
            rows.append(row)
    if not rows:
        raise SystemExit(f"no prediction files in {d}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "prediction_analysis.csv"), index=False)
    cols = ["threshold", "brier", "mean_p_pos", "mean_p_neg", "at05_precision", "at05_recall",
            "at05_pred_pos_rate", "calib_precision", "calib_recall"]
    print(df.groupby("strategy", sort=False)[cols].mean().to_string(float_format="%.3f"))

    z = np.load(os.path.join(d, f"fl_test_preds_seed{args.plot_seed}.npz"))
    y = z["y_test"].astype(int)
    present = [s for s in STRATEGIES if s in z]
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 2.8), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, s in zip(np.atleast_1d(axes), present):
        p = z[s]
        ax.hist(p[y == 0], bins=bins, density=True, alpha=0.6, label="Survived")
        ax.hist(p[y == 1], bins=bins, density=True, alpha=0.6, label="Died")
        t = float(df[(df.strategy == s) & (df.seed == args.plot_seed)]["threshold"].iloc[0])
        ax.axvline(0.5, color="k", ls=":", lw=1)
        ax.axvline(t, color="k", ls="--", lw=1)
        ax.set_title(s, fontsize=10)
        ax.set_xlabel("Predicted probability")
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].set_ylabel("Density (log)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "prob_distributions.png"), dpi=200)


if __name__ == "__main__":
    main()
