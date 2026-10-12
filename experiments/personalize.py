"""Personalization by local fine-tuning.

Starting from each strategy's selected global model (saved by
run_fl_experiments.py as fl_models_seed*.pt), every client trains for
FT_EPOCHS further epoch(s) on its own training partition, with the same
local optimizer, learning rate, batch size and class weighting as during
federated training, and the resulting personalized model is evaluated on
that client's own test partition. The global model is evaluated on the same
rows, so each (seed, strategy, client) gives a paired global-vs-personalized
comparison. Pooling every client's personalized predictions over its own
test rows also gives a whole-test-set AUC comparable to the global one.

    python experiments/personalize.py --dataset eicu --experiment main

Writes personalized_per_client.csv and personalized_pooled.csv into
results/runs/<dataset>/<experiment>/.
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
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import data as data_mod  # noqa: E402
import federated as fed  # noqa: E402
from models import get_model  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs")
FT_EPOCHS = 1


def _auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)


def personalize_seed(dataset, d, seed, lr=0.001, batch_size=512):
    final = pd.read_csv(os.path.join(d, f"fl_final_results_seed{seed}.csv"))
    run = final.iloc[0]
    df = data_mod.load_cohort(dataset)
    bundle = data_mod.build_clients_cached(df, seed=seed, config=run["config"],
                                           imbalance=run["imbalance"], dataset=dataset)
    weights = torch.load(os.path.join(d, f"fl_models_seed{seed}.pt"))
    model = get_model(run["model"], bundle["n_features"])

    per_client, pooled = [], []
    for strategy, global_flat in weights.items():
        ys, p_glob, p_pers = [], [], []
        for unit, c in bundle["clients"].items():
            if len(c["y_test"]) == 0:
                continue
            g = fed.predict_proba(model, c["X_test"], global_flat)
            flat, _ = fed.train_local(model, global_flat, c["X_train_res"], c["y_train_res"],
                                      c["pos_weight"], local_epochs=FT_EPOCHS,
                                      batch_size=batch_size, lr=lr, seed=seed)
            p = fed.predict_proba(model, c["X_test"], flat)
            (ga, gp), (pa, pp) = _auc(c["y_test"], g), _auc(c["y_test"], p)
            per_client.append(dict(seed=seed, strategy=strategy, client=unit,
                                   n_test=len(c["y_test"]), n_pos_test=int(c["y_test"].sum()),
                                   global_auc_roc=ga, global_auc_pr=gp,
                                   pers_auc_roc=pa, pers_auc_pr=pp))
            ys.append(c["y_test"]), p_glob.append(g), p_pers.append(p)
        y = np.concatenate(ys)
        (ga, gp), (pa, pp) = _auc(y, np.concatenate(p_glob)), _auc(y, np.concatenate(p_pers))
        pooled.append(dict(seed=seed, strategy=strategy, global_auc_roc=ga, global_auc_pr=gp,
                           pers_auc_roc=pa, pers_auc_pr=pp))
        print(f"[seed {seed}] {strategy:10s} pooled AUC-ROC {ga:.4f} -> {pa:.4f}  "
              f"AUC-PR {gp:.4f} -> {pp:.4f}", flush=True)
    return per_client, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic", choices=data_mod.DATASETS)
    ap.add_argument("--experiment", default="main")
    args = ap.parse_args()
    d = os.path.join(RUNS, args.dataset, args.experiment)

    per_client, pooled = [], []
    for path in sorted(glob.glob(os.path.join(glob.escape(d), "fl_models_seed*.pt"))):
        seed = int(path.rsplit("seed", 1)[1][:-3])
        pc, po = personalize_seed(args.dataset, d, seed)
        per_client.extend(pc)
        pooled.extend(po)
    if not pooled:
        raise SystemExit(f"no saved models in {d}")
    pc = pd.DataFrame(per_client)
    pc.to_csv(os.path.join(d, "personalized_per_client.csv"), index=False)
    po = pd.DataFrame(pooled)
    po.to_csv(os.path.join(d, "personalized_pooled.csv"), index=False)
    print(po.groupby("strategy", sort=False).mean(numeric_only=True).drop(columns="seed")
            .to_string(float_format="%.4f"))
    gain = (pc["pers_auc_pr"] - pc["global_auc_pr"]).groupby(pc["strategy"]).agg(["mean", "median"])
    print("\nPer-client AUC-PR gain from fine-tuning:\n", gain.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
