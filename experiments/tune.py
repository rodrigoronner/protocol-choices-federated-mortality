"""Per-strategy hyperparameter tuning on the global VALIDATION partition.

Each strategy gets a small grid over its own hyperparameters (the shared
client-side settings -- local learning rate, local epochs, batch size,
rounds -- are held fixed here and studied separately in the sensitivity
experiments). The selection criterion is validation AUC-PR, the primary
metric under this class imbalance, with validation AUC-ROC as tie-breaker;
the test partition is never looked at.

The grid is split across workers so it can run in parallel:

    python experiments/tune.py --dataset mimic --worker 0 --n_workers 3 &
    python experiments/tune.py --dataset mimic --worker 1 --n_workers 3 &
    python experiments/tune.py --dataset mimic --worker 2 --n_workers 3 &
    wait
    python experiments/tune.py --dataset mimic --select

--select writes results/tuning/<dataset>/best_hp.json, which
run_fl_experiments.py reads through --hp_json. Each dataset is tuned
separately.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _threading_limits  # noqa: E402,F401  (must be imported before numpy/sklearn/torch)

import pandas as pd

import data as data_mod  # noqa: E402
from run_fl_experiments import RESULTS_DIR, run_strategy  # noqa: E402

TUNING_DIR = os.path.join(RESULTS_DIR, "tuning")

# Server learning rate and adaptivity constant follow the log-scale grids of
# Reddi et al. (2021); mu covers the range explored by Li et al. (2020).
GRID = (
    [("FedAvg", {})]
    + [("FedProx", {"mu": mu}) for mu in (0.001, 0.01, 0.1, 1.0)]
    + [("FedAdagrad", {"eta_adagrad": eta, "tau_adagrad": tau})
       for eta in (0.001, 0.01, 0.1) for tau in (1e-3, 1e-2)]
    + [("FedAdam", {"eta_adam": eta, "tau_adam": tau})
       for eta in (0.001, 0.01, 0.1) for tau in (1e-3, 1e-2)]
    + [("FedCluster", {"eps_quantile": q}) for q in (0.25, 0.5, 0.75)]
    # Added after the first tuning pass; appended so that the worker
    # assignment (index modulo n_workers) of the configs above is unchanged.
    # SCAFFOLD trains locally with SGD (see federated.Scaffold), so its local
    # learning rate is tuned together with the global step.
    + [("SCAFFOLD", {"lr_scaffold": lr, "eta_g": eta}) for lr in (0.003, 0.01, 0.03) for eta in (0.5, 1.0)]
    + [("FedYogi", {"eta_yogi": eta, "tau_yogi": tau})
       for eta in (0.001, 0.01, 0.1) for tau in (1e-3, 1e-2)]
)


# Larger local SGD rates for SCAFFOLD. In a first tuning pass (with SMOTE-
# Tomek) its optimum on eICU-CRD landed on the largest rate of GRID (0.03,
# best round = last), and 0.3 won once 0.1-1.0 were added; the rates are
# appended for both datasets so that both are tuned on the same grid. With
# class weighting, eICU-CRD's optimum moved to 1.0, so 3.0 is added there.
EXTRA_GRID = {
    ds: [("SCAFFOLD", {"lr_scaffold": lr, "eta_g": eta}) for lr in (0.1, 0.3, 1.0) for eta in (0.5, 1.0)]
    for ds in ("mimic", "eicu")
}
EXTRA_GRID["eicu"] += [("SCAFFOLD", {"lr_scaffold": 3.0, "eta_g": eta}) for eta in (0.5, 1.0)]


def run_worker(worker, n_workers, seed, config, rounds, dataset, imbalance="weighted"):
    tdir = os.path.join(TUNING_DIR, dataset)
    os.makedirs(tdir, exist_ok=True)
    out = os.path.join(tdir, f"grid_worker{worker}.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    # A config counts as done when any worker's file has it, so the grid can
    # be resumed or extended with a different number of workers.
    done = set()
    for p in glob.glob(os.path.join(glob.escape(tdir), "grid_worker*.csv")):
        done |= {(r["strategy"], r["hp_grid"]) for r in pd.read_csv(p).to_dict("records")}

    df = data_mod.load_cohort(dataset)
    bundle = data_mod.build_clients_cached(df, seed=seed, config=config, imbalance=imbalance,
                                           dataset=dataset)
    for i, (strategy, hp) in enumerate(GRID + EXTRA_GRID.get(dataset, [])):
        if i % n_workers != worker:
            continue
        key = (strategy, json.dumps(hp, sort_keys=True))
        if key in done:
            continue
        print(f"\n=== [{worker}] {strategy} {hp} ===", flush=True)
        _, final, _, _, _ = run_strategy(strategy, bundle, rounds=rounds, hp=hp, seed=seed,
                                         eval_every=rounds)
        final.update(hp_grid=key[1], seed=seed, dataset=dataset, config=config, imbalance=imbalance)
        rows.append(final)
        pd.DataFrame(rows).to_csv(out, index=False)


def select(dataset):
    tdir = os.path.join(TUNING_DIR, dataset)
    grid = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(glob.escape(tdir), "grid_worker*.csv")))],
                     ignore_index=True)
    grid = grid.sort_values(["strategy", "val_auc_pr", "val_auc_roc"], ascending=[True, False, False])
    grid.to_csv(os.path.join(tdir, "grid_all.csv"), index=False)
    best = {}
    for strategy, sub in grid.groupby("strategy"):
        best[strategy] = json.loads(sub.iloc[0]["hp_grid"])
    with open(os.path.join(tdir, "best_hp.json"), "w") as fh:
        json.dump(best, fh, indent=2, sort_keys=True)
    cols = ["strategy", "hp_grid", "val_auc_pr", "val_auc_roc"]
    print(grid[cols].to_string(index=False))
    print("\nSelected:", json.dumps(best, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic", choices=["mimic", "eicu"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="min_deaths", choices=data_mod.CLIENT_CONFIGS)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--imbalance", default="weighted", choices=data_mod.IMBALANCE_MODES)
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    if args.select:
        select(args.dataset)
    else:
        run_worker(args.worker, args.n_workers, args.seed, args.config, args.rounds, args.dataset,
                   args.imbalance)


if __name__ == "__main__":
    main()
