"""Lists (or, with --run N, executes N at a time) one shell command per
pending experiment run, for both datasets:

    python experiments/jobs.py            # print the pending commands
    python experiments/jobs.py --run 3    # run them, 3 in parallel

Runs whose final CSV already lists every strategy are skipped, so the queue
can be stopped and resumed at any time. Every FL run uses the hyperparameters
selected for its dataset by experiments/tune.py
(results/tuning/<dataset>/best_hp.json). The queue is ordered by priority:
the main experiments of both datasets first.

Experiments (results/runs/<dataset>/<name>/, one file set per seed):
  main          clients = sites with >= 100 deaths (MIMIC-IV: 12 care units;
                eICU: 49 hospitals), class-weighted loss, 10 seeds, + centralized
  imb_smote     ablation: SMOTE-Tomek on each client's training data, 10 seeds
  imb_none      ablation: no imbalance handling at all, 10 seeds
  cfg_other     + one client pooling every other site, 5 seeds (+ centralized)
  cfg_min200    sites with >= 200 deaths only (eICU: 17 hospitals), 5 seeds
  cfg_edsplit4  MIMIC-IV: Emergency Department split into 4 clients, 5 seeds
  cfg_top5      MIMIC-IV: the five largest care units, 5 seeds
  sens_e1/e5    1 or 5 local epochs instead of 3, 3 seeds
  sens_part50   half of the clients sampled each round, 3 seeds
  sens_part20   eICU: a fifth of the clients sampled each round, 3 seeds
  sens_r100     MIMIC-IV: 100 rounds instead of 50, 1 seed
  arch_logreg   logistic regression instead of the MLP, 3 seeds
  eicu_aps/main eICU + first-day vitals/GCS/urine/ventilation, 5 seeds (+ centralized)
"""
import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
PY = sys.executable
N_STRATEGIES = 7

SEEDS10 = list(range(42, 52))
SEEDS5 = SEEDS10[:5]
SEEDS3 = SEEDS10[:3]

# (dataset, experiment, seeds, extra FL arguments, centralized config or None)
EXPERIMENTS = [
    ("mimic", "main", SEEDS10, [], "min_deaths"),
    ("eicu", "main", SEEDS10, [], "min_deaths"),
    ("mimic", "imb_smote", SEEDS10, ["--imbalance", "smote_tomek"], None),
    ("eicu", "imb_smote", SEEDS10, ["--imbalance", "smote_tomek"], None),
    ("mimic", "imb_none", SEEDS10, ["--imbalance", "none"], None),
    ("eicu", "imb_none", SEEDS10, ["--imbalance", "none"], None),
    ("mimic", "cfg_other", SEEDS5, ["--config", "min_deaths_other"], "min_deaths_other"),
    ("eicu", "cfg_other", SEEDS5, ["--config", "min_deaths_other"], "min_deaths_other"),
    ("eicu", "cfg_min200", SEEDS5, ["--config", "min_deaths200"], "min_deaths200"),
    ("mimic", "cfg_edsplit4", SEEDS5, ["--config", "min_deaths_edsplit4"], None),
    ("mimic", "cfg_top5", SEEDS5, ["--config", "top5"], "top5"),
    ("mimic", "sens_part50", SEEDS3, ["--participation", "0.5"], None),
    ("eicu", "sens_part50", SEEDS3, ["--participation", "0.5"], None),
    ("eicu", "sens_part20", SEEDS3, ["--participation", "0.2"], None),
    ("mimic", "sens_e1", SEEDS3, ["--local_epochs", "1"], None),
    ("eicu", "sens_e1", SEEDS3, ["--local_epochs", "1"], None),
    ("mimic", "sens_e5", SEEDS3, ["--local_epochs", "5"], None),
    ("eicu", "sens_e5", SEEDS3, ["--local_epochs", "5"], None),
    ("mimic", "arch_logreg", SEEDS3, ["--model", "logreg"], None),
    ("eicu", "arch_logreg", SEEDS3, ["--model", "logreg"], None),
    ("mimic", "sens_r100", SEEDS10[:1], ["--rounds", "100"], None),
    # eICU with first-day vitals, GCS, urine output and ventilation added
    # (apacheApsVar): checks that the conclusions hold with a richer feature
    # set like the prior work's. Reuses the hyperparameters tuned for "eicu".
    ("eicu_aps", "main", SEEDS5, [], "min_deaths"),
]

# Datasets that reuse another dataset's tuned hyperparameters.
TUNED_ON = {"eicu_aps": "eicu"}


def done(path, n_rows):
    if not os.path.exists(path):
        return False
    try:
        return len(pd.read_csv(path)) >= n_rows
    except Exception:
        return False


def q(args):
    return " ".join(shlex.quote(a) for a in args)


def pending():
    cmds = []
    for dataset, name, seeds, extra, central in EXPERIMENTS:
        out = os.path.join(RUNS, dataset, name)
        hp = os.path.join(ROOT, "results", "tuning", TUNED_ON.get(dataset, dataset), "best_hp.json")
        for seed in seeds:
            tag = f"_seed{seed}"
            if not done(os.path.join(out, f"fl_final_results{tag}.csv"), N_STRATEGIES):
                args = [PY, os.path.join(ROOT, "experiments", "run_fl_experiments.py"),
                        "--dataset", dataset, "--seed", str(seed), "--hp_json", hp,
                        "--out_dir", out, "--out_tag", tag, "--save_preds"] + extra
                log = os.path.join(out, f"log{tag}.txt")
                cmds.append(f"mkdir -p {shlex.quote(out)} && {q(args)} > {shlex.quote(log)} 2>&1")
            if central and not done(os.path.join(out, f"centralized_results{tag}.csv"), 3):
                args = [PY, os.path.join(ROOT, "experiments", "run_centralized.py"),
                        "--dataset", dataset, "--seed", str(seed), "--config", central,
                        "--out_dir", out, "--out_tag", tag]
                log = os.path.join(out, f"log_centralized{tag}.txt")
                cmds.append(f"mkdir -p {shlex.quote(out)} && {q(args)} > {shlex.quote(log)} 2>&1")
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=0, help="execute with this many parallel runs")
    args = ap.parse_args()
    cmds = pending()
    if not args.run:
        print("\n".join(cmds))
        return
    print(f"{len(cmds)} pending runs, {args.run} in parallel", flush=True)

    def run(cmd):
        rc = subprocess.call(cmd, shell=True)
        print(f"[exit {rc}] {cmd.split(' > ')[-1].split()[0]}", flush=True)
        return rc

    with ThreadPoolExecutor(args.run) as pool:
        codes = list(pool.map(run, cmds))
    print(f"done: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed", flush=True)


if __name__ == "__main__":
    main()
