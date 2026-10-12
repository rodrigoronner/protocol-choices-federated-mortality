"""Control experiment for SCAFFOLD: FedAvg with local SGD at a tuned learning
rate (strategy "FedAvgSGD"), on the main configuration of both datasets.

SCAFFOLD trains locally with SGD and its local learning rate is tuned,
whereas the other strategies use Adam at a fixed rate. FedAvgSGD gets the
same optimizer and the same learning-rate grid as SCAFFOLD but no control
variates, so the paired difference SCAFFOLD - FedAvgSGD isolates the effect
of the drift correction.

    python experiments/run_control.py --parallel 3

Steps: tune FedAvgSGD on seed 42 (experiments/tune.py, configs not yet in the
grid), select, then run the ten seeds and compare with SCAFFOLD from the main
experiment. Results: results/runs/<dataset>/ctl_fedavg_sgd/ and
results/runs/<dataset>/control_scaffold_vs_fedavgsgd.csv.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUNS = os.path.join(ROOT, "results", "runs")
SEEDS = list(range(42, 52))


def sh(args, log):
    with open(log, "w") as fh:
        return subprocess.call(args, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)


def tune(ds, n):
    tdir = os.path.join(ROOT, "results", "tuning", ds)
    with ThreadPoolExecutor(n) as pool:
        codes = list(pool.map(lambda w: sh([PY, "experiments/tune.py", "--dataset", ds, "--worker", str(w),
                                            "--n_workers", str(n)], os.path.join(tdir, f"log_control_worker{w}.txt")),
                              range(n)))
    assert not any(codes), codes
    assert sh([PY, "experiments/tune.py", "--dataset", ds, "--select"], os.path.join(tdir, "selection.txt")) == 0


def runs(ds, n):
    out = os.path.join(RUNS, ds, "ctl_fedavg_sgd")
    os.makedirs(out, exist_ok=True)
    hp = os.path.join(ROOT, "results", "tuning", ds, "best_hp.json")

    def one(seed):
        if os.path.exists(os.path.join(out, f"fl_final_results_seed{seed}.csv")):
            return 0
        return sh([PY, "experiments/run_fl_experiments.py", "--dataset", ds, "--seed", str(seed),
                   "--strategies", "FedAvgSGD", "--hp_json", hp, "--out_dir", out,
                   "--out_tag", f"_seed{seed}", "--save_preds"], os.path.join(out, f"log_seed{seed}.txt"))
    with ThreadPoolExecutor(n) as pool:
        codes = list(pool.map(one, SEEDS))
    assert not any(codes), codes


def compare(ds):
    main = pd.concat([pd.read_csv(os.path.join(RUNS, ds, "main", f"fl_final_results_seed{s}.csv")) for s in SEEDS])
    ctl = pd.concat([pd.read_csv(os.path.join(RUNS, ds, "ctl_fedavg_sgd", f"fl_final_results_seed{s}.csv"))
                     for s in SEEDS])
    rows = []
    for m in ("auc_roc", "auc_pr"):
        a = main[main["strategy"] == "SCAFFOLD"].set_index("seed")[m]
        for name, b in (("FedAvgSGD", ctl.set_index("seed")[m]),
                        ("FedAvg", main[main["strategy"] == "FedAvg"].set_index("seed")[m])):
            d = (a - b.reindex(a.index)).to_numpy()
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(metric=m, a="SCAFFOLD", b=name, mean_a=a.mean(), mean_b=b.mean(), mean_diff=d.mean(),
                             ci95_low=d.mean() - half, ci95_high=d.mean() + half,
                             t_p=stats.ttest_rel(a, b.reindex(a.index)).pvalue,
                             wilcoxon_p=stats.wilcoxon(d).pvalue, wins=int((d > 0).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RUNS, ds, "control_scaffold_vs_fedavgsgd.csv"), index=False)
    print(f"\n### {ds}\n", out.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--datasets", nargs="+", default=["mimic", "eicu"])
    args = ap.parse_args()
    for ds in args.datasets:
        tune(ds, args.parallel)
        runs(ds, args.parallel)
        compare(ds)
    print("CONTROL DONE")


if __name__ == "__main__":
    main()
