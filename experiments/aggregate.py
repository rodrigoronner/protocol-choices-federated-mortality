"""Aggregates the results/runs/<dataset>/<experiment>/ runs
(python experiments/aggregate.py --dataset mimic|eicu).

For every experiment: mean, std and 95% t-interval per strategy over seeds.
For experiments with >= 5 seeds, also the statistical comparison of the
strategies, paired by seed (within a seed every strategy is evaluated on the
same held-out test partition):
  - Friedman test across the federated strategies (omnibus);
  - following Demsar (2006), the best-ranked federated strategy (the
    control) against each other federated strategy, and, as a separate
    family, against each centralized baseline: the mean paired difference
    with its 95% t-interval, the paired t-test and the Wilcoxon signed-rank
    test, with Holm's correction within each family and metric. (Correcting
    over all 45 pairs would make significance unreachable with 10 seeds: the
    smallest exact two-sided Wilcoxon p is 2/2^10 = 0.002, and 45 x 0.002 > 0.05.)
The ablation table compares each strategy across imbalance treatments
(main = class weighting, imb_smote = SMOTE-Tomek, imb_none), paired by seed.

Writes summary_<experiment>.csv, pairwise_<experiment>.csv, friedman.csv and
ablation_imbalance*.csv into results/runs/<dataset>/.
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs")
METRICS = ["auc_roc", "auc_pr", "f1", "precision", "recall", "accuracy"]
PRIMARY = ["auc_roc", "auc_pr"]
STRATEGIES = ["FedAvg", "FedProx", "SCAFFOLD", "FedAdagrad", "FedAdam", "FedYogi", "FedCluster"]


def load(exp, prefix="fl_final_results"):
    paths = sorted(glob.glob(os.path.join(glob.escape(os.path.join(RUNS, exp)), f"{prefix}_seed*.csv")))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summarize(df, by="strategy"):
    rows = []
    for s, sub in df.groupby(by, sort=False):
        row = {by: s, "n_seeds": sub["seed"].nunique()}
        for m in METRICS:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=1)
            row[f"{m}_ci95"] = ci95(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def pairwise(df, pairs, by="strategy", metrics=PRIMARY, family=""):
    """Paired comparisons for an explicit list of (a, b) pairs per metric
    (a callable metric -> pairs is also accepted), Holm-corrected within the
    list."""
    rows = []
    for m in metrics:
        wide = df.pivot_table(index="seed", columns=by, values=m).dropna()
        block = []
        for a, b in (pairs(m) if callable(pairs) else pairs):
            if a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).values
            t_p = stats.ttest_rel(wide[a], wide[b]).pvalue
            w_p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            block.append(dict(family=family, metric=m, a=a, b=b, n=len(d), mean_diff=d.mean(),
                              ci95_low=d.mean() - ci95(d), ci95_high=d.mean() + ci95(d),
                              t_p=t_p, wilcoxon_p=w_p))
        if block:
            t_adj = holm([r["t_p"] for r in block])
            w_adj = holm([r["wilcoxon_p"] for r in block])
            for r, ta, wa in zip(block, t_adj, w_adj):
                r.update(t_p_holm=ta, wilcoxon_p_holm=wa)
            rows.extend(block)
    return pd.DataFrame(rows)


def control(df, metric):
    """Federated strategy with the best mean rank over seeds."""
    wide = df.pivot_table(index="seed", columns="strategy", values=metric).dropna()
    return wide.rank(axis=1, ascending=False).mean().idxmin()


def friedman(df, exp, by="strategy"):
    rows = []
    for m in PRIMARY:
        wide = df.pivot_table(index="seed", columns=by, values=m).dropna()
        if wide.shape[0] < 3 or wide.shape[1] < 3:
            continue
        res = stats.friedmanchisquare(*[wide[c] for c in wide])
        ranks = wide.rank(axis=1, ascending=False).mean()
        rows.append(dict(experiment=exp, metric=m, n_seeds=wide.shape[0], chi2=res.statistic,
                         p=res.pvalue, **{f"mean_rank_{c}": ranks[c] for c in wide}))
    return rows


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic", choices=["mimic", "eicu", "eicu_aps"])
    RUNS = os.path.join(RUNS, ap.parse_args().dataset)
    exps = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
    fried = []
    for exp in exps:
        fl = load(exp)
        if fl is None:
            continue
        cen = load(exp, "centralized_results")
        both = fl if cen is None else pd.concat([fl, cen], ignore_index=True)
        order = [s for s in STRATEGIES + ["naive", "global_smote", "weighted_loss"] if s in set(both["strategy"])]
        summ = summarize(both)
        summ["strategy"] = pd.Categorical(summ["strategy"], order)
        summ.sort_values("strategy").to_csv(os.path.join(RUNS, f"summary_{exp}.csv"), index=False)
        if fl["seed"].nunique() >= 5:
            fl_names = [s for s in order if s in STRATEGIES]
            cen_names = [s for s in order if s not in STRATEGIES]
            vs_fl = pairwise(both, lambda m: [(control(fl, m), s) for s in fl_names if s != control(fl, m)],
                             family="control_vs_federated")
            vs_cen = pairwise(both, lambda m: [(control(fl, m), s) for s in cen_names],
                              family="control_vs_centralized")
            pd.concat([vs_fl, vs_cen]).to_csv(os.path.join(RUNS, f"pairwise_{exp}.csv"), index=False)
            fried.extend(friedman(fl, exp))
        print(f"\n### {exp}")
        cols = ["strategy", "n_seeds"] + [f"{m}_{k}" for m in PRIMARY + ["f1"] for k in ("mean", "std")]
        print(summ.sort_values("strategy")[cols].to_string(index=False, float_format="%.4f"))
    if fried:
        pd.DataFrame(fried).to_csv(os.path.join(RUNS, "friedman.csv"), index=False)
        print("\n### Friedman")
        print(pd.DataFrame(fried).to_string(index=False, float_format="%.4g"))

    parts = []
    for exp, label in (("main", "weighted"), ("imb_smote", "smote_tomek"), ("imb_none", "none")):
        df = load(exp)
        if df is not None:
            parts.append(df.assign(imbalance=label))
    if len(parts) > 1:
        abl = pd.concat(parts, ignore_index=True)
        rows = []
        for s in STRATEGIES:
            sub = abl[abl["strategy"] == s]
            summ = summarize(sub, by="imbalance").assign(strategy=s)
            pw = pairwise(sub, list(itertools.combinations(["weighted", "smote_tomek", "none"], 2)),
                          by="imbalance", family="imbalance")
            rows.append((summ, pw.assign(strategy=s)))
        pd.concat([r[0] for r in rows]).to_csv(os.path.join(RUNS, "ablation_imbalance.csv"), index=False)
        pd.concat([r[1] for r in rows]).to_csv(os.path.join(RUNS, "ablation_imbalance_pairwise.csv"), index=False)
        print("\n### Imbalance ablation (AUC-PR mean by strategy x treatment)")
        print(abl.pivot_table(index="strategy", columns="imbalance", values="auc_pr").round(4))


if __name__ == "__main__":
    main()
