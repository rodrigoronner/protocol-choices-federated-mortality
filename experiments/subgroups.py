"""Performance by sociodemographic subgroup (TRIPOD+AI fairness items).

For every seed of the main experiment, the test rows are rebuilt in the
order in which data.build_clients concatenates them (clients by decreasing
size, each client's test partition in cohort order), matched to the saved
test predictions (fl_test_preds_seed*.npz), and AUC-ROC and AUC-PR are
computed separately by sex, age group (<65 / >=65 years) and race or
ethnicity.

    python experiments/subgroups.py

Writes results/runs/<dataset>/main/subgroups.csv and prints the mean over
seeds.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import _threading_limits  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import data as data_mod  # noqa: E402

RUNS = os.path.join(ROOT, "results", "runs")
STRATEGIES = ["SCAFFOLD", "FedAvg"]


def race_group(r):
    r = str(r).upper()
    if "WHITE" in r or "CAUCASIAN" in r:
        return "White"
    if "BLACK" in r or "AFRICAN" in r:
        return "Black"
    if "HISPANIC" in r or "LATINO" in r:
        return "Hispanic"
    if "ASIAN" in r:
        return "Asian"
    return "Other or unknown"


def test_rows(df, seed, config="min_deaths"):
    df = df.copy()
    df["_client"] = data_mod.assign_clients(df, config, seed=seed)
    df = df[df["_client"].notna()]
    names = list(df["_client"].value_counts().index)
    df["_split"] = data_mod._patient_level_split(df, seed, 0.2, 0.1)
    parts = [df[(df["_client"] == u) & (df["_split"] == "test")] for u in names]
    return pd.concat(parts, ignore_index=True)


def main():
    for ds, female in (("mimic", "F"), ("eicu", "Female")):
        cohort = data_mod.load_cohort(ds)
        d = os.path.join(RUNS, ds, "main")
        rows = []
        for path in sorted(glob.glob(os.path.join(glob.escape(d), "fl_test_preds_seed*.npz"))):
            seed = int(path.rsplit("seed", 1)[1][:-4])
            z = np.load(path)
            t = test_rows(cohort, seed)
            assert len(t) == len(z["y_test"]) and np.array_equal(
                t[data_mod.TARGET_COL].to_numpy(), z["y_test"].astype(int)), "test order mismatch"
            groups = {
                "Sex": np.where(t["gender"] == female, "Female", "Male"),
                "Age": np.where(t["age"] >= 65, "65 years or older", "Under 65 years"),
                "Race or ethnicity": t["race"].map(race_group).to_numpy(),
            }
            y = z["y_test"].astype(int)
            for var, g in groups.items():
                for level in pd.unique(g):
                    m = g == level
                    if y[m].sum() < 10 or (1 - y[m]).sum() < 10:
                        continue
                    for s in STRATEGIES:
                        p = z[s][m]
                        rows.append(dict(seed=seed, variable=var, group=level, strategy=s,
                                         n=int(m.sum()), deaths=int(y[m].sum()),
                                         auc_roc=roc_auc_score(y[m], p), auc_pr=average_precision_score(y[m], p)))
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(d, "subgroups.csv"), index=False)
        summ = out.groupby(["variable", "group", "strategy"])[["n", "deaths", "auc_roc", "auc_pr"]].mean()
        print(f"\n### {ds}\n", summ.round(3).to_string())


if __name__ == "__main__":
    main()
