"""Size of the discharge-coded feature leakage on MIMIC-IV.

Counts of an admission's diagnosis and procedure codes (num_diagnoses,
num_procedures) are a tempting summary of comorbidity, but MIMIC-IV assigns
those codes at discharge, so the counts are not available within 24 hours of
admission. This script
measures what they add, with a centralized gradient-boosting reference model
on the main client set and split (seeds 42-46): the 24-hour feature set with
and without the two counts, and the counts alone. Needs the MIMIC-IV
database (see src/data.py) for the code counts.

    python experiments/leakage_check.py

Writes results/runs/mimic/leakage_check.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import _threading_limits  # noqa: E402,F401  (must be imported before numpy/sklearn/torch)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import data as data_mod  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs",
                   "mimic", "leakage_check.csv")
COUNTS = ["num_diagnoses", "num_procedures"]


def icd_counts(dsn=data_mod.DEFAULT_DSN):
    """Number of diagnosis and procedure codes of each admission, as coded at
    discharge in MIMIC-IV (admissions without codes count as 0)."""
    from sqlalchemy import create_engine
    query = """
        SELECT a.hadm_id,
               COALESCE(d.n, 0) AS num_diagnoses,
               COALESCE(p.n, 0) AS num_procedures
        FROM mimiciv_hosp.admissions a
        LEFT JOIN (SELECT hadm_id, COUNT(icd_code) AS n FROM mimiciv_hosp.diagnoses_icd GROUP BY hadm_id) d
               ON a.hadm_id = d.hadm_id
        LEFT JOIN (SELECT hadm_id, COUNT(icd_code) AS n FROM mimiciv_hosp.procedures_icd GROUP BY hadm_id) p
               ON a.hadm_id = p.hadm_id
    """
    return pd.read_sql(query, create_engine(dsn))


def main():
    raw = data_mod.extract_cohort().merge(icd_counts(), on="hadm_id", how="left")
    rows = []
    for seed in range(42, 47):
        df = raw.copy()
        df["_client"] = data_mod.assign_clients(df, "min_deaths", seed=seed)
        df = df[df["_client"].notna()]
        split = data_mod._patient_level_split(df, seed, 0.2, 0.1)
        tr, te = df[split == "train"], df[split == "test"]
        ohe, scaler, med = data_mod._fit_shared_preprocessor(tr)
        Xtr, ytr = data_mod._transform(tr, ohe, scaler, med)
        Xte, yte = data_mod._transform(te, ohe, scaler, med)
        ctr, cte = tr[COUNTS].to_numpy(np.float32), te[COUNTS].to_numpy(np.float32)
        for name, A, B in (("24h features", Xtr, Xte),
                           ("24h features + ICD counts", np.hstack([Xtr, ctr]), np.hstack([Xte, cte])),
                           ("ICD counts only", ctr, cte)):
            m = HistGradientBoostingClassifier(random_state=seed).fit(A, ytr)
            p = m.predict_proba(B)[:, 1]
            rows.append(dict(seed=seed, features=name, auc_roc=roc_auc_score(yte, p),
                             auc_pr=average_precision_score(yte, p)))
        print(f"seed {seed} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.groupby("features", sort=False)[["auc_roc", "auc_pr"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
