"""LaTeX table bodies for the RITA manuscript, generated from the result CSVs
so that no number in a table is typed by hand.

    python experiments/make_tables.py

Writes manuscript/tables/*.tex (tabular environments only; captions and
labels live in main.tex). Best federated value per column in bold.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import _threading_limits  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as data_mod  # noqa: E402

RUNS = os.path.join(ROOT, "results", "runs")
TUNING = os.path.join(ROOT, "results", "tuning")
OUT = os.path.join(ROOT, "manuscript", "tables")
FL = ["FedAvg", "FedProx", "SCAFFOLD", "FedAdagrad", "FedAdam", "FedYogi", "FedCluster"]
CEN = {"naive": "Centralized, no correction", "weighted_loss": "Centralized, class weights",
       "global_smote": "Centralized, global SMOTE"}


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.tex"), "w") as fh:
        fh.write(body)
    print("wrote", name)


def load(ds, exp, prefix="fl_final_results"):
    d = os.path.join(RUNS, ds, exp)
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(glob.escape(d), f"{prefix}_seed*.csv")))],
                     ignore_index=True)


def fmt(mean, sd=None, bold=False, nd=3):
    s = f"{mean:.{nd}f}" + (f" $\\pm$ {sd:.{nd}f}" if sd is not None else "")
    return f"\\textbf{{{s}}}" if bold else s


# ---------------------------------------------------------------------------
# Table 2: cohorts and federations
# ---------------------------------------------------------------------------
def table_cohorts():
    cols = []
    for ds, gender_f in (("mimic", "F"), ("eicu", "Female")):
        df = data_mod.load_cohort(ds)
        df["_client"] = data_mod.assign_clients(df, "min_deaths")
        cl = df[df["_client"].notna()]
        per = cl.groupby("_client")[data_mod.TARGET_COL].agg(["size", "mean"])
        labs = cl[data_mod.LAB_COLS].notna().mean()
        cols.append({
            "Hospital stays (all sites)": f"{len(df):,}",
            "Hospital stays (clients)": f"{len(cl):,}",
            "Patients (clients)": f"{cl['subject_id'].nunique():,}",
            "In-hospital mortality": f"{100 * cl[data_mod.TARGET_COL].mean():.2f}\\%",
            "Age, mean (SD), years": f"{cl['age'].mean():.1f} ({cl['age'].std():.1f})",
            "Female": f"{100 * (cl['gender'] == gender_f).mean():.1f}\\%",
            "Clients (sites with $\\geq$100 deaths)": f"{len(per)}",
            "Stays per client, min--max": f"{per['size'].min():,}--{per['size'].max():,}",
            "Largest client's share of stays": f"{100 * per['size'].max() / per['size'].sum():.1f}\\%",
            "Mortality per client, min--max": f"{100 * per['mean'].min():.1f}--{100 * per['mean'].max():.1f}\\%",
            "Lab panel coverage, min--max\\textsuperscript{a}": f"{100 * labs.min():.0f}--{100 * labs.max():.0f}\\%",
        })
    rows = "\n".join(f"{k} & {a} & {b} \\\\" for (k, a), b in zip(cols[0].items(), cols[1].values()))
    write("tab2_cohorts", "\\begin{tabular}{@{}lrr@{}}\n\\toprule\n & MIMIC-IV & eICU-CRD \\\\\n\\midrule\n"
          + rows + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Table 4: main results
# ---------------------------------------------------------------------------
def table_main():
    lines = []
    blocks = {}
    for ds in ("mimic", "eicu"):
        s = pd.read_csv(os.path.join(RUNS, ds, "summary_main.csv")).set_index("strategy")
        fr = pd.read_csv(os.path.join(RUNS, ds, "friedman.csv"))
        fr = fr[fr["experiment"] == "main"].set_index("metric")
        rc = pd.read_csv(os.path.join(RUNS, ds, "main", "recalibration.csv"))
        brier = rc[rc["method"] == "platt"].groupby("strategy")["brier"].mean()
        blocks[ds] = (s, fr, brier)
    best = {(ds, m): blocks[ds][0].loc[FL, f"{m}_mean"].max() for ds in blocks for m in ("auc_roc", "auc_pr")}
    best_b = {ds: blocks[ds][2].loc[FL].min() for ds in blocks}
    for name in FL + list(CEN):
        cells = [name if name in FL else CEN[name]]
        for ds in ("mimic", "eicu"):
            s, fr, brier = blocks[ds]
            for m in ("auc_roc", "auc_pr"):
                mu, sd = s.loc[name, f"{m}_mean"], s.loc[name, f"{m}_std"]
                cells.append(fmt(mu, sd, bold=name in FL and np.isclose(mu, best[(ds, m)])))
            if name in FL:
                cells.append(f"{fr.loc['auc_roc', f'mean_rank_{name}']:.1f}")
                cells.append(fmt(brier[name], bold=np.isclose(brier[name], best_b[ds]), nd=4))
            else:
                cells += ["--", "--"]
        if name == "naive":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{4}{c}{MIMIC-IV (12 care units)} & \\multicolumn{4}{c}{eICU-CRD (49 hospitals)} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n"
            "Strategy & AUC-ROC & AUC-PR & Rank & Brier\\textsuperscript{a} & AUC-ROC & AUC-PR & Rank & Brier\\textsuperscript{a} \\\\\n\\midrule\n")
    write("tab4_main", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Table 5: imbalance treatment ablation (AUC-PR and raw Brier)
# ---------------------------------------------------------------------------
def table_imbalance():
    exps = (("main", "Weights"), ("imb_smote", "SMOTE"), ("imb_none", "None"))
    vals = {}
    for ds in ("mimic", "eicu"):
        for exp, _ in exps:
            df = load(ds, exp)
            vals[(ds, exp, "auc_pr")] = df.groupby("strategy")["auc_pr"].mean()
            pa = pd.read_csv(os.path.join(RUNS, ds, exp, "prediction_analysis.csv")) \
                if os.path.exists(os.path.join(RUNS, ds, exp, "prediction_analysis.csv")) else None
            vals[(ds, exp, "brier")] = pa.groupby("strategy")["brier"].mean() if pa is not None else None
    lines = []
    for s in FL:
        cells = [s]
        for ds in ("mimic", "eicu"):
            ap = [vals[(ds, e, "auc_pr")][s] for e, _ in exps]
            cells += [fmt(v, bold=np.isclose(v, max(ap))) for v in ap]
            br = [vals[(ds, e, "brier")][s] if vals[(ds, e, "brier")] is not None else np.nan for e, _ in exps]
            cells += [fmt(v, bold=np.isclose(v, np.nanmin(br)), nd=3) for v in br]
        lines.append(" & ".join(cells) + " \\\\")
    sub = " & ".join(lab for _, lab in exps)
    head = ("\\begin{tabular}{@{}lcccccccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{6}{c}{MIMIC-IV} & \\multicolumn{6}{c}{eICU-CRD} \\\\\n"
            "\\cmidrule(lr){2-7}\\cmidrule(l){8-13}\n"
            " & \\multicolumn{3}{c}{AUC-PR} & \\multicolumn{3}{c}{Brier (raw)} & \\multicolumn{3}{c}{AUC-PR} & \\multicolumn{3}{c}{Brier (raw)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}\n"
            f"Strategy & {sub} & {sub} & {sub} & {sub} \\\\\n\\midrule\n")
    write("tab5_imbalance", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Table 6: calibration before and after Platt scaling
# ---------------------------------------------------------------------------
def table_calibration():
    lines = []
    rc = {ds: pd.read_csv(os.path.join(RUNS, ds, "main", "recalibration.csv")) for ds in ("mimic", "eicu")}
    for s in FL:
        cells = [s]
        for ds in ("mimic", "eicu"):
            g = rc[ds][rc[ds]["strategy"] == s].groupby("method")[["brier", "citl", "slope", "ece"]].mean()
            for m, nd in (("brier", 4), ("citl", 2), ("slope", 2), ("ece", 3)):
                cells.append(f"{g.loc['raw', m]:.{nd}f} $\\to$ {g.loc['platt', m]:.{nd}f}")
        lines.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{4}{c}{MIMIC-IV} & \\multicolumn{4}{c}{eICU-CRD} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n"
            "Strategy & Brier & Intercept & Slope & ECE & Brier & Intercept & Slope & ECE \\\\\n\\midrule\n")
    write("tab6_calibration", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Appendix A1: tuning grids and selected values
# ---------------------------------------------------------------------------
def table_tuning():
    grids = {
        "FedProx": "$\\mu \\in \\{0.001, 0.01, 0.1, 1\\}$",
        "SCAFFOLD": "local SGD lr $\\in \\{0.003, 0.01, 0.03, 0.1, 0.3, 1\\}$\\textsuperscript{a}, $\\eta_g \\in \\{0.5, 1\\}$",
        "FedAdagrad": "$\\eta \\in \\{0.001, 0.01, 0.1\\}$, $\\tau \\in \\{10^{-3}, 10^{-2}\\}$",
        "FedAdam": "$\\eta \\in \\{0.001, 0.01, 0.1\\}$, $\\tau \\in \\{10^{-3}, 10^{-2}\\}$",
        "FedYogi": "$\\eta \\in \\{0.001, 0.01, 0.1\\}$, $\\tau \\in \\{10^{-3}, 10^{-2}\\}$",
        "FedCluster": "DBSCAN radius quantile $\\in \\{0.25, 0.5, 0.75\\}$",
        "FedAvgSGD": "local SGD lr $\\in \\{0.003, 0.01, 0.03, 0.1, 0.3, 1\\}$\\textsuperscript{a}",
    }
    labels = {"FedAvgSGD": "FedAvg-SGD (control)"}
    names = {"mu": "$\\mu$", "lr_scaffold": "lr", "eta_g": "$\\eta_g$", "eta_adagrad": "$\\eta$",
             "tau_adagrad": "$\\tau$", "eta_adam": "$\\eta$", "tau_adam": "$\\tau$", "eta_yogi": "$\\eta$",
             "tau_yogi": "$\\tau$", "eps_quantile": "q", "lr_sgd": "lr"}
    best = {ds: json.load(open(os.path.join(TUNING, ds, "best_hp.json"))) for ds in ("mimic", "eicu")}
    n_cfg = {ds: pd.read_csv(os.path.join(TUNING, ds, "grid_all.csv")).groupby("strategy").size()
             for ds in ("mimic", "eicu")}
    lines = []
    for s, g in grids.items():
        if s not in best["mimic"] or s not in best["eicu"]:
            continue
        sel = []
        for ds in ("mimic", "eicu"):
            sel.append(", ".join(f"{names[k]} = {v:g}" for k, v in sorted(best[ds][s].items())))
        if s == "FedAvgSGD":
            lines.append("\\midrule")
        lines.append(f"{labels.get(s, s)} & {g} & {n_cfg['mimic'][s]}/{n_cfg['eicu'][s]} & {sel[0]} & {sel[1]} \\\\")
    head = ("\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
            "Strategy & Grid & Configs & Selected, MIMIC-IV & Selected, eICU-CRD \\\\\n\\midrule\n")
    write("tabA1_tuning", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Appendix A2: robustness - AUC-ROC of every strategy in every variant
# ---------------------------------------------------------------------------
def table_robustness():
    variants = [
        ("mimic", "main", "MIMIC-IV, main"), ("mimic", "cfg_other", "+ pooled other units"),
        ("mimic", "cfg_edsplit4", "Emergency Dept.\\ split in 4"), ("mimic", "cfg_top5", "Five largest units"),
        ("mimic", "sens_e1", "1 local epoch"), ("mimic", "sens_e5", "5 local epochs"),
        ("mimic", "sens_part50", "50\\% participation"), ("mimic", "sens_r100", "100 rounds"),
        ("mimic", "arch_logreg", "Logistic regression"),
        ("eicu", "main", "eICU-CRD, main"), ("eicu", "cfg_other", "+ pooled other hospitals"),
        ("eicu", "cfg_min200", "Hospitals with $\\geq$200 deaths"), ("eicu", "sens_e1", "1 local epoch"),
        ("eicu", "sens_e5", "5 local epochs"), ("eicu", "sens_part50", "50\\% participation"),
        ("eicu", "sens_part20", "20\\% participation"), ("eicu", "arch_logreg", "Logistic regression"),
        ("eicu_aps", "main", "eICU-CRD + day-1 physiology"),
    ]
    lines = []
    for ds, exp, label in variants:
        df = load(ds, exp)
        m = df.groupby("strategy")["auc_roc"].mean().reindex(FL)
        n = df["seed"].nunique()
        cells = [f"{label} ({n})"] + [fmt(v, bold=np.isclose(v, m.max())) for v in m]
        if exp == "main" and ds != "mimic":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}l" + "c" * len(FL) + "@{}}\n\\toprule\n"
            "Variant (seeds) & " + " & ".join(FL) + " \\\\\n\\midrule\n")
    write("tabA2_robustness", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Appendix A3: performance by sociodemographic subgroup (experiments/subgroups.py)
# ---------------------------------------------------------------------------
def table_subgroups():
    order = [("Sex", "Female"), ("Sex", "Male"), ("Age", "Under 65 years"), ("Age", "65 years or older"),
             ("Race or ethnicity", "White"), ("Race or ethnicity", "Black"), ("Race or ethnicity", "Hispanic"),
             ("Race or ethnicity", "Asian"), ("Race or ethnicity", "Other or unknown")]
    sg = {ds: pd.read_csv(os.path.join(RUNS, ds, "main", "subgroups.csv")).groupby(["variable", "group", "strategy"])
          [["deaths", "auc_roc"]].mean() for ds in ("mimic", "eicu")}
    lines, prev = [], None
    for var, grp in order:
        if prev and var != prev:
            lines.append("\\midrule")
        prev = var
        cells = [grp]
        for ds in ("mimic", "eicu"):
            d = sg[ds]
            cells.append(f"{d.loc[(var, grp, 'SCAFFOLD'), 'deaths']:.0f}")
            cells.append(f"{d.loc[(var, grp, 'SCAFFOLD'), 'auc_roc']:.3f}")
            cells.append(f"{d.loc[(var, grp, 'FedAvg'), 'auc_roc']:.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{MIMIC-IV} & \\multicolumn{3}{c}{eICU-CRD} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Subgroup & Deaths & SCAFFOLD & FedAvg & Deaths & SCAFFOLD & FedAvg \\\\\n\\midrule\n")
    write("tabA3_subgroups", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Appendix A4: synthetic positives created by client-level SMOTE-Tomek
# ---------------------------------------------------------------------------
def table_synthetic():
    def load_clients(ds):
        d = os.path.join(RUNS, ds, "imb_smote")
        df = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(glob.escape(d), "clients_seed*.csv"))])
        return df.groupby("client")[["n_train", "n_pos_train", "n_synthetic_pos", "synthetic_share_of_pos"]].mean()
    m = load_clients("mimic").sort_values("n_train", ascending=False)
    lines = []
    for unit, r in m.iterrows():
        lines.append(f"{unit} & {r.n_train:,.0f} & {r.n_pos_train:,.0f} & {r.n_synthetic_pos:,.0f} & "
                     f"{100 * r.synthetic_share_of_pos:.1f}\\% \\\\")
    e = load_clients("eicu")
    lines.append("\\midrule")
    lines.append(f"eICU-CRD, 49 hospitals (median) & {e.n_train.median():,.0f} & {e.n_pos_train.median():,.0f} & "
                 f"{e.n_synthetic_pos.median():,.0f} & {100 * e.synthetic_share_of_pos.median():.1f}\\% \\\\")
    lines.append(f"eICU-CRD, 49 hospitals (range) & {e.n_train.min():,.0f}--{e.n_train.max():,.0f} & "
                 f"{e.n_pos_train.min():,.0f}--{e.n_pos_train.max():,.0f} & {e.n_synthetic_pos.min():,.0f}--"
                 f"{e.n_synthetic_pos.max():,.0f} & {100 * e.synthetic_share_of_pos.min():.1f}--"
                 f"{100 * e.synthetic_share_of_pos.max():.1f}\\% \\\\")
    head = ("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
            "Client & Training stays & Real deaths & Synthetic deaths & Synthetic share \\\\\n\\midrule\n")
    write("tabA4_synthetic", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    table_cohorts()
    table_main()
    table_imbalance()
    table_calibration()
    table_tuning()
    table_robustness()
    table_subgroups()
    table_synthetic()
