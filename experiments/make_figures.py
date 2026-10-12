"""Figures for the RITA manuscript (vector PDF, sized for the RITA template:
3.35 in per column, 6.9 in full width).

One visual system for every figure: sans 7.5 pt, solid hairline grid, no top
or right spines, SCAFFOLD as the emphasized series (blue) and the other
strategies in gray, at most three hues at once (the first three slots of the
validated categorical palette), every colored series named by a legend or a
direct label.

    python experiments/make_figures.py

Writes manuscript/figures/fig_*.pdf.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import _threading_limits  # noqa: E402,F401  (must be imported before numpy/sklearn/torch)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import data as data_mod  # noqa: E402
import federated as fed  # noqa: E402
from models import get_model  # noqa: E402

RUNS = os.path.join(ROOT, "results", "runs")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from run_fl_experiments import DEFAULT_HP  # noqa: E402
OWN_HP = {"FedProx": ["mu"], "SCAFFOLD": ["lr_scaffold", "eta_g"], "FedAdagrad": ["eta_adagrad", "tau_adagrad"],
          "FedAdam": ["eta_adam", "tau_adam"], "FedYogi": ["eta_yogi", "tau_yogi"], "FedCluster": ["eps_quantile"]}
TUNING = os.path.join(ROOT, "results", "tuning")
OUT = os.path.join(ROOT, "manuscript", "figures")

COL, FULL = 3.35, 6.9
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, MUTED, GRID = "#a3a19b", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STRATEGIES = ["FedAvg", "FedProx", "SCAFFOLD", "FedAdagrad", "FedAdam", "FedYogi", "FedCluster"]
DATASETS = {"mimic": "MIMIC-IV (12 care units)", "eicu": "eICU-CRD (49 hospitals)",
            "eicu_aps": "eICU-CRD + day-1 physiology"}
METRICS = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def load(dataset, exp, prefix="fl_final_results"):
    d = os.path.join(RUNS, dataset, exp)
    paths = sorted(glob.glob(os.path.join(glob.escape(d), f"{prefix}_seed*.csv")))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 1: the two federations - every site, size vs mortality, eligibility
# ---------------------------------------------------------------------------
def fig_federations():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for ax, ds in zip(axes, ["mimic", "eicu"]):
        df = data_mod.load_cohort(ds)
        df = df[df["site"].notna() & ~df["site"].isin(["UNKNOWN", "Unknown"])]
        g = df.groupby("site")[data_mod.TARGET_COL].agg(n="size", deaths="sum")
        g["mort"] = 100 * g["deaths"] / g["n"]
        ok = g["deaths"] >= data_mod.MIN_DEATHS_PER_CLIENT
        ax.scatter(g.loc[~ok, "n"], g.loc[~ok, "mort"], s=10, color=GRAY, linewidths=0,
                   alpha=0.8, label=f"Not eligible (< {data_mod.MIN_DEATHS_PER_CLIENT} deaths)")
        ax.scatter(g.loc[ok, "n"], g.loc[ok, "mort"], s=16, color=BLUE, edgecolor="white",
                   linewidths=0.6, label=f"Client ({ok.sum()} sites)", zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Hospital stays per site (log scale)")
        ax.set_ylabel("In-hospital mortality (%)")
        ax.set_title(("a  " if ds == "mimic" else "b  ") + DATASETS[ds])
        ax.legend(loc="upper right" if ds == "mimic" else "upper left", handletextpad=0.2)
        if ds == "mimic":
            for name, short, off in [("Emergency Department", "Emergency Dept.", (-12, 9)),
                                     ("Medical Intensive Care Unit (MICU)", "MICU", (4, 2)),
                                     ("Discharge Lounge", "Discharge Lounge", (-62, 16)),
                                     ("Labor & Delivery", "Labor & Delivery (1 death)", (-118, 6))]:
                if name in g.index:
                    x, y = g.loc[name, "n"], g.loc[name, "mort"]
                    ax.annotate(short, (x, y), xytext=off, textcoords="offset points",
                                fontsize=6.5, color=INK2,
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
            ax.set_ylim(-3, 36)
    fig.tight_layout(w_pad=2)
    save(fig, "fig1_federations")


# ---------------------------------------------------------------------------
# Figure 3: SCAFFOLD minus each alternative, paired over seeds, 95% CI
# ---------------------------------------------------------------------------
def fig_paired_differences():
    rows = []
    for ds in DATASETS:
        pw = pd.read_csv(os.path.join(RUNS, ds, "pairwise_main.csv"))
        pw = pw[(pw["a"] == "SCAFFOLD") & pw["b"].isin(STRATEGIES + ["naive"])]
        rows.append(pw.assign(dataset=ds))
    pw = pd.concat(rows)
    order = [s for s in STRATEGIES if s != "SCAFFOLD"] + ["naive"]
    labels = order[:-1] + ["Centralized"]
    colors = {"mimic": BLUE, "eicu": ORANGE, "eicu_aps": AQUA}
    offs = {"mimic": -0.22, "eicu": 0.0, "eicu_aps": 0.22}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    for ax, (m, mlabel) in zip(axes, METRICS.items()):
        sub = pw[pw["metric"] == m]
        for ds in DATASETS:
            s = sub[sub["dataset"] == ds].set_index("b").reindex(order)
            y = np.arange(len(order)) + offs[ds]
            ax.errorbar(s["mean_diff"], y, xerr=[s["mean_diff"] - s["ci95_low"], s["ci95_high"] - s["mean_diff"]],
                        fmt="o", ms=3.5, color=colors[ds], ecolor=colors[ds], elinewidth=1,
                        capsize=0, label=DATASETS[ds], mec="white", mew=0.5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axhline(len(order) - 1.5, color=GRID, lw=0.8)
        ax.set_yticks(range(len(order)), labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"SCAFFOLD minus alternative, {mlabel}")
        ax.set_title(("a  " if m == "auc_roc" else "b  ") + mlabel)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, -0.42), ncol=3, handletextpad=0.1)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_paired_differences")


# ---------------------------------------------------------------------------
# Figure 4: size of each protocol choice next to the choice of strategy
# ---------------------------------------------------------------------------
def protocol_effects(ds, m):
    """(label, center, low, high) for one dataset and metric; low/high span the
    strategies (or seeds, for leakage)."""
    out = []
    main = load(ds, "main")
    means = main.groupby("strategy")[m].mean()
    spread = means.max() - means.min()
    out.append(("Choice of aggregation strategy", spread, spread, spread))
    if ds == "mimic":
        lk = pd.read_csv(os.path.join(RUNS, "mimic", "leakage_check.csv"))
        w = lk.pivot(index="seed", columns="features", values=m)
        d = w["24h features + ICD counts"] - w["24h features"]
        out.append(("Discharge-coded features\n(leakage, centralized)", d.mean(), d.min(), d.max()))
    smote = load(ds, "imb_smote")
    d = (main.groupby("strategy")[m].mean() - smote.groupby("strategy")[m].mean())
    out.append(("Client-level SMOTE\n(vs class weights)", d.mean(), d.min(), d.max()))
    last = (main[m] - main[f"last_{m}"]).groupby(main["strategy"]).mean()
    out.append(("Last round instead of\nbest validation round", last.mean(), last.min(), last.max()))
    # Untuned = each strategy run with common default hyperparameters
    # (run_fl_experiments.DEFAULT_HP), tuned = the grid's validation choice;
    # both on the same seed-42 tuning run, scored on the test set.
    grid = pd.read_csv(os.path.join(TUNING, ds, "grid_all.csv"))
    best = json.load(open(os.path.join(TUNING, ds, "best_hp.json")))
    diffs = []
    for s, keys in OWN_HP.items():
        sub = grid[grid["strategy"] == s].copy()
        sub["hp"] = sub["hp_grid"].map(json.loads)
        dflt = sub[sub["hp"].map(lambda h: all(np.isclose(h.get(k), DEFAULT_HP[k]) for k in keys))]
        tuned = sub[sub["hp"].map(lambda h: h == best[s])]
        if len(dflt) and len(tuned):
            diffs.append(tuned[m].iloc[0] - dflt[m].iloc[0])
    d = pd.Series(diffs)
    out.append(("Default instead of\ntuned hyperparameters", d.mean(), d.min(), d.max()))
    return out


def fig_protocol_effects():
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.6), sharex="col")
    for i, ds in enumerate(["mimic", "eicu"]):
        for j, (m, mlabel) in enumerate(METRICS.items()):
            ax = axes[i, j]
            eff = protocol_effects(ds, m)
            labels = [e[0] for e in eff]
            y = np.arange(len(eff))
            for k, (lab, c, lo, hi) in enumerate(eff):
                ref = k == 0
                ax.barh(k, c, height=0.55, color=GRAY if ref else BLUE, linewidth=0)
                if hi > lo:
                    ax.plot([lo, hi], [k, k], color=INK, lw=0.8)
                ax.text(max(c, hi) + 0.002, k, f"{abs(c) if abs(c) < 5e-4 else c:.3f}", va="center",
                        fontsize=6.5, color=INK2)
            ax.set_yticks(y, labels if j == 0 else [""] * len(labels))
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.axvline(0, color=INK2, lw=0.6)
            letter = "abcd"[2 * i + j]
            ax.set_title(f"{letter}  {DATASETS[ds]} - {mlabel}")
            if i == 1:
                ax.set_xlabel(f"Change in {mlabel} (absolute)")
    fig.text(0.01, -0.02, "Gray: largest difference between the seven strategies' means. Blue: mean change caused by the "
             "protocol choice (positive = the recommended protocol is better);\nblack lines: range over strategies "
             "(over seeds for leakage). Hyperparameters: six strategies with a tuning grid, seed 42.",
             fontsize=6.5, color=INK2)
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "fig4_protocol_effects")


# ---------------------------------------------------------------------------
# Figure 5: validation AUC-PR per round
# ---------------------------------------------------------------------------
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    emph = {"SCAFFOLD": BLUE, "FedAvg": ORANGE, "FedProx": AQUA}
    for ax, ds in zip(axes, ["mimic", "eicu"]):
        h = load(ds, "main", "fl_round_history")
        curves = h.groupby(["strategy", "round"])["val_auc_pr"].mean().unstack(0)
        for s in STRATEGIES:
            if s in emph:
                continue
            ax.plot(curves.index, curves[s], color=GRAY, lw=0.8, alpha=0.9)
        for s, c in emph.items():
            ax.plot(curves.index, curves[s], color=c, lw=1.6, label=s)
        last = curves.index.max()
        ax.text(last + 0.8, curves[[x for x in STRATEGIES if x not in emph]].iloc[-1].mean(),
                "other\nstrategies", fontsize=6.5, color=MUTED, va="center")
        ax.set_xlim(1, last + 9)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Validation AUC-PR (mean of 10 seeds)")
        ax.set_title(("a  " if ds == "mimic" else "b  ") + DATASETS[ds])
        ax.legend(loc="lower right")
    fig.tight_layout(w_pad=2)
    save(fig, "fig5_convergence")


# ---------------------------------------------------------------------------
# Figure 6: reliability diagram, SCAFFOLD, raw vs Platt-recalibrated (seed 42)
# ---------------------------------------------------------------------------
def reliability(y, p, bins=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    return (np.array([p[idx == b].mean() for b in range(len(q) - 1)]),
            np.array([y[idx == b].mean() for b in range(len(q) - 1)]))


def fig_calibration(seed=42, strategy="SCAFFOLD"):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    for ax, ds in zip(axes, ["mimic", "eicu"]):
        d = os.path.join(RUNS, ds, "main")
        run = pd.read_csv(os.path.join(d, f"fl_final_results_seed{seed}.csv")).iloc[0]
        bundle = data_mod.build_clients_cached(data_mod.load_cohort(ds), seed=seed, config=run["config"],
                                               imbalance=run["imbalance"], dataset=ds)
        flat = torch.load(os.path.join(d, f"fl_models_seed{seed}.pt"))[strategy]
        model = get_model(run["model"], bundle["n_features"])
        pv = fed.predict_proba(model, bundle["X_val_global"], flat)
        pt = fed.predict_proba(model, bundle["X_test_global"], flat)
        yv, yt = bundle["y_val_global"], bundle["y_test_global"]
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(lg(pv).reshape(-1, 1), yv)
        pp = platt.predict_proba(lg(pt).reshape(-1, 1))[:, 1]
        hi = 1.0
        sq = (lambda v: np.sqrt(np.clip(v, 0, None)), np.square)
        ax.set_xscale("function", functions=sq)
        ax.set_yscale("function", functions=sq)
        diag = np.linspace(0, hi, 200)
        ax.plot(diag, diag, color=INK2, lw=0.8, label="Perfect calibration")
        for p, c, lab in ((pt, ORANGE, "Raw (class-weighted training)"), (pp, BLUE, "Platt-recalibrated")):
            mx, my = reliability(yt, p)
            ax.plot(mx, my, "-o", color=c, lw=1.4, ms=3.5, mec="white", mew=0.5, label=lab)
        ticks = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("Mean predicted probability (decile bins, square-root scale)")
        ax.set_ylabel("Observed mortality (square-root scale)")
        ax.set_title(("a  " if ds == "mimic" else "b  ") + f"{DATASETS[ds]}, {strategy}")
        ax.legend(loc="upper left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig6_calibration")


if __name__ == "__main__":
    fig_federations()
    fig_paired_differences()
    fig_protocol_effects()
    fig_convergence()
    fig_calibration()
