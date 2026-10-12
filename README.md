# Protocol Choices Matter More Than the Aggregation Algorithm

Code and aggregated results for the study *"Protocol Choices Matter More Than the
Aggregation Algorithm: Federated Learning for In-Hospital Mortality Prediction in
Care-Unit and Multi-Hospital Federations"* (under review).

The study compares seven federated aggregation strategies (FedAvg, FedProx, SCAFFOLD,
FedAdagrad, FedAdam, FedYogi and FedCluster) for in-hospital mortality prediction on two
federations built with the same pipeline:

- **MIMIC-IV v3.1**: 12 care units of one hospital (an intra-hospital federation in which
  one client holds 79% of the stays);
- **eICU-CRD v2.0**: 49 hospitals (an inter-hospital federation).

Every strategy is tuned on validation data and selected by the same validation rule,
every comparison is repeated over ten seeds and tested with a Friedman test followed by
Holm-corrected comparisons, and calibration is evaluated before and after Platt
recalibration. The code also measures how much four protocol choices change the
conclusions: predictors coded at discharge, client-level SMOTE, evaluating the last round
instead of the best validation round, and untuned hyperparameters.

## Repository layout

| Path | Content |
|---|---|
| `src/data.py` | MIMIC-IV cohort extraction (self-contained SQL), client construction, patient-level splits, preprocessing, class-imbalance treatments |
| `src/data_eicu.py` | eICU-CRD cohort built from the PhysioNet CSV files (and the day-1 physiology sensitivity analysis) |
| `src/federated.py` | Local training, the aggregation rules (including SCAFFOLD control variates and FedCluster) and evaluation |
| `src/centralized.py`, `src/models.py` | Centralized baselines; MLP and logistic regression |
| `experiments/run_all.sh` | Runs the whole study: tuning, all experiments, aggregation, tables and figures |
| `experiments/tune.py` | Per-strategy hyperparameter grid search on validation AUC-PR |
| `experiments/jobs.py` | The experiment list (main, ablations, federation variants, sensitivity analyses) and a parallel runner |
| `experiments/run_fl_experiments.py`, `run_centralized.py` | One federated or centralized run (one dataset, configuration and seed) |
| `experiments/aggregate.py` | Means, confidence intervals, Friedman and Holm-corrected paired tests |
| `experiments/analyze_predictions.py`, `recalibrate.py`, `personalize.py`, `subgroups.py`, `leakage_check.py` | Calibration, Platt/isotonic recalibration, local fine-tuning, subgroup performance, leakage of discharge-coded features |
| `experiments/make_tables.py`, `make_figures.py` | Every table and figure of the manuscript |
| `results/tuning/` | Tuning grids and the selected hyperparameters per dataset |
| `results/runs/<dataset>/<experiment>/` | Per-seed metrics and the aggregated statistics |

Patient-level files (cohorts, test predictions, trained models and per-client counts) are
not included, as required by the PhysioNet data use agreements; `run_all.sh` regenerates
them.

## Data access

Both databases are available from [PhysioNet](https://physionet.org) to credentialed users
who complete the required training and sign the data use agreements:

- MIMIC-IV v3.1: <https://doi.org/10.13026/kpb9-mt58>. Load it into PostgreSQL with the
  MIT-LCP [mimic-code](https://github.com/MIT-LCP/mimic-code) scripts (schema `mimiciv_hosp`)
  and set `MIMIC_DSN`, e.g. `export MIMIC_DSN=postgresql+psycopg2://localhost/mimiciv`.
- eICU-CRD v2.0: <https://doi.org/10.13026/C2WM1R>. Only `patient.csv.gz`, `hospital.csv.gz`,
  `lab.csv.gz` and, for the sensitivity analysis, `apacheApsVar.csv.gz` are needed. Set
  `EICU_DIR` to the folder that holds them (default
  `~/Data/eicu-crd/physionet.org/files/eicu-crd/2.0`).

The derived cohorts are cached locally in `data/` (MIMIC-IV) and next to the eICU-CRD
files (`EICU_COHORT_CSV`); client datasets are cached in `~/Library/Caches/FedMIMIC`
(`FEDMIMIC_CACHE`).

## Running

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PY=$(which python) N_PARALLEL=3 bash experiments/run_all.sh
```

`run_all.sh` can be stopped and restarted: finished tuning configurations and runs are
skipped. The full study (2 tuning grids, 118 federated and 45 centralized runs) took about
two days on a 10-core laptop with three runs in parallel; training runs on the CPU, which
was faster than Apple's MPS device for this small model (`FEDMIMIC_DEVICE` overrides it).

Single runs, for example:

```bash
python experiments/run_fl_experiments.py --dataset eicu --seed 42 \
    --hp_json results/tuning/eicu/best_hp.json --out_dir results/runs/eicu/main --out_tag _seed42
python experiments/aggregate.py --dataset eicu
```

## Software

Python 3.12, PyTorch 2.13, scikit-learn 1.9, imbalanced-learn 0.14 (exact versions in
`requirements.txt`).

## License

MIT (see `LICENSE`).
