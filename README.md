# Protocol Choices Matter More Than the Aggregation Algorithm

Code and aggregated results for the study *"Protocol Choices Matter More Than the
Aggregation Algorithm: Federated Learning for In-Hospital Mortality Prediction in
Care-Unit and Multi-Hospital Federations"* (under review).

## In plain language

Hospitals hold records that could train software to flag which admitted patients are at
highest risk of dying, so that care teams can act earlier. Such models work better when
trained on many hospitals, but patient privacy prevents the hospitals from pooling their
records. *Federated learning* avoids pooling: each hospital trains the model on its own
records and sends only the model updates to a server, which combines them into a shared
model. There are several ways of combining the updates (*aggregation strategies*), and many
studies try to find the best one.

This study shows that **how the comparison is run matters more than which strategy is
chosen**. Seven strategies were compared on two federations, the 12 care units of one
hospital (where the Emergency Department holds 79% of the stays) and 49 independent
hospitals, under an evaluation protocol that uses only information a real deployment would
have. In the care-unit federation, three choices common in published comparisons each
changed the results more than the choice of strategy, whose spread was 0.019 in AUC-ROC: using diagnosis codes that are only
recorded at discharge (+0.038), creating synthetic patients within each hospital to balance
the rare deaths (up to −0.074), and reporting the model from the last training round instead
of the best one (up to −0.084).

Under the realistic protocol, SCAFFOLD was the most accurate strategy in both federations,
but only by a small margin, and federated training was as accurate as pooling all the data
across the 49 hospitals. Common corrections for rare outcomes made the models overestimate
risk; a simple recalibration step (Platt scaling), which the hospitals can fit without
sharing records, restored accurate risk estimates. The practical lesson: fix a realistic
evaluation protocol first, and only then rank federated learning strategies.

## Study summary

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
instead of the best validation round, and untuned hyperparameters. A control run of FedAvg
with SCAFFOLD's local optimizer (FedAvg-SGD) separates the effect of SCAFFOLD's control
variates from that of its optimizer.

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
| `experiments/run_control.py` | Control for SCAFFOLD: FedAvg with local SGD tuned on SCAFFOLD's learning-rate grid, and the paired comparison with SCAFFOLD |
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
skipped. The full study (2 tuning grids, 118 federated and 45 centralized runs, plus the tuning and 20 runs of
the FedAvg-SGD control) took about
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
