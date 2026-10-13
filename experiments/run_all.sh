#!/bin/bash
# Full experiment pipeline for MIMIC-IV and eICU-CRD:
# per-strategy tuning on each dataset's validation set, every run listed in
# experiments/jobs.py (N in parallel), then aggregation and the prediction
# analysis, the personalization (local fine-tuning) evaluation and the
# post-hoc recalibration. Safe to re-run: finished tuning configs and runs are skipped.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venvs/FedMIMIC/bin/python}"
N="${N_PARALLEL:-3}"

for DS in mimic eicu; do
    mkdir -p "results/tuning/$DS"
    if [ ! -f "results/tuning/$DS/best_hp.json" ]; then
        for w in $(seq 0 $((N - 1))); do
            "$PY" experiments/tune.py --dataset "$DS" --worker "$w" --n_workers "$N" \
                > "results/tuning/$DS/log_worker$w.txt" 2>&1 &
        done
        wait
        "$PY" experiments/tune.py --dataset "$DS" --select > "results/tuning/$DS/selection.txt"
    fi
    echo "== $DS tuning selected =="
    tail -25 "results/tuning/$DS/selection.txt"
done

"$PY" experiments/jobs.py --run "$N"
# Control for SCAFFOLD (FedAvg-SGD): its tuning, ten seeds and the paired comparison.
"$PY" experiments/run_control.py --parallel "$N" > results/runs/control_log.txt 2>&1

for DS in mimic eicu eicu_aps; do
    "$PY" experiments/aggregate.py --dataset "$DS" > "results/runs/$DS/aggregate.txt"
    "$PY" experiments/analyze_predictions.py --dataset "$DS" > "results/runs/$DS/prediction_analysis.txt"
    "$PY" experiments/personalize.py --dataset "$DS" --experiment main > "results/runs/$DS/personalization.txt"
    "$PY" experiments/recalibrate.py --dataset "$DS" --experiment main > "results/runs/$DS/recalibration.txt"
done
# Numbers, tables and figures of the manuscript.
"$PY" experiments/leakage_check.py > results/runs/mimic/leakage_check.txt
"$PY" experiments/subgroups.py > results/runs/subgroups.txt
"$PY" experiments/make_tables.py
"$PY" experiments/make_figures.py
echo "ALL DONE"
