import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import _threading_limits  # noqa: E402,F401  (must be imported before numpy/sklearn/torch)

import pandas as pd

import data as data_mod  # noqa: E402
from centralized import run_centralized_baseline  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="mimic", choices=data_mod.DATASETS)
    ap.add_argument("--config", default="min_deaths", choices=data_mod.CLIENT_CONFIGS)
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = data_mod.load_cohort(args.dataset)
    # Centralized training uses the clients' raw (pre-SMOTE) rows, so the
    # federated imbalance mode is irrelevant here; "weighted" skips the
    # per-client SMOTE pass that would otherwise run for nothing.
    bundle = data_mod.build_clients_cached(df, seed=args.seed, config=args.config, imbalance="weighted",
                                           dataset=args.dataset)
    pooled = data_mod.build_pooled_centralized_splits(bundle, seed=args.seed)

    results = []
    all_per_client = []
    for strategy in ["naive", "global_smote", "weighted_loss"]:
        print(f"\n=== Centralized: {strategy} ===")
        m, per_client_rows = run_centralized_baseline(
            strategy, pooled, bundle["n_features"], seed=args.seed, clients=bundle["clients"])
        m.update(seed=args.seed, dataset=args.dataset, config=args.config)
        print(m, flush=True)
        results.append(m)
        all_per_client.extend(per_client_rows)
        pd.DataFrame(results).to_csv(
            os.path.join(args.out_dir, f"centralized_results{args.out_tag}.csv"), index=False)
        pd.DataFrame(all_per_client).to_csv(
            os.path.join(args.out_dir, f"centralized_per_client_results{args.out_tag}.csv"), index=False)

    print("\n=== FINAL CENTRALIZED RESULTS ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
