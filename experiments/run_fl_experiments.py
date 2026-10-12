"""
Runs the federated learning strategies (FedAvg, FedProx, SCAFFOLD,
FedAdagrad, FedAdam, FedYogi, FedCluster) on the MIMIC-IV or eICU-CRD
federation and writes per-round and final metrics, per-client metrics, test
predictions and the selected global models.

Shared settings: 50 rounds, 3 local epochs, mini-batches of 512 rows, Adam
with learning rate 1e-3 for local training (SGD for SCAFFOLD).

Each strategy's own hyperparameters (FedProx mu, SCAFFOLD's global step
eta_g, the server learning rate eta and stability constant tau of
FedAdagrad/FedAdam/FedYogi, FedCluster's DBSCAN radius quantile) are command-line options, so that experiments/tune.py can
select them on the validation set; the defaults are common untuned settings; the
tuned values are passed with --hp_json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import _threading_limits  # noqa: E402,F401  (must be imported before numpy/sklearn/torch)

import numpy as np
import pandas as pd
import torch

import data as data_mod  # noqa: E402
import federated as fed  # noqa: E402
from models import get_model  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

STRATEGIES = ["FedAvg", "FedProx", "SCAFFOLD", "FedAdagrad", "FedAdam", "FedYogi", "FedCluster"]

DEFAULT_HP = dict(mu=0.1, eta_g=1.0, lr_scaffold=0.03, eta_adagrad=0.01, tau_adagrad=1e-3, eta_adam=0.01, tau_adam=1e-3,
                  eta_yogi=0.01, tau_yogi=1e-3, eps_quantile=0.5)


def run_strategy(strategy, bundle, model_name="dnn", rounds=50, local_epochs=3,
                  batch_size=512, lr=0.001, hp=None, participation=1.0, seed=42, eval_every=1):
    hp = dict(DEFAULT_HP, **(hp or {}))
    torch.manual_seed(seed)
    np.random.seed(seed)

    clients = bundle["clients"]
    names = list(clients)
    n_features = bundle["n_features"]
    model = get_model(model_name, n_features)
    global_flat = fed.get_flat_params(model)

    server_opt = None
    if strategy == "FedAdagrad":
        server_opt = fed.ServerAdagrad(eta=hp["eta_adagrad"], tau=hp["tau_adagrad"])
    elif strategy == "FedAdam":
        server_opt = fed.ServerAdam(eta=hp["eta_adam"], tau=hp["tau_adam"])
    elif strategy == "FedYogi":
        server_opt = fed.ServerYogi(eta=hp["eta_yogi"], tau=hp["tau_yogi"])
    elif strategy == "SCAFFOLD":
        server_opt = fed.Scaffold(names, global_flat.numel(), eta_g=hp["eta_g"],
                                  client_sizes={k: len(clients[k]["y_train_res"]) for k in names})

    # Clients sampled per round when participation < 1; the same sequence of
    # samples for every strategy within a seed, so comparisons stay paired.
    # SCAFFOLD trains locally with SGD at its own (tuned) learning rate.
    local_opt, local_lr = ("sgd", hp["lr_scaffold"]) if strategy == "SCAFFOLD" else ("adam", lr)

    n_per_round = max(1, int(round(participation * len(names))))
    sampler = np.random.RandomState(seed + 10_000)

    # Model selection: the global model is checkpointed at the round with the
    # best validation AUC-PR, so the number of rounds is an upper bound rather
    # than a choice that favours whichever strategy peaks near it. The last
    # round's model is still evaluated and reported alongside.
    best_val, best_round, best_flat = -1.0, 0, global_flat.clone()

    history = []
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        if n_per_round < len(names):
            selected = [names[i] for i in sorted(sampler.choice(len(names), n_per_round, replace=False))]
        else:
            selected = names
        client_flats, client_ns, c_deltas = [], [], []
        for unit in selected:
            c = clients[unit]
            mu_eff = hp["mu"] if strategy == "FedProx" else 0.0
            correction = None
            if strategy == "SCAFFOLD":
                correction = server_opt.correction(unit)
                new_c_i = fed.full_gradient(model, global_flat, c["X_train_res"], c["y_train_res"],
                                            c["pos_weight"])
                c_deltas.append(server_opt.update_client(unit, new_c_i))
            flat, n = fed.train_local(
                model, global_flat, c["X_train_res"], c["y_train_res"], c["pos_weight"],
                local_epochs=local_epochs, batch_size=batch_size, lr=local_lr, mu=mu_eff,
                seed=seed + rnd, correction=correction, optimizer=local_opt,
            )
            client_flats.append(flat)
            client_ns.append(n)

        if strategy in ("FedAvg", "FedProx"):
            global_flat = fed.aggregate_fedavg(client_flats, client_ns)
        elif strategy in ("FedAdagrad", "FedAdam", "FedYogi"):
            global_flat = server_opt.step(global_flat, client_flats, client_ns)
        elif strategy == "SCAFFOLD":
            global_flat = server_opt.step(global_flat, client_flats, client_ns, c_deltas)
        elif strategy == "FedCluster":
            global_flat = fed.aggregate_fedcluster(global_flat, client_flats, client_ns,
                                                   eps_quantile=hp["eps_quantile"])
        else:
            raise ValueError(strategy)

        val_auc_pr = fed.evaluate_ranking(model, global_flat, bundle["X_val_global"], bundle["y_val_global"])["auc_pr"]
        if val_auc_pr > best_val:
            best_val, best_round, best_flat = val_auc_pr, rnd, global_flat.clone()

        if rnd % eval_every == 0 or rnd == rounds:
            metrics = fed.evaluate(model, global_flat, bundle["X_test_global"], bundle["y_test_global"], threshold=0.5)
            metrics.update(round=rnd, strategy=strategy, elapsed_s=time.time() - t0, val_auc_pr=val_auc_pr)
            history.append(metrics)
            print(f"[{strategy}] round {rnd:3d}/{rounds}  "
                  f"AUC-ROC={metrics['auc_roc']:.4f} AUC-PR={metrics['auc_pr']:.4f}  "
                  f"F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} "
                  f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}  "
                  f"val AUC-PR={val_auc_pr:.4f}  ({metrics['elapsed_s']:.1f}s elapsed)", flush=True)

    total_time = time.time() - t0
    last = fed.evaluate_ranking(model, global_flat, bundle["X_test_global"], bundle["y_test_global"])
    global_flat = best_flat

    best_t = fed.find_best_threshold(model, global_flat, bundle["X_val_global"], bundle["y_val_global"])
    final_metrics = fed.evaluate(model, global_flat, bundle["X_test_global"], bundle["y_test_global"], threshold=best_t)
    val_metrics = fed.evaluate(model, global_flat, bundle["X_val_global"], bundle["y_val_global"], threshold=best_t)
    final_metrics.update(strategy=strategy, time_s=total_time, threshold=best_t,
                         val_auc_roc=val_metrics["auc_roc"], val_auc_pr=val_metrics["auc_pr"],
                         best_round=best_round, last_auc_roc=last["auc_roc"], last_auc_pr=last["auc_pr"],
                         hp=json.dumps(hp, sort_keys=True))

    per_client = fed.evaluate_per_client(model, global_flat, clients, threshold=best_t)
    per_client_rows = []
    for unit, m in per_client.items():
        row = dict(m)
        row.update(strategy=strategy, care_unit=unit)
        per_client_rows.append(row)

    return history, final_metrics, per_client_rows, global_flat, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--dataset", default="mimic", choices=data_mod.DATASETS)
    ap.add_argument("--config", default="min_deaths", choices=data_mod.CLIENT_CONFIGS)
    ap.add_argument("--imbalance", default="weighted", choices=data_mod.IMBALANCE_MODES)
    ap.add_argument("--participation", type=float, default=1.0)
    ap.add_argument("--model", default="dnn")
    ap.add_argument("--hp_json", default=None,
                    help="JSON file mapping strategy -> {hyperparameter: value}")
    for k, v in DEFAULT_HP.items():
        ap.add_argument(f"--{k}", type=float, default=None, help=f"default {v}")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tuned = {}
    if args.hp_json:
        with open(args.hp_json) as fh:
            tuned = json.load(fh)
    overrides = {k: getattr(args, k) for k in DEFAULT_HP if getattr(args, k) is not None}

    df = data_mod.load_cohort(args.dataset)
    bundle = data_mod.build_clients_cached(df, seed=args.seed, config=args.config,
                                           imbalance=args.imbalance, dataset=args.dataset)
    data_mod.client_summary(bundle).to_csv(
        os.path.join(args.out_dir, f"clients{args.out_tag}.csv"), index=False)

    all_history = []
    all_final = []
    all_per_client = []
    preds = {"y_test": bundle["y_test_global"], "client": bundle["test_client_global"]}
    weights = {}
    for strategy in args.strategies:
        hp = dict(tuned.get(strategy, {}), **overrides)
        print(f"\n=== Running {strategy} ({args.rounds} rounds, config={args.config}, "
              f"imbalance={args.imbalance}, hp={hp}) ===", flush=True)
        history, final_metrics, per_client_rows, global_flat, model = run_strategy(
            strategy, bundle, model_name=args.model, rounds=args.rounds,
            local_epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr, hp=hp,
            participation=args.participation, seed=args.seed, eval_every=args.eval_every,
        )
        final_metrics.update(seed=args.seed, dataset=args.dataset, config=args.config,
                             imbalance=args.imbalance,
                             participation=args.participation, local_epochs=args.local_epochs,
                             rounds=args.rounds, model=args.model)
        all_history.extend(history)
        all_final.append(final_metrics)
        all_per_client.extend(per_client_rows)
        if args.save_preds:
            preds[strategy] = fed.predict_proba(model, bundle["X_test_global"], global_flat)
            np.savez_compressed(os.path.join(args.out_dir, f"fl_test_preds{args.out_tag}.npz"), **preds)
            # The selected global model, for experiments/personalize.py.
            weights[strategy] = global_flat
            torch.save(weights, os.path.join(args.out_dir, f"fl_models{args.out_tag}.pt"))

        pd.DataFrame(all_history).to_csv(
            os.path.join(args.out_dir, f"fl_round_history{args.out_tag}.csv"), index=False)
        pd.DataFrame(all_final).to_csv(
            os.path.join(args.out_dir, f"fl_final_results{args.out_tag}.csv"), index=False)
        pd.DataFrame(all_per_client).to_csv(
            os.path.join(args.out_dir, f"fl_per_client_results{args.out_tag}.csv"), index=False)

    print("\n=== FINAL RESULTS ===")
    print(pd.DataFrame(all_final).drop(columns=["hp"]).to_string(index=False))


if __name__ == "__main__":
    main()
