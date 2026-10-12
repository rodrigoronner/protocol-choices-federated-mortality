"""Centralized (non-federated) baselines, trained on the pooled 5-careunit
cohort and evaluated on the EXACT SAME global test set as the FL strategies
(see data.build_pooled_centralized_splits). Mirrors the manuscript's
appendix `train_centralized_model`: early stopping on validation loss,
patience=5, up to 100 epochs, Adam(lr=0.001, weight_decay=1e-5).
"""
import time

import numpy as np
import torch
import torch.nn as nn
from imblearn.combine import SMOTETomek

import federated as fed
from models import get_model


def _train_until_early_stop(model, X_train, y_train, X_val, y_val, pos_weight=1.0,
                             batch_size=512, lr=0.001, max_epochs=100, patience=5, seed=42):
    torch.manual_seed(seed)
    model.to(fed.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_w = torch.tensor([pos_weight], dtype=torch.float32, device=fed.DEVICE)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(fed.DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(fed.DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        loader = fed.make_loader(X_train, y_train, batch_size, seed + epoch)
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(fed.DEVICE), yb.to(fed.DEVICE)
            opt.zero_grad()
            out = model(xb).clamp(1e-7, 1 - 1e-7)
            weight = yb * pos_w + (1 - yb)
            loss = nn.functional.binary_cross_entropy(out, yb, weight=weight)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).clamp(1e-7, 1 - 1e-7)
            val_loss = nn.functional.binary_cross_entropy(val_out, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, epoch + 1


def run_centralized_baseline(strategy_name, pooled, n_features, model_name="dnn", seed=42, clients=None):
    """strategy_name in {"naive", "global_smote", "weighted_loss"}. If
    `clients` (the per-care-unit dict from data.build_clients) is given,
    also evaluates the trained centralized model on each client's own local
    test partition, for a direct centralized-vs-federated comparison at the
    same per-client granularity (see federated.evaluate_per_client).
    """
    t0 = time.time()
    X_train, y_train = pooled["X_train"], pooled["y_train"]
    pos_weight = 1.0

    if strategy_name == "global_smote":
        smt = SMOTETomek(random_state=seed)
        X_train, y_train = smt.fit_resample(X_train, y_train)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
    elif strategy_name == "weighted_loss":
        n_pos = max(int(y_train.sum()), 1)
        n_neg = max(int((1 - y_train).sum()), 1)
        pos_weight = min(n_neg / n_pos, 20.0)
    elif strategy_name != "naive":
        raise ValueError(strategy_name)

    model = get_model(model_name, n_features)
    model, n_epochs = _train_until_early_stop(
        model, X_train, y_train, pooled["X_val"], pooled["y_val"],
        pos_weight=pos_weight, seed=seed,
    )

    global_flat = fed.get_flat_params(model)
    best_t = fed.find_best_threshold(model, global_flat, pooled["X_val"], pooled["y_val"])
    metrics = fed.evaluate(model, global_flat, pooled["X_test"], pooled["y_test"], threshold=best_t)
    metrics.update(strategy=strategy_name, time_s=time.time() - t0, threshold=best_t,
                    n_epochs=n_epochs, n_train=len(y_train))

    per_client_rows = []
    if clients is not None:
        per_client = fed.evaluate_per_client(model, global_flat, clients, threshold=best_t)
        for unit, m in per_client.items():
            row = dict(m)
            row.update(strategy=strategy_name, care_unit=unit)
            per_client_rows.append(row)

    return metrics, per_client_rows
