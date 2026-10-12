"""
Synchronous federated learning simulation: local training, server-side
aggregation rules and evaluation.

A plain Python loop is used instead of a framework simulator: with tens of
clients on one machine there is no benefit to a process pool, and a manual
loop keeps every aggregation rule (including FedCluster and the SCAFFOLD
control variates) short and easy to inspect. Functionally it is the standard
synchronous protocol: in each round every participating client trains from
the global model and the server aggregates the returned models.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

import os

# CPU by default: for this small MLP the per-batch overhead dominates, and one
# CPU process with 3 threads runs a round ~2.4x faster than Apple's MPS
# device (4.3s vs 10.2s on an M-series laptop), which also lets several runs
# share the machine. FEDMIMIC_DEVICE=mps (or cuda) overrides it.
DEVICE = torch.device(os.environ.get("FEDMIMIC_DEVICE", "cpu"))


def as_tensor(a):
    """Zero-copy float32 view of a numpy array (torch.tensor would copy it:
    ~80 MB per call for the Emergency Department client, every round)."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).clone()


def set_flat_params(model, flat):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view_as(p))
        offset += n


def make_loader(X, y, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    X_t = as_tensor(X)
    y_t = as_tensor(y).view(-1, 1)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


def _weighted_bce(out, yb, pos_w):
    out_c = out.clamp(1e-7, 1 - 1e-7)
    weight = yb * pos_w + (1 - yb)
    return nn.functional.binary_cross_entropy(out_c, yb, weight=weight)


def full_gradient(model, flat, X, y, pos_weight, batch_size=4096):
    """Gradient of the client's (class-weighted) loss at parameters `flat`,
    over its whole training set: SCAFFOLD's option-I control variate."""
    model.to(DEVICE)
    set_flat_params(model, flat.to(DEVICE))
    model.train()
    model.zero_grad()
    pos_w = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    X_t = as_tensor(X)
    y_t = as_tensor(y).view(-1, 1)
    n = len(y_t)
    for i in range(0, n, batch_size):
        xb, yb = X_t[i:i + batch_size].to(DEVICE), y_t[i:i + batch_size].to(DEVICE)
        loss = _weighted_bce(model(xb), yb, pos_w) * (len(yb) / n)
        loss.backward()
    return torch.cat([p.grad.view(-1) for p in model.parameters()]).detach().cpu().clone()


def train_local(model, global_flat, X, y, pos_weight, local_epochs=3, batch_size=32,
                 lr=0.001, mu=0.0, seed=42, correction=None, optimizer="adam"):
    """One client's local training round. `mu>0` activates the FedProx
    proximal term. `pos_weight` implements per-client class-weighted BCE
    (used for the Labor & Delivery client, which SMOTE cannot touch — see
    src/data.py) and, more mildly, for every other client on top of its
    already-SMOTE-Tomek-balanced training partition. `correction` (a flat
    vector, c - c_i) is SCAFFOLD's drift correction, added to every
    mini-batch gradient before the optimizer step; SCAFFOLD runs with
    optimizer="sgd", as in its original formulation (see Scaffold).
    """
    model.to(DEVICE)
    set_flat_params(model, global_flat.to(DEVICE))
    global_params = [p.detach().clone() for p in model.parameters()]

    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    loader = make_loader(X, y, batch_size, seed)
    corr = None
    if correction is not None:
        corr, offset = [], 0
        for p in model.parameters():
            corr.append(correction[offset:offset + p.numel()].view_as(p).to(DEVICE))
            offset += p.numel()

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = _weighted_bce(model(xb), yb, pos_w)
            if mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (mu / 2) * prox
            loss.backward()
            if corr is not None:
                for p, c in zip(model.parameters(), corr):
                    p.grad.add_(c)
            opt.step()

    return get_flat_params(model).cpu(), len(y)


@torch.no_grad()
def predict_proba(model, X, global_flat=None, batch_size=4096):
    model.to(DEVICE)
    if global_flat is not None:
        set_flat_params(model, global_flat.to(DEVICE))
    model.eval()
    X_t = as_tensor(X)
    preds = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(DEVICE)
        preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds) if preds else np.array([])


def find_best_threshold(model, global_flat, X_val, y_val):
    if len(y_val) == 0:
        return 0.5
    probs = predict_proba(model, X_val, global_flat)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate(model, global_flat, X, y, threshold=0.5):
    """Accuracy/Precision/Recall/F1 at a fixed decision threshold, plus
    threshold-independent AUC-ROC and AUC-PR (average precision) -- the
    latter two are the primary metrics for this task following Harutyunyan
    et al. (2019), whose "decompensation" benchmark task has a near-identical
    2.06% prevalence to ours (2.16%) and, for exactly this reason, also
    reports AUC-ROC/AUC-PR as primary rather than a fixed-threshold metric
    like F1 (see Section~\\ref{sec:eval_metrics}). AUC-ROC/AUC-PR are
    undefined when y contains only one class (e.g. the Labor & Delivery
    client's test partition has zero positives by construction) -- reported
    as NaN in that case rather than raising. Also guards against a fully
    EMPTY y (Labor & Delivery's val/test partitions have zero ROWS, not
    just zero positives, since with only 1 total positive there is nothing
    to stratify a three-way split on -- see data.build_clients).
    """
    if len(y) == 0:
        nan = float("nan")
        return dict(accuracy=nan, precision=nan, recall=nan, f1=nan, loss=nan,
                    auc_roc=nan, auc_pr=nan, prevalence=nan)
    probs = predict_proba(model, X, global_flat)
    preds = (probs >= threshold).astype(int)
    eps = 1e-7
    probs_c = np.clip(probs, eps, 1 - eps)
    bce = -np.mean(y * np.log(probs_c) + (1 - y) * np.log(1 - probs_c))
    if len(np.unique(y)) < 2:
        auc_roc, auc_pr = float("nan"), float("nan")
    else:
        auc_roc = roc_auc_score(y, probs)
        auc_pr = average_precision_score(y, probs)
    return dict(
        accuracy=accuracy_score(y, preds),
        precision=precision_score(y, preds, zero_division=0),
        recall=recall_score(y, preds, zero_division=0),
        f1=f1_score(y, preds, zero_division=0),
        loss=float(bce),
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        prevalence=float(np.mean(y)) if len(y) else float("nan"),
    )


def evaluate_ranking(model, global_flat, X, y):
    """AUC-ROC and AUC-PR only, cheap enough to run on the validation set
    after every round for model selection."""
    probs = predict_proba(model, X, global_flat)
    return dict(auc_roc=roc_auc_score(y, probs), auc_pr=average_precision_score(y, probs))


def evaluate_per_client(model, global_flat, clients, threshold=0.5):
    """Evaluates the final global model separately on each client's own
    local (natural-distribution) test partition -- the federated analogue
    of Harutyunyan et al. (2019) Table 2, which breaks down performance by
    per-category prevalence. Here the "categories" are the five care units,
    whose test-set mortality rates range from 0.00% (Labor & Delivery) to
    2.99% (Medicine); Labor & Delivery's row will show NaN for AUC-ROC/PR
    since its test partition has zero positives by construction.
    """
    rows = {}
    for unit, c in clients.items():
        m = evaluate(model, global_flat, c["X_test"], c["y_test"], threshold=threshold)
        m["n_test"] = len(c["y_test"])
        m["n_pos_test"] = int(c["y_test"].sum())
        rows[unit] = m
    return rows


# ---------------------------------------------------------------------------
# Server-side aggregation strategies
# ---------------------------------------------------------------------------

def aggregate_fedavg(client_flats, client_ns):
    ns = np.array(client_ns, dtype=np.float64)
    weights = ns / ns.sum()
    stacked = torch.stack(client_flats)
    w = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    return (stacked * w).sum(dim=0)


class ServerAdagrad:
    def __init__(self, eta=0.01, tau=1e-3):
        self.eta = eta
        self.tau = tau
        self.v = None

    def step(self, global_flat, client_flats, client_ns):
        avg_new = aggregate_fedavg(client_flats, client_ns)
        delta = avg_new - global_flat
        if self.v is None:
            self.v = torch.zeros_like(delta)
        self.v = self.v + delta ** 2
        return global_flat + self.eta * delta / (torch.sqrt(self.v) + self.tau)


class ServerAdam:
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.99, tau=1e-3):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None
        self.v = None

    def step(self, global_flat, client_flats, client_ns):
        avg_new = aggregate_fedavg(client_flats, client_ns)
        delta = avg_new - global_flat
        if self.m is None:
            self.m = torch.zeros_like(delta)
            self.v = torch.zeros_like(delta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta ** 2)
        return global_flat + self.eta * self.m / (torch.sqrt(self.v) + self.tau)


class ServerYogi(ServerAdam):
    """FedYogi (Reddi et al., 2021): FedAdam with Yogi's additive second-
    moment update, which grows v more cautiously than Adam's EMA."""

    def step(self, global_flat, client_flats, client_ns):
        avg_new = aggregate_fedavg(client_flats, client_ns)
        delta = avg_new - global_flat
        if self.m is None:
            self.m = torch.zeros_like(delta)
            self.v = torch.zeros_like(delta)
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        d2 = delta ** 2
        self.v = self.v - (1 - self.beta2) * d2 * torch.sign(self.v - d2)
        return global_flat + self.eta * self.m / (torch.sqrt(self.v) + self.tau)


class Scaffold:
    """SCAFFOLD (Karimireddy et al., 2020) server and client control
    variates. Each sampled client trains with local SGD, its gradients
    corrected by c - c_i; its new control variate is option I of the
    paper, the full local gradient at the round's global model.

    Local SGD is required: with Adam, the per-coordinate normalization turns
    the (round-constant) correction into a drift of ~lr per step whatever
    its size, which over the Emergency Department's ~3,000 local steps per
    round made c blow up and the model collapse within three rounds.

    The objective weights clients by training-set size (as the aggregation
    does), so c is the size-weighted average of the c_i: the server moves c
    by the weighted sum of the sampled clients' changes and the global model
    by eta_g times the weighted average client update."""

    def __init__(self, client_names, n_params, eta_g=1.0, client_sizes=None):
        self.eta_g = eta_g
        self.c = torch.zeros(n_params)
        self.c_i = {k: torch.zeros(n_params) for k in client_names}
        sizes = client_sizes or {k: 1 for k in client_names}
        total = float(sum(sizes.values()))
        self.w = {k: sizes[k] / total for k in client_names}

    def correction(self, client):
        return self.c - self.c_i[client]

    def update_client(self, client, new_c_i):
        delta = new_c_i - self.c_i[client]
        self.c_i[client] = new_c_i
        return self.w[client] * delta

    def step(self, global_flat, client_flats, client_ns, weighted_c_deltas):
        avg_new = aggregate_fedavg(client_flats, client_ns)
        self.c = self.c + torch.stack(weighted_c_deltas).sum(dim=0)
        return global_flat + self.eta_g * (avg_new - global_flat)


def aggregate_fedcluster(global_flat, client_flats, client_ns, eps=None, min_samples=2,
                          eps_quantile=0.5):
    """DBSCAN over the clients' delta vectors, L2-normalized to unit length
    first so clustering reflects update DIRECTION (client drift) rather than
    magnitude, which otherwise dominates distance in a small-model, high-
    dimensional parameter space; keeps only the largest cluster's deltas,
    averaged (unweighted, per the manuscript's Eq. in Section 3.5).
    """
    deltas = torch.stack([f - global_flat for f in client_flats]).numpy()
    n = deltas.shape[0]
    if n <= min_samples:
        # Too few clients for a meaningful density-based cluster split;
        # fall back to averaging everyone (== FedAvg for this round).
        avg_delta = torch.tensor(deltas.mean(axis=0), dtype=torch.float32)
        return global_flat + avg_delta

    flat_norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    flat_norms[flat_norms == 0] = 1.0
    normed = deltas / flat_norms  # direction-only, scale-invariant clustering
    if eps is None:
        from sklearn.metrics import pairwise_distances
        d = pairwise_distances(normed)
        # DBSCAN radius = a quantile of the pairwise distances (the median
        # by default); the quantile is FedCluster's tuned hyperparameter.
        eps = np.quantile(d[np.triu_indices(n, k=1)], eps_quantile) if n > 1 else 0.5
        eps = max(eps, 1e-3)

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(normed)
    if (labels >= 0).sum() == 0:
        avg_delta = torch.tensor(deltas.mean(axis=0), dtype=torch.float32)
        return global_flat + avg_delta

    vals, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_cluster = vals[np.argmax(counts)]
    mask = labels == largest_cluster
    avg_delta = torch.tensor(deltas[mask].mean(axis=0), dtype=torch.float32)
    return global_flat + avg_delta
