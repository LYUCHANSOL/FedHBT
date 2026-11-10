"""Tabular federations with one rare class held by only a few clients.

Datasets: covtype (Covertype), shuttle (Statlog Shuttle), drybean (Dry Bean).
Classes with fewer than MIN_CLASS samples are dropped; the rare class is the
smallest remaining class.
"""

import os
import numpy as np

DATA_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIN_CLASS = 100


def _raw(name):
    if name == "covtype":
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype(data_home=DATA_HOME)
        return d.data.astype(np.float32), d.target.astype(str), 10
    if name == "shuttle":
        from sklearn.datasets import fetch_openml
        d = fetch_openml("shuttle", version=1, as_frame=False, data_home=DATA_HOME, parser="liac-arff")
        return d.data.astype(np.float32), d.target.astype(str), d.data.shape[1]
    if name == "drybean":
        cache = os.path.join(DATA_HOME, "drybean.npz")
        if not os.path.exists(cache):
            from ucimlrepo import fetch_ucirepo
            d = fetch_ucirepo(id=602)
            np.savez(cache, X=d.data.features.to_numpy(np.float32),
                     y=d.data.targets.iloc[:, 0].astype(str).to_numpy())
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"].astype(str), z["X"].shape[1]
    raise ValueError(name)


def load_dataset(name):
    """Returns X (standardized continuous columns), y in 0..C-1, rare class index."""
    X, labels, n_cont = _raw(name)
    names, y = np.unique(labels, return_inverse=True)
    keep = np.flatnonzero(np.bincount(y) >= MIN_CLASS)
    mask = np.isin(y, keep)
    X, y = X[mask], np.searchsorted(keep, y[mask])
    mu, sd = X[:, :n_cont].mean(0), X[:, :n_cont].std(0) + 1e-8
    X[:, :n_cont] = (X[:, :n_cont] - mu) / sd
    rare = int(np.argmin(np.bincount(y)))
    return X, y, rare


def make_federation(X, y, rare_class, seed=0, K=100, k_rare=2, rare_per_holder=30,
                    per_class_cap=6000, classes_per_client=(2, 3), test_per_class=400,
                    natural_test_size=20000, dirichlet_alpha=1.0):
    """Returns (clients, X_test, y_test, X_nat, y_nat, holders).

    clients[k] = (X_k, y_k). Every client holds 2-3 common classes, split by a
    Dirichlet over the clients that hold each class. Only `k_rare` clients hold
    the rare class, each with `rare_per_holder` samples.
    (X_nat, y_nat) is drawn first, the same fraction from every class, so it keeps
    the full-data class frequencies; it is used for overall accuracy.
    (X_test, y_test) is class-balanced (up to a third of what remains of each
    class), for per-class accuracy.
    No sample appears in more than one of train, balanced test, natural test.
    """
    rng = np.random.default_rng(seed)
    C = int(y.max()) + 1
    nat_frac = min(0.15, natural_test_size / len(y))
    test_idx, nat_idx, pool = [], [], {}
    for c in range(C):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_nat = int(round(nat_frac * len(idx)))
        nat_idx.append(idx[:n_nat])
        idx = idx[n_nat:]
        n_test = min(test_per_class, len(idx) // 3)
        test_idx.append(idx[:n_test])
        cap = k_rare * rare_per_holder if c == rare_class else per_class_cap
        pool[c] = idx[n_test:n_test + cap]
    test_idx = np.concatenate(test_idx)
    nat_idx = np.concatenate(nat_idx)

    common = [c for c in range(C) if c != rare_class]
    lo, hi = min(classes_per_client[0], len(common)), min(classes_per_client[1], len(common))
    owners = {c: [] for c in range(C)}
    for k in range(K):
        for c in rng.choice(common, size=rng.integers(lo, hi + 1), replace=False):
            owners[c].append(k)
    holders = rng.choice(K, size=k_rare, replace=False)
    owners[rare_class] = list(holders)

    parts = [[] for _ in range(K)]
    for c in range(C):
        own = owners[c]
        if not own:
            continue
        if c == rare_class:
            splits = np.array_split(pool[c], len(own))
        else:
            props = rng.dirichlet(np.full(len(own), dirichlet_alpha))
            cuts = (np.cumsum(props)[:-1] * len(pool[c])).astype(int)
            splits = np.split(pool[c], cuts)
        for k, s in zip(own, splits):
            parts[k].append(s)

    clients = []
    for k in range(K):
        idx = np.concatenate(parts[k]) if parts[k] else np.array([], dtype=int)
        clients.append((X[idx], y[idx]))
    return clients, X[test_idx], y[test_idx], X[nat_idx], y[nat_idx], holders


def class_counts(clients, C):
    return np.array([np.bincount(yk, minlength=C) for _, yk in clients], dtype=float)
