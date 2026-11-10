"""Numpy MLP, privacy mechanisms, and federated training routines.

Routing methods follow Stratify's batch version: every global step visits each
class once; for a class turn, one or more clients compute a gradient on their
samples of that class, and the server averages the class gradients.

  oracle       server picks a holder from the true counts n_i(l)/N(l) (no privacy)
  counts_dp    server picks from counts released once as [n + Lap(1/eps)]_+
  bits_rr      server picks uniformly among clients whose randomized-response
               presence bit is 1 (pp-CFL style metadata)
  self_select  server broadcasts the class turn; every true holder trains and
               uploads. Each non-holder uploads a zero with probability dummy_q
               (cover traffic). The server receives only the SecAgg sum of
               uploaders, then divides by the holder count n_c, not |uploaders|.
               With eps given, n_c is released once as count + Lap(1/eps)
               (floored at 1). dummy_q=1 recovers all-client uploads.
A picked client that does not hold the class returns no gradient: the turn is lost.

Privacy accounting is per class entry for every method, matching pp-CFL's per-bit
randomized response.

Communication per global step: server routing uploads C models (one per class
turn). self_select uploads about sum_c [n_c + dummy_q*(K-n_c)] vectors, because
holders always send and a random fraction dummy_q of non-holders send zeros.
"""

import numpy as np


# ---------------------------------------------------------------- model
def init_mlp(rng, d, h, C):
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / d), (d, h)).astype(np.float32),
        "b1": np.zeros(h, np.float32),
        "W2": rng.normal(0, np.sqrt(2.0 / h), (h, C)).astype(np.float32),
        "b2": np.zeros(C, np.float32),
    }


def forward(p, X):
    H = np.maximum(X @ p["W1"] + p["b1"], 0.0)
    Z = H @ p["W2"] + p["b2"]
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    return H, P


def grad(p, X, y):
    H, P = forward(p, X)
    n = len(y)
    G = P.copy()
    G[np.arange(n), y] -= 1.0
    G /= n
    dW2 = H.T @ G
    db2 = G.sum(0)
    dH = (G @ p["W2"].T) * (H > 0)
    return {"W1": X.T @ dH, "b1": dH.sum(0), "W2": dW2, "b2": db2}


def add(p, g, s=1.0):
    return {k: p[k] + s * g[k] for k in p}


def sub(p, q):
    return {k: p[k] - q[k] for k in p}


def copy_params(p):
    return {k: v.copy() for k, v in p.items()}


def zeros_like(p):
    return {k: np.zeros_like(v) for k, v in p.items()}


def evaluate(p, X, y, C):
    pred = forward(p, X)[1].argmax(1)
    recall = np.array([(pred[y == c] == c).mean() for c in range(C)])
    return {"acc": float((pred == y).mean()), "bal_acc": float(recall.mean()), "recall": recall}


# ---------------------------------------------------------------- privacy
def rr_bits(rng, B, eps):
    p = 1.0 / (np.exp(eps) + 1.0)
    return np.where(rng.random(B.shape) < p, 1 - B, B)


def laplace_counts(rng, N, eps):
    return np.maximum(N + rng.laplace(0, 1.0 / eps, N.shape), 0.0)


# ---------------------------------------------------------------- training
def _class_batch(rng, Xk, yk, c, batch):
    idx = np.flatnonzero(yk == c)
    if len(idx) == 0:
        return None
    sel = rng.choice(idx, size=min(batch, len(idx)), replace=False)
    return Xk[sel], yk[sel]


def selection_table(rng, counts, routing, eps):
    """Per-class probability of picking each client, from what the server sees."""
    if routing == "oracle":
        S = counts.copy()
    elif routing == "counts_dp":
        S = laplace_counts(rng, counts, eps)
    elif routing == "bits_rr":
        S = rr_bits(rng, (counts > 0).astype(int), eps).astype(float)
    else:
        raise ValueError(routing)
    col = S.sum(0, keepdims=True)
    S = np.where(col > 0, S / np.maximum(col, 1e-12), 1.0 / S.shape[0])
    return S


def train_class_turns(clients, counts, C, d, routing, eps=None, steps=1500, lr=0.1,
                      batch=64, hidden=64, seed=0, dummy_q=0.1, log_every=0,
                      X_test=None, y_test=None):
    rng = np.random.default_rng(seed)
    p = init_mlp(rng, d, hidden, C)
    K = len(clients)
    holders = [np.flatnonzero(counts[:, c] > 0) for c in range(C)]
    if routing == "self_select":
        n_hold = np.array([len(h) for h in holders], dtype=float)
        if eps is not None:
            n_hold = np.maximum(n_hold + rng.laplace(0, 1.0 / eps, C), 1.0)
    else:
        S = selection_table(rng, counts, routing, eps)
    lost = np.zeros(C)
    n_up = 0
    dummy_rng = np.random.default_rng(seed + 99991)
    for t in range(steps):
        g_total = zeros_like(p)
        for c in rng.permutation(C):
            if routing == "self_select":
                if len(holders[c]) == 0:
                    continue
                g_sum = zeros_like(p)
                for k in holders[c]:
                    g_sum = add(g_sum, grad(p, *_class_batch(rng, *clients[k], c, batch)))
                n_dummy = dummy_rng.binomial(K - len(holders[c]), dummy_q)
                n_up += len(holders[c]) + int(n_dummy)
                g_total = add(g_total, g_sum, 1.0 / n_hold[c])
            else:
                k = rng.choice(K, p=S[:, c])
                b = _class_batch(rng, *clients[k], c, batch)
                n_up += 1
                if b is None:
                    lost[c] += 1
                    continue
                g_total = add(g_total, grad(p, *b))
        p = add(p, g_total, -lr / C)
    return p, lost / steps, n_up / steps


def n_params(p):
    return int(sum(v.size for v in p.values()))


def train_fedavg(clients, C, d, rounds=150, frac=0.1, local_steps=10, lr=0.1, batch=64,
                 hidden=64, seed=0):
    return train_fed_family(clients, C, d, method="fedavg", rounds=rounds, frac=frac,
                            local_steps=local_steps, lr=lr, batch=batch, hidden=hidden,
                            seed=seed)


def train_fed_family(clients, C, d, method="fedavg", rounds=150, frac=0.1, local_steps=10,
                     lr=0.1, batch=64, hidden=64, seed=0, mu=0.01, alpha_dyn=0.01,
                     moon_mu=1.0, moon_t=0.5):
    """Label-blind aggregators that share FedAvg's clock: 150 x 10 x 10.

    fedavg    sample-size weighted model average
    fedprox   FedProx proximal term mu ||w - w_t||^2  (Li et al., 2020)
    scaffold  SCAFFOLD control variates                (Karimireddy et al., 2020)
    feddyn    FedDyn dynamic regularizer               (Acar et al., ICLR 2021)
    moon      MOON model-contrastive loss on hidden    (Li et al., CVPR 2021)
    """
    rng = np.random.default_rng(seed)
    p = init_mlp(rng, d, hidden, C)
    K = len(clients)
    sizes = np.array([len(yk) for _, yk in clients], dtype=float)
    c_glob = zeros_like(p) if method == "scaffold" else None
    c_k = [zeros_like(p) for _ in range(K)] if method == "scaffold" else None
    h_k = [zeros_like(p) for _ in range(K)] if method == "feddyn" else None
    prev_k = [copy_params(p) for _ in range(K)] if method == "moon" else None
    n_sel = max(1, int(frac * K))
    for r in range(rounds):
        sel = rng.choice(K, size=n_sel, replace=False)
        new, w = zeros_like(p), 0.0
        dc_sum = zeros_like(p) if method == "scaffold" else None
        for k in sel:
            Xk, yk = clients[k]
            q = copy_params(p)
            p_t = p
            for _ in range(local_steps):
                i = rng.choice(len(yk), size=min(batch, len(yk)), replace=False)
                g = grad(q, Xk[i], yk[i])
                if method == "fedprox":
                    g = add(g, sub(q, p_t), mu)
                elif method == "scaffold":
                    g = add(g, sub(c_glob, c_k[k]))
                elif method == "feddyn":
                    g = add(g, h_k[k])
                    g = add(g, sub(q, p_t), alpha_dyn)
                elif method == "moon":
                    g = add(g, _moon_grad(q, p_t, prev_k[k], Xk[i], moon_t), moon_mu)
                q = add(q, g, -lr)
            new = add(new, q, sizes[k])
            w += sizes[k]
            if method == "scaffold":
                c_new = add(add(c_k[k], c_glob, -1.0),
                            scale_params(sub(p_t, q), 1.0 / (lr * local_steps)))
                dc_sum = add(dc_sum, sub(c_new, c_k[k]))
                c_k[k] = c_new
            elif method == "feddyn":
                h_k[k] = add(h_k[k], sub(q, p_t), alpha_dyn)
            elif method == "moon":
                prev_k[k] = copy_params(q)
        p = {kk: v / w for kk, v in new.items()}
        if method == "scaffold":
            c_glob = add(c_glob, dc_sum, 1.0 / K)
    return p


def scale_params(p, s):
    return {k: s * v for k, v in p.items()}


def _moon_grad(q, p_glob, p_prev, X, temperature):
    """Contrastive pull of local hidden toward global, push from previous local."""
    H, _ = forward(q, X)
    Hg, _ = forward(p_glob, X)
    Hp, _ = forward(p_prev, X)
    z = _l2norm(H)
    zg = _l2norm(Hg)
    zp = _l2norm(Hp)
    t = temperature
    pos = (z * zg).sum(1, keepdims=True) / t
    neg = (z * zp).sum(1, keepdims=True) / t
    m = np.maximum(pos, neg)
    epos, eneg = np.exp(pos - m), np.exp(neg - m)
    # dL/dpos = -eneg/(epos+eneg), dL/dneg = epos/(epos+eneg) wait:
    # L = -pos + log(epos+eneg); dL/dpos = -1 + epos/(epos+eneg) = -eneg/(epos+eneg)
    dpos = -eneg / (epos + eneg)
    dneg = eneg * 0 + epos / (epos + eneg)
    dpos /= t
    dneg /= t
    n, h = z.shape
    dz = (dpos * zg + dneg * zp) / n
    dH = _l2norm_backward(H, dz)
    # local params q only; freeze global/prev (stop-grad)
    return _hidden_to_param_grad(q, X, H, dH)


def _l2norm(H):
    nrm = np.sqrt((H * H).sum(1, keepdims=True) + 1e-8)
    return H / nrm


def _l2norm_backward(H, dZ):
    nrm = np.sqrt((H * H).sum(1, keepdims=True) + 1e-8)
    Z = H / nrm
    return (dZ - (Z * dZ).sum(1, keepdims=True) * Z) / nrm


def _hidden_to_param_grad(p, X, H, dH):
    """d ell / d(W1,b1) from d ell / dH, with W2/b2 zero (contrastive uses hidden only)."""
    dW1 = X.T @ dH
    db1 = dH.sum(0)
    return {
        "W1": dW1.astype(np.float32),
        "b1": db1.astype(np.float32),
        "W2": np.zeros_like(p["W2"]),
        "b2": np.zeros_like(p["b2"]),
    }
