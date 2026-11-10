"""Reproduce the FedHBT tables.

  python run.py --dataset covtype
  python run.py --dataset shuttle --family fed
  python run.py --dataset drybean --family class
  python run.py --dataset covtype --quick
  python run.py --dataset covtype --plot-only

--family class  class-turn methods + FedAvg (Stratify/pp-CFL routing, FedHBT, Oracle)
--family fed    FedProx, SCAFFOLD, FedDyn, MOON (same clock as FedAvg)
--family all    both (default)
"""

import argparse
import csv
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset, make_federation, class_counts
from fl import train_class_turns, train_fed_family, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
PRIVATE = ["counts_dp", "bits_rr", "fedhbt_dp"]
FED_FAMILY = ["fedavg", "fedprox", "scaffold", "feddyn", "moon"]


def _paths(dataset, rare_per_holder, quick):
    tag = f"{dataset}_rare{rare_per_holder}" + ("_quick" if quick else "")
    os.makedirs(OUT, exist_ok=True)
    return (
        os.path.join(OUT, f"{tag}.csv"),
        os.path.join(OUT, f"{tag}_fed.csv"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="covtype", choices=["covtype", "shuttle", "drybean"])
    ap.add_argument("--rare-per-holder", type=int, default=30)
    ap.add_argument("--family", default="all", choices=["all", "class", "fed"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    class_csv, fed_csv = _paths(args.dataset, args.rare_per_holder, args.quick)
    if args.plot_only:
        plot(class_csv, args.dataset, args.rare_per_holder)
        return

    seeds = [0] if args.quick else [0, 1, 2]
    eps_grid = [0.5, 2.0, 8.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    steps = 300 if args.quick else 1500
    rounds = 40 if args.quick else 150

    X, y, rare = load_dataset(args.dataset)
    C, d = int(y.max()) + 1, X.shape[1]
    do_class = args.family in ("all", "class")
    do_fed = args.family in ("all", "fed")
    class_rows, fed_rows = [], []

    for seed in seeds:
        clients, Xte, yte, Xnat, ynat, _ = make_federation(
            X, y, rare, seed=seed, rare_per_holder=args.rare_per_holder)
        counts = class_counts(clients, C)
        t0 = time.time()

        def record(bucket, method, eps, p, lost=None, uploads=np.nan):
            bal = evaluate(p, Xte, yte, C)
            nat = evaluate(p, Xnat, ynat, C)
            row = dict(seed=seed, method=method, eps=eps,
                       overall_acc=nat["acc"], rare_acc=bal["recall"][rare],
                       balanced_acc=bal["bal_acc"],
                       rare_lost=(lost[rare] if lost is not None else np.nan),
                       uploads_per_step=uploads)
            bucket.append(row)
            print(f"seed={seed} {method:15s} eps={eps!s:5s} overall={nat['acc']:.3f} "
                  f"rare={bal['recall'][rare]:.3f} balanced={bal['bal_acc']:.3f}"
                  + (f" rare_turns_lost={lost[rare]:.2f}" if lost is not None else ""),
                  flush=True)

        if do_fed:
            for method in FED_FAMILY:
                p = train_fed_family(clients, C, d, method=method, rounds=rounds, seed=seed)
                record(fed_rows, method, "-", p)
        if do_class:
            for routing in ["oracle", "fedhbt"]:
                p, lost, uploads = train_class_turns(
                    clients, counts, C, d, routing, steps=steps, seed=seed)
                record(class_rows, routing, "-", p, lost, uploads)
            for method in PRIVATE:
                routing = "fedhbt" if method == "fedhbt_dp" else method
                for eps in eps_grid:
                    p, lost, uploads = train_class_turns(
                        clients, counts, C, d, routing, eps=eps, steps=steps, seed=seed)
                    record(class_rows, method, eps, p, lost, uploads)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)

    if class_rows:
        _write(class_csv, class_rows)
        plot(class_csv, args.dataset, args.rare_per_holder)
    if fed_rows:
        _write(fed_csv, fed_rows)
        print(f"saved {fed_csv}")


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(csv_path, dataset, rare_per_holder):
    if not os.path.exists(csv_path):
        print(f"missing {csv_path}")
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    eps_vals = {float(r["eps"]) for r in rows if r["eps"] != "-"}
    if not eps_vals:
        return
    eps_grid = sorted(eps_vals)

    def mean_of(method, key, eps="-"):
        vals = [float(r[key]) for r in rows
                if r["method"] == method and r["eps"] == (eps if eps == "-" else str(eps))]
        return np.mean(vals) if vals else np.nan

    keys = ["overall_acc", "rare_acc", "balanced_acc"]
    titles = [f"{dataset}: overall accuracy",
              f"{dataset}: rare-class accuracy ({rare_per_holder} x 2 holders)",
              f"{dataset}: balanced accuracy"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, key in enumerate(keys):
        plot_name = {
            "counts_dp": "Stratify",
            "bits_rr": "pp-CFL",
            "fedhbt_dp": "FedHBT",
            "oracle": "Oracle",
            "fedhbt": "FedHBT (no noise)",
            "fedavg": "FedAvg",
        }
        for method, mk in [("counts_dp", "o-"), ("bits_rr", "s-"), ("fedhbt_dp", "^-")]:
            ax[j].semilogx(
                eps_grid, [mean_of(method, key, e) for e in eps_grid], mk,
                label=plot_name[method],
            )
        for ref, ls in [("oracle", "k--"), ("fedhbt", "g-"), ("fedavg", "r:")]:
            y = mean_of(ref, key)
            if np.isfinite(y):
                ax[j].axhline(y, ls=ls[1:], c=ls[0], label=plot_name[ref])
        ax[j].set_xticks(eps_grid)
        ax[j].set_xticklabels([f"{e:g}" for e in eps_grid])
        ax[j].minorticks_off()
        ax[j].set(xlabel=r"$\varepsilon$", ylabel="accuracy", title=titles[j])
        ax[j].legend(fontsize=8)
    fig.tight_layout()
    png = csv_path.replace(".csv", ".png")
    fig.savefig(png, dpi=150)
    print(f"saved {csv_path}\nsaved {png}")


if __name__ == "__main__":
    main()
