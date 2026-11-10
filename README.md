# FedSelfSelect

Identification-free rare-class learning in federated classification.

Paper: `FedSelfSelect.pdf`. Speaker script: `FedSelfSelect_script.pdf`.

Working title is close to FedSelect (CVPR 2024). Candidate names: **FedHBT** (Holder-Blind Turn), **FedNCT** (Nameless Class Turn), **FedBCT** (Broadcast Class Turn).

## Setup

```bash
pip install numpy scikit-learn matplotlib ucimlrepo liac-arff
```

Covertype and Shuttle download on first run. Dry Bean uses `data/drybean.npz` if present.

## Reproduce tables

```bash
python run.py --dataset covtype
python run.py --dataset shuttle
python run.py --dataset drybean
```

`--family class` runs Stratify / pp-CFL / FedSelfSelect / Oracle (`T=1500`).  
`--family fed` runs FedAvg / FedProx / SCAFFOLD / FedDyn / MOON (`150×10×10`).  
`--family all` is the default.

Paper numbers are already in `results/{covtype,shuttle,drybean}_rare30.csv` and `*_fed.csv`. Use `--quick` only as a smoke test; it overwrites CSVs with a shorter run.
