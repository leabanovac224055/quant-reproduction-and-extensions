"""
QUANT reproduction: fold-0 sanity check.

Runs QUANT on a handful of UCR datasets using the default train/test split
(= resample 0) and compares the accuracies against the published results
shipped in the QUANT repo. Resample 0 uses no index files, so this isolates
the transform and classifier from the resampling scheme.

Three classifier seeds per dataset, to show the run-to-run spread that
run.py's missing random_state leaves in the published numbers.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Set QUANT_ROOT if they live outside the repo.

  python src/check_fold0.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

QUANT_CODE  = str(ROOT / "Quant/quant-main/quant-main/code")
DATA_DIR    = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
RESULTS_CSV = str(ROOT / "Quant/quant-main/quant-main/results/accuracy_resamples_ucr142.csv")
OUT         = str(ROOT / "results/fold0_comparison.csv")

DATASETS = [
    "GunPoint",
    "Coffee",
    "ItalyPowerDemand",
    "ECG200",
    "SyntheticControl",
    "Beef",
]

SEEDS = [0, 1, 2]   # repeat the classifier a few times to gauge run-to-run spread

# -----------------------------------------------------------------------------

sys.path.append(QUANT_CODE)

import torch                                        # noqa: E402
from quant import Quant                             # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier   # noqa: E402
from aeon.datasets import load_from_ts_file         # noqa: E402


def load_split(name, split):
    path = f"{DATA_DIR}/{name}/{name}_{split}.ts"
    X, y = load_from_ts_file(path)
    return torch.tensor(np.asarray(X), dtype=torch.float32), np.asarray(y)


def run_quant(name, seed):
    """Fit QUANT + ExtraTrees on the default split. Returns (accuracy, seconds)."""
    X_train, y_train = load_split(name, "TRAIN")
    X_test,  y_test  = load_split(name, "TEST")

    start = time.perf_counter()

    transform = Quant()
    Z_train = transform.fit_transform(X_train, y_train)
    Z_test = transform.transform(X_test)

    clf = ExtraTreesClassifier(
        n_estimators=200,
        max_features=0.1,
        criterion="entropy",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(Z_train, y_train)
    accuracy = clf.score(Z_test, y_test)

    return accuracy, time.perf_counter() - start, Z_train.shape[1]


def load_published(path):
    """
    Load the repo's per-resample accuracies.

    The exact layout isn't documented, so this is deliberately defensive: it
    tries a few plausible shapes and, failing that, dumps the head so you can
    see what you're dealing with.
    """
    df = pd.read_csv(path)

    # Case 1: first column is the dataset name, remaining columns are resamples.
    first = df.columns[0]
    rest = df.columns[1:]
    looks_numeric = all(str(c).strip().lstrip("-").isdigit() for c in rest)

    if looks_numeric and df[first].dtype == object:
        out = df.set_index(first)
        out.columns = [int(str(c).strip()) for c in out.columns]
        return out

    # Case 2: no header row for the resamples, index is unnamed.
    if df[first].dtype == object:
        out = df.set_index(first)
        out.columns = range(len(out.columns))
        return out

    print("Could not infer the CSV layout. First few rows:\n")
    print(df.head())
    raise SystemExit(
        "\nUnrecognised CSV layout; expected a dataset-name column followed "
        "by one column per resample."
    )


def main():
    published = load_published(RESULTS_CSV)
    print(f"Published results: {published.shape[0]} datasets, "
          f"{published.shape[1]} resample columns\n")

    rows = []
    for name in DATASETS:
        if name not in published.index:
            print(f"  ! {name} not found in the published CSV, skipping")
            continue

        accs, times = [], []
        for seed in SEEDS:
            acc, secs, n_features = run_quant(name, seed)
            accs.append(acc)
            times.append(secs)

        theirs = float(published.loc[name].iloc[0])   # resample 0 = default split
        mine = float(np.mean(accs))

        rows.append({
            "dataset": name,
            "published": round(theirs, 4),
            "mine": round(mine, 4),
            "diff": round(mine - theirs, 4),
            "spread": round(float(np.max(accs) - np.min(accs)), 4),
            "features": n_features,
            "secs": round(float(np.mean(times)), 1),
        })

        print(f"  {name:<20} published {theirs:.4f}   mine {mine:.4f}   "
              f"diff {mine - theirs:+.4f}")

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    if len(table):
        print(f"\nmean absolute difference: {table['diff'].abs().mean():.4f}")
        print(f"max absolute difference:  {table['diff'].abs().max():.4f}")
        print(f"mean seed-to-seed spread: {table['spread'].mean():.4f}")

    table.to_csv(OUT, index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()