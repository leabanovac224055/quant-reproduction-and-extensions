"""
SUPERSEDED. Does the v1 feature-selection result hold on small datasets?

The v1 experiment used 24 large datasets, chosen because the classifier
dominates runtime there. Most of the UCR archive is small, where the transform
can be over half the runtime and selection cannot help with any of it. This
reran the same protocol on 20 small ones.

Not reported: choosing the dataset list after seeing a result is selection on
the evaluation set, which is what prompted the rebuild. The reported
experiment, run_reduction_v2.py, forms its small, large and random groups by a
rule fixed before any run and reports all three.

  python src/superseded/run_reduction_small.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("QUANT_ROOT",
                           Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(ROOT / "Quant/quant-main/quant-main/code"))
sys.path.append(str(ROOT / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import run_reduction as R

# small datasets from the paper's eligible pool, spanning 128 to 621 cases
R.DATASETS = [
    "Herring", "Lightning7", "HouseTwenty", "ToeSegmentation2", "UMD", "BME",
    "GunPoint", "Trace", "ShapeletSim", "ECG200", "Plane", "ArrowHead",
    "Ham", "Worms", "ToeSegmentation1", "DiatomSizeReduction",
    "Fish", "PowerCons", "Chinatown", "OSULeaf",
]
R.OUT = str(ROOT / "results/superseded/reduction_results_small.csv")

if __name__ == "__main__":
    names = [n for n in R.DATASETS
             if os.path.isdir(os.path.join(R.DATA_DIR, n))]

    done = set()
    if os.path.exists(R.OUT):
        prior = pd.read_csv(R.OUT)
        counts = prior.groupby("dataset")["fold"].nunique()
        done = set(counts[counts >= 5].index)

    for i, name in enumerate([n for n in names if n not in done], 1):
        t0 = time.perf_counter()
        rows = R.run_dataset(name, 5)
        pd.DataFrame(rows).to_csv(
            R.OUT, mode="a", header=not os.path.exists(R.OUT), index=False)
        acc = pd.DataFrame(rows).query("strategy == 'full'")["accuracy"].mean()
        print(f"[{i}] {name:<24} full {acc:.4f}   "
              f"{time.perf_counter() - t0:6.1f}s")

    R.summarise(R.OUT)