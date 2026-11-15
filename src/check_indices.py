"""
Do the official resample indices fit the data we have?

run.py reads the tsml/Java resample index files for resamples 1-29. Those
files were generated against particular versions of each dataset. If a dataset
was later revised, its index files could point at cases that no longer exist,
or produce splits of the wrong size, and nothing in run.py would notice.

For every dataset in results/ucr142_local.txt and every resample 1-29, this
checks that the TRAIN and TEST index files

  - exist,
  - have exactly as many entries as the shipped train and test splits,
  - stay within the pooled case count (0 .. n_train + n_test - 1),
  - contain no index twice, across TRAIN and TEST together.

Resample 0 is the original split and uses no index files, so it is skipped.

Expects the UCR archive and the Java resample indices in data/ (see README).
Set QUANT_ROOT if they live outside the repo.

  python src/check_indices.py
"""

import os
from pathlib import Path

import numpy as np
from aeon.datasets import load_from_ts_file

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data/Univariate2018_ts/Univariate_ts"
INDEX_DIR = ROOT / "data/JavaResampleIndices/JavaResampleIndices"
NAMES = ROOT / "results/ucr142_local.txt"

RESAMPLES = range(1, 30)


def main():
    names = [n.strip() for n in open(NAMES) if n.strip()]

    checked = missing = 0
    problems = []

    for name in names:
        base = DATA_DIR / name / name
        _, y_train = load_from_ts_file(f"{base}_TRAIN.ts")
        _, y_test = load_from_ts_file(f"{base}_TEST.ts")
        n_train, n_test = len(y_train), len(y_test)
        total = n_train + n_test

        for r in RESAMPLES:
            p_train = INDEX_DIR / name / f"resample{r}Indices_TRAIN.txt"
            p_test = INDEX_DIR / name / f"resample{r}Indices_TEST.txt"
            if not (p_train.exists() and p_test.exists()):
                missing += 1
                problems.append((name, r, "index file missing"))
                continue

            tr = np.loadtxt(p_train, dtype=int, ndmin=1).ravel()
            te = np.loadtxt(p_test, dtype=int, ndmin=1).ravel()
            both = np.concatenate([tr, te])
            checked += 1

            if len(tr) != n_train or len(te) != n_test:
                problems.append((name, r, f"sizes {len(tr)}/{len(te)}, "
                                          f"expected {n_train}/{n_test}"))
            elif both.min() < 0 or both.max() >= total:
                problems.append((name, r, f"index out of range 0..{total - 1}"))
            elif len(np.unique(both)) != len(both):
                problems.append((name, r, "duplicate index"))

    for name, r, why in problems[:20]:
        print(f"  {name:<30} resample {r:>2}: {why}")
    if len(problems) > 20:
        print(f"  ... and {len(problems) - 20} more")

    print(f"\n{len(names)} datasets, resamples {RESAMPLES.start}-{RESAMPLES.stop - 1}")
    print(f"{checked} resample pairs (TRAIN + TEST) checked, "
          f"{missing} missing, {len(problems) - missing} problems")
    if not problems:
        print("every index file fits its dataset")


if __name__ == "__main__":
    main()