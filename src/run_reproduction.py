"""
QUANT reproduction harness.

An independent reimplementation of the experiment in the QUANT repo's run.py.
Runs QUANT over the 112 equal-length univariate UCR datasets across 30
resamples and compares the per-dataset means against the accuracies published
in the repo (results/accuracy_resamples_ucr142.csv).

Resample 0 is the original train/test split. Resamples 1-29 are stratified
reshuffles of the pooled train+test data.

One deliberate difference from the published protocol: resamples 1-29 use
aeon's stratified_resample_data, not the precomputed tsml index files the
published results use. The cost of that substitution is measured separately
(see compare_three_way.py).

  python run_reproduction.py                 # run everything (resumes if interrupted)
  python run_reproduction.py --smoke         # 8 small datasets, 5 resamples
  python run_reproduction.py --compare-only  # just rebuild the comparison table

Results are appended to RAW_OUT after every dataset, so the run can be stopped
and restarted without losing work. Edit the paths in the CONFIG block below.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG ----

ROOT = os.environ.get("QUANT_ROOT", "D:/AAI/TimeSeries/QuantProject")

QUANT_CODE  = f"{ROOT}/Quant/quant-main/quant-main/code"
DATA_DIR    = f"{ROOT}/data/Univariate2018_ts/Univariate_ts"
RESULTS_CSV = f"{ROOT}/Quant/quant-main/quant-main/results/accuracy_resamples_ucr142.csv"
RAW_OUT     = f"{ROOT}/results/raw_results.csv"
SUMMARY_OUT = f"{ROOT}/results/comparison.csv"

N_RESAMPLES = 30
# classifier seed = SEED_OFFSET + resample index.
# Their run.py sets no random_state, so their published results carry seed
# noise and ours do not. This is a deliberate deviation.
SEED_OFFSET = 0

# -----------------------------------------------------------------------------

sys.path.append(QUANT_CODE)

import torch                                        # noqa: E402
from quant import Quant                             # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier   # noqa: E402
from aeon.datasets import load_from_ts_file         # noqa: E402

try:
    from aeon.benchmarking.resampling import stratified_resample_data
except ImportError:                                  # older aeon
    from aeon.datasets.resampling import stratified_resample_data


# ------------------------------------------------------------- datasets ----

def dataset_list():
    """The 112 equal-length, no-missing-values univariate UCR datasets."""
    from aeon.datasets import tsc_datasets

    for attr in ("univariate_equal_length", "univariate_equal_length_datasets"):
        if hasattr(tsc_datasets, attr):
            names = sorted(getattr(tsc_datasets, attr))
            break
    else:
        raise SystemExit(
            "Could not find the 112-dataset list in aeon.datasets.tsc_datasets. "
            f"Available: {[a for a in dir(tsc_datasets) if not a.startswith('_')]}"
        )

    present = [n for n in names if os.path.isdir(os.path.join(DATA_DIR, n))]
    missing = set(names) - set(present)
    if missing:
        print(f"warning: {len(missing)} of {len(names)} datasets not on disk "
              f"(e.g. {sorted(missing)[:3]})")
    return present


def load_dataset(name):
    base = f"{DATA_DIR}/{name}/{name}"
    X_train, y_train = load_from_ts_file(f"{base}_TRAIN.ts")
    X_test,  y_test  = load_from_ts_file(f"{base}_TEST.ts")
    return (np.asarray(X_train), np.asarray(y_train),
            np.asarray(X_test),  np.asarray(y_test))


# ------------------------------------------------------------------ run ----

def run_one(X_train, y_train, X_test, y_test, seed):
    """One QUANT fit/predict. Returns accuracy, elapsed seconds, feature count."""
    # timed end to end: the returned seconds cover transform and classifier
    # together, not the fit alone
    start = time.perf_counter()

    # defaults are the paper's: depth 6, m/4 quantiles per interval
    transform = Quant()
    # fit on train only: fit_transform derives the interval structure from the
    # training data, transform applies that same structure to test. No test
    # data influences how features are built.
    Z_train = transform.fit_transform(
        torch.tensor(X_train, dtype=torch.float32), y_train)
    Z_test = transform.transform(
        torch.tensor(X_test, dtype=torch.float32))

    # Sect. 3.4: 200 trees, 10% of features per split, entropy
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


def run_dataset(name, n_resamples):
    """All resamples for one dataset. Resample 0 is the original split."""
    X_train, y_train, X_test, y_test = load_dataset(name)
    rows = []

    for i in range(n_resamples):
        if i == 0:
            Xtr, ytr, Xte, yte = X_train, y_train, X_test, y_test
        else:
            Xtr, ytr, Xte, yte = stratified_resample_data(
                X_train, y_train, X_test, y_test, random_state=i)

        acc, secs, n_feat = run_one(Xtr, ytr, Xte, yte, SEED_OFFSET + i)
        rows.append({
            "dataset": name,
            "resample": i,
            "accuracy": acc,
            "seconds": secs,
            "n_features": n_feat,
            "n_train": len(ytr),
            "n_test": len(yte),
            "length": X_train.shape[-1],
        })

    return rows


def append_rows(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# -------------------------------------------------------------- compare ----

def load_published(path):
    df = pd.read_csv(path)
    first = df.columns[0]
    out = df.set_index(first)
    out.columns = range(len(out.columns))
    return out


def build_comparison(raw_path, published_path, expected_resamples):
    raw = pd.read_csv(raw_path).drop_duplicates(["dataset", "resample"])
    published = load_published(published_path)

    rows = []
    for name, group in raw.groupby("dataset"):
        if name not in published.index:
            continue

        theirs_all = published.loc[name].astype(float)
        resamples_done = sorted(group["resample"].unique())

        # Average their results over the same NUMBER of resamples we ran, not the
        # same splits: our resamples 1-29 come from aeon, theirs from tsml index
        # files, so index i does not refer to the same split on both sides. Only the
        # means are comparable. Fold 0 is the exception - both use the original split.
        theirs = theirs_all.iloc[resamples_done].mean()
        mine = group["accuracy"].mean()

        rows.append({
            "dataset": name,
            "n_resamples": len(resamples_done),
            "published_mean": theirs,
            "my_mean": mine,
            "diff": mine - theirs,
            "my_std": group["accuracy"].std(),
            "fold0_published": theirs_all.iloc[0],
            "fold0_mine": group.loc[group["resample"] == 0, "accuracy"].mean(),
            "mean_seconds": group["seconds"].mean(),
            "length": group["length"].iloc[0],
            "n_features": group["n_features"].iloc[0],
        })

    # sorted so head(5) and tail(5) in report() give the extremes
    table = pd.DataFrame(rows).sort_values("diff")
    table["fold0_diff"] = table["fold0_mine"] - table["fold0_published"]
    return table


def report(table, expected_resamples):
    complete = table[table.n_resamples == expected_resamples]
    print(f"\n{len(table)} datasets, {len(complete)} with all "
          f"{expected_resamples} resamples\n")

    print("most negative differences (we do worse than published):")
    print(table.head(5)[["dataset", "published_mean", "my_mean", "diff"]]
          .to_string(index=False))
    print("\nmost positive differences:")
    print(table.tail(5)[["dataset", "published_mean", "my_mean", "diff"]]
          .to_string(index=False))

    print(f"\n{'':<24}{'published':>12}{'mine':>12}{'diff':>12}")
    print(f"{'mean accuracy':<24}{table.published_mean.mean():>12.4f}"
          f"{table.my_mean.mean():>12.4f}"
          f"{table.my_mean.mean() - table.published_mean.mean():>+12.4f}")

    print(f"\nmean |diff| over all resamples : {table['diff'].abs().mean():.4f}")
    # fold 0 uses the original split on both sides, so its difference excludes
    # the resampling scheme; comparing the two figures isolates its effect
    print(f"mean |diff| on fold 0 only     : {table['fold0_diff'].abs().mean():.4f}")
    print(f"datasets where we do better    : {(table['diff'] > 0).sum()}")
    print(f"datasets where we do worse     : {(table['diff'] < 0).sum()}")
    # transform + classifier, multi-core; not comparable with the paper's
    # single-core figures
    print(f"total compute                  : {table.mean_seconds.sum() * expected_resamples / 60:.1f} min")

    print("\nNote: if |diff| on fold 0 is small but |diff| over all resamples")
    print("is larger and one-sided, the resampling scheme is the likely cause,")
    print("not the transform or the classifier. In our 112-dataset run the")
    print("reverse held, so resampling was not the dominant factor.")


# ----------------------------------------------------------------- main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="8 small datasets, 5 resamples")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()

    # --smoke overrides --resamples
    n_resamples = 5 if args.smoke else args.resamples

    if not args.compare_only:
        names = dataset_list()
        if args.smoke:
            names = ["Coffee", "GunPoint", "ItalyPowerDemand", "ECG200",
                     "SyntheticControl", "Beef", "BirdChicken", "Chinatown"]
            names = [n for n in names if os.path.isdir(os.path.join(DATA_DIR, n))]

        # resume: a dataset counts as done only if it has at least n_resamples
        # distinct resamples. Changing --resamples between runs will therefore
        # re-run datasets and append duplicates - delete RAW_OUT instead.
        done = set()
        if os.path.exists(RAW_OUT):
            prior = pd.read_csv(RAW_OUT)
            counts = prior.groupby("dataset")["resample"].nunique()
            done = set(counts[counts >= n_resamples].index)
            if done:
                print(f"resuming: {len(done)} datasets already complete")

        todo = [n for n in names if n not in done]
        print(f"running {len(todo)} datasets x {n_resamples} resamples\n")

        t0 = time.perf_counter()
        for i, name in enumerate(todo, 1):
            start = time.perf_counter()
            try:
                rows = run_dataset(name, n_resamples)
            except Exception as exc:           # keep going; note the failure
                print(f"[{i}/{len(todo)}] {name:<28} FAILED: {exc}")
                continue

            append_rows(rows, RAW_OUT)
            mean_acc = np.mean([r["accuracy"] for r in rows])
            print(f"[{i}/{len(todo)}] {name:<28} acc {mean_acc:.4f}   "
                  f"{time.perf_counter() - start:6.1f}s")

        print(f"\ntotal wall clock: {(time.perf_counter() - t0) / 60:.1f} min")

    if not os.path.exists(RAW_OUT):
        raise SystemExit("no results to compare")

    table = build_comparison(RAW_OUT, RESULTS_CSV, n_resamples)
    table.to_csv(SUMMARY_OUT, index=False)
    report(table, n_resamples)
    print(f"\nwritten to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
