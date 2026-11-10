"""
Reproduce Figure 8 of the QUANT paper under the paper's own protocol.

Our earlier weighting experiment found none of the depth-by-quantile pattern
Fig. 8 reports. That experiment used a different protocol from the paper's, so
it could not distinguish "the pattern is not real" from "the pattern depends on
the protocol". This script removes that ambiguity by matching Sect. 4.2 exactly.

What the paper specifies (Sect. 4.2):

    Following Herrmann et al. (2023), in an effort to avoid the peculiarities
    of the smallest datasets and the original training/test splits, we conduct
    the sensitivity analysis using stratified 5-fold cross-validation (such
    that, for each fold, 80% of the data is used for training, and 20% of the
    data is used for validation), using a random sample of 50 of the datasets
    from the subset of 112 datasets from the UCR archive [...] In particular,
    from the subset of 112 datasets, we randomly sample 50 of the 100 datasets
    where there are at least 100 training examples on an 80/20 split, and at
    least 5 examples of each class.

So:
  * train and test are POOLED, then split by stratified 5-fold CV
  * eligibility: >= 100 training cases at 80/20, i.e. >= 125 cases total,
    and >= 5 cases of every class
  * 50 are sampled at random from the eligible pool

The paper does not list which 50 it sampled, so that part cannot be matched.
This script reports the eligible pool size (the paper says 100, which is a
check on the filter), samples with a recorded seed, and can run the whole pool
with --all to remove the sampling question altogether.

Figure 8 sweeps depth 4-8 against m, m/2, m/4, m/8, m/16 quantiles per
interval, with the default classifier throughout.

  python reproduce_fig8.py --list        show the eligible pool and stop
  python reproduce_fig8.py --smoke       6 datasets, depths 5-7, divs 4/8/16
  python reproduce_fig8.py               the paper's 50
  python reproduce_fig8.py --all         every eligible dataset

Appends to fig8_results.csv after each dataset; safe to interrupt and resume.

A NOTE ON DEPTH CAPPING, which matters for reading any depth sweep:
quant.py caps depth internally at min(depth, int(log2(L)) + 1) for each
representation of length L. Short series therefore cannot reach the deeper
levels, and requesting depth 8 on them yields exactly the features of depth 7.
Including such datasets dilutes any depth effect. This script detects it by
comparing feature counts between consecutive depths and reports results both
with and without the affected datasets.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG ----

ROOT = "D:/AAI/TimeSeries/QuantProject"
QUANT_CODE = f"{ROOT}/Quant/quant-main/quant-main/code"
DATA_DIR = f"{ROOT}/data/Univariate2018_ts/Univariate_ts"
OUT = f"{ROOT}/fig8_results.csv"

DEPTHS = [4, 5, 6, 7, 8]            # Fig. 8 x-axis
DIVS = [1, 2, 4, 8, 16]             # m, m/2, m/4, m/8, m/16 - the five lines

N_FOLDS = 5
N_SAMPLE = 50                       # the paper's sample size
SAMPLE_SEED = 0                     # recorded, since the paper's 50 are unknown

MIN_TRAIN_AT_80_20 = 100            # -> at least 125 cases in total
MIN_PER_CLASS = 5

# classifier defaults, Sect. 3.4
N_ESTIMATORS = 200
MAX_FEATURES = 0.1

MAX_FEATURE_COLUMNS = 400_000       # refuse configurations that will not fit

# -----------------------------------------------------------------------------

sys.path.append(QUANT_CODE)
sys.path.append(ROOT)

import torch                                            # noqa: E402
from quant import Quant                                 # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier       # noqa: E402
from sklearn.model_selection import StratifiedKFold     # noqa: E402
from sklearn.preprocessing import LabelEncoder          # noqa: E402
from aeon.datasets import load_from_ts_file, tsc_datasets   # noqa: E402

from quant_provenance import feature_provenance         # noqa: E402


# --------------------------------------------------------------- loading ----

def pooled(name):
    """Train and test concatenated, as the paper's CV protocol requires."""
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, y


def eligibility():
    """Apply the paper's filter to the 112 and report who qualifies and why."""
    names = sorted(tsc_datasets.univariate_equal_length)
    rows = []

    for name in names:
        if not os.path.isdir(os.path.join(DATA_DIR, name)):
            continue
        try:
            X, y = pooled(name)
        except Exception as exc:
            rows.append({"dataset": name, "eligible": False,
                         "reason": f"load failed: {exc}"})
            continue

        n_cases = len(y)
        train_at_80 = int(np.floor(n_cases * 0.8))
        _, counts = np.unique(y, return_counts=True)
        min_class = int(counts.min())

        reasons = []
        if train_at_80 < MIN_TRAIN_AT_80_20:
            reasons.append(f"only {train_at_80} training cases at 80/20")
        if min_class < MIN_PER_CLASS:
            reasons.append(f"smallest class has {min_class}")

        rows.append({
            "dataset": name,
            "n_cases": n_cases,
            "length": X.shape[-1],
            "n_classes": len(counts),
            "train_at_80": train_at_80,
            "min_class": min_class,
            "eligible": not reasons,
            "reason": "; ".join(reasons),
        })

    return pd.DataFrame(rows)


def depth_reach(series_length, depths, div):
    """
    Which requested depths actually add features for this series length.

    quant.py caps depth per representation at min(depth, int(log2(L)) + 1),
    so on short series a deeper request can produce an identical transform.
    Compare feature counts between consecutive depths to detect it.
    """
    counts = {}
    for d in depths:
        try:
            counts[d] = len(feature_provenance(series_length, depth=d, div=div))
        except ValueError:
            counts[d] = None

    saturated_from = None
    ordered = sorted(d for d in counts if counts[d] is not None)
    for prev, nxt in zip(ordered, ordered[1:]):
        if counts[nxt] == counts[prev] and saturated_from is None:
            saturated_from = prev
    return counts, saturated_from


# ------------------------------------------------------------------- run ----

def run_dataset(name, depths, divs, n_folds, n_jobs, seed=0):
    X, y = pooled(name)
    y = LabelEncoder().fit_transform(y)
    series_length = X.shape[-1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y))

    rows = []
    for div in divs:
        counts, saturated_from = depth_reach(series_length, depths, div)

        for depth in depths:
            n_expected = counts.get(depth)
            if n_expected is None:
                continue
            if n_expected > MAX_FEATURE_COLUMNS:
                rows.append({
                    "dataset": name, "depth": depth, "div": div,
                    "fold": -1, "accuracy": np.nan,
                    "n_features": n_expected, "length": series_length,
                    "n_cases": len(y),
                    "saturated_from": saturated_from,
                    "note": f"skipped: {n_expected} columns",
                })
                continue

            for fold, (tr, te) in enumerate(folds):
                t0 = time.perf_counter()
                try:
                    q = Quant(depth=depth, div=div)
                    Ztr = q.fit_transform(
                        torch.tensor(X[tr], dtype=torch.float32), y[tr]).numpy()
                    Zte = q.transform(
                        torch.tensor(X[te], dtype=torch.float32)).numpy()

                    clf = ExtraTreesClassifier(
                        n_estimators=N_ESTIMATORS,
                        max_features=MAX_FEATURES,
                        criterion="entropy",
                        n_jobs=n_jobs,
                        random_state=seed,
                    )
                    clf.fit(Ztr, y[tr])
                    acc = float(clf.score(Zte, y[te]))
                    n_features, note = Ztr.shape[1], ""
                except Exception as exc:
                    acc, n_features = np.nan, n_expected
                    note = f"{type(exc).__name__}: {exc}"

                rows.append({
                    "dataset": name, "depth": depth, "div": div,
                    "fold": fold, "accuracy": acc,
                    "seconds": time.perf_counter() - t0,
                    "n_features": n_features, "length": series_length,
                    "n_cases": len(y),
                    "saturated_from": saturated_from,
                    "note": note,
                })

    return rows


# --------------------------------------------------------------- summary ----

def summarise(path):
    df = pd.read_csv(path).drop_duplicates(
        ["dataset", "depth", "div", "fold"])
    df = df[df["fold"] >= 0]

    per_dataset = (df.groupby(["dataset", "div", "depth"])["accuracy"]
                   .mean().reset_index())

    # which datasets saturate before the deepest level requested?
    sat = (df.groupby("dataset")["saturated_from"].first())
    clean = sat[sat.isna()].index
    print(f"\n{df['dataset'].nunique()} datasets, "
          f"{len(clean)} of which reach every requested depth\n")

    def fig8(frame, title):
        print(title)
        print("=" * 62)
        table = frame.pivot_table(index="depth", columns="div",
                                  values="accuracy", aggfunc="mean")
        table.columns = [f"m/{c}" if c > 1 else "m" for c in table.columns]
        print(table.round(4).to_string())

        depths = sorted(frame["depth"].unique())
        if len(depths) > 1:
            print("\n  peak depth per line, and change from depth 6 onward:")
            for col in table.columns:
                series = table[col].dropna()
                peak = series.idxmax()
                tail = (series.loc[depths[-1]] - series.loc[6]
                        if 6 in series.index and depths[-1] in series.index
                        else np.nan)
                print(f"    {col:<6} peak at depth {peak}"
                      f"   depth 6 -> {depths[-1]}: {tail:+.4f}")

    fig8(per_dataset, "mean accuracy (all datasets)")

    sub = per_dataset[per_dataset["dataset"].isin(clean)]
    if len(clean) and len(clean) < df["dataset"].nunique():
        print()
        fig8(sub, f"mean accuracy ({len(clean)} datasets that reach every depth)")

    print("\n\nwhat the paper reports (Fig. 8):")
    print("  - accuracy rises with depth to about 6, then falls, but only")
    print("    for small numbers of quantiles per interval")
    print("  - m/4 and above are 'relatively immune' to the decline")
    print("  - the whole accuracy range across the sweep is ~0.910 to 0.917")

    print("\n\ntotal compute by configuration (minutes)")
    print("-" * 62)
    t = df.pivot_table(index="depth", columns="div",
                       values="seconds", aggfunc="sum") / 60
    t.columns = [f"m/{c}" if c > 1 else "m" for c in t.columns]
    print(t.round(1).to_string())

    bad = df[df["note"].notna() & (df["note"] != "")]
    if len(bad):
        print(f"\n{len(bad)} problem rows:")
        print(bad.groupby("note").size().head(10).to_string())


# ------------------------------------------------------------------ main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="show the eligible pool and exit")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="run every eligible dataset, not a sample of 50")
    p.add_argument("--sample", type=int, default=N_SAMPLE)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="use 1 to make timings comparable with the paper")
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args()

    if args.summarise_only:
        summarise(OUT)
        return

    elig = eligibility()
    pool = elig[elig["eligible"]].sort_values("n_cases")

    print(f"{len(elig)} datasets checked, {len(pool)} eligible")
    print(f"  (the paper reports 100 eligible; a different number means the "
          f"filter is being read differently)\n")

    if args.list:
        pd.set_option("display.max_rows", 200)
        print(pool[["dataset", "n_cases", "length", "n_classes",
                    "train_at_80", "min_class"]].to_string(index=False))
        print("\nexcluded:")
        print(elig[~elig["eligible"]][["dataset", "reason"]]
              .to_string(index=False))
        return

    if args.smoke:
        names = list(pool["dataset"].head(6))
        depths, divs, n_folds = [5, 6, 7], [4, 8, 16], 3
    else:
        depths, divs, n_folds = DEPTHS, DIVS, N_FOLDS
        if args.all:
            names = list(pool["dataset"])
        else:
            rng = np.random.default_rng(args.seed)
            take = min(args.sample, len(pool))
            names = sorted(rng.choice(pool["dataset"].to_numpy(),
                                      size=take, replace=False))
            print(f"sampled {take} datasets with seed {args.seed}\n")

    # how many will saturate before the deepest level?
    lengths = elig.set_index("dataset")["length"]
    saturating = [n for n in names
                  if depth_reach(int(lengths[n]), depths, 4)[1] is not None]
    if saturating:
        print(f"{len(saturating)} of {len(names)} datasets are too short to "
              f"reach depth {max(depths)};")
        print("  for those, the deeper levels give an identical transform.")
        print("  Results are reported with and without them.\n")

    done = set()
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        counts = prior.groupby("dataset")["fold"].nunique()
        done = set(counts[counts >= n_folds].index)
        if done:
            print(f"resuming: {len(done)} datasets already complete\n")

    todo = [n for n in names if n not in done]
    total = len(todo) * len(depths) * len(divs) * n_folds
    print(f"{len(todo)} datasets x {len(depths)} depths x {len(divs)} divs "
          f"x {n_folds} folds = {total} fits\n")

    t_start = time.perf_counter()
    for i, name in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            rows = run_dataset(name, depths, divs, n_folds, args.n_jobs,
                               seed=args.seed)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {name:<28} FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue

        pd.DataFrame(rows).to_csv(
            OUT, mode="a", header=not os.path.exists(OUT), index=False)

        got = pd.DataFrame(rows)
        got = got[got["fold"] >= 0]
        acc = got[(got["depth"] == 6) & (got["div"] == 4)]["accuracy"].mean()
        print(f"[{i}/{len(todo)}] {name:<28} d6 m/4: {acc:.4f}   "
              f"{time.perf_counter() - t0:7.1f}s")

    print(f"\ntotal wall clock: {(time.perf_counter() - t_start) / 60:.1f} min")
    summarise(OUT)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
