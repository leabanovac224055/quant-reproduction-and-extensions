"""
Feature selection, with a class-balanced metric.

run_reduction_v2.py records accuracy only, matching the paper. Accuracy can
hide a cost to minority classes: a dataset with 90% of cases in one class gives
0.90 to a model that ignores every other class. If selecting 12.5% of features
preserves accuracy by keeping what identifies the large classes while losing
what catches the rare ones, accuracy alone will not show it.

Same protocol and same datasets as run_reduction_v2.py --group random, so the
accuracy column should reproduce that run. Predictions are kept so macro-F1
and balanced accuracy can be computed alongside, plus the majority-class
baseline for each fold.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Set QUANT_ROOT if they live outside the repo.

  python src/run_reduction_f1.py            the random group
  python src/run_reduction_f1.py --summarise

Appends to results/reduction_f1_results.csv after each dataset; safe to
interrupt and resume.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.append(str(ROOT / "Quant/quant-main/quant-main/code"))
sys.path.append(str(ROOT / "src"))

import torch
from quant import Quant
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             balanced_accuracy_score)
from aeon.datasets import load_from_ts_file, tsc_datasets

from quant_provenance import feature_provenance

DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
OUT = str(ROOT / "results/reduction_f1_results.csv")

DEPTH, DIV = 6, 4
FAST_DEPTH, FAST_DIV = 5, 8
BUDGETS = [0.0625, 0.125, 0.25, 0.50]
N_FOLDS = 5
GROUP_SIZE = 20
SEED = 0

N_ESTIMATORS, MAX_FEATURES = 200, 0.1
PROBE_TREES = 20
N_JOBS = 1                      # as in run_reduction_v2.py

MIN_TRAIN_AT_80_20, MIN_PER_CLASS = 100, 5

# Only the strategies the report compares. The levers are omitted: their
# accuracy behaviour is already established by run_reduction_v2.py, and the
# question here is whether SELECTION costs minority-class performance.
SELECTION = ["importance_sqrt", "balanced_sqrt", "random"]
PROBES = {"importance_sqrt": "sqrt"}


def pooled(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, LabelEncoder().fit_transform(y)


def eligible():
    rows = []
    for name in sorted(tsc_datasets.univariate_equal_length):
        if not os.path.isdir(os.path.join(DATA_DIR, name)):
            continue
        X, y = pooled(name)
        _, counts = np.unique(y, return_counts=True)
        if int(np.floor(len(y) * 0.8)) < MIN_TRAIN_AT_80_20:
            continue
        if counts.min() < MIN_PER_CLASS:
            continue
        rows.append(dict(dataset=name, n_cases=len(y), length=X.shape[-1],
                         workload=len(y) * X.shape[-1]))
    return pd.DataFrame(rows).sort_values("workload").reset_index(drop=True)


def random_group(pool):
    """The same 20 datasets as run_reduction_v2.py --group random."""
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(pool["dataset"].to_numpy(), size=GROUP_SIZE,
                             replace=False))


def forest(n_trees, max_features, seed):
    return ExtraTreesClassifier(n_estimators=n_trees, max_features=max_features,
                                criterion="entropy", n_jobs=N_JOBS,
                                random_state=seed)


def transform(Xtr, Xte, depth, div):
    t0 = time.perf_counter()
    q = Quant(depth=depth, div=div)
    Ztr = q.fit_transform(torch.tensor(Xtr, dtype=torch.float32), None).numpy()
    Zte = q.transform(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return Ztr, Zte, time.perf_counter() - t0


def fit_and_measure(Ztr, ytr, Zte, yte, seed):
    """Keep the predictions, so every metric comes from one fit."""
    t0 = time.perf_counter()
    clf = forest(N_ESTIMATORS, MAX_FEATURES, seed).fit(Ztr, ytr)
    pred = clf.predict(Zte)
    secs = time.perf_counter() - t0
    return dict(
        accuracy=float(accuracy_score(yte, pred)),
        macro_f1=float(f1_score(yte, pred, average="macro", zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(yte, pred)),
    ), secs


def majority_accuracy(ytr, yte):
    """Accuracy of always predicting the most frequent training class."""
    values, counts = np.unique(ytr, return_counts=True)
    return float((yte == values[counts.argmax()]).mean())


def select(strategy, Ztr, ytr, k, prov, rng, seed):
    t0 = time.perf_counter()
    n = Ztr.shape[1]

    if strategy == "random":
        idx = rng.choice(n, size=k, replace=False)
    elif strategy in PROBES:
        probe = forest(PROBE_TREES, PROBES[strategy], seed).fit(Ztr, ytr)
        idx = np.argsort(probe.feature_importances_)[-k:]
    elif strategy == "balanced_sqrt":
        probe = forest(PROBE_TREES, "sqrt", seed).fit(Ztr, ytr)
        imp = probe.feature_importances_
        depths = prov["depth"].to_numpy()[:n]
        levels = np.unique(depths)
        per_level = max(1, k // len(levels))
        picked = []
        for d in levels:
            cols = np.flatnonzero(depths == d)
            picked.append(cols[np.argsort(imp[cols])[-min(per_level, len(cols)):]])
        idx = np.concatenate(picked)
        if len(idx) > k:
            idx = idx[np.argsort(imp[idx])[-k:]]
        elif len(idx) < k:
            rest = np.setdiff1d(np.arange(n), idx)
            idx = np.concatenate([idx, rest[np.argsort(imp[rest])[-(k - len(idx)):]]])
    else:
        raise ValueError(strategy)

    return np.sort(idx), time.perf_counter() - t0


def run_dataset(name):
    X, y = pooled(name)
    length = X.shape[-1]
    prov = feature_provenance(length, depth=DEPTH, div=DIV)
    n_default = len(prov)
    n_classes = len(np.unique(y))
    _, class_counts = np.unique(y, return_counts=True)
    imbalance = float(class_counts.max() / class_counts.min())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        rng = np.random.default_rng(SEED * 100 + fold)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        meta = dict(dataset=name, fold=fold, length=length, n_cases=len(y),
                    n_classes=n_classes, imbalance=imbalance,
                    majority_acc=majority_accuracy(ytr, yte))

        Ztr, Zte, t_tf = transform(Xtr, Xte, DEPTH, DIV)
        m, t_fit = fit_and_measure(Ztr, ytr, Zte, yte, SEED)
        rows.append(dict(**meta, strategy="full", budget=1.0,
                         n_features=Ztr.shape[1], **m,
                         transform_s=t_tf, select_s=0.0, fit_s=t_fit))

        Ftr, Fte, t_tf_f = transform(Xtr, Xte, FAST_DEPTH, FAST_DIV)
        m, t_fit_f = fit_and_measure(Ftr, ytr, Fte, yte, SEED)
        rows.append(dict(**meta, strategy="fast", budget=-1.0,
                         n_features=Ftr.shape[1], **m,
                         transform_s=t_tf_f, select_s=0.0, fit_s=t_fit_f))

        for budget in BUDGETS:
            k = max(2, int(round(n_default * budget)))
            for strategy in SELECTION:
                idx, t_sel = select(strategy, Ztr, ytr, k, prov, rng, SEED)
                m, t_fit_s = fit_and_measure(
                    Ztr[:, idx], ytr, Zte[:, idx], yte, SEED)
                rows.append(dict(**meta, strategy=strategy, budget=budget,
                                 n_features=len(idx), **m,
                                 transform_s=t_tf, select_s=t_sel,
                                 fit_s=t_fit_s))
    return rows


METRICS = ["accuracy", "macro_f1", "balanced_acc"]


def summarise():
    from scipy.stats import wilcoxon

    df = pd.read_csv(OUT).drop_duplicates(
        ["dataset", "fold", "strategy", "budget"])

    per = (df.groupby(["dataset", "strategy", "budget"])[METRICS]
           .mean().reset_index())
    full = per[per.strategy == "full"].set_index("dataset")
    fast = per[per.strategy == "fast"].set_index("dataset")

    info = df.groupby("dataset")[["n_classes", "imbalance", "majority_acc"]].first()

    print(f"\n{df.dataset.nunique()} datasets, {df.fold.nunique()} folds")
    print(f"class counts {info.n_classes.min():.0f} to {info.n_classes.max():.0f}, "
          f"imbalance ratio up to {info.imbalance.max():.1f}:1")
    print(f"majority-class baseline: mean accuracy "
          f"{info.majority_acc.mean():.4f}\n")

    print(f"{'':<18}" + "".join(f"{m:>14}" for m in METRICS))
    for label, frame in [("full", full), ("QUANT-FAST", fast)]:
        print(f"{label:<18}" + "".join(f"{frame[m].mean():>14.4f}"
                                       for m in METRICS))
    print(f"{'majority baseline':<18}{info.majority_acc.mean():>14.4f}")

    print("\n\nchange from the full model, by metric")
    print("=" * 76)
    print(f"  {'strategy':<18}{'budget':>8}" +
          "".join(f"{m:>20}" for m in METRICS))
    for strategy in SELECTION:
        for budget in BUDGETS:
            s = per[(per.strategy == strategy) & (per.budget == budget)]
            if not len(s):
                continue
            s = s.set_index("dataset")
            line = f"  {strategy:<18}{budget:>8.1%}"
            for m in METRICS:
                pair = pd.concat([s[m], full[m]], axis=1).dropna()
                pair.columns = ["a", "b"]
                d = (pair.a - pair.b).mean()
                p = wilcoxon(pair.a, pair.b).pvalue if len(pair) > 5 else np.nan
                line += f"{d:>+12.4f}{p:>8.3f}"
            print(line)

    print("\n  If accuracy holds but macro-F1 falls, selection is keeping the")
    print("  features that identify large classes and losing those that catch")
    print("  rare ones. If both hold, the reduction is genuinely free.")

    print("\n\ndoes the cost grow with class imbalance?")
    print("-" * 76)
    from scipy.stats import spearmanr
    for budget in BUDGETS:
        s = per[(per.strategy == "importance_sqrt") & (per.budget == budget)]
        s = s.set_index("dataset")
        pair = pd.concat([s["macro_f1"], full["macro_f1"], info], axis=1).dropna()
        pair.columns = ["a", "b", "n_classes", "imbalance", "majority_acc"]
        delta = pair.a - pair.b
        rho, p = spearmanr(pair["imbalance"], delta)
        rho_c, p_c = spearmanr(pair["n_classes"], delta)
        print(f"  budget {budget:>6.1%}  macro-F1 change vs imbalance "
              f"rho={rho:+.3f} (p={p:.3f})   vs n_classes "
              f"rho={rho_c:+.3f} (p={p_c:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summarise", action="store_true")
    args = p.parse_args()

    if args.summarise:
        summarise()
        return

    names = random_group(eligible())

    done = set()
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        c = prior.groupby("dataset")["fold"].nunique()
        done = set(c[c >= N_FOLDS].index)

    todo = [n for n in names if n not in done]
    print(f"{len(todo)} datasets to run ({len(done)} already done)\n")

    t_start = time.perf_counter()
    for i, name in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            rows = run_dataset(name)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {name:<26} FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue

        pd.DataFrame(rows).to_csv(OUT, mode="a",
                                  header=not os.path.exists(OUT), index=False)
        got = pd.DataFrame(rows)
        f = got[got["strategy"] == "full"]
        print(f"[{i}/{len(todo)}] {name:<26} acc {f['accuracy'].mean():.4f}  "
              f"F1 {f['macro_f1'].mean():.4f}  "
              f"{time.perf_counter() - t0:7.1f}s")

    print(f"\ntotal {(time.perf_counter() - t_start) / 60:.1f} min")
    summarise()


if __name__ == "__main__":
    main()