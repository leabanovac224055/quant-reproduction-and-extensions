"""
Feature selection for QUANT: large, small and random dataset groups.

Supersedes run_reduction.py, whose dataset list was changed after seeing that
the method did not help on small datasets. That made the reported speedup a
best case rather than a representative one.

Here the three groups are defined by rules fixed before running, and all three
are reported:

  large    the 20 eligible datasets with the highest n_cases x series_length
  small    the 20 with the lowest
  random   20 drawn at random from all 100 eligible, seed 0

Both of QUANT's own reduction levers are tested separately (depth, div) as
well as QUANT-FAST, which moves both at once.

Eligibility is the paper's own filter (Sect. 4.2): at least 100 training cases
on an 80/20 split and at least 5 cases of every class. n_cases x length is a
proxy for classifier workload, computed from the data before any experiment is
run, so it does not depend on any result.

The random group overlaps the other two by construction. That is intended: it
is the unbiased estimate, while large and small bound the range.

The probe is not tuned. Both candidate settings are run as separate strategies
so that neither is selected using evaluation data.

All fits are single-core, matching the paper's reported protocol.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Set QUANT_ROOT if they live outside the repo.

  python src/run_reduction_v2.py --group large
  python src/run_reduction_v2.py --group small
  python src/run_reduction_v2.py --group random
  python src/run_reduction_v2.py --summarise
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
from aeon.datasets import load_from_ts_file, tsc_datasets

from quant_provenance import feature_provenance

DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
OUT = str(ROOT / "results/reduction_v2_results.csv")

DEPTH, DIV = 6, 4
FAST_DEPTH, FAST_DIV = 5, 8
BUDGETS = [0.0625, 0.125, 0.25, 0.50]
N_FOLDS = 5
GROUP_SIZE = 20
SEED = 0

N_ESTIMATORS, MAX_FEATURES = 200, 0.1
PROBE_TREES = 20

# The paper reports timings on a single CPU core per dataset per resample
# (Sect. 4.1), so every fit here is single-core too. doc.md's example code uses
# n_jobs=-1, but the published figures are single-core and the point of these
# timings is comparability with them.
N_JOBS = 1

MIN_TRAIN_AT_80_20, MIN_PER_CLASS = 100, 5


def pooled(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, LabelEncoder().fit_transform(y)


def eligible():
    """The paper's filter, plus the workload proxy used to form the groups."""
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


def groups(pool):
    rng = np.random.default_rng(SEED)
    return {
        "small": list(pool["dataset"].head(GROUP_SIZE)),
        "large": list(pool["dataset"].tail(GROUP_SIZE)),
        "random": sorted(rng.choice(pool["dataset"].to_numpy(),
                                    size=GROUP_SIZE, replace=False)),
    }


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


def fit_score(Ztr, ytr, Zte, yte, seed):
    t0 = time.perf_counter()
    clf = forest(N_ESTIMATORS, MAX_FEATURES, seed).fit(Ztr, ytr)
    return float(clf.score(Zte, yte)), time.perf_counter() - t0


# strategy -> the probe's max_features setting; non-probe strategies are
# handled separately in select()
PROBES = {"importance_sqrt": "sqrt", "importance_p10": MAX_FEATURES}


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


SELECTION = ["importance_sqrt", "importance_p10", "balanced_sqrt", "random"]


def lever_config(lever, length, target_k):
    """
    The closest (depth, div) under the budget using one of QUANT's own levers.

    QUANT-FAST moves both at once (depth 6->5, div 4->8), but it is a single
    fixed point, not a lever aimed at a budget. Three strategies are tested:

      depth   fewer interval levels; drops whole scales
      div     fewer quantiles per interval; keeps every scale, coarser
      both    the closest (depth, div) pair under the budget - the best QUANT
              can reach by tuning its own hyperparameters

    "both" is the fair baseline: without it, the comparison would handicap
    QUANT by forcing it to use one knob at a time. Each lever has a floor
    imposed by the method's arithmetic, so some budgets are unreachable and
    those rows are skipped.
    """
    if lever == "depth":
        depths, divs = range(1, DEPTH + 1), [DIV]
    elif lever == "div":
        depths, divs = [DEPTH], [4, 8, 16, 32, 64, 128]
    else:                       # "both": the best QUANT can do with its own
        depths, divs = range(1, DEPTH + 1), [4, 8, 16, 32, 64, 128]

    best = None
    for d in depths:
        for v in divs:
            try:
                n = len(feature_provenance(length, depth=d, div=v))
            except ValueError:
                continue
            if n <= target_k and (best is None or n > best[2]):
                best = (d, v, n)
    return best


LEVERS = ["depth", "div", "both"]


def run_dataset(name, group):
    X, y = pooled(name)
    length = X.shape[-1]
    prov = feature_provenance(length, depth=DEPTH, div=DIV)
    n_default = len(prov)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        rng = np.random.default_rng(SEED * 100 + fold)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        Ztr, Zte, t_tf = transform(Xtr, Xte, DEPTH, DIV)
        acc, t_fit = fit_score(Ztr, ytr, Zte, yte, SEED)
        rows.append(dict(group=group, dataset=name, fold=fold, strategy="full",
                         budget=1.0, n_features=Ztr.shape[1], accuracy=acc,
                         transform_s=t_tf, select_s=0.0, fit_s=t_fit,
                         length=length, n_cases=len(y)))

        Ftr, Fte, t_tf_f = transform(Xtr, Xte, FAST_DEPTH, FAST_DIV)
        acc_f, t_fit_f = fit_score(Ftr, ytr, Fte, yte, SEED)
        rows.append(dict(group=group, dataset=name, fold=fold, strategy="fast",
                         budget=-1.0, n_features=Ftr.shape[1], accuracy=acc_f,
                         transform_s=t_tf_f, select_s=0.0, fit_s=t_fit_f,
                         length=length, n_cases=len(y)))

        for budget in BUDGETS:
            k = max(2, int(round(n_default * budget)))

            for strategy in SELECTION:
                idx, t_sel = select(strategy, Ztr, ytr, k, prov, rng, SEED)
                acc_s, t_fit_s = fit_score(Ztr[:, idx], ytr, Zte[:, idx], yte, SEED)
                rows.append(dict(group=group, dataset=name, fold=fold,
                                 strategy=strategy, budget=budget,
                                 n_features=len(idx), accuracy=acc_s,
                                 transform_s=t_tf, select_s=t_sel,
                                 fit_s=t_fit_s, length=length, n_cases=len(y)))

            for lever in LEVERS:
                cfg = lever_config(lever, length, k)
                if cfg is None:        # no setting of this lever fits the budget
                    continue
                d, v, n_feat = cfg
                Ltr, Lte, t_tf_l = transform(Xtr, Xte, d, v)
                acc_l, t_fit_l = fit_score(Ltr, ytr, Lte, yte, SEED)
                rows.append(dict(group=group, dataset=name, fold=fold,
                                 strategy=lever, budget=budget,
                                 n_features=n_feat, accuracy=acc_l,
                                 transform_s=t_tf_l, select_s=0.0,
                                 fit_s=t_fit_l, length=length, n_cases=len(y),
                                 config=f"depth={d} div={v}"))
    return rows


def summarise():
    from scipy.stats import wilcoxon

    df = pd.read_csv(OUT).drop_duplicates(
        ["group", "dataset", "fold", "strategy", "budget"])
    df["classifier_s"] = df.select_s + df.fit_s
    df["total_s"] = df.transform_s + df.classifier_s

    per = (df.groupby(["group", "dataset", "strategy", "budget"])
           [["accuracy", "classifier_s", "total_s", "transform_s"]]
           .mean().reset_index())

    for group in ["small", "random", "large"]:
        g = per[per.group == group]
        if not len(g):
            continue
        full = g[g.strategy == "full"]
        fast = g[g.strategy == "fast"]

        print(f"\n{'=' * 74}\n{group.upper()}  ({g.dataset.nunique()} datasets)")
        print(f"transform is "
              f"{full.transform_s.mean() / full.total_s.mean():.0%} of runtime")
        print(f"  full        acc {full.accuracy.mean():.4f}  "
              f"classifier {full.classifier_s.mean():6.2f}s  "
              f"total {full.total_s.mean():6.2f}s")
        print(f"  QUANT-FAST  acc {fast.accuracy.mean():.4f}  "
              f"classifier {fast.classifier_s.mean():6.2f}s  "
              f"total {fast.total_s.mean():6.2f}s")

        print(f"\n  {'strategy':<18}{'budget':>8}{'n':>4}{'acc':>9}{'vs full':>10}"
              f"{'p':>8}{'clf x':>8}{'total x':>9}")
        for strategy in SELECTION + LEVERS:
            for budget in BUDGETS:
                s = g[(g.strategy == strategy) & (g.budget == budget)]
                if not len(s):
                    continue
                pair = s.set_index("dataset")["accuracy"].to_frame("a").join(
                    full.set_index("dataset")["accuracy"].to_frame("b")).dropna()
                p = wilcoxon(pair.a, pair.b).pvalue if len(pair) > 5 else np.nan

                # n is the number of datasets this cell covers. The levers can
                # only reach a budget on some series lengths, so a lever row
                # with small n is a mean over a different, smaller set of
                # datasets and is NOT comparable with the selection rows.
                flag = "  <- partial" if len(pair) < len(full) else ""

                print(f"  {strategy:<18}{budget:>8.1%}{len(pair):>4}"
                      f"{s.accuracy.mean():>9.4f}"
                      f"{(pair.a - pair.b).mean():>+10.4f}{p:>8.3f}"
                      f"{full.classifier_s.mean() / s.classifier_s.mean():>8.2f}"
                      f"{full.total_s.mean() / s.total_s.mean():>9.2f}{flag}")

        print("\n  'partial' rows cover fewer datasets than the full model,")
        print("  because that lever cannot reach the budget on every series")
        print("  length. Their means are over a different subset.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["small", "large", "random"])
    p.add_argument("--summarise", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.summarise:
        summarise()
        return

    pool = eligible()
    g = groups(pool)

    if args.list:
        print(f"{len(pool)} eligible datasets\n")
        for name, members in g.items():
            w = pool.set_index("dataset").loc[members, "workload"]
            print(f"{name:<8} n={len(members)}  "
                  f"workload {w.min():,} to {w.max():,}")
            print(f"  {', '.join(members)}\n")
        overlap = set(g["random"]) & (set(g["small"]) | set(g["large"]))
        print(f"random overlaps the other groups on {len(overlap)}: "
              f"{sorted(overlap)}")
        return

    if not args.group:
        p.error("choose --group, or --list, or --summarise")

    names = g[args.group]
    done = set()
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        prior = prior[prior.group == args.group]
        c = prior.groupby("dataset")["fold"].nunique()
        done = set(c[c >= N_FOLDS].index)

    todo = [n for n in names if n not in done]
    print(f"{args.group}: {len(todo)} datasets to run "
          f"({len(done)} already done)\n")

    for i, name in enumerate(todo, 1):
        t0 = time.perf_counter()
        rows = run_dataset(name, args.group)
        pd.DataFrame(rows).to_csv(OUT, mode="a",
                                  header=not os.path.exists(OUT), index=False)
        acc = pd.DataFrame(rows).query("strategy == 'full'").accuracy.mean()
        print(f"[{i}/{len(todo)}] {name:<26} full {acc:.4f}  "
              f"{time.perf_counter() - t0:7.1f}s")

    summarise()


if __name__ == "__main__":
    main()