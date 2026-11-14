"""
SUPERSEDED. The v1 feature-selection experiment, which produced
results/superseded/reduction_results_sqrt.csv. Not reported: its dataset list
was changed after a first run showed no speedup on small datasets, and its
probe setting was chosen using datasets that also appeared in the evaluation.
Both are forms of selection on the evaluation set. The reported experiment is
run_reduction_v2.py, which fixes the dataset groups and strategies in advance
and runs single-core.

Sect. 3.5 of the paper:

    The majority of compute time is spent in training the classifier. In other
    words, any attempt at optimising total compute time should concentrate on
    reducing the size of the feature space, and/or improving the efficiency of
    classifier training. We leave further optimisation for future work.

The paper's own attempt at this, QUANT-FAST, simply halves two hyperparameters
(depth 6 -> 5, m/4 -> m/8). That is a reduction, but an unprincipled one: it
removes features by category rather than by usefulness.

Our Fig. 8 reproduction showed the accuracy surface is remarkably flat - 25
configurations spanning 0.0059, with no depth transition significant - while
compute varies 20-fold across the same grid. That gap is the opportunity: if
accuracy barely moves, features can be removed aggressively.

THE QUESTION: at a matched feature budget, which way of reducing the feature
space preserves the most accuracy?

Strategies, all evaluated at the same budgets:

  full          depth 6, m/4. The default.
  fast          depth 5, m/8. The paper's own QUANT-FAST.
  div           raise the quantile divisor until the budget is met
  depth         lower the depth until the budget is met
                  ^ the two levers the paper itself uses
  variance      keep the highest-variance columns (unsupervised, very cheap)
  importance    keep the columns a cheap 20-tree probe ranks highest
  balanced      as importance, but keeping each depth's share of the budget
                equal - motivated by the feature-count skew we measured
  random        keep a random subset. THE CONTROL: if random does as well as
                importance, the features are simply redundant and no selection
                method deserves credit.

The transform is computed once per (dataset, fold, configuration) and shared
across the selection strategies, so they differ only in which columns survive.
Selection overhead is measured and included in the reported cost, otherwise
"importance" would be getting its ranking for free.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Set QUANT_ROOT if they live outside the repo.

  python src/superseded/run_reduction.py --smoke      6 datasets, 3 folds
  python src/superseded/run_reduction.py              24 datasets, 5 folds
  python src/superseded/run_reduction.py --summarise-only

Appends to results/superseded/reduction_results_sqrt.csv after each dataset;
safe to interrupt.
"""

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT",
                           Path(__file__).resolve().parent.parent.parent))
QUANT_CODE = str(ROOT / "Quant/quant-main/quant-main/code")
DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
OUT = str(ROOT / "results/superseded/reduction_results_sqrt.csv")

DEFAULT_DEPTH, DEFAULT_DIV = 6, 4          # QUANT's defaults
FAST_DEPTH, FAST_DIV = 5, 8                # QUANT-FAST

# fraction of the default feature count to keep
BUDGETS = [0.50, 0.25, 0.125, 0.0625]

SELECTION_STRATEGIES = ["variance", "importance", "balanced", "random"]
LEVER_STRATEGIES = ["div", "depth"]

N_FOLDS = 5
N_ESTIMATORS = 200
MAX_FEATURES = 0.1
PROBE_TREES = 20            # cheap forest used only to rank features

# 24 datasets spanning size and difficulty, all from the paper's eligible pool
# The paper's claim is that the CLASSIFIER dominates compute. That is only
# true once a dataset is big enough for the fit to outweigh the transform, so
# these are chosen for size (cases x length) rather than for variety. On
# 200-case datasets the transform and the selection probe swamp any saving,
# which the smoke run demonstrated.
DATASETS = [
    "CinCECGTorso", "Mallat", "MixedShapesSmallTrain", "StarLightCurves",
    "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
    "InsectWingbeatSound", "FaceAll", "FacesUCR", "SwedishLeaf", "Adiac",
    "ECG5000", "Yoga", "Strawberry", "FreezerRegularTrain",
    "ChlorineConcentration", "TwoPatterns", "Symbols", "EthanolLevel",
    "SemgHandMovementCh2", "EOGHorizontalSignal", "CricketX", "ShapesAll",
]
# a smoke set that is still big enough for the classifier to dominate
SMOKE = ["Symbols", "Adiac", "SwedishLeaf", "CricketX",
         "InsectWingbeatSound", "Strawberry"]

# -----------------------------------------------------------------------------

sys.path.append(QUANT_CODE)
sys.path.append(str(ROOT / "src"))

import torch                                            # noqa: E402
from quant import Quant                                 # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier       # noqa: E402
from sklearn.model_selection import StratifiedKFold     # noqa: E402
from sklearn.preprocessing import LabelEncoder          # noqa: E402
from aeon.datasets import load_from_ts_file             # noqa: E402

from quant_provenance import feature_provenance         # noqa: E402


def pooled(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, LabelEncoder().fit_transform(y)


def transform(Xtr, Xte, depth, div):
    t0 = time.perf_counter()
    q = Quant(depth=depth, div=div)
    Ztr = q.fit_transform(torch.tensor(Xtr, dtype=torch.float32), None).numpy()
    Zte = q.transform(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return Ztr, Zte, time.perf_counter() - t0


def fit_score(Ztr, ytr, Zte, yte, seed):
    t0 = time.perf_counter()
    clf = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES,
        criterion="entropy", n_jobs=-1, random_state=seed)
    clf.fit(Ztr, ytr)
    acc = float(clf.score(Zte, yte))
    return acc, time.perf_counter() - t0


# ------------------------------------------------------------- selection ----

def select(strategy, Ztr, ytr, k, prov, rng, seed):
    """
    Return (column indices to keep, seconds spent choosing them).

    Selection uses training data only.
    """
    t0 = time.perf_counter()
    n = Ztr.shape[1]

    if strategy == "random":
        idx = rng.choice(n, size=k, replace=False)

    elif strategy == "variance":
        idx = np.argsort(Ztr.var(axis=0))[-k:]

    elif strategy in ("importance", "balanced"):
        probe = ExtraTreesClassifier(
            n_estimators=PROBE_TREES, max_features="sqrt",
            criterion="entropy", n_jobs=-1, random_state=seed)
        probe.fit(Ztr, ytr)
        imp = probe.feature_importances_

        if strategy == "importance":
            idx = np.argsort(imp)[-k:]
        else:
            # equal budget per depth, then best-by-importance within each
            depths = prov["depth"].to_numpy()[:n]
            levels = np.unique(depths)
            per_level = max(1, k // len(levels))
            picked = []
            for d in levels:
                cols = np.flatnonzero(depths == d)
                take = min(per_level, len(cols))
                picked.append(cols[np.argsort(imp[cols])[-take:]])
            idx = np.concatenate(picked)
            # top up to k with the best remaining
            if len(idx) < k:
                rest = np.setdiff1d(np.arange(n), idx)
                extra = rest[np.argsort(imp[rest])[-(k - len(idx)):]]
                idx = np.concatenate([idx, extra])

    else:
        raise ValueError(strategy)

    return np.sort(idx), time.perf_counter() - t0


def lever_config(strategy, series_length, target_k):
    """
    The nearest (depth, div) that gets within the budget using QUANT's own
    levers, so the comparison is like for like.
    """
    best = None
    depths = range(1, DEFAULT_DEPTH + 1) if strategy == "depth" else [DEFAULT_DEPTH]
    divs = [DEFAULT_DIV] if strategy == "depth" else [4, 8, 16, 32, 64, 128]

    for depth in depths:
        for div in divs:
            try:
                n = len(feature_provenance(series_length, depth=depth, div=div))
            except ValueError:
                continue
            if n <= target_k and (best is None or n > best[2]):
                best = (depth, div, n)
    return best


# ------------------------------------------------------------------- run ----

def run_dataset(name, n_folds, seed=0):
    X, y = pooled(name)
    series_length = X.shape[-1]
    prov = feature_provenance(series_length, depth=DEFAULT_DEPTH,
                              div=DEFAULT_DIV)
    n_default = len(prov)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        rng = np.random.default_rng(seed * 100 + fold)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        # --- the two reference points -------------------------------------
        Ztr, Zte, t_tf = transform(Xtr, Xte, DEFAULT_DEPTH, DEFAULT_DIV)
        acc, t_fit = fit_score(Ztr, ytr, Zte, yte, seed)
        rows.append(dict(dataset=name, fold=fold, strategy="full", budget=1.0,
                         actual_frac=1.0,
                         n_features=Ztr.shape[1], accuracy=acc,
                         transform_s=t_tf, select_s=0.0, fit_s=t_fit,
                         length=series_length, n_cases=len(y), note=""))

        Ftr, Fte, t_tf_f = transform(Xtr, Xte, FAST_DEPTH, FAST_DIV)
        acc_f, t_fit_f = fit_score(Ftr, ytr, Fte, yte, seed)
        rows.append(dict(dataset=name, fold=fold, strategy="fast",
                         budget=-1.0,       # fixed label; real fraction below
                         actual_frac=Ftr.shape[1] / n_default,
                         n_features=Ftr.shape[1], accuracy=acc_f,
                         transform_s=t_tf_f, select_s=0.0, fit_s=t_fit_f,
                         length=series_length, n_cases=len(y), note=""))

        # --- budgets ------------------------------------------------------
        for budget in BUDGETS:
            k = max(2, int(round(n_default * budget)))

            for strategy in SELECTION_STRATEGIES:
                try:
                    idx, t_sel = select(strategy, Ztr, ytr, k, prov, rng, seed)
                    acc_s, t_fit_s = fit_score(
                        Ztr[:, idx], ytr, Zte[:, idx], yte, seed)
                    note = ""
                except Exception as exc:
                    acc_s, t_sel, t_fit_s = np.nan, np.nan, np.nan
                    idx, note = [], f"{type(exc).__name__}: {exc}"

                rows.append(dict(
                    dataset=name, fold=fold, strategy=strategy, budget=budget,
                    actual_frac=len(idx) / n_default,
                    n_features=len(idx), accuracy=acc_s,
                    transform_s=t_tf,          # shared full transform
                    select_s=t_sel, fit_s=t_fit_s,
                    length=series_length, n_cases=len(y), note=note))

            for strategy in LEVER_STRATEGIES:
                cfg = lever_config(strategy, series_length, k)
                if cfg is None:
                    continue
                depth, div, n_feat = cfg
                try:
                    Ltr, Lte, t_tf_l = transform(Xtr, Xte, depth, div)
                    acc_l, t_fit_l = fit_score(Ltr, ytr, Lte, yte, seed)
                    note = f"depth={depth} div={div}"
                except Exception as exc:
                    acc_l, t_tf_l, t_fit_l = np.nan, np.nan, np.nan
                    n_feat, note = 0, f"{type(exc).__name__}: {exc}"

                rows.append(dict(
                    dataset=name, fold=fold, strategy=strategy, budget=budget,
                    actual_frac=n_feat / n_default,
                    n_features=n_feat, accuracy=acc_l,
                    transform_s=t_tf_l, select_s=0.0, fit_s=t_fit_l,
                    length=series_length, n_cases=len(y), note=note))

    return rows


# --------------------------------------------------------------- summary ----

def summarise(path):
    from scipy.stats import wilcoxon

    df = pd.read_csv(path).drop_duplicates(
        ["dataset", "fold", "strategy", "budget"])
    df["classifier_s"] = df["select_s"].fillna(0) + df["fit_s"]
    df["total_s"] = df["transform_s"] + df["classifier_s"]

    cols = ["accuracy", "total_s", "classifier_s", "fit_s", "transform_s",
            "n_features", "actual_frac"]
    per = (df.groupby(["dataset", "strategy", "budget"])[cols]
           .mean().reset_index())

    full = per[per["strategy"] == "full"].set_index("dataset")
    fast = per[per["strategy"] == "fast"].set_index("dataset")

    print(f"\n{df['dataset'].nunique()} datasets, {df['fold'].nunique()} folds")
    print(f"\ntransform is {full['transform_s'].mean() / full['total_s'].mean():.0%} "
          f"of default runtime; the classifier is the rest.")
    print("Selection cannot reduce the transform, so the classifier column is")
    print("the one that matters for any selection strategy.\n")

    print(f"default (depth {DEFAULT_DEPTH}, m/{DEFAULT_DIV}): "
          f"acc {full['accuracy'].mean():.4f}, "
          f"{full['n_features'].mean():.0f} features, "
          f"classifier {full['classifier_s'].mean():.2f}s, "
          f"total {full['total_s'].mean():.2f}s")
    if len(fast):
        print(f"QUANT-FAST (depth {FAST_DEPTH}, m/{FAST_DIV}): "
              f"acc {fast['accuracy'].mean():.4f}, "
              f"{fast['n_features'].mean():.0f} features "
              f"({fast['actual_frac'].mean():.0%} of default), "
              f"classifier {fast['classifier_s'].mean():.2f}s, "
              f"total {fast['total_s'].mean():.2f}s")

    budgeted = per[per["budget"] > 0]

    print("\n\naccuracy at each budget")
    print("=" * 76)
    acc = budgeted.pivot_table(index="budget", columns="strategy",
                               values="accuracy")
    print(acc.round(4).to_string())

    print("\nloss against the default, in accuracy points")
    loss = acc - full["accuracy"].mean()
    if len(fast):
        print(f"  (QUANT-FAST's own loss: "
              f"{fast['accuracy'].mean() - full['accuracy'].mean():+.4f})")
    print(loss.round(4).to_string())

    print("\n\nclassifier speedup against the default (selection cost included)")
    print("=" * 76)
    c = budgeted.pivot_table(index="budget", columns="strategy",
                             values="classifier_s")
    print((full["classifier_s"].mean() / c).round(2).to_string())

    print("\nend-to-end speedup, transform included")
    print("-" * 76)
    t = budgeted.pivot_table(index="budget", columns="strategy",
                             values="total_s")
    print((full["total_s"].mean() / t).round(2).to_string())

    print("\n\nis each strategy better than the random control?")
    print("=" * 76)
    print("  (if nothing beats random, the feature space is simply redundant")
    print("   and no selection method deserves credit)\n")
    for budget in sorted(budgeted["budget"].unique()):
        sub = budgeted[budgeted["budget"] == budget].pivot(
            index="dataset", columns="strategy", values="accuracy")
        if "random" not in sub:
            continue
        print(f"  budget {budget:>6.1%}")
        for s in SELECTION_STRATEGIES + LEVER_STRATEGIES:
            if s == "random" or s not in sub:
                continue
            pair = sub[[s, "random"]].dropna()
            if len(pair) < 6:
                continue
            p = wilcoxon(pair[s], pair["random"]).pvalue
            flag = "  *" if p < 0.05 else ""
            print(f"    {s:<12}{(pair[s] - pair['random']).mean():+.4f}"
                  f"  (n={len(pair)}, p={p:.3f}){flag}")

    if len(fast):
        print("\n\ncan any strategy beat QUANT-FAST on both accuracy and cost?")
        print("=" * 76)
        fast_acc = fast["accuracy"].mean()
        fast_clf = fast["classifier_s"].mean()
        fast_tot = fast["total_s"].mean()
        print(f"  QUANT-FAST: acc {fast_acc:.4f}, "
              f"classifier {fast_clf:.2f}s, total {fast_tot:.2f}s\n")
        found = False
        for s in SELECTION_STRATEGIES + LEVER_STRATEGIES:
            sub = budgeted[budgeted["strategy"] == s]
            for budget in sorted(sub["budget"].unique()):
                cell = sub[sub["budget"] == budget]
                a = cell["accuracy"].mean()
                c_ = cell["classifier_s"].mean()
                t_ = cell["total_s"].mean()
                if a >= fast_acc and c_ < fast_clf:
                    found = True
                    print(f"  {s:<12} budget {budget:>6.1%}: acc {a:.4f}, "
                          f"classifier {c_:.2f}s, total {t_:.2f}s"
                          f"{'  (also faster end to end)' if t_ < fast_tot else ''}")
        if not found:
            print("  none - QUANT-FAST is on the frontier for these datasets")

    notes = df["note"].fillna("")
    bad = df[(notes != "") & ~notes.str.startswith("depth=")]
    if len(bad):
        print(f"\n{len(bad)} problem rows:")
        print(bad.groupby("note").size().head().to_string())


# ------------------------------------------------------------------ main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--folds", type=int, default=N_FOLDS)
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args()

    if args.summarise_only:
        summarise(OUT)
        return

    names = SMOKE if args.smoke else DATASETS
    n_folds = 3 if args.smoke else args.folds
    names = [n for n in names if os.path.isdir(os.path.join(DATA_DIR, n))]

    done = set()
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        counts = prior.groupby("dataset")["fold"].nunique()
        done = set(counts[counts >= n_folds].index)
        if done:
            print(f"resuming: {len(done)} datasets done")

    todo = [n for n in names if n not in done]
    print(f"{len(todo)} datasets x {n_folds} folds x "
          f"({len(BUDGETS)} budgets x "
          f"{len(SELECTION_STRATEGIES) + len(LEVER_STRATEGIES)} strategies "
          f"+ 2 references)\n")

    t_start = time.perf_counter()
    for i, name in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            rows = run_dataset(name, n_folds)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {name:<24} FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue

        pd.DataFrame(rows).to_csv(
            OUT, mode="a", header=not os.path.exists(OUT), index=False)
        got = pd.DataFrame(rows)
        base = got[got["strategy"] == "full"]["accuracy"].mean()
        print(f"[{i}/{len(todo)}] {name:<24} full {base:.4f}   "
              f"{time.perf_counter() - t0:6.1f}s")

    print(f"\ntotal: {(time.perf_counter() - t_start) / 60:.1f} min")
    summarise(OUT)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()