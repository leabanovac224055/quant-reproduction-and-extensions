"""
Interval-weighted feature sampling, on a properly drawn sample.

Supersedes run_weighting.py, whose 20 datasets were chosen by an invented rule
(accuracy headroom) and included five too short to reach depth 8, diluting the
comparison that mattered by 25%.

The output is named v3 because v2 used the same design but a faulty subset
draw: weighted sampling without replacement does not realise the intended
weights (see interval_weighting.py). v2's results are kept in
results/superseded/.

Tests the suggestion in Sect. 4.2.1 that the paper never tests:

    It may be possible to counteract this effect by sampling features from
    larger intervals with higher probability when training the classifier.
    We leave this for future work.

Protocol follows Sect. 4.2, the same as our Fig. 8 reproduction:
  - eligibility: the paper's filter (>= 100 training cases at 80/20, >= 5 per
    class), which returns exactly the 100 datasets it reports
  - 20 drawn at random from those, seed recorded
  - 5-fold stratified CV on pooled train+test

Depths 4-8 rather than the default 6, because the effect under test is a
decline that is supposed to appear past depth 6. Datasets too short to reach
the deepest level are detected and reported separately, since for those the
deeper transforms are identical and would flatten any depth effect.

Multi-core: this experiment makes no timing claim, only an accuracy one.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Set QUANT_ROOT if they live outside the repo.

  python src/run_weighting_v2.py --list      show the sample and stop
  python src/run_weighting_v2.py --smoke     6 datasets, depths 5-7, 3 folds
                                             (all three quantile settings, so
                                             still 135 fits per dataset)
  python src/run_weighting_v2.py             the full grid
  python src/run_weighting_v2.py --summarise

Appends to results/weighting_v3_results.csv after each dataset; safe to
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_from_ts_file, tsc_datasets

from quant_provenance import feature_provenance
from interval_weighting import WeightedQuant

DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
OUT = str(ROOT / "results/weighting_v3_results.csv")

DEPTHS = [4, 5, 6, 7, 8]
DIVS = [4, 8, 16]                 # m/4 (default), m/8, m/16
N_FOLDS = 5
SAMPLE = 20
SEED = 0

MIN_TRAIN_AT_80_20, MIN_PER_CLASS = 100, 5

# (label, scheme, strategy). pertree_none is the control: the per-tree
# machinery with uniform weights, which separates the effect of weighting from
# the effect of per-tree ensembling.
VARIANTS = [
    ("baseline",       "none",   "baseline"),
    ("dup_equal",      "equal",  "duplicate"),
    ("pertree_none",   "none",   "per_tree"),
    ("pertree_equal",  "equal",  "per_tree"),
    ("pertree_length", "length", "per_tree"),
]


def pooled(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, LabelEncoder().fit_transform(y)


def eligible():
    """The paper's filter. Should return 100 datasets."""
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
        rows.append(dict(dataset=name, n_cases=len(y), length=X.shape[-1]))
    return pd.DataFrame(rows)


def saturates_at(length, depths, div):
    """
    The depth beyond which a deeper request adds no features.

    quant.py caps depth at min(depth, int(log2(L)) + 1) per representation, so
    on short series the deeper levels produce an identical transform. Returns
    None if every requested depth adds something.
    """
    counts = {d: len(feature_provenance(length, depth=d, div=div))
              for d in depths}
    ordered = sorted(counts)
    for prev, nxt in zip(ordered, ordered[1:]):
        if counts[nxt] == counts[prev]:
            return prev
    return None


def run_dataset(name, depths, divs, n_folds):
    X, y = pooled(name)
    length = X.shape[-1]
    n_channels = X.shape[1] if X.ndim == 3 else 1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    rows = []
    for div in divs:
        saturated = saturates_at(length, depths, div)

        for depth in depths:
            prov = feature_provenance(length, depth=depth, div=div,
                                      n_channels=n_channels)

            for fold, (tr, te) in enumerate(folds):
                q = Quant(depth=depth, div=div)
                Ztr = q.fit_transform(
                    torch.tensor(X[tr], dtype=torch.float32), y[tr]).numpy()
                Zte = q.transform(
                    torch.tensor(X[te], dtype=torch.float32)).numpy()

                if Ztr.shape[1] != len(prov):
                    raise AssertionError(
                        f"{name} depth={depth} div={div}: provenance has "
                        f"{len(prov)} rows, transform gave {Ztr.shape[1]}")

                for label, scheme, strategy in VARIANTS:
                    t0 = time.perf_counter()
                    try:
                        model = WeightedQuant(prov, scheme=scheme,
                                              strategy=strategy,
                                              random_state=SEED).fit(Ztr, y[tr])
                        acc, note = model.score(Zte, y[te]), ""
                    except Exception as exc:
                        acc, note = np.nan, f"{type(exc).__name__}: {exc}"

                    rows.append(dict(
                        dataset=name, depth=depth, div=div, fold=fold,
                        variant=label, accuracy=acc,
                        seconds=time.perf_counter() - t0,
                        n_features=Ztr.shape[1], length=length,
                        saturated_from=saturated, note=note))
    return rows


def summarise():
    from scipy.stats import wilcoxon

    df = pd.read_csv(OUT).drop_duplicates(
        ["dataset", "depth", "div", "fold", "variant"])

    names = [v[0] for v in VARIANTS]
    depths = sorted(df["depth"].unique())

    sat = df.groupby("dataset")["saturated_from"].first()
    full_depth = sorted(sat[sat.isna()].index)

    print(f"\n{df['dataset'].nunique()} datasets, {df['fold'].nunique()} folds")
    print(f"{len(full_depth)} reach every requested depth\n")

    per = (df.groupby(["dataset", "depth", "div", "variant"])["accuracy"]
           .mean().unstack())[names]

    for subset, label in [(sorted(df["dataset"].unique()), "all datasets"),
                          (full_depth, f"{len(full_depth)} reaching every depth")]:
        if len(subset) < 6 or (label.startswith(str(len(full_depth)))
                               and len(subset) == df["dataset"].nunique()):
            continue

        sub = per.loc[subset]
        print(f"\n{'=' * 72}\n{label}")

        print("\nbaseline accuracy by depth and quantile setting")
        print(sub["baseline"].groupby(level=["div", "depth"]).mean()
              .unstack("div").round(4).to_string())

        print("\n  the paper predicts a decline past depth 6 at m/8 and m/16,")
        print("  and immunity at m/4")

        print("\ndifference from baseline, by depth")
        rows = []
        for variant in names[1:]:
            for depth in depths:
                a = sub.xs(depth, level="depth")[variant]
                b = sub.xs(depth, level="depth")["baseline"]
                ok = a.notna() & b.notna()
                rows.append({
                    "variant": variant, "depth": depth,
                    "n": int(ok.sum()),
                    "diff": (a[ok] - b[ok]).mean(),
                    "p": wilcoxon(a[ok], b[ok]).pvalue if ok.sum() > 5 else np.nan,
                })
        out = pd.DataFrame(rows)
        print(out.pivot(index="depth", columns="variant",
                        values="diff").round(4).to_string())
        print("\np")
        print(out.pivot(index="depth", columns="variant",
                        values="p").round(3).to_string())

        n_tests = len(out)
        thresh = 0.05 / n_tests
        sig = out[out["p"] < thresh]
        print(f"\n{n_tests} tests, Bonferroni threshold {thresh:.5f}, "
              f"{len(sig)} surviving")
        if len(sig):
            print(sig.round(4).to_string(index=False))

    bad = df[df["note"].fillna("") != ""]
    if len(bad):
        print(f"\n{len(bad)} failed runs")
        print(bad.groupby("note").size().head().to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--summarise", action="store_true")
    args = p.parse_args()

    if args.summarise:
        summarise()
        return

    pool = eligible()
    rng = np.random.default_rng(SEED)
    names = sorted(rng.choice(pool["dataset"].to_numpy(), SAMPLE,
                              replace=False))

    depths, divs, n_folds = DEPTHS, DIVS, N_FOLDS
    if args.smoke:
        names, depths, n_folds = names[:6], [5, 6, 7], 3

    if args.list:
        print(f"{len(pool)} eligible datasets "
              f"(the paper reports 100)\n")
        info = pool.set_index("dataset").loc[names]
        info["saturates"] = [saturates_at(int(l), DEPTHS, 4)
                             for l in info["length"]]
        print(info.to_string())
        n_sat = info["saturates"].notna().sum()
        print(f"\n{n_sat} of {len(names)} cannot reach depth {max(DEPTHS)}; "
              "results are reported with and without them")
        return

    done = set()
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        c = prior.groupby("dataset")["fold"].nunique()
        done = set(c[c >= n_folds].index)

    todo = [n for n in names if n not in done]
    fits = len(depths) * len(divs) * len(VARIANTS) * n_folds
    print(f"{len(todo)} datasets x {fits} fits each "
          f"({len(todo) * fits} total)\n")

    t_start = time.perf_counter()
    for i, name in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            rows = run_dataset(name, depths, divs, n_folds)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {name:<28} FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue

        pd.DataFrame(rows).to_csv(OUT, mode="a",
                                  header=not os.path.exists(OUT), index=False)
        got = pd.DataFrame(rows)
        acc = got[(got["variant"] == "baseline") & (got["depth"] == 6)
                & (got["div"] == 4)]["accuracy"].mean()
        print(f"[{i}/{len(todo)}] {name:<28} baseline(d6,m/4) {acc:.4f}   "
              f"{time.perf_counter() - t0:7.1f}s")

    print(f"\ntotal {(time.perf_counter() - t_start) / 60:.1f} min")
    summarise()


if __name__ == "__main__":
    main()