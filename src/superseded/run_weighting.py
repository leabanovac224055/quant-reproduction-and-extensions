"""
SUPERSEDED. The v1 interval-weighting experiment, which produced
results/superseded/weighting_results.csv.

Not reported, for two reasons. Its 20 datasets were chosen by an accuracy
headroom rule invented for the purpose, which is selection on the evaluation
set; the baseline accuracies in the comments below are what that rule used.
And five of the 20 are too short to reach depth 8, which went undetected.

The reported experiment is run_weighting_v2.py: datasets drawn at random from
the paper's eligible pool, 5-fold cross-validation on pooled data as in the
paper's own sensitivity protocol, and saturation detected and reported.

Tests the hypothesis from Sect. 4.2.1 of the QUANT paper: that sampling
features from larger intervals with higher probability counteracts the accuracy
drop seen at low quantile counts.

The paper's own observation gives a sharp prediction:

    Configurations using more quantiles per interval appear to be relatively
    immune to this effect.

So weighting should help at m/8 and m/16, and do little or nothing at m/4.
A variant that helps everywhere equally, or nowhere, would not match the stated
mechanism.

Each (dataset, div, variant) is run over several resamples. The transform is
computed once per (dataset, resample, div) and shared across variants, so the
comparison differs only in the classifier.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Set QUANT_ROOT if they live outside the repo.

  python src/superseded/run_weighting.py --smoke        6 datasets, 3 resamples
  python src/superseded/run_weighting.py                20 datasets, 10 resamples
  python src/superseded/run_weighting.py --resamples 30 --datasets all

Appends to results/superseded/weighting_results.csv after every dataset;
safe to interrupt.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT",
                           Path(__file__).resolve().parent.parent.parent))
QUANT_CODE = str(ROOT / "Quant/quant-main/quant-main/code")
DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
OUT = str(ROOT / "results/superseded/weighting_results.csv")

DIVS = [4, 8, 16]          # m/4 (default), m/8, m/16

# Fig. 8 shows accuracy rising with depth to ~6 then FALLING, but only at low
# quantile counts. Depth 6 is the peak, so there is nothing there to repair -
# the decline the weighting is meant to counteract happens at depth 7 and 8.
# Testing only at depth 6 would make a null result almost tautological.
DEPTHS = [6, 7, 8]

# (name, scheme, strategy) - see interval_weighting.py
#   duplicate+equal is the only scheme duplication can realise faithfully;
#   length and linear need per_tree, which uses exact weights with no cap
VARIANTS = [
    ("baseline",        "none",   "baseline"),
    ("dup_equal",       "equal",  "duplicate"),
    ("pertree_none",    "none",   "per_tree"),   # control for per_tree itself
    ("pertree_equal",   "equal",  "per_tree"),
    ("pertree_length",  "length", "per_tree"),
]

# Datasets near 100% accuracy cannot show an improvement, so the smoke set is
# chosen for headroom rather than for being small. Baseline accuracies (from
# our own 112-dataset run) are in comments.
SMALL = [
    "ECG200",            # 0.882
    "ArrowHead",         # 0.844
    "Lightning7",        # 0.743
    "MedicalImages",     # 0.812
    "DistalPhalanxTW",   # 0.704
    "Herring",           # 0.615
]

SUBSET = SMALL + [
    "Car",                           # 0.893
    "Beef",                          # 0.779
    "Worms",                         # 0.730
    "WormsTwoClass",                 # 0.799
    "Haptics",                       # 0.519
    "InlineSkate",                   # 0.578
    "MiddlePhalanxTW",               # 0.589
    "MiddlePhalanxOutlineAgeGroup",  # 0.703
    "ScreenType",                    # 0.671
    "RefrigerationDevices",          # 0.747
    "Computers",                     # 0.837
    "Earthquakes",                   # 0.756
    "CricketX",                      # 0.772
    "FiftyWords",                    # 0.773
]

# -----------------------------------------------------------------------------

sys.path.append(QUANT_CODE)
sys.path.append(str(ROOT / "src"))

import torch                                        # noqa: E402
from quant import Quant                             # noqa: E402
from aeon.datasets import load_from_ts_file         # noqa: E402

try:
    from aeon.benchmarking.resampling import stratified_resample_data
except ImportError:
    from aeon.datasets.resampling import stratified_resample_data

from quant_provenance import feature_provenance     # noqa: E402
from interval_weighting import WeightedQuant        # noqa: E402


def load_dataset(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    return (np.asarray(Xtr), np.asarray(ytr),
            np.asarray(Xte), np.asarray(yte))


def transform(Xtr, ytr, Xte, div, depth):
    """One QUANT transform, shared by every variant at this (dataset, div, depth)."""
    t0 = time.perf_counter()
    q = Quant(depth=depth, div=div)
    Ztr = q.fit_transform(torch.tensor(Xtr, dtype=torch.float32), ytr).numpy()
    Zte = q.transform(torch.tensor(Xte, dtype=torch.float32)).numpy()
    return Ztr, Zte, time.perf_counter() - t0


def run_dataset(name, divs, depths, n_resamples, seed_offset=0):
    Xtr0, ytr0, Xte0, yte0 = load_dataset(name)
    series_length = Xtr0.shape[-1]
    n_channels = Xtr0.shape[1] if Xtr0.ndim == 3 else 1

    rows = []
    for depth in depths:
        for div in divs:
            prov = feature_provenance(series_length, depth=depth, div=div,
                                      n_channels=n_channels)

            for i in range(n_resamples):
                if i == 0:
                    Xtr, ytr, Xte, yte = Xtr0, ytr0, Xte0, yte0
                else:
                    Xtr, ytr, Xte, yte = stratified_resample_data(
                        Xtr0, ytr0, Xte0, yte0, random_state=i)

                Ztr, Zte, t_transform = transform(Xtr, ytr, Xte, div, depth)

                if Ztr.shape[1] != len(prov):
                    raise AssertionError(
                        f"{name} div={div} depth={depth}: provenance has "
                        f"{len(prov)} rows but transform produced "
                        f"{Ztr.shape[1]} columns")

                for label, scheme, strategy in VARIANTS:
                    t0 = time.perf_counter()
                    try:
                        model = WeightedQuant(
                            prov, scheme=scheme, strategy=strategy,
                            random_state=seed_offset + i,
                        ).fit(Ztr, ytr)
                        acc = model.score(Zte, yte)
                        note = ""
                    except MemoryError as exc:
                        acc, note = np.nan, f"memory: {exc}"
                    except Exception as exc:              # keep going
                        acc, note = np.nan, f"{type(exc).__name__}: {exc}"

                    rows.append({
                        "dataset": name,
                        "div": div,
                        "depth": depth,
                        "variant": label,
                        "resample": i,
                        "accuracy": acc,
                        "fit_seconds": time.perf_counter() - t0,
                        "transform_seconds": t_transform,
                        "n_features": Ztr.shape[1],
                        "length": series_length,
                        "note": note,
                    })

    return rows


def summarise(path):
    df = pd.read_csv(path).drop_duplicates(
        ["dataset", "div", "depth", "variant", "resample"])

    order = [v[0] for v in VARIANTS]
    present = [c for c in order if c in set(df["variant"])]
    divs = sorted(df["div"].unique())
    depths = sorted(df["depth"].unique())

    print(f"\n{df['dataset'].nunique()} datasets, "
          f"{df['resample'].nunique()} resamples\n")

    # ---- the actual hypothesis --------------------------------------------
    #
    # Does the baseline decline with depth at low quantile counts, and does
    # weighting flatten that decline?

    print("accuracy vs depth, for each quantile setting")
    print("=" * 74)
    for div in divs:
        sub = df[df["div"] == div]
        acc = (sub.pivot_table(index="depth", columns="variant",
                               values="accuracy", aggfunc="mean")
               .reindex(columns=present))
        print(f"\nm/{div}")
        print(acc.round(4).to_string())

        if len(depths) > 1:
            decline = acc.loc[depths[-1]] - acc.loc[depths[0]]
            print(f"  depth {depths[0]} -> {depths[-1]}:  " +
                  "  ".join(f"{v:+.4f}" for v in decline))

    print("\n\n  Reading it: the paper's mechanism predicts the BASELINE")
    print("  declines from depth 6 to 8 at m/8 and m/16 but not at m/4, and")
    print("  that weighting makes that decline shallower. A variant that is")
    print("  uniformly better or worse at every depth is doing something")
    print("  else - probably just changing feature diversity.")

    # ---- difference from baseline -----------------------------------------

    print("\n\ndifference from baseline, by depth and quantile setting")
    print("=" * 74)
    pivot = df.pivot_table(index=["div", "depth"], columns="variant",
                           values="accuracy", aggfunc="mean")
    diff = pivot.sub(pivot["baseline"], axis=0).drop(columns="baseline")
    print(diff.round(4).to_string())

    # ---- win/draw/loss ----------------------------------------------------

    print("\n\nwin/draw/loss vs baseline, per-dataset means")
    print("=" * 74)
    per = (df.groupby(["dataset", "div", "depth", "variant"])["accuracy"]
           .mean().unstack())
    for variant in present[1:]:
        print(f"\n  {variant}")
        for div in divs:
            cells = []
            for depth in depths:
                try:
                    sub = per.xs((div, depth), level=("div", "depth"))
                except KeyError:
                    continue
                a, b = sub[variant], sub["baseline"]
                ok = a.notna() & b.notna()
                cells.append(f"d{depth}: {int((a[ok] > b[ok]).sum())}/"
                             f"{int((a[ok] == b[ok]).sum())}/"
                             f"{int((a[ok] < b[ok]).sum())}")
            print(f"    m/{div:<3} " + "   ".join(cells))

    # ---- cost -------------------------------------------------------------

    print("\n\nmean fit seconds, and feature count")
    print("=" * 74)
    t = df.pivot_table(index="depth", columns="variant",
                       values="fit_seconds", aggfunc="mean").reindex(
        columns=present)
    print(t.round(2).to_string())
    print("\nmean transform width (columns before any duplication):")
    print(df.groupby(["div", "depth"])["n_features"].mean()
          .round(0).to_string())

    failures = df[df["note"].notna() & (df["note"] != "")]
    if len(failures):
        print(f"\n\n{len(failures)} failed runs:")
        print(failures.groupby(["variant", "note"]).size().to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resamples", type=int, default=10)
    p.add_argument("--datasets", default="subset",
                   choices=["smoke", "subset", "all"])
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args()

    if args.summarise_only:
        summarise(OUT)
        return

    if args.smoke:
        names, n_resamples = SMALL, 3
    else:
        n_resamples = args.resamples
        if args.datasets == "all":
            from aeon.datasets import tsc_datasets
            names = sorted(tsc_datasets.univariate_equal_length)
        elif args.datasets == "smoke":
            names = SMALL
        else:
            names = SUBSET

    names = [n for n in names if os.path.isdir(os.path.join(DATA_DIR, n))]

    done = set()
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        counts = prior.groupby("dataset").apply(
            lambda g: g["resample"].nunique(), include_groups=False)
        done = set(counts[counts >= n_resamples].index)
        if done:
            print(f"resuming: {len(done)} datasets already done")

    todo = [n for n in names if n not in done]
    per_dataset = len(DIVS) * len(DEPTHS) * len(VARIANTS) * n_resamples
    print(f"{len(todo)} datasets x {len(DIVS)} divs x {len(DEPTHS)} depths x "
          f"{len(VARIANTS)} variants x {n_resamples} resamples")
    print(f"= {per_dataset} fits per dataset, "
          f"{per_dataset * len(todo)} total\n")

    for i, name in enumerate(todo, 1):
        start = time.perf_counter()
        try:
            rows = run_dataset(name, DIVS, DEPTHS, n_resamples)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {name:<24} FAILED: "
                  f"{type(exc).__name__}: {exc}")
            continue

        pd.DataFrame(rows).to_csv(
            OUT, mode="a", header=not os.path.exists(OUT), index=False)

        got = pd.DataFrame(rows)
        # NB: got.div is pandas' division method, not the column - use [""]
        base = got[(got["variant"] == "baseline")
                   & (got["div"] == 4)]["accuracy"].mean()
        print(f"[{i}/{len(todo)}] {name:<24} baseline(m/4) {base:.4f}   "
              f"{time.perf_counter() - start:6.1f}s")

    summarise(OUT)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()