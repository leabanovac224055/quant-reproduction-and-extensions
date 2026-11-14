"""
SUPERSEDED. Three follow-up questions asked of the v1 feature-selection run
(results/superseded/reduction_results.csv), whose dataset list changed
mid-experiment and whose probe setting was chosen using datasets that also
appeared in the evaluation. Its numbers are v1 numbers and are not the ones
reported; the reported analysis is in run_reduction_v2.py --summarise.

Kept because Q1 and Q3 shaped the v2 design: Q1 motivated charging the probe
cost in every reported runtime, and Q3 became the depth-collapse argument.

Q1  How much of the selection overhead is avoidable?
Q2  What decides whether the transform or the classifier dominates?
Q3  Why does lowering depth lose to random selection at the same budget?

  python src/superseded/reduction_followups.py           all three
  python src/superseded/reduction_followups.py --q 2     just one
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(os.environ.get("QUANT_ROOT",
                           Path(__file__).resolve().parent.parent.parent))
QUANT_CODE = str(ROOT / "Quant/quant-main/quant-main/code")
DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
RESULTS = str(ROOT / "results/superseded/reduction_results.csv")
PROBE_OUT = str(ROOT / "results/superseded/probe_cost_results.csv")

sys.path.append(QUANT_CODE)
sys.path.append(str(ROOT / "src"))

import torch                                            # noqa: E402
from quant import Quant                                 # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier       # noqa: E402
from sklearn.model_selection import StratifiedKFold     # noqa: E402
from sklearn.preprocessing import LabelEncoder          # noqa: E402
from aeon.datasets import load_from_ts_file             # noqa: E402

from quant_provenance import feature_provenance         # noqa: E402

DEPTH, DIV = 6, 4
N_ESTIMATORS, MAX_FEATURES = 200, 0.1
BUDGET = 0.125                      # the budget where selection looked best

# probe configurations to compare: (label, n_trees, row_fraction, max_features)
PROBES = [
    ("probe_20_full", 20, 1.0, MAX_FEATURES),      # what we used
    ("probe_10_full", 10, 1.0, MAX_FEATURES),
    ("probe_5_full", 5, 1.0, MAX_FEATURES),
    ("probe_20_sqrt", 20, 1.0, "sqrt"),            # cheaper candidate split
    ("probe_10_half", 10, 0.5, MAX_FEATURES),      # half the rows
    ("probe_10_sqrt", 10, 1.0, "sqrt"),
]

Q1_DATASETS = ["StarLightCurves", "Yoga", "EthanolLevel", "ShapesAll",
               "MixedShapesSmallTrain", "ChlorineConcentration",
               "SemgHandMovementCh2", "UWaveGestureLibraryX"]


def pooled(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, LabelEncoder().fit_transform(y)


# ------------------------------------------------------------------- Q1 ----

def q1(n_folds=3, seed=0):
    """How cheap can the probe be before the ranking it produces degrades?"""
    print("\nQ1  How much of the selection overhead is avoidable?")
    print("=" * 74)
    print(f"budget {BUDGET:.1%}, {len(Q1_DATASETS)} large datasets, "
          f"{n_folds} folds\n")

    rows = []
    for name in Q1_DATASETS:
        if not os.path.isdir(os.path.join(DATA_DIR, name)):
            print(f"  skipping {name}: not on disk")
            continue

        X, y = pooled(name)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed)

        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            q = Quant(depth=DEPTH, div=DIV)
            Ztr = q.fit_transform(
                torch.tensor(X[tr], dtype=torch.float32), None).numpy()
            Zte = q.transform(
                torch.tensor(X[te], dtype=torch.float32)).numpy()
            ytr, yte = y[tr], y[te]
            k = max(2, int(round(Ztr.shape[1] * BUDGET)))

            for label, n_trees, row_frac, mf in PROBES:
                t0 = time.perf_counter()
                if row_frac < 1.0:
                    rng = np.random.default_rng(seed)
                    take = max(10, int(len(ytr) * row_frac))
                    sub = rng.choice(len(ytr), size=take, replace=False)
                    Ptr, Pty = Ztr[sub], ytr[sub]
                else:
                    Ptr, Pty = Ztr, ytr

                probe = ExtraTreesClassifier(
                    n_estimators=n_trees, max_features=mf,
                    criterion="entropy", n_jobs=-1, random_state=seed)
                probe.fit(Ptr, Pty)
                idx = np.sort(np.argsort(probe.feature_importances_)[-k:])
                t_select = time.perf_counter() - t0

                t0 = time.perf_counter()
                clf = ExtraTreesClassifier(
                    n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES,
                    criterion="entropy", n_jobs=-1, random_state=seed)
                clf.fit(Ztr[:, idx], ytr)
                acc = float(clf.score(Zte[:, idx], yte))
                t_fit = time.perf_counter() - t0

                rows.append(dict(dataset=name, fold=fold, probe=label,
                                 accuracy=acc, select_s=t_select,
                                 fit_s=t_fit, n_features=Ztr.shape[1]))

        print(f"  {name} done")

    df = pd.DataFrame(rows)
    df.to_csv(PROBE_OUT, index=False)

    per = df.groupby(["dataset", "probe"])[
        ["accuracy", "select_s", "fit_s"]].mean().reset_index()
    per["classifier_s"] = per["select_s"] + per["fit_s"]

    summary = per.groupby("probe").agg(
        accuracy=("accuracy", "mean"),
        select_s=("select_s", "mean"),
        fit_s=("fit_s", "mean"),
        classifier_s=("classifier_s", "mean"),
    )
    summary["overhead"] = summary["select_s"] / summary["classifier_s"]
    order = [p[0] for p in PROBES]
    summary = summary.reindex(order)

    print()
    print(summary.round(4).to_string())

    ref = "probe_20_full"
    print(f"\naccuracy against {ref}, per dataset (n={per.dataset.nunique()})")
    pivot = per.pivot(index="dataset", columns="probe", values="accuracy")
    for label in order:
        if label == ref or label not in pivot:
            continue
        pair = pivot[[label, ref]].dropna()
        p = wilcoxon(pair[label], pair[ref]).pvalue if len(pair) > 5 else np.nan
        saving = (summary.loc[ref, "classifier_s"]
                  - summary.loc[label, "classifier_s"])
        print(f"  {label:<16}{(pair[label] - pair[ref]).mean():+.4f}  "
              f"(p={p:.3f})   classifier {saving:+.2f}s")

    print("\n  A cheaper probe is worth taking only if accuracy is unchanged")
    print("  (p well above 0.05) while classifier time drops.")


# ------------------------------------------------------------------- Q2 ----

def q2():
    """Does case count predict the transform's share of runtime?"""
    print("\n\nQ2  What decides whether the transform or the classifier dominates?")
    print("=" * 74)

    df = pd.read_csv(RESULTS).drop_duplicates(
        ["dataset", "fold", "strategy", "budget"])
    full = df[df["strategy"] == "full"]

    per = full.groupby("dataset").agg(
        transform_s=("transform_s", "mean"),
        fit_s=("fit_s", "mean"),
        n_features=("n_features", "mean"),
        n_cases=("n_cases", "mean"),
        length=("length", "mean"),
    )
    per["total_s"] = per["transform_s"] + per["fit_s"]
    per["transform_share"] = per["transform_s"] / per["total_s"]

    print(f"\ntransform share ranges {per['transform_share'].min():.0%} to "
          f"{per['transform_share'].max():.0%} "
          f"(mean {per['transform_share'].mean():.0%})\n")

    print("Spearman correlation with transform share:")
    for col in ["n_cases", "n_features", "length"]:
        rho, p = spearmanr(per[col], per["transform_share"])
        flag = "  *" if p < 0.05 else ""
        print(f"  {col:<12}rho = {rho:+.3f}   p = {p:.4f}{flag}")

    print("\nthe classifier dominates when there are many cases:")
    ranked = per.sort_values("n_cases")
    print(ranked[["n_cases", "n_features", "transform_s", "fit_s",
                  "transform_share"]].round(3).to_string())

    median = per["n_cases"].median()
    small = per[per["n_cases"] <= median]["transform_share"].mean()
    large = per[per["n_cases"] > median]["transform_share"].mean()
    print(f"\n  datasets with <= {median:.0f} cases: "
          f"transform is {small:.0%} of runtime")
    print(f"  datasets with >  {median:.0f} cases: "
          f"transform is {large:.0%} of runtime")
    print("\n  Feature selection can only address the classifier portion, so")
    print("  its end-to-end value is bounded by this split.")


# ------------------------------------------------------------------- Q3 ----

def q3():
    """Why does lowering depth lose to random selection at the same budget?"""
    print("\n\nQ3  Why does lowering depth lose so badly?")
    print("=" * 74)

    df = pd.read_csv(RESULTS).drop_duplicates(
        ["dataset", "fold", "strategy", "budget"])
    per = (df.groupby(["dataset", "strategy", "budget"])["accuracy"]
           .mean().reset_index())

    sub = per[per["budget"] == BUDGET].pivot(
        index="dataset", columns="strategy", values="accuracy")
    full = per[per["strategy"] == "full"].set_index("dataset")["accuracy"]

    print(f"\nat a {BUDGET:.1%} budget, loss against the full model:")
    for s in ["importance", "balanced", "random", "variance", "depth", "div"]:
        if s not in sub:
            continue
        pair = pd.concat([sub[s], full], axis=1).dropna()
        pair.columns = [s, "full"]
        print(f"  {s:<12}{(pair[s] - pair['full']).mean():+.4f}  "
              f"(n={len(pair)})")

    # what does each strategy actually keep, by depth?
    print(f"\nwhat survives at a {BUDGET:.1%} budget, by interval depth")
    print("-" * 74)

    lengths = df.groupby("dataset")["length"].first()
    example = lengths.index[len(lengths) // 2]
    L = int(lengths[example])

    prov = feature_provenance(L, depth=DEPTH, div=DIV)
    n_default = len(prov)
    k = int(round(n_default * BUDGET))
    base_share = prov.groupby("depth").size() / n_default

    # `depth` strategy: the largest depth whose feature count fits the budget
    fitting = None
    for d in range(1, DEPTH + 1):
        if len(feature_provenance(L, depth=d, div=DIV)) <= k:
            fitting = d
    depth_prov = feature_provenance(L, depth=fitting, div=DIV)
    depth_share = depth_prov.groupby("depth").size() / len(depth_prov)

    table = pd.DataFrame({
        "full model": base_share,
        "random (expected)": base_share,          # random preserves shares
        f"depth={fitting}": depth_share,
    }).fillna(0.0)

    print(f"  example: {example}, series length {L}, "
          f"{n_default} features, budget {k}\n")
    print((table * 100).round(1).to_string())

    print(f"\n  Lowering depth to {fitting} deletes depths "
          f"{fitting + 1}-{DEPTH} entirely: those scales are simply gone.")
    print("  Random thinning keeps every depth, just more sparsely.")
    print("  That is the difference - coverage of scales, not feature count.")
    print("\n  So the depth lever does not merely reduce the feature space,")
    print("  it removes the method's multi-scale structure, which is the part")
    print("  the paper argues is doing the work (Fig. 2, Sect. 3.2).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", type=int, choices=[1, 2, 3])
    p.add_argument("--folds", type=int, default=3)
    args = p.parse_args()

    if args.q in (None, 2):
        q2()
    if args.q in (None, 3):
        q3()
    if args.q in (None, 1):
        q1(n_folds=args.folds)


if __name__ == "__main__":
    main()