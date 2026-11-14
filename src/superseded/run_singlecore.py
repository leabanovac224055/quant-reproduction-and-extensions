"""
SUPERSEDED. Single-core timings, run as a check before the v2 experiment.

Every v1 result used n_jobs=-1, but the paper's compute figures are
single-core, and the speedup ratio can differ between the two: the probe and
the full fit parallelise differently, so a multi-core ratio is not
transferable. This checked how much that mattered on a subset of the large
datasets.

The answer was that it mattered, so run_reduction_v2.py runs single-core
throughout and this script is not needed. Its numbers are not reported.

  python src/superseded/run_singlecore.py
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

import torch
from quant import Quant
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_from_ts_file

DATA_DIR = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
OUT = str(ROOT / "results/superseded/singlecore_results.csv")

DATASETS = ["Yoga", "ShapesAll", "MixedShapesSmallTrain", "EthanolLevel",
            "ChlorineConcentration", "UWaveGestureLibraryX",
            "SemgHandMovementCh2", "Symbols"]
BUDGETS = [0.125, 0.25]
N_FOLDS = 3
N_JOBS = 1


def pooled(name):
    base = f"{DATA_DIR}/{name}/{name}"
    Xtr, ytr = load_from_ts_file(f"{base}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{base}_TEST.ts")
    X = np.concatenate([np.asarray(Xtr), np.asarray(Xte)])
    y = np.concatenate([np.asarray(ytr), np.asarray(yte)])
    return X, LabelEncoder().fit_transform(y)


def forest(n_trees, max_features, seed):
    return ExtraTreesClassifier(n_estimators=n_trees, max_features=max_features,
                                criterion="entropy", n_jobs=N_JOBS,
                                random_state=seed)


rows = []
for name in DATASETS:
    if not os.path.isdir(os.path.join(DATA_DIR, name)):
        continue
    X, y = pooled(name)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    t_ds = time.perf_counter()

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        q = Quant(depth=6, div=4)
        t0 = time.perf_counter()
        Ztr = q.fit_transform(torch.tensor(X[tr], dtype=torch.float32), None).numpy()
        Zte = q.transform(torch.tensor(X[te], dtype=torch.float32)).numpy()
        t_tf = time.perf_counter() - t0
        ytr, yte = y[tr], y[te]

        # full model
        t0 = time.perf_counter()
        clf = forest(200, 0.1, 0).fit(Ztr, ytr)
        acc = clf.score(Zte, yte)
        rows.append(dict(dataset=name, fold=fold, strategy="full", budget=1.0,
                         accuracy=acc, transform_s=t_tf, select_s=0.0,
                         fit_s=time.perf_counter() - t0,
                         n_features=Ztr.shape[1]))

        # QUANT-FAST
        qf = Quant(depth=5, div=8)
        t0 = time.perf_counter()
        Ftr = qf.fit_transform(torch.tensor(X[tr], dtype=torch.float32), None).numpy()
        Fte = qf.transform(torch.tensor(X[te], dtype=torch.float32)).numpy()
        t_tf_f = time.perf_counter() - t0
        t0 = time.perf_counter()
        clf = forest(200, 0.1, 0).fit(Ftr, ytr)
        rows.append(dict(dataset=name, fold=fold, strategy="fast", budget=-1.0,
                         accuracy=clf.score(Fte, yte), transform_s=t_tf_f,
                         select_s=0.0, fit_s=time.perf_counter() - t0,
                         n_features=Ftr.shape[1]))

        # importance selection
        for budget in BUDGETS:
            k = max(2, int(round(Ztr.shape[1] * budget)))
            t0 = time.perf_counter()
            probe = forest(20, "sqrt", 0).fit(Ztr, ytr)
            idx = np.sort(np.argsort(probe.feature_importances_)[-k:])
            t_sel = time.perf_counter() - t0

            t0 = time.perf_counter()
            clf = forest(200, 0.1, 0).fit(Ztr[:, idx], ytr)
            rows.append(dict(dataset=name, fold=fold, strategy="importance",
                             budget=budget, accuracy=clf.score(Zte[:, idx], yte),
                             transform_s=t_tf, select_s=t_sel,
                             fit_s=time.perf_counter() - t0, n_features=k))

    print(f"{name:<26}{time.perf_counter() - t_ds:7.1f}s")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

df["classifier_s"] = df.select_s + df.fit_s
df["total_s"] = df.transform_s + df.classifier_s
per = df.groupby(["dataset", "strategy", "budget"])[
    ["accuracy", "classifier_s", "total_s", "transform_s"]].mean().reset_index()

full = per[per.strategy == "full"]
fast = per[per.strategy == "fast"]

print(f"\nsingle core, {df.dataset.nunique()} datasets, {N_FOLDS} folds")
print(f"transform is {full.transform_s.mean() / full.total_s.mean():.0%} of runtime\n")
print(f"{'':<22}{'acc':>8}{'classifier':>12}{'total':>9}{'clf speedup':>13}")
print(f"{'full':<22}{full.accuracy.mean():>8.4f}"
      f"{full.classifier_s.mean():>12.2f}{full.total_s.mean():>9.2f}{1.0:>13.2f}")
print(f"{'QUANT-FAST':<22}{fast.accuracy.mean():>8.4f}"
      f"{fast.classifier_s.mean():>12.2f}{fast.total_s.mean():>9.2f}"
      f"{full.classifier_s.mean() / fast.classifier_s.mean():>13.2f}")
for budget in BUDGETS:
    s = per[(per.strategy == "importance") & (per.budget == budget)]
    print(f"{'importance@' + f'{budget:.1%}':<22}{s.accuracy.mean():>8.4f}"
          f"{s.classifier_s.mean():>12.2f}{s.total_s.mean():>9.2f}"
          f"{full.classifier_s.mean() / s.classifier_s.mean():>13.2f}")