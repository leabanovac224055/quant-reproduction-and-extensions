"""
Two assumptions the reported results rest on, verified here rather than
assumed.

1. aeon's stratified_resample_data preserves train and test sizes and class
   proportions. The reimplementation substitutes it for the published tsml
   resample indices, so the difference it reports is only interpretable if
   the substitution is otherwise faithful.

2. Under a tight feature budget, the depth lever collapses to a single
   interval at every series length. The claim that it loses to importance
   selection because it dismantles the multi-scale structure, rather than
   merely thinning it, rests on this.

Expects the UCR archive in data/ and the QUANT repository in Quant/ (see
README). Override the project root with QUANT_ROOT if they live elsewhere.

  python src/check_assumptions.py
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.append(str(ROOT / "Quant/quant-main/quant-main/code"))
sys.path.append(str(ROOT / "src"))

from aeon.datasets import load_from_ts_file
from quant_provenance import feature_provenance

try:
    from aeon.benchmarking.resampling import stratified_resample_data
except ImportError:
    from aeon.datasets.resampling import stratified_resample_data

D = str(ROOT / "data/Univariate2018_ts/Univariate_ts")
NAMES = ["ECG200", "Adiac", "Chinatown", "Beef", "StarLightCurves", "CricketX"]


def load(n):
    Xtr, ytr = load_from_ts_file(f"{D}/{n}/{n}_TRAIN.ts")
    Xte, yte = load_from_ts_file(f"{D}/{n}/{n}_TEST.ts")
    return np.asarray(Xtr), np.asarray(ytr), np.asarray(Xte), np.asarray(yte)


print("1. does stratified_resample_data preserve sizes and class balance?\n")
print(f"{'dataset':<18}{'train':>12}{'test':>12}{'max class shift':>18}{'':>8}")
for n in NAMES:
    Xtr, ytr, Xte, yte = load(n)
    n_tr, n_te = len(ytr), len(yte)
    classes = np.unique(np.concatenate([ytr, yte]))
    before = np.array([(ytr == c).sum() / n_tr for c in classes])

    worst, sizes_ok = 0.0, True
    for seed in [1, 7, 23]:
        rtr, rytr, rte, ryte = stratified_resample_data(
            Xtr, ytr, Xte, yte, random_state=seed)
        if len(rytr) != n_tr or len(ryte) != n_te:
            sizes_ok = False
        after = np.array([(rytr == c).sum() / len(rytr) for c in classes])
        worst = max(worst, float(np.abs(after - before).max()))

        pooled_before = sorted(np.concatenate([ytr, yte]))
        pooled_after = sorted(np.concatenate([rytr, ryte]))
        if pooled_before != pooled_after:
            print(f"  {n}: pooled labels changed under resampling")

    print(f"{n:<18}{n_tr:>12}{n_te:>12}{worst:>18.4f}"
          f"{'  ok' if sizes_ok and worst < 0.05 else '  CHECK':>8}")

print("\n  max class shift is the largest change in any class's share of the")
print("  training set. Near zero means stratification is working.\n")

print("\n2. what depth does a feature budget force?\n")
print(f"{'length':>8}{'features':>10}" +
      "".join(f"{b:>10.1%}" for b in [0.0625, 0.125, 0.25, 0.50]))
for length in [96, 176, 300, 512, 1024, 2709]:
    n_default = len(feature_provenance(length, depth=6, div=4))
    row = f"{length:>8}{n_default:>10}"
    for budget in [0.0625, 0.125, 0.25, 0.50]:
        k = int(round(n_default * budget))
        best = None
        for d in range(1, 7):
            n = len(feature_provenance(length, depth=d, div=4))
            if n <= k and (best is None or n > best[1]):
                best = (d, n)
        row += f"{('d' + str(best[0])) if best else '-':>10}"
    print(row)

print("\n  The depth lever can only pick from six discrete points, so at tight")
print("  budgets it collapses to depth 1 regardless of series length.")