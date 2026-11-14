"""
SUPERSEDED. Significance tests for the v1 feature-selection run
(results/superseded/reduction_results_sqrt.csv), whose dataset list changed
mid-experiment and whose probe setting was chosen using datasets that also
appeared in the evaluation. Kept for the record; none of its numbers are
reported. The analysis that is reported lives in
run_reduction_v2.py --summarise.

Paired Wilcoxon on per-dataset means, since folds within a dataset are not
independent.

  python src/superseded/check_claims.py
"""

import os
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("QUANT_ROOT",
                           Path(__file__).resolve().parent.parent.parent))
RESULTS = ROOT / "results/superseded/reduction_results_sqrt.csv"

df = pd.read_csv(RESULTS).drop_duplicates(["dataset", "fold", "strategy", "budget"])
per = df.groupby(["dataset", "strategy", "budget"])["accuracy"].mean().reset_index()

full = per[per.strategy == "full"].set_index("dataset")["accuracy"]
fast = per[per.strategy == "fast"].set_index("dataset")["accuracy"]
budgets = sorted(b for b in per.budget.unique() if 0 < b < 1)


def test(a, b):
    pair = pd.concat([a, b], axis=1).dropna()
    pair.columns = ["a", "b"]
    d = pair.a - pair.b
    return d.mean(), wilcoxon(pair.a, pair.b).pvalue, len(d)


print(f"{len(full)} datasets")
print(f"full {full.mean():.4f}   fast {fast.mean():.4f}\n")

print(f"{'':<24}{'vs QUANT-FAST':>24}{'vs full model':>24}")
for strategy in ["importance", "balanced"]:
    for budget in budgets:
        sel = per[(per.strategy == strategy) & (per.budget == budget)]
        sel = sel.set_index("dataset")["accuracy"]

        d1, p1, n = test(sel, fast)
        d2, p2, _ = test(sel, full)

        v1 = "beats" if p1 < 0.05 and d1 > 0 else "loses" if p1 < 0.05 else "matches"
        v2 = "no loss" if p2 >= 0.05 else f"costs {-d2:.4f}"

        print(f"{strategy}@{budget:.1%}".ljust(24) +
              f"{d1:+.4f} p={p1:.3f} {v1}".rjust(24) +
              f"{d2:+.4f} p={p2:.3f} {v2}".rjust(24))

print(f"\n(n={n}, paired Wilcoxon on per-dataset means)")