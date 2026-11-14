"""
Does results/reproduce_fig8.py compute the same thing as the reproduction harness?

Both runs include QUANT at its defaults (depth 6, m/4). If the Fig. 8 script
were computing accuracy wrongly, that configuration would disagree with the
reproduction run on the datasets they share.

The two use different protocols - 5-fold CV on pooled data vs 30 resamples of
the original split - so they will not match exactly. CV trains on 80% of the
pooled data, which is usually more than the shipped training set, so the CV
figure should be equal or higher on most datasets. A large systematic gap in
the other direction would indicate a problem.

  python src/check_fig8_consistency.py
"""

import os
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

fig8 = pd.read_csv(ROOT / "results/fig8_results.csv")
repro = pd.read_csv(ROOT / "results/raw_results.csv")

a = (fig8[(fig8["fold"] >= 0) & (fig8["depth"] == 6) & (fig8["div"] == 4)]
     .groupby("dataset")["accuracy"].mean())
b = repro.drop_duplicates(["dataset", "resample"]).groupby("dataset")["accuracy"].mean()

both = pd.DataFrame({"fig8_cv": a, "repro_30resamples": b}).dropna()
both["diff"] = both["fig8_cv"] - both["repro_30resamples"]

print(f"{len(both)} datasets in both runs\n")
print(f"  fig8 (5-fold CV)        {both.fig8_cv.mean():.4f}")
print(f"  repro (30 resamples)    {both.repro_30resamples.mean():.4f}")
print(f"  mean difference         {both['diff'].mean():+.4f}")
print(f"  mean |difference|       {both['diff'].abs().mean():.4f}")
print(f"  higher on CV            {(both['diff'] > 0).sum()} of {len(both)}")
print(f"  Wilcoxon p              {wilcoxon(both.fig8_cv, both.repro_30resamples).pvalue:.4f}")

print("\nlargest disagreements")
print(both.reindex(both["diff"].abs().sort_values(ascending=False).index)
      .head(8).round(4).to_string())

print("\nA positive mean difference is expected: CV trains on 80% of the pooled")
print("data, which is more than most of these datasets ship as a training set.")
print("A negative mean, or outliers of several points, would be a warning.")