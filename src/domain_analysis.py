"""
Per-domain analysis of QUANT's results.

The paper reports a single mean over 112 datasets. This breaks that down by the
domain labels from timeseriesclassification.com, to see whether QUANT's
performance is uniform across domains or carried by particular ones.

Inputs:
  results/dataset_types.csv   - from build_types.py
  results/raw_results.csv     - from run_reproduction.py
  the published CSV           - from the QUANT repo (see README)

  python src/domain_analysis.py
      -> results/domain_summary.csv, results/domain_per_dataset.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

TYPES     = ROOT / "results/dataset_types.csv"
MINE      = ROOT / "results/raw_results.csv"
PUBLISHED = ROOT / "Quant/quant-main/quant-main/results/accuracy_resamples_ucr142.csv"
SUMMARY_OUT     = ROOT / "results/domain_summary.csv"
PER_DATASET_OUT = ROOT / "results/domain_per_dataset.csv"

# -----------------------------------------------------------------------------


def load_published(path):
    df = pd.read_csv(path)
    out = df.set_index(df.columns[0])
    out.columns = range(len(out.columns))
    return out


def try_baselines():
    """
    Report whether aeon's bake-off results are available in this
    installation. The per-dataset baseline comparison is done in
    domain_sensitivity.ipynb, which assembles them into
    results/results_with_baselines.csv; this only checks the loader works.
    """
    try:
        from aeon.benchmarking.published_results import (
            load_classification_bake_off_2023_results,
        )
        res = load_classification_bake_off_2023_results()
    except Exception as exc:
        print(f"(baselines unavailable: {exc})\n")
        return None

    print(f"baseline loader returned {type(res).__name__}", end="")
    if isinstance(res, tuple):
        print(f" of {len(res)} parts: {[type(p).__name__ for p in res]}")
    elif hasattr(res, "shape"):
        print(f" with shape {res.shape}")
    else:
        print()
    return res


def main():
    types = pd.read_csv(TYPES)
    published = load_published(PUBLISHED)
    raw = pd.read_csv(MINE).drop_duplicates(["dataset", "resample"])

    try_baselines()

    mine = raw.groupby("dataset")["accuracy"].agg(["mean", "std", "count"])
    mine.columns = ["my_mean", "my_std", "n_resamples"]

    rows = []
    for _, r in types.iterrows():
        name = r["Dataset"]
        if name not in published.index or name not in mine.index:
            continue

        pub = published.loc[name].astype(float).mean()
        rows.append({
            "dataset": name,
            "type": r["Type"],
            "group": r["Group"],
            "length": r["Length"],
            "train_size": r["TrainSize"],
            "n_classes": r["NumberClasses"],
            "published": pub,
            "mine": mine.loc[name, "my_mean"],
            "diff": mine.loc[name, "my_mean"] - pub,
        })

    per_dataset = pd.DataFrame(rows)
    print(f"{len(per_dataset)} datasets matched across all three sources\n")

    # ---- per-group summary -------------------------------------------------

    summary = []
    for group, g in per_dataset.groupby("group"):
        row = {
            "group": group,
            "n": len(g),
            "published": g["published"].mean(),
            "mine": g["mine"].mean(),
            "diff": g["diff"].mean(),
            "mean_abs_diff": g["diff"].abs().mean(),
            "worst": g.loc[g["published"].idxmin(), "dataset"],
            "worst_acc": g["published"].min(),
            "best_acc": g["published"].max(),
        }

        # is our reproduction systematically off within this group?
        if len(g) >= 6:
            try:
                row["wilcoxon_p"] = wilcoxon(g["mine"], g["published"]).pvalue
            except ValueError:
                row["wilcoxon_p"] = np.nan
        else:
            row["wilcoxon_p"] = np.nan

        summary.append(row)

    summary = pd.DataFrame(summary).sort_values("published")

    print("QUANT accuracy by domain (published, 30 resamples)")
    print("-" * 78)
    print(summary[["group", "n", "published", "mine", "diff",
                   "mean_abs_diff", "wilcoxon_p"]]
          .round(4).to_string(index=False))

    overall = per_dataset["published"].mean()
    print(f"\noverall published mean: {overall:.4f}")
    print("\nspread across domains: "
          f"{summary['published'].min():.4f} ({summary.iloc[0]['group']}) to "
          f"{summary['published'].max():.4f} "
          f"({summary.iloc[-1]['group']})")

    print("\nhardest dataset in each domain:")
    for _, r in summary.iterrows():
        print(f"  {r['group']:<15} {r['worst']:<28} {r['worst_acc']:.4f}")

    # ---- does our reproduction error depend on domain? ----------------------

    print("\nreproduction error by domain (|mine - published|):")
    for _, r in summary.sort_values("mean_abs_diff", ascending=False).iterrows():
        print(f"  {r['group']:<15} {r['mean_abs_diff']:.4f}  (n={int(r['n'])})")

    summary.to_csv(SUMMARY_OUT, index=False)
    per_dataset.to_csv(PER_DATASET_OUT, index=False)
    print(f"\nwritten to {SUMMARY_OUT} and {PER_DATASET_OUT}")


if __name__ == "__main__":
    main()