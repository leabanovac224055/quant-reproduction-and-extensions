"""
Does QUANT's advantage over other interval methods hold across domains?

Figure 1 of the paper reports a single win/draw/loss for QUANT against TSF,
STSF, rSTSF, CIF and DrCIF over all 112 datasets. This breaks that comparison
down by domain.

Baselines come from aeon's copy of the bake-off results, as
dict[method][dataset] -> array of 30 resample accuracies.

The means and win/draw/loss counts reproduce Fig. 1 exactly. The p-values
differ slightly (DrCIF 0.1595 vs 0.1442, rSTSF 0.0294 vs 0.0277) because the
Wilcoxon tests here use zero_method="wilcox", which drops zero differences;
the paper's values correspond to zero_method="pratt". The per-dataset table
used for the critical difference diagram is assembled in
domain_sensitivity.ipynb instead.

    python src/domain_vs_baselines.py   ->  results/domain_vs_baselines.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from aeon.benchmarking.published_results import (
    load_classification_bake_off_2023_results,
)

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

TYPES     = ROOT / "results/dataset_types.csv"
PUBLISHED = ROOT / "Quant/quant-main/quant-main/results/accuracy_resamples_ucr142.csv"
MINE      = ROOT / "results/raw_results.csv"
OUT       = ROOT / "results/domain_vs_baselines.csv"

# the interval methods QUANT is compared against in Fig. 1
WANTED = ["TSF", "STSF", "RSTSF", "rSTSF", "CIF", "DrCIF"]

# -----------------------------------------------------------------------------


def load_published(path):
    df = pd.read_csv(path)
    out = df.set_index(df.columns[0])
    out.columns = range(len(out.columns))
    return out


def resolve_methods(results):
    """Match the methods we want against whatever aeon actually calls them."""
    available = list(results.keys())
    lower = {k.lower(): k for k in available}

    found = {}
    for want in WANTED:
        if want.lower() in lower:
            found[want] = lower[want.lower()]

    missing = [w for w in WANTED if w not in found]
    if missing:
        print(f"not in aeon's results: {missing}")
        print(f"available: {available}\n")

    # de-duplicate RSTSF/rSTSF if both matched the same key
    seen, out = set(), {}
    for label, key in found.items():
        if key not in seen:
            out[label] = key
            seen.add(key)
    return out


def wdl(a, b):
    """win / draw / loss of a against b, over per-dataset means."""
    return int((a > b).sum()), int((a == b).sum()), int((a < b).sum())


def main():
    types = pd.read_csv(TYPES).set_index("Dataset")
    published = load_published(PUBLISHED)
    results = load_classification_bake_off_2023_results()
    methods = resolve_methods(results)

    mine = (pd.read_csv(MINE)
            .drop_duplicates(["dataset", "resample"])
            .groupby("dataset")["accuracy"].mean())

    # per-dataset mean accuracy for quant and each baseline
    frame = {"quant_published": published.mean(axis=1),
             "quant_mine": mine}
    for label, key in methods.items():
        frame[label] = pd.Series(
            {d: np.mean(v) for d, v in results[key].items()})

    df = pd.DataFrame(frame)
    df = df.join(types[["Type", "Group"]], how="inner").dropna(
        subset=["quant_published"] + list(methods))

    print(f"{len(df)} datasets with results for every method\n")

    # ---- overall, as a check against Fig. 1 --------------------------------

    print("overall (means and W/D/L match Fig. 1; p uses zero_method='wilcox')")
    print("-" * 64)
    print(f"{'method':<12}{'mean':>9}{'quant-them':>13}{'W/D/L':>14}{'p':>10}")
    q = df["quant_published"]
    for label in methods:
        b = df[label]
        w, d, l = wdl(q, b)
        p = wilcoxon(q, b, zero_method="wilcox").pvalue
        print(f"{label:<12}{b.mean():>9.4f}{q.mean() - b.mean():>+13.4f}"
              f"{f'{w}/{d}/{l}':>14}{p:>10.4f}")
    print(f"{'QUANT':<12}{q.mean():>9.4f}")

    # ---- by domain ---------------------------------------------------------

    rows = []
    for group, g in df.groupby("Group"):
        if len(g) < 4:
            continue
        gq = g["quant_published"]
        for label in methods:
            gb = g[label]
            w, d, l = wdl(gq, gb)
            try:
                p = wilcoxon(gq, gb, zero_method="wilcox").pvalue if len(g) >= 6 else np.nan
            except ValueError:
                p = np.nan
            rows.append({
                "group": group, "n": len(g), "baseline": label,
                "quant": gq.mean(), "baseline_acc": gb.mean(),
                "diff": gq.mean() - gb.mean(),
                "win": w, "draw": d, "loss": l,
                "p": p,
            })

    table = pd.DataFrame(rows)

    print("\n\nQUANT vs each interval method, by domain")
    print("=" * 72)
    for group, g in table.groupby("group"):
        n = g["n"].iloc[0]
        print(f"\n{group}  (n={n})")
        print(f"  {'baseline':<10}{'QUANT':>8}{'them':>8}{'diff':>9}"
              f"{'W/D/L':>12}{'p':>9}")
        for _, r in g.iterrows():
            wdl_s = f"{r['win']}/{r['draw']}/{r['loss']}"
            p_s = "-" if pd.isna(r["p"]) else f"{r['p']:.3f}"
            print(f"  {r['baseline']:<10}{r['quant']:>8.4f}"
                  f"{r['baseline_acc']:>8.4f}{r['diff']:>+9.4f}"
                  f"{wdl_s:>12}{p_s:>9}")

    # ---- the headline: where does the advantage live? ----------------------

    print("\n\nmean advantage over the two strongest baselines, by domain")
    print("-" * 58)
    strong = [m for m in ("DrCIF", "rSTSF", "RSTSF") if m in methods]
    pivot = (table[table["baseline"].isin(strong)]
             .groupby("group")[["diff", "n"]].mean()
             .sort_values("diff"))
    for group, r in pivot.iterrows():
        bar = "#" * max(1, int(abs(r["diff"]) * 1000))
        sign = "+" if r["diff"] >= 0 else "-"
        print(f"  {group:<15}{r['diff']:>+8.4f}  {sign}{bar}")

    table.to_csv(OUT, index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()