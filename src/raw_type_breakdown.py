"""
The same comparison as domain_vs_baselines.py, but on aeon's raw Type labels
rather than the merged groups.

The merged PHYSIOLOGICAL group (ECG + HEMODYNAMICS + EOG + EPG) is where the
largest advantage showed up, so this checks whether the four subcategories
agree or whether one of them is carrying it.

Groups smaller than MIN_N are listed but flagged, since means over 2-3 datasets
say very little.

    python src/raw_type_breakdown.py   ->  results/raw_type_breakdown.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from aeon.benchmarking.published_results import (
    load_classification_bake_off_2023_results,
)

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

TYPES     = ROOT / "results/dataset_types.csv"
PUBLISHED = ROOT / "Quant/quant-main/quant-main/results/accuracy_resamples_ucr142.csv"
OUT       = ROOT / "results/raw_type_breakdown.csv"

BASELINES = ["DrCIF", "RSTSF", "CIF", "TSF"]
STRONG    = ["DrCIF", "RSTSF"]          # the two QUANT is closest to
MIN_N     = 5                            # below this, flag as unreliable

# the four categories merged into PHYSIOLOGICAL
PHYSIO = ["ECG", "HEMODYNAMICS", "EOG", "EPG"]

# -----------------------------------------------------------------------------


def load_published(path):
    df = pd.read_csv(path)
    out = df.set_index(df.columns[0])
    out.columns = range(len(out.columns))
    return out


def main():
    types = pd.read_csv(TYPES).set_index("Dataset")
    published = load_published(PUBLISHED)
    results = load_classification_bake_off_2023_results()

    frame = {"QUANT": published.mean(axis=1)}
    for m in BASELINES:
        frame[m] = pd.Series({d: np.mean(v) for d, v in results[m].items()})

    df = pd.DataFrame(frame).join(types[["Type", "Group"]], how="inner")
    df = df.dropna(subset=["QUANT"] + BASELINES)

    rows = []
    for type_, g in df.groupby("Type"):
        row = {"type": type_, "n": len(g), "quant": g["QUANT"].mean()}
        for m in BASELINES:
            row[f"vs_{m}"] = g["QUANT"].mean() - g[m].mean()
            row[f"wdl_{m}"] = (f"{int((g['QUANT'] > g[m]).sum())}/"
                               f"{int((g['QUANT'] == g[m]).sum())}/"
                               f"{int((g['QUANT'] < g[m]).sum())}")
        row["vs_strong"] = np.mean([row[f"vs_{m}"] for m in STRONG])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("vs_strong", ascending=False)

    print("QUANT vs baselines, by raw type")
    print("=" * 76)
    print(f"{'type':<15}{'n':>4}{'QUANT':>9}{'vsDrCIF':>10}{'vsRSTSF':>10}"
          f"{'vsCIF':>9}{'vsTSF':>9}{'':>3}")
    print("-" * 76)
    for _, r in table.iterrows():
        flag = "  *" if r["n"] < MIN_N else ""
        print(f"{r['type']:<15}{int(r['n']):>4}{r['quant']:>9.4f}"
              f"{r['vs_DrCIF']:>+10.4f}{r['vs_RSTSF']:>+10.4f}"
              f"{r['vs_CIF']:>+9.4f}{r['vs_TSF']:>+9.4f}{flag}")
    print(f"\n* fewer than {MIN_N} datasets - treat as indicative only")

    # ---- does the PHYSIOLOGICAL merge hold up? -----------------------------

    print("\n\nis the PHYSIOLOGICAL advantage uniform across its parts?")
    print("-" * 62)
    present = [p for p in PHYSIO if p in set(df["Type"])]
    sub = table[table["type"].isin(present)]

    for _, r in sub.sort_values("vs_strong", ascending=False).iterrows():
        print(f"  {r['type']:<15}n={int(r['n']):<4}"
              f"vs strong baselines: {r['vs_strong']:>+8.4f}   "
              f"DrCIF {r['wdl_DrCIF']}")

    merged = df[df["Type"].isin(present)]
    merged_diff = np.mean([
        merged["QUANT"].mean() - merged[m].mean() for m in STRONG])
    print(f"\n  merged (n={len(merged)}): {merged_diff:>+8.4f}")

    spread = sub["vs_strong"].max() - sub["vs_strong"].min()
    print(f"\n  spread across subcategories: {spread:.4f}")
    if spread > abs(merged_diff):
        print("  -> the subcategories disagree more than the merged effect size;")
        print("     the merge is doing real work and should be reported as a choice.")
    else:
        print("  -> the subcategories broadly agree; the merge is defensible.")

    # per-dataset detail for the physiological ones, since n is small
    print("\n\nevery physiological dataset:")
    print(f"  {'dataset':<28}{'type':<14}{'QUANT':>8}{'DrCIF':>8}{'diff':>9}")
    detail = df[df["Type"].isin(present)].sort_values(["Type", "QUANT"])
    for name, r in detail.iterrows():
        print(f"  {name:<28}{r['Type']:<14}{r['QUANT']:>8.4f}"
              f"{r['DrCIF']:>8.4f}{r['QUANT'] - r['DrCIF']:>+9.4f}")

    table.to_csv(OUT, index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()