"""
Three-way comparison for the QUANT reproduction.

  published   accuracies shipped in the QUANT repo
              (results/accuracy_resamples_ucr142.csv, 142 datasets)
  theirs      the authors' run.py, run here on all 142 datasets with the
              official Java resample indices
  mine        run_reproduction.py, an independent reimplementation using
              aeon's stratified resampling (112 datasets only)

The comparison covers every dataset in `theirs`. The `mine` column is empty for
the 30 datasets added by the bake-off redux, since run_reproduction.py covers
only the 112 equal-length UCR datasets; n_mine shows which those are.

The published/theirs gap tests whether the released code reproduces the
released numbers. Some difference is expected: run.py sets no random_state.

The theirs/mine gap combines two deliberate changes, the resampling scheme and
a fixed classifier seed, and does not separate them.

Dataset names differ between sources: our data folders use the Feb-2024
underscore names (AconityMINIPrinterLarge_eq) while the published CSV uses the
later camel-case ones (AconityMINIPrinterLargeEq). Same datasets, renamed.

All sources are assumed to have 30 resamples per dataset; the script warns if
not, since the means would otherwise not be comparable.

  python src/compare_three_way.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG ----

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))

# the published accuracies ship inside the QUANT repository, which is not
# redistributed here; clone it into Quant/ (see README)
PUBLISHED = ROOT / "Quant/quant-main/quant-main/results/accuracy_resamples_ucr142.csv"
THEIRS    = ROOT / "results/theirs142_results.csv"
MINE      = ROOT / "results/raw_results.csv"

OUT = ROOT / "results/three_way_comparison.csv"

N_RESAMPLES = 30

# the four datasets whose test sets the bake-off's published Correction
# revises, reported separately
CORRECTION = ["Covid3MonthDiscrete", "FloodModeling1Discrete",
              "FloodModeling2Discrete", "FloodModeling3Discrete"]

# -----------------------------------------------------------------------------


def camel(name):
    """Our folder names -> the published CSV's names."""
    return (name.replace("_eq", "Eq")
                .replace("_disc", "Discrete")
                .replace("_nmv", "Nmv"))


def load_published(path):
    df = pd.read_csv(path)
    out = df.set_index(df.columns[0])
    out.columns = range(len(out.columns))
    return out


def load_runs(path, label):
    """run.py writes dataset,resample,accuracy; so does run_reproduction.py."""
    df = pd.read_csv(path)
    df = df[["dataset", "resample", "accuracy"]].drop_duplicates(
        ["dataset", "resample"])
    return df.rename(columns={"accuracy": label})


def main():
    published = load_published(PUBLISHED)
    theirs = load_runs(THEIRS, "theirs")
    mine = load_runs(MINE, "mine")

    datasets = sorted(set(theirs["dataset"]))
    print(f"comparing {len(datasets)} datasets\n")

    rows = []
    for name in datasets:
        pub_name = camel(name)
        if pub_name not in published.index:
            print(f"  ! {name} ({pub_name}) missing from published results")
            continue

        pub_all = published.loc[pub_name].astype(float)
        t = theirs.loc[theirs["dataset"] == name, "theirs"]
        m = mine.loc[mine["dataset"] == name, "mine"]

        rows.append({
            "dataset": pub_name,
            "published": pub_all.mean(),
            "theirs": t.mean(),
            "mine": m.mean() if len(m) else np.nan,
            "n_theirs": len(t),
            "n_mine": len(m),
        })

    table = pd.DataFrame(rows)
    table["theirs-pub"] = table["theirs"] - table["published"]
    table["mine-pub"] = table["mine"] - table["published"]
    table["mine-theirs"] = table["mine"] - table["theirs"]

    # the means assume every source ran the same number of resamples
    odd = table[table["n_theirs"] != N_RESAMPLES]
    if len(odd):
        print(f"! {len(odd)} datasets do not have {N_RESAMPLES} resamples in "
              "`theirs`; their means are not directly comparable")
        print(odd[["dataset", "n_theirs"]].to_string(index=False))

    print("largest |theirs - published| differences")
    print("-" * 62)
    worst = table.reindex(
        table["theirs-pub"].abs().sort_values(ascending=False).index)
    print(worst.head(8)[["dataset", "published", "theirs", "theirs-pub"]]
          .round(4).to_string(index=False))

    # ---- headline, with and without the correction datasets ------------------

    clean = table[~table["dataset"].isin(CORRECTION)]

    for label, frame in [("all datasets", table),
                         (f"excluding the {len(CORRECTION)} Correction datasets",
                          clean)]:
        print(f"\n{label}  (n={len(frame)})")
        print("-" * 62)
        print(f"  published {frame.published.mean():.4f}   "
              f"theirs {frame.theirs.mean():.4f}   "
              f"mine {frame['mine'].mean():.4f}")
        for col in ["theirs-pub", "mine-pub", "mine-theirs"]:
            d = frame[col].abs()
            print(f"  {col:<14} mean |diff| {d.mean():.4f}   "
                  f"max {d.max():.4f}   n {d.notna().sum()}")

    print("\nReading it:")
    print("  theirs-pub  -> whether the released code reproduces the released")
    print("                 numbers. run.py sets no random_state, so some")
    print("                 difference is expected from classifier noise")
    print("  mine-theirs -> the combined effect of aeon resampling instead of")
    print("                 the official Java indices and of fixing a seed")
    print("\n  The four Correction datasets are reported separately because")
    print("  the bake-off revised their test sets after publication.")

    table.to_csv(OUT, index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()