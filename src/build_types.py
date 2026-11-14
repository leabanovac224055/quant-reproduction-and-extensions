"""
Build a complete domain-label table for the 112 UCR datasets.

Pulls metadata from aeon, fills the two datasets aeon has no row for, and adds a
coarser grouping that merges the very small categories into usable sizes.

  python src/build_types.py   ->  results/dataset_types.csv
"""

import os
from pathlib import Path
import pandas as pd

from aeon.datasets import get_dataset_meta_data, tsc_datasets

ROOT = Path(os.environ.get("QUANT_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "results/dataset_types.csv"

# aeon has no metadata row for these two
MISSING = {
    "InsectWingbeatSound": "AUDIO",
    "MixedShapesRegularTrain": "IMAGE",
}

# merge the small categories so every group is big enough to say anything about
GROUPS = {
    "ECG": "PHYSIOLOGICAL",
    "HEMODYNAMICS": "PHYSIOLOGICAL",
    "EOG": "PHYSIOLOGICAL",
    "EPG": "PHYSIOLOGICAL",
    "MOTION": "HAR",
    "AUDIO": "OTHER",
    "TRAFFIC": "OTHER",
}


def main():
    names = sorted(tsc_datasets.univariate_equal_length)
    meta = get_dataset_meta_data(names)

    rows = meta[["Dataset", "Type", "TrainSize", "TestSize",
                 "Length", "NumberClasses"]].copy()

    # add the two aeon is missing
    for name, type_ in MISSING.items():
        if name not in set(rows["Dataset"]):
            rows = pd.concat([rows, pd.DataFrame([{
                "Dataset": name, "Type": type_,
                "TrainSize": None, "TestSize": None,
                "Length": None, "NumberClasses": None,
            }])], ignore_index=True)

    rows["Group"] = rows["Type"].replace(GROUPS)
    rows = rows.sort_values("Dataset").reset_index(drop=True)

    missing = set(names) - set(rows["Dataset"])
    if missing:
        print(f"still unlabelled: {sorted(missing)}")

    print(f"{len(rows)} datasets\n")
    print("raw types:")
    print(rows["Type"].value_counts().to_string())
    print("\ngrouped:")
    print(rows["Group"].value_counts().to_string())

    rows.to_csv(OUT, index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()