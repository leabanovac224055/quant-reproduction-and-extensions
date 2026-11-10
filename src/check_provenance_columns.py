"""
Does quant_provenance map EVERY column correctly, or only the first?

verify() in quant_provenance.py checks the total column count and recomputes
column 0. That constrains the total and one endpoint, but not the ordering: a
mapping could be right at column 0 and wrong at column 900.

This recomputes a sample of columns spread across all four representations,
every depth, both shifted and unshifted intervals, and several quantile
indices, and compares each against what Quant actually produced.

This verifies the INDEXING, not the representations: expected_value uses the
same four representation functions as quant.py, so a fault in those would be
reproduced rather than caught. What it establishes is that column c holds the
value the provenance table says it should.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "D:/AAI/TimeSeries/QuantProject"
sys.path.append(f"{ROOT}/Quant/quant-main/quant-main/code")
sys.path.append(ROOT)

from quant import Quant
from aeon.datasets import load_from_ts_file
from quant_provenance import feature_provenance

DATA = f"{ROOT}/data/Univariate2018_ts/Univariate_ts"
DATASETS = ["GunPoint", "ECG200", "Chinatown", "Adiac",
            "StarLightCurves", "HandOutlines"]
N_CASES = 20          # enough to compare values, small enough to be quick
TOL = 1e-4

REPRESENTATIONS = {
    "raw": lambda X: X,
    "diff1_smoothed": lambda X: F.avg_pool1d(
        F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
    "diff2": lambda X: X.diff(n=2),
    "fft_magnitude": lambda X: torch.fft.rfft(X).abs(),
}


def expected_value(X, row):
    """Recompute one feature from the raw data, following the paper."""
    Z = REPRESENTATIONS[row["representation"]](X)
    seg = Z[:, row["channel"]:row["channel"] + 1, row["start"]:row["end"]]

    m, k, qi = int(row["length"]), int(row["n_quantiles"]), int(row["quantile_index"])

    if m == 1:
        return seg[:, 0, 0]
    if k == 1:
        return seg.quantile(torch.tensor([0.5]), dim=-1)[0, :, 0]

    q = seg.quantile(torch.linspace(0, 1, k), dim=-1).permute(1, 2, 0)
    if m >= 2 and k >= 2:
        q[..., 1::2] = q[..., 1::2] - seg.mean(-1, keepdims=True)
    return q[:, 0, qi]


def sample_columns(prov, rng, per_cell=2):
    """At least a couple of columns from every (representation, depth,
    shifted) combination, so nothing is checked only at its easy end."""
    picked = []
    for _, cell in prov.groupby(["representation", "depth", "shifted"]):
        take = min(per_cell, len(cell))
        picked.extend(rng.choice(cell["column"].to_numpy(), take, replace=False))
    # plus the extremes, which are the most likely to be off by one
    picked += [prov["column"].min(), prov["column"].max()]
    return sorted(set(int(c) for c in picked))


rng = np.random.default_rng(0)
total_checked = total_bad = 0

for name in DATASETS:
    X, _ = load_from_ts_file(f"{DATA}/{name}/{name}_TRAIN.ts")
    X = torch.tensor(np.asarray(X)[:N_CASES], dtype=torch.float32)
    length = X.shape[-1]

    Z = Quant(depth=6, div=4).fit_transform(X, None)
    prov = feature_provenance(length, depth=6, div=4, n_channels=X.shape[1])

    if len(prov) != Z.shape[1]:
        print(f"{name:<20} COUNT MISMATCH {len(prov)} vs {Z.shape[1]}")
        total_bad += 1
        continue

    cols = sample_columns(prov, rng)
    bad = []
    for c in cols:
        row = prov.iloc[c]
        got = Z[:, c]
        want = expected_value(X, row)
        if not torch.allclose(want, got, atol=TOL):
            bad.append((c, row["representation"], int(row["depth"]),
                        bool(row["shifted"]), int(row["quantile_index"]),
                        float((want - got).abs().max())))

    total_checked += len(cols)
    total_bad += len(bad)
    status = "ok" if not bad else f"{len(bad)} WRONG"
    print(f"{name:<20} length {length:>5}  {len(prov):>6} cols  "
          f"checked {len(cols):>3}  {status}")
    for b in bad[:5]:
        print(f"    col {b[0]}: {b[1]} depth {b[2]} shifted {b[3]} "
              f"q{b[4]}  max diff {b[5]:.2e}")

print(f"\n{total_checked} columns checked across {len(DATASETS)} datasets, "
      f"{total_bad} wrong")
if total_bad == 0:
    print("the mapping holds at every representation, depth and shift tested")
