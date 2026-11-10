"""
Feature provenance for QUANT.

`Quant.fit_transform` returns a flat (n_cases, n_features) matrix with no record
of where each column came from. Interval-weighted feature sampling needs that
record: which representation, which depth, which interval, which quantile.

This module reconstructs the layout analytically by replaying the same interval
construction `quant.py` performs, then verifies the reconstruction against the
real transform.

    from quant_provenance import feature_provenance, verify

    prov = feature_provenance(series_length=150, depth=6, div=4)
    verify(X, depth=6, div=4)          # X: torch tensor (cases, channels, time)

Nothing here modifies quant.py; it is a parallel description of what it does.
"""

import numpy as np
import pandas as pd

# The four representations, and how each changes series length.
# Matches the lambdas in Quant.representation_functions:
#   X                                                   -> n
#   avg_pool1d(pad(X.diff(), (2,2), "replicate"), 5, 1)  -> n - 1
#   X.diff(n=2)                                          -> n - 2
#   rfft(X).abs()                                        -> n // 2 + 1
REPRESENTATIONS = [
    ("raw", lambda n: n),
    ("diff1_smoothed", lambda n: n - 1),
    ("diff2", lambda n: n - 2),
    ("fft_magnitude", lambda n: n // 2 + 1),
]


def interval_metadata(input_length, depth=6):
    """
    Replay make_intervals, recording what each interval is.

    Returns a list of dicts in the same order quant.py produces them, which is
    the order their features appear in the output matrix.
    """
    exponent = min(depth, int(np.log2(input_length)) + 1)
    out = []

    for level in range(exponent):
        n = 2**level
        indices = np.linspace(0, input_length, n + 1).astype(int)
        base = list(zip(indices[:-1], indices[1:]))

        for a, b in base:
            out.append({
                "depth": level + 1,
                "shifted": False,
                "start": int(a),
                "end": int(b),
                "length": int(b - a),
            })

        # quant.py adds a shifted copy when n > 1 and the median interval
        # length is more than 1; the last interval is dropped as it runs off
        # the end of the series.
        #
        # NOTE: quant.py uses torch.median, which for an even number of values
        # returns the LOWER of the two middle ones. np.median averages them.
        # On short series the interval lengths alternate 1, 2, 1, 2, ... and
        # the two conventions disagree (torch 1, numpy 1.5), which changes
        # whether the shifted set is added at all. Replicate torch here.
        lengths = np.sort(np.array([b - a for a, b in base]))
        torch_median = lengths[(len(lengths) - 1) // 2]
        if n > 1 and torch_median > 1:
            shift = int(np.ceil(input_length / n / 2))
            for a, b in base[:-1]:
                out.append({
                    "depth": level + 1,
                    "shifted": True,
                    "start": int(a + shift),
                    "end": int(b + shift),
                    "length": int(b - a),
                })

    return out


def n_quantiles(interval_length, div=4):
    """How many quantiles f_quantile returns for an interval of this length."""
    if interval_length == 1:
        return 1
    return max(1, 1 + (interval_length - 1) // div)


def feature_provenance(series_length, depth=6, div=4, n_channels=1):
    """
    One row per feature column, in the order the columns appear.

    Columns:
      column           position in the feature matrix
      representation   raw | diff1_smoothed | diff2 | fft_magnitude
      depth            1..6, the dyadic level the interval came from
      shifted          whether this is a half-offset interval
      start, end       interval bounds within that representation
      length           interval length in points (m in the paper)
      channel          channel index
      quantile_index   0..k-1 within the interval
      n_quantiles      k for this interval
      mean_subtracted  whether the interval mean was subtracted (odd indices)
    """
    rows = []
    column = 0

    for rep_name, length_fn in REPRESENTATIONS:
        rep_length = length_fn(series_length)
        if rep_length < 1:
            raise ValueError(
                f"series_length={series_length} too short for {rep_name}")

        for interval in interval_metadata(rep_length, depth=depth):
            m = interval["length"]
            k = n_quantiles(m, div=div)

            # quant.py reshapes (cases, channels, k) -> (cases, channels * k),
            # so channel is the outer index and quantile the inner one
            for channel in range(n_channels):
                for qi in range(k):
                    rows.append({
                        "column": column,
                        "representation": rep_name,
                        "depth": interval["depth"],
                        "shifted": interval["shifted"],
                        "start": interval["start"],
                        "end": interval["end"],
                        "length": m,
                        "channel": channel,
                        "quantile_index": qi,
                        "n_quantiles": k,
                        # mean is subtracted from every second quantile, and
                        # only where m >= 2 and k >= 2
                        "mean_subtracted": bool(m >= 2 and k >= 2 and qi % 2 == 1),
                    })
                    column += 1

    return pd.DataFrame(rows)


def verify(X, depth=6, div=4, quant_code_path=None, verbose=True):
    """
    Check the reconstruction against the real transform.

    Compares the predicted number of columns against what Quant actually
    produces, and spot-checks one feature value by recomputing it by hand.

    X: torch tensor, shape (n_cases, n_channels, n_timepoints)
    """
    import sys

    if quant_code_path and quant_code_path not in sys.path:
        sys.path.append(quant_code_path)

    import torch
    from quant import Quant

    n_cases, n_channels, n_time = X.shape

    transform = Quant(depth=depth, div=div)
    Z = transform.fit_transform(X, None)

    prov = feature_provenance(n_time, depth=depth, div=div,
                              n_channels=n_channels)

    ok = len(prov) == Z.shape[1]
    if verbose:
        print(f"predicted columns: {len(prov)}")
        print(f"actual columns:    {Z.shape[1]}")
        print(f"match: {ok}")

    if not ok:
        raise AssertionError(
            f"layout mismatch: predicted {len(prov)}, got {Z.shape[1]}. "
            "The reconstruction is wrong; do not use it."
        )

    # spot-check: the first column should be a quantile of the depth-1 raw
    # interval, i.e. of the whole series
    first = prov.iloc[0]
    assert first["representation"] == "raw" and first["depth"] == 1, first

    m = first["length"]
    k = int(first["n_quantiles"])
    expected = torch.quantile(
        X[0, 0, first["start"]:first["end"]],
        torch.linspace(0, 1, k)[0],
    )
    got = Z[0, 0]

    close = bool(torch.isclose(expected, got, atol=1e-4))
    if verbose:
        print(f"\nspot-check column 0 (raw, depth 1, quantile 0):")
        print(f"  recomputed: {float(expected):.6f}")
        print(f"  transform:  {float(got):.6f}")
        print(f"  match: {close}")

    if verbose:
        print("\nfeatures per depth (all representations, all intervals):")
        summary = prov.groupby("depth").agg(
            n_features=("column", "count"),
            mean_interval_len=("length", "mean"),
        )
        print(summary.to_string())

        print("\nfeatures per representation:")
        print(prov["representation"].value_counts().to_string())

    return prov, Z


if __name__ == "__main__":
    # quick self-test with no data: just print the layout for a 150-point series
    prov = feature_provenance(150)
    print(f"150-point series -> {len(prov)} features\n")
    print(prov.groupby(["representation", "depth"]).size().to_string())
    print(f"\nshare of features by depth:")
    share = prov.groupby("depth").size() / len(prov)
    for depth, frac in share.items():
        print(f"  depth {depth}: {frac:6.1%}  {'#' * int(frac * 100)}")
