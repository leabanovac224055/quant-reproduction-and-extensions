
import numpy as np


def calculate_temporal_features(series):
    """
    Calculate temporal-order descriptors for one time-series interval.

    Returns:
        A numpy array containing temporal features.
    """

    x = np.asarray(series, dtype=np.float64).flatten()

    if len(x) < 2:
        return np.zeros(10, dtype=np.float32)

    # Remove non-finite values if present.
    x = np.where(np.isfinite(x), x, np.nan)

    if np.isnan(x).any():
        valid = np.flatnonzero(np.isfinite(x))

        if len(valid) < 2:
            return np.zeros(10, dtype=np.float32)

        x = np.interp(
            np.arange(len(x)),
            valid,
            x[valid]
        )

    n = len(x)

    # Differences between consecutive observations.
    diff = np.diff(x)

    # Identify upward, downward and unchanged movements.
    signs = np.sign(diff)

    up_count = np.sum(signs > 0)
    down_count = np.sum(signs < 0)

    up_ratio = up_count / max(len(diff), 1)
    down_ratio = down_count / max(len(diff), 1)

    # Ignore unchanged movements when detecting direction changes.
    nonzero_signs = signs[signs != 0]

    if len(nonzero_signs) > 1:
        direction_changes = np.sum(
            nonzero_signs[1:] != nonzero_signs[:-1]
        )

        direction_ratio = (
            direction_changes / (len(nonzero_signs) - 1)
        )
    else:
        direction_ratio = 0.0

    # Calculate increasing and decreasing run lengths.
    positive_runs = []
    negative_runs = []

    current_sign = 0
    current_length = 0

    for sign in signs:
        if sign == current_sign and sign != 0:
            current_length += 1

        else:
            if current_sign == 1 and current_length > 0:
                positive_runs.append(current_length)

            elif current_sign == -1 and current_length > 0:
                negative_runs.append(current_length)

            if sign == 0:
                current_sign = 0
                current_length = 0
            else:
                current_sign = sign
                current_length = 1

    if current_sign == 1:
        positive_runs.append(current_length)

    elif current_sign == -1:
        negative_runs.append(current_length)

    mean_increasing_run = (
        np.mean(positive_runs) / n if positive_runs else 0.0
    )

    mean_decreasing_run = (
        np.mean(negative_runs) / n if negative_runs else 0.0
    )

    # Count local peaks and valleys.
    peaks = np.sum(
        (x[1:-1] > x[:-2]) &
        (x[1:-1] > x[2:])
    )

    valleys = np.sum(
        (x[1:-1] < x[:-2]) &
        (x[1:-1] < x[2:])
    )

    peak_ratio = peaks / max(n - 2, 1)
    valley_ratio = valleys / max(n - 2, 1)

    # Calculate autocorrelation at lags 1, 2 and 3.
    autocorrelations = []

    centered = x - np.mean(x)
    denominator = np.sum(centered ** 2)

    for lag in [1, 2, 3]:

        if n <= lag or denominator < 1e-12:
            autocorrelations.append(0.0)

        else:
            numerator = np.sum(
                centered[:-lag] * centered[lag:]
            )

            autocorrelations.append(
                numerator / denominator
            )

    features = [
        up_ratio,
        down_ratio,
        direction_ratio,
        mean_increasing_run,
        mean_decreasing_run,
        peak_ratio,
        valley_ratio,
        autocorrelations[0],
        autocorrelations[1],
        autocorrelations[2]
    ]

    return np.asarray(features, dtype=np.float32)


def extract_temporal_features(X, intervals, feature_set="all"):
    """
    Extract temporal features from the same fixed intervals
    used by the original QUANT implementation.

    X shape: [number_of_samples, 1, time_series_length]

    feature_set:
        direction -> upward, downward and direction changes
        autocorrelation -> autocorrelation at lags 1, 2, 3
        all -> all ten descriptors
    """

    X = X.detach().cpu().numpy()

    intervals = intervals.detach().cpu().numpy()

    if feature_set == "direction":
        selected = [0, 1, 2]

    elif feature_set == "autocorrelation":
        selected = [7, 8, 9]

    elif feature_set == "all":
        selected = list(range(10))

    else:
        raise ValueError("Unknown feature_set")

    all_features = []

    for series in X:

        sample_features = []

        for start, end in intervals:

            interval = series[0, start:end]

            features = calculate_temporal_features(interval)

            sample_features.extend(features[selected])

        all_features.append(sample_features)

    return np.asarray(all_features, dtype=np.float32)