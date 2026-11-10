"""
Interval-weighted feature sampling for QUANT.

Sect. 4.2.1 of the paper observes that with few quantiles per interval, accuracy
falls as depth increases past 6, and suggests a fix it never tests:

    It may be possible to counteract this effect by sampling features from
    larger intervals with higher probability when training the classifier.
    We leave this for future work.

This module implements that. scikit-learn's `max_features` samples candidate
features uniformly at each split, with no hook for per-feature weights, so the
weighting has to be engineered around it. Two strategies are provided:

  "duplicate"  Replicate columns from large intervals, so that uniform
               sampling over the widened matrix is weighted sampling over the
               original. Exact, works with stock sklearn, costs memory.

  "per_tree"   Fit many small forests, each on a feature subset drawn to match
               the target depth shares, and average their probabilities.
               Cheaper, but moves the weighting from per-split to per-tree,
               which is a different algorithm.

CAVEAT on "duplicate": a duplicated column is not only more likely to be drawn
as a candidate, it also receives more random split points when drawn more than
once in the same split (extremely randomised trees draw one threshold per
candidate). So duplication raises both selection probability and threshold
quality. The two effects cannot be separated within stock sklearn. Comparing
"duplicate" against "per_tree", which has no such confound, is part of the
experiment rather than an afterthought.

NOTE on how "per_tree" draws its subsets. An earlier version used
`rng.choice(n, size=k, replace=False, p=weights)`. That is wrong for this
purpose: weighted sampling *without replacement* draws sequentially and
renormalises the remaining weights after each draw, so a column's marginal
chance of ending up in the subset is not its weight. Extreme weights are
compressed toward uniform - for a scheme with an 89:1 weight ratio the
shallowest depth received 44% of the subset where 51% was intended. The subset
is now drawn by quota instead: exactly round(share_d * k) columns are taken
uniformly at random from within each depth, which realises the target shares to
within integer rounding. `achieved_shares()` reports what was actually drawn so
this can be checked rather than assumed.
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from quant_provenance import feature_provenance

# ---------------------------------------------------------------- schemes ----


def depth_weights(prov, scheme="equal", alpha=1.0):
    """
    Desired relative sampling weight for each depth.

    "none"    current behaviour: weight proportional to how many features a
              depth happens to have, i.e. no reweighting at all
    "equal"   every depth gets the same total probability mass
    "length"  weight proportional to mean interval length ** alpha, so larger
              intervals are favoured in proportion to how large they are
    "linear"  weight proportional to (7 - depth), a gentler version of "equal"
    """
    depths = sorted(prov["depth"].unique())
    counts = prov.groupby("depth").size()

    if scheme == "none":
        return {d: counts[d] / counts.sum() for d in depths}

    if scheme == "equal":
        raw = {d: 1.0 for d in depths}
    elif scheme == "length":
        means = prov.groupby("depth")["length"].mean()
        raw = {d: float(means[d]) ** alpha for d in depths}
    elif scheme == "linear":
        top = max(depths) + 1
        raw = {d: float(top - d) for d in depths}
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    total = sum(raw.values())
    return {d: raw[d] / total for d in depths}


def duplication_factors(prov, scheme="equal", alpha=1.0, max_factor=128,
                        precision=4):
    """
    Integer duplication count per depth that moves the feature-mass
    distribution towards the target weights.

    Depths already over-represented get the smallest factor; the rest are
    scaled up. `precision` multiplies every factor before rounding, which buys
    resolution (the factors are small integers, so without it `equal` lands
    several points away from its target) at the cost of a wider matrix.

    Returns (factors, capped), where `capped` lists any depth whose required
    factor exceeded max_factor. A capped scheme is not the scheme you asked
    for: the cap is shaping the result, so either raise it, lower `alpha`, or
    use strategy="per_tree", which has no cap.
    """
    target = depth_weights(prov, scheme=scheme, alpha=alpha)
    counts = prov.groupby("depth").size()
    current = counts / counts.sum()

    # how much each depth needs to grow, normalised so the smallest is 1
    ratio = {d: target[d] / current[d] for d in target}
    smallest = min(ratio.values())
    wanted = {d: ratio[d] / smallest * precision for d in ratio}

    capped = [d for d, w in wanted.items() if w > max_factor]
    factors = {d: max(1, min(max_factor, int(round(w))))
               for d, w in wanted.items()}
    return factors, capped


def expansion_index(prov, scheme="equal", alpha=1.0, max_factor=128,
                    precision=4, warn=True):
    """
    Column indices to build the widened matrix: X[:, index].

    Returns (index, factors, achieved_shares).
    """
    factors, capped = duplication_factors(
        prov, scheme=scheme, alpha=alpha,
        max_factor=max_factor, precision=precision)

    if capped and warn:
        print(f"  ! scheme '{scheme}' capped at depth(s) {capped}; "
              f"achieved shares will not match the target "
              f"(raise max_factor, lower alpha, or use per_tree)")

    depth_of = prov["depth"].to_numpy()

    index = np.concatenate([
        np.repeat(np.flatnonzero(depth_of == d), factors[d])
        for d in sorted(factors)
    ])

    expanded_depths = depth_of[index]
    achieved = {d: float((expanded_depths == d).mean()) for d in sorted(factors)}
    return index, factors, achieved


def target_vs_achieved(prov, scheme="equal", alpha=1.0, max_factor=128,
                       precision=4):
    """Max absolute gap between the target depth shares and what duplication
    actually delivers. Use it to check a configuration before running it."""
    target = depth_weights(prov, scheme=scheme, alpha=alpha)
    _, _, achieved = expansion_index(prov, scheme, alpha, max_factor,
                                     precision, warn=False)
    return max(abs(target[d] - achieved[d]) for d in target)


def depth_quotas(target, available, n_cols):
    """
    How many columns to draw from each depth so the subset matches `target`.

    `available` maps depth -> number of columns that depth has. Quotas are
    floor(share * n_cols), capped at availability, with the remainder handed to
    whichever depths are furthest below their fractional entitlement. The
    result always sums to n_cols unless every depth is exhausted.
    """
    depths = sorted(target)
    quota = {d: min(int(np.floor(target[d] * n_cols)), available[d])
             for d in depths}

    for _ in range(10):
        short = n_cols - sum(quota.values())
        if short == 0:
            break
        if short > 0:
            room = [d for d in depths if quota[d] < available[d]]
            if not room:
                break
            order = sorted(room, key=lambda d: quota[d] - target[d] * n_cols)
            for d in order[:short]:
                quota[d] += 1
        else:
            room = [d for d in depths if quota[d] > 0]
            if not room:
                break
            order = sorted(room, key=lambda d: target[d] * n_cols - quota[d])
            for d in order[:-short]:
                quota[d] -= 1

    return quota


# ------------------------------------------------------------- classifier ----


class WeightedQuant:
    """
    QUANT's classifier stage with interval-weighted feature sampling.

    Takes an already-transformed feature matrix plus its provenance table.
    The transform itself is untouched.
    """

    def __init__(self, prov, scheme="equal", strategy="duplicate",
                 alpha=1.0, max_factor=128, precision=4, n_estimators=200,
                 max_features=0.1, n_subforests=20, random_state=None,
                 n_jobs=-1, memory_limit_mb=2000):
        self.prov = prov
        self.scheme = scheme
        self.strategy = strategy
        self.alpha = alpha
        self.max_factor = max_factor
        self.precision = precision
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_subforests = n_subforests
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.memory_limit_mb = memory_limit_mb

        self.index_ = None
        self.factors_ = None
        self.achieved_ = None
        self.models_ = None
        self.subsets_ = None
        self.quotas_ = None
        self.classes_ = None

    # -- baseline path -------------------------------------------------------

    def _plain_forest(self, seed):
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            criterion="entropy",
            n_jobs=self.n_jobs,
            random_state=seed,
        )

    # -- fit -----------------------------------------------------------------

    def fit(self, X, y):
        # scheme="none" means "do not reweight". With strategy="baseline" that
        # is stock QUANT; with strategy="per_tree" it is the per-tree ensemble
        # under uniform weights, which is the control needed to separate the
        # effect of weighting from the effect of per-tree ensembling.
        # Reweighting nothing under duplication is a no-op, so that combination
        # falls through to baseline.
        if self.scheme == "none" and self.strategy == "duplicate":
            self.strategy = "baseline"

        if self.strategy == "baseline":
            self.models_ = [self._plain_forest(self.random_state).fit(X, y)]
            self.classes_ = self.models_[0].classes_
            return self

        if self.strategy == "duplicate":
            self.index_, self.factors_, self.achieved_ = expansion_index(
                self.prov, self.scheme, self.alpha, self.max_factor,
                self.precision)

            projected_mb = X.shape[0] * len(self.index_) * 4 / 1e6
            if projected_mb > self.memory_limit_mb:
                raise MemoryError(
                    f"widened matrix would be ~{projected_mb:.0f} MB "
                    f"({X.shape[0]} x {len(self.index_)}); "
                    f"limit is {self.memory_limit_mb} MB. "
                    "Lower max_factor or use strategy='per_tree'."
                )

            Xw = np.asarray(X)[:, self.index_]
            self.models_ = [self._plain_forest(self.random_state).fit(Xw, y)]
            self.classes_ = self.models_[0].classes_
            return self

        if self.strategy == "per_tree":
            rng = np.random.default_rng(self.random_state)
            target = depth_weights(self.prov, self.scheme, self.alpha)
            depth_of = self.prov["depth"].to_numpy()[:X.shape[1]]
            by_depth = {d: np.flatnonzero(depth_of == d) for d in target}

            n_cols = max(1, int(self.max_features * X.shape[1]))
            self.quotas_ = depth_quotas(
                target, {d: len(v) for d, v in by_depth.items()}, n_cols)
            per_forest = max(1, self.n_estimators // self.n_subforests)

            X = np.asarray(X)
            self.models_, self.subsets_ = [], []
            for i in range(self.n_subforests):
                # draw each depth's quota uniformly from within that depth, so
                # the subset's depth shares match the target by construction
                cols = np.sort(np.concatenate([
                    rng.choice(by_depth[d], size=self.quotas_[d], replace=False)
                    for d in sorted(self.quotas_) if self.quotas_[d] > 0
                ]))

                clf = ExtraTreesClassifier(
                    n_estimators=per_forest,
                    max_features=1.0,          # subset already drawn
                    criterion="entropy",
                    n_jobs=self.n_jobs,
                    random_state=None if self.random_state is None
                    else self.random_state + i,
                )
                clf.fit(X[:, cols], y)
                self.models_.append(clf)
                self.subsets_.append(cols)

            self.classes_ = self.models_[0].classes_
            return self

        raise ValueError(f"unknown strategy: {self.strategy}")

    def achieved_shares(self):
        """
        The depth shares the fitted model actually used, so the weighting can
        be checked rather than assumed. Returns None for the baseline.
        """
        depth_of = self.prov["depth"].to_numpy()
        depths = sorted(self.prov["depth"].unique())

        if self.strategy == "baseline":
            return None
        if self.strategy == "duplicate":
            return dict(self.achieved_)

        pooled = np.concatenate(self.subsets_)
        d = depth_of[pooled]
        return {k: float((d == k).mean()) for k in depths}

    # -- predict -------------------------------------------------------------

    def predict_proba(self, X):
        X = np.asarray(X)

        if self.strategy == "baseline":
            return self.models_[0].predict_proba(X)

        if self.strategy == "duplicate":
            return self.models_[0].predict_proba(X[:, self.index_])

        probas = [m.predict_proba(X[:, cols])
                  for m, cols in zip(self.models_, self.subsets_)]
        return np.mean(probas, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


# ------------------------------------------------------------------ demo ----

if __name__ == "__main__":
    import pandas as pd

    SERIES_LENGTH = 150
    SCHEMES = ["equal", "length", "linear"]

    for div, label in [(4, "m/4 (default)"), (8, "m/8"), (16, "m/16")]:
        prov = feature_provenance(SERIES_LENGTH, depth=6, div=div)
        current = prov.groupby("depth").size() / len(prov)

        print(f"\n{'=' * 78}")
        print(f"{label}: {len(prov)} features")
        print("  as built:  " +
              "  ".join(f"d{d} {v:5.1%}" for d, v in current.items()))

        print("\n  targets:")
        for scheme in SCHEMES:
            t = depth_weights(prov, scheme)
            print(f"    {scheme:<8}" +
                  "  ".join(f"d{d} {v:5.1%}" for d, v in t.items()))

        print("\n  achieved by duplication:")
        rows = []
        for scheme in SCHEMES:
            target = depth_weights(prov, scheme)
            index, _, achieved = expansion_index(prov, scheme)
            rows.append({
                "scheme": scheme,
                **{f"d{d}": f"{achieved[d]:.1%}" for d in sorted(achieved)},
                "max gap": f"{max(abs(target[d] - achieved[d]) for d in target):.1%}",
                "width": f"{len(index) / len(prov):.1f}x",
            })
        print(pd.DataFrame(rows).to_string(index=False))

        print("\n  achieved by per-tree quota sampling:")
        rows = []
        depth_of = prov["depth"].to_numpy()
        by_depth = {d: np.flatnonzero(depth_of == d)
                    for d in sorted(prov["depth"].unique())}
        n_cols = max(1, int(0.1 * len(prov)))
        for scheme in SCHEMES:
            target = depth_weights(prov, scheme)
            quota = depth_quotas(target, {d: len(v) for d, v in by_depth.items()},
                                 n_cols)
            achieved = {d: quota[d] / sum(quota.values()) for d in quota}
            rows.append({
                "scheme": scheme,
                **{f"d{d}": f"{achieved[d]:.1%}" for d in sorted(achieved)},
                "max gap": f"{max(abs(target[d] - achieved[d]) for d in target):.1%}",
                "cols": sum(quota.values()),
            })
        print(pd.DataFrame(rows).to_string(index=False))
