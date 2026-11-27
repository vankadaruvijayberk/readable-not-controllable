"""Bootstrap intervals and paired tests for the intervention analysis."""

import numpy as np
from scipy import stats


def finite(values):
    return np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)


def _percentile_interval(draws):
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean_ci(values, n_boot, rng):
    a = finite(values)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(1)
    return (float(a.mean()), *_percentile_interval(draws))


def bootstrap_diff_ci(x, y, n_boot, rng):
    a, b = finite(x), finite(y)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    draws_a = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(1)
    draws_b = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(1)
    return (float(a.mean() - b.mean()), *_percentile_interval(draws_a - draws_b))


def mcnemar_p(b, c):
    """McNemar test on discordant counts: exact binomial below 25 pairs, continuity-corrected chi-square otherwise."""
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        return float(min(2.0 * stats.binom.cdf(min(b, c), n, 0.5), 1.0))
    return float(stats.chi2.sf((abs(b - c) - 1.0) ** 2 / n, 1))


def holm(p_values):
    """Holm step-down adjusted p-values, in the input order. NaN entries stay NaN."""
    p = np.asarray([np.nan if v is None else v for v in p_values], dtype=float)
    m = int(np.sum(~np.isnan(p)))
    adjusted = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(i for i in np.argsort(p) if not np.isnan(p[i])):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()
