"""Statistical comparison of models over repeated partitions."""
from __future__ import annotations
import itertools
import numpy as np
from scipy import stats


def wilcoxon_pair(a, b):
    """Two-sided Wilcoxon signed-rank test on paired scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0, 0.0
    try:
        w, p = stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(p), float(w)


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def paired_cohen_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def friedman(score_matrix):
    """score_matrix : (n_partitions, n_models). Returns statistic, p, mean ranks."""
    M = np.asarray(score_matrix, float)
    stat, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)
    return float(stat), float(p), ranks.mean(0)


def nemenyi_critical_difference(n_models, n_datasets, alpha=0.05):
    """Critical difference of the Nemenyi post-hoc test."""
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
         9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
         15: 3.391, 16: 3.426}.get(n_models, 3.45)
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(adj, 0, 1)
    return rej, out


def summarise(scores):
    a = np.asarray(scores, float)
    n = len(a)
    m, sd = a.mean(), a.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return dict(mean=float(m), sd=float(sd), ci_lo=float(m - half), ci_hi=float(m + half))
