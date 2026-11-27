"""Simplified reference detectors on the residual stream.

These follow the idea of each cited method but are not reimplementations of the published code.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from actground.probes import probe_scores


def folded_auroc(y, scores):
    auroc = roc_auc_score(y, scores)
    return max(auroc, 1 - auroc)


def saplma_style(X, y, folds):
    """Supervised logistic regression on one residual-stream layer (after Azaria & Mitchell, 2023).

    Returns the mean and standard deviation of the fold AUROCs.
    """
    scores = probe_scores(X, y, folds)
    return float(scores.mean()), float(scores.std())


def haloscope_style(X, y, n_directions=10):
    """Projection on a top singular direction (after Du et al., 2024).

    The direction is the one of the first `n_directions` with the highest AUROC, so labels are used for selection.
    """
    Xc = X - X.mean(0, keepdims=True)
    directions = np.linalg.svd(Xc, full_matrices=False)[2]
    aurocs = [folded_auroc(y, Xc @ directions[d]) for d in range(min(n_directions, len(directions)))]
    best = int(np.argmax(aurocs))
    return float(aurocs[best]), best


def ccs_style(X, y, seed, n_iter=300, lr=1e-2):
    """Unsupervised probe trained to be confident (after Burns et al., 2022).

    CCS proper needs contrast pairs. This single-view version only pushes predictions away from 0.5.
    """
    Xs = StandardScaler().fit_transform(X)
    w = np.random.default_rng(seed).normal(size=Xs.shape[1]) * 0.01
    b = 0.0
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        grad = (p - 0.5) * p * (1 - p)
        w -= lr * (-(Xs * grad[:, None]).mean(0) + 1e-3 * w)
        b -= lr * -grad.mean()
    return float(folded_auroc(y, 1 / (1 + np.exp(-(Xs @ w + b)))))
