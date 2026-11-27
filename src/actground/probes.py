"""Linear probes and neuron-set statistics used by the detection experiments."""

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler


def make_folds(seed):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def neuron_aurocs(X, y):
    """AUROC of each column on its own, folded to max(auc, 1 - auc).

    Uses the Mann-Whitney identity AUC = (R1 - n1(n1+1)/2) / (n1 n0) so all columns are scored at once.
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], 0.5)
    rank_sum = rankdata(X, axis=0)[y == 1].sum(axis=0)
    auc = (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    folded = np.maximum(auc, 1.0 - auc)
    folded[X.max(axis=0) == X.min(axis=0)] = 0.5
    return folded


def top_neurons(scores, k):
    return np.argsort(scores)[::-1][:k]


def probe_scores(X, y, folds, n_jobs=-1):
    """Per-fold AUROC of an L2 logistic regression on standardized features."""
    Xs = StandardScaler().fit_transform(X)
    return cross_val_score(LogisticRegression(max_iter=500), Xs, y, cv=folds, scoring="roc_auc", n_jobs=n_jobs)


def out_of_fold_probabilities(X, y, folds):
    Xs = StandardScaler().fit_transform(X)
    return cross_val_predict(LogisticRegression(max_iter=500), Xs, y, cv=folds, method="predict_proba", n_jobs=-1)[:, 1]


def random_subset_aurocs(X, y, k, n_draws, folds, rng):
    return np.array(
        [probe_scores(X[:, rng.choice(X.shape[1], size=k, replace=False)], y, folds).mean() for _ in range(n_draws)]
    )


def l1_probe(X, y, C, folds, seed):
    """L1 logistic regression over all neurons. Returns (mean AUROC, std, number of nonzero weights)."""
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    clf = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, max_iter=500, random_state=seed)
    scores = cross_val_score(clf, Xs, y, cv=folds, scoring="roc_auc", n_jobs=2)
    nonzero = int(np.sum(clf.fit(Xs, y).coef_[0] != 0))
    return float(scores.mean()), float(scores.std()), nonzero


def bootstrap_auroc_ci(scores, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aurocs.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    cross = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    norm = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(cross / norm) if norm > 0 else 0.0


def jaccard(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / len(a | b) if a | b else 0.0
