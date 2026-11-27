"""Detection experiments for one model: layer scan, baselines, cross-position structure, transfer, calibration.

Writes <out>/<model>.json. The top-level keys follow the saved result files in results/detection
and are described in docs/RESULTS.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from actground import baselines, probes
from actground.activations import POSITIONS, VIEWS, build_layer_store, open_activations, read_labels, reliable_mask
from actground.config import add_config_arg, load_config

PRIMARY_VIEW = "exact_cett"


def dataset_summary(labels, y):
    per_dataset = {
        name: {
            "n": int(len(group)),
            "hall_rate": float(group["hallucinated"].mean()),
            "n_positives": int(group["hallucinated"].sum()),
        }
        for name, group in labels.groupby("source_dataset")
    }
    return {
        "n_samples_after_filter": int(len(labels)),
        "overall_hall_rate": float(y.mean()),
        "per_dataset": per_dataset,
    }


def baseline_table(labels, residual, residual_layer, y, folds, seed):
    saplma_mean, saplma_std = baselines.saplma_style(residual, y, folds)
    haloscope_auroc, haloscope_direction = baselines.haloscope_style(residual, y)
    return {
        "token_neg_prob": {"auroc": float(roc_auc_score(y, -labels["first_tok_prob"].values))},
        "token_entropy": {"auroc": float(roc_auc_score(y, labels["first_tok_entropy"].values))},
        "saplma_style_resid": {"auroc": saplma_mean, "auroc_std": saplma_std, "ref_layer": residual_layer},
        "haloscope_style": {"auroc": haloscope_auroc, "best_dir": haloscope_direction, "ref_layer": residual_layer},
        "ccs_style": {"auroc": baselines.ccs_style(residual, y, seed), "ref_layer": residual_layer},
    }


def layer_scan(store, y, ks, folds):
    """Top-k probe AUROC at every layer. Returns the scan, per-neuron AUROCs and the best (layer, k) per condition."""
    scan, neuron_scores, best = {}, {}, {}
    for view in VIEWS:
        for position in POSITIONS:
            for layer in tqdm(range(store.n_layers), desc=f"{view}/{position}"):
                X = store.matrix(view, position, layer)
                scores = probes.neuron_aurocs(X, y)
                neuron_scores[view, position, layer] = scores
                for k in ks:
                    fold_aurocs = probes.probe_scores(X[:, probes.top_neurons(scores, k)], y, folds)
                    scan[view, position, layer, k] = (float(fold_aurocs.mean()), float(fold_aurocs.std()))
            cells = [(layer, k) for layer in range(store.n_layers) for k in ks]
            best_layer, best_k = max(cells, key=lambda cell: scan[view, position, *cell][0])
            mean, std = scan[view, position, best_layer, best_k]
            best[view, position] = {"layer": best_layer, "k": best_k, "auroc": mean, "auroc_std": std}
    return scan, neuron_scores, best


def best_condition_detail(store, neuron_scores, best, y, folds, cfg, seed):
    """Bootstrap CI, random-subset comparison and L1 probe at the best layer of each view and position."""
    detail = {}
    for (view, position), cond in best.items():
        X = store.matrix(view, position, cond["layer"])
        top = probes.top_neurons(neuron_scores[view, position, cond["layer"]], cond["k"])
        ci = probes.bootstrap_auroc_ci(
            probes.out_of_fold_probabilities(X[:, top], y, folds), y, cfg["n_bootstrap"], seed
        )
        random_aurocs = probes.random_subset_aurocs(
            X, y, cond["k"], cfg["n_random_draws"], folds, np.random.default_rng(cfg["random_subset_seed"])
        )
        l1_mean, l1_std, l1_nonzero = probes.l1_probe(X, y, cfg["l1_C"], folds, seed)
        detail[f"{view}_{position}"] = {
            "best_auroc": cond["auroc"],
            "ci95": list(ci),
            "random_mean": float(random_aurocs.mean()),
            "random_std": float(random_aurocs.std()),
            "random_p95": float(np.percentile(random_aurocs, 95)),
            "lift_over_random": float(cond["auroc"] - random_aurocs.mean()),
            "l1_sparse_auroc": l1_mean,
            "l1_sparse_std": l1_std,
            "l1_nonzero_neurons": l1_nonzero,
        }
    return detail


def mean_random_jaccard(d_inter, k, rng, n_pairs=50):
    pairs = [(rng.choice(d_inter, k, replace=False), rng.choice(d_inter, k, replace=False)) for _ in range(n_pairs)]
    return float(np.mean([probes.jaccard(a, b) for a, b in pairs]))


def cross_position_structure(store, neuron_scores, ks, small_k, cka_k, seed):
    """Do the prompt and gen0 positions select the same neurons, and is the selected geometry similar?"""
    d_inter = store.shape[1]
    rng = np.random.default_rng(seed)
    random_jaccard = {k: mean_random_jaccard(d_inter, k, rng) for k in ks}
    jaccard, spearman, cka = {}, {}, {}
    for layer in tqdm(range(store.n_layers), desc="cross-position"):
        prompt_scores = neuron_scores[PRIMARY_VIEW, "prompt", layer]
        gen0_scores = neuron_scores[PRIMARY_VIEW, "gen0", layer]
        jaccard[f"L{layer}"] = probes.jaccard(
            probes.top_neurons(prompt_scores, small_k), probes.top_neurons(gen0_scores, small_k)
        )
        spearman[f"L{layer}"] = float(spearmanr(prompt_scores, gen0_scores)[0])
        cka[f"L{layer}"] = probes.linear_cka(
            store.matrix("raw", "prompt", layer)[:, probes.top_neurons(prompt_scores, cka_k)],
            store.matrix("raw", "gen0", layer)[:, probes.top_neurons(gen0_scores, cka_k)],
        )
    return {
        "primary_view": PRIMARY_VIEW,
        f"jaccard_by_layer_k{small_k}": jaccard,
        f"mean_jaccard_k{small_k}": float(np.mean(list(jaccard.values()))),
        "random_jaccard": random_jaccard,
        "spearman_by_layer": spearman,
        "mean_spearman": float(np.mean(list(spearman.values()))),
        "cka_by_layer": cka,
        "mean_cka": float(np.mean(list(cka.values()))),
    }


def usable(y, cfg):
    return len(y) >= cfg["min_samples"] and min(y.sum(), (1 - y).sum()) >= cfg["min_positives"]


def position_fusion(X_prompt, X_gen0, y, masks, cfg, seed):
    """Per dataset: top-k probe on the prompt position, the gen0 position, and both concatenated."""
    fusion = {}
    for name, mask in masks.items():
        ys = y[mask]
        if not usable(ys, cfg):
            continue
        folds = probes.make_folds(seed)
        Xp = X_prompt[mask][:, probes.top_neurons(probes.neuron_aurocs(X_prompt[mask], ys), cfg["small_k"])]
        Xg = X_gen0[mask][:, probes.top_neurons(probes.neuron_aurocs(X_gen0[mask], ys), cfg["small_k"])]
        prompt, gen0, concat = (float(probes.probe_scores(X, ys, folds).mean()) for X in (Xp, Xg, np.hstack([Xp, Xg])))
        fusion[name] = {"prompt": prompt, "gen0": gen0, "concat": concat, "additivity": concat - max(prompt, gen0)}
    return fusion


def cross_dataset_transfer(X, y, masks, cfg):
    """Fit a top-k probe on each dataset and score it on every dataset. Diagonal entries are in-sample."""
    fitted = {}
    for name, mask in masks.items():
        if name == "all" or not usable(y[mask], cfg):
            continue
        top = probes.top_neurons(probes.neuron_aurocs(X[mask], y[mask]), cfg["small_k"])
        scaler = StandardScaler().fit(X[mask][:, top])
        clf = LogisticRegression(max_iter=500).fit(scaler.transform(X[mask][:, top]), y[mask])
        fitted[name] = (top, scaler, clf)
    matrix = {
        train: {
            test: float(roc_auc_score(y[masks[test]], clf.decision_function(scaler.transform(X[masks[test]][:, top]))))
            for test in fitted
        }
        for train, (top, scaler, clf) in fitted.items()
    }
    within = [matrix[a][b] for a in matrix for b in matrix if a == b]
    cross = [matrix[a][b] for a in matrix for b in matrix if a != b]
    return {
        "transfer_matrix": matrix,
        "within_mean": float(np.mean(within)),
        "cross_mean": float(np.mean(cross)),
        "fragmentation_gap": float(np.mean(within) - np.mean(cross)),
    }


def dimensionality(X, top, y, folds, seed):
    """How much of the top-k signal sits on the first principal component, plus effect sizes."""
    Xt = StandardScaler().fit_transform(X[:, top])
    pca = PCA(n_components=min(10, len(top)), random_state=seed).fit(Xt)
    pc1_auroc = baselines.folded_auroc(y, pca.transform(Xt)[:, 0])
    full_auroc = float(probes.probe_scores(X[:, top], y, folds).mean())
    best = X[:, top[0]]
    cohens_d = (best[y == 1].mean() - best[y == 0].mean()) / (
        np.sqrt((best[y == 1].var() + best[y == 0].var()) / 2) + 1e-9
    )
    mean_hall, mean_correct = X[:, top][y == 1].mean(0), X[:, top][y == 0].mean(0)
    return {
        "pc1_variance_explained": float(pca.explained_variance_ratio_[0]),
        "pc1_only_auroc": float(pc1_auroc),
        "full_topk_auroc": full_auroc,
        "nonlinearity_premium": float(full_auroc - pc1_auroc),
        "cohens_d_best_neuron": float(cohens_d),
        "diff_correct_norm_ratio": float(
            np.linalg.norm(mean_hall - mean_correct) / (np.linalg.norm(mean_correct) + 1e-9)
        ),
    }


def calibration(probabilities, y):
    precision, recall, _ = precision_recall_curve(y, probabilities)
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities >= low) & (probabilities < high)
        if in_bin.any():
            ece += in_bin.mean() * abs(probabilities[in_bin].mean() - y[in_bin].mean())
    result = {"brier": float(brier_score_loss(y, probabilities)), "ece": float(ece)}
    for target in (50, 70, 80, 90):
        result[f"precision_at_recall_{target}"] = float(precision[recall >= target / 100].max())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--activations-dir", help="local activations; default streams from the Hub repo in the config")
    parser.add_argument("--work-dir", default="work", help="scratch space for the per-layer memmaps")
    parser.add_argument("--out", default="results/detection")
    args = parser.parse_args()
    cfg = load_config(args.config)
    det, seed = cfg["detection"], cfg["seed"]

    source = open_activations(args.model, args.activations_dir, cfg["activations_repo"])
    work_dir = Path(args.work_dir) / args.model
    all_labels = read_labels(source, work_dir)
    mask = reliable_mask(all_labels, cfg["datasets"])
    labels = all_labels[mask].reset_index(drop=True)
    y = labels["hallucinated"].astype(int).to_numpy()
    masks = {"all": np.ones(len(y), dtype=bool)}
    masks.update(
        {name: (labels["source_dataset"] == name).to_numpy() for name in sorted(labels["source_dataset"].unique())}
    )

    store, residual, residual_layer = build_layer_store(source, mask, work_dir, VIEWS, POSITIONS, with_residual=True)
    folds = probes.make_folds(seed)

    results = {
        "model_name": args.model,
        "config": det,
        "n_total_samples_raw": int(len(all_labels)),
        "n_layers": store.n_layers,
        "d_inter": store.shape[1],
        "d_model": residual.shape[1],
    }
    results["section_A"] = dataset_summary(labels, y)
    results["baseline_table"] = baseline_table(labels, residual, residual_layer, y, folds, seed)

    scan, neuron_scores, best = layer_scan(store, y, det["ks"], folds)
    results["section_B"] = {
        "best_conditions": {f"{v}_{p}": cond for (v, p), cond in best.items()},
        "layer_scan": {f"{v}_{p}_L{layer}_k{k}": mean for (v, p, layer, k), (mean, _) in scan.items()},
    }
    results["section_B_detail"] = best_condition_detail(store, neuron_scores, best, y, folds, det, seed)

    results["section_C"] = cross_position_structure(store, neuron_scores, det["ks"], det["small_k"], det["cka_k"], seed)
    prompt_best, gen0_best = best[PRIMARY_VIEW, "prompt"], best[PRIMARY_VIEW, "gen0"]
    X_gen0 = store.matrix(PRIMARY_VIEW, "gen0", gen0_best["layer"])
    results["section_C"]["fusion"] = position_fusion(
        store.matrix(PRIMARY_VIEW, "prompt", prompt_best["layer"]), X_gen0, y, masks, det, seed
    )
    results["section_D"] = cross_dataset_transfer(X_gen0, y, masks, det)

    cett_layer = gen0_best["layer"]
    results["section_E"] = {
        "neuron_overlap_raw_vs_cett": {
            f"k{k}": probes.jaccard(
                probes.top_neurons(neuron_scores["raw", "gen0", cett_layer], k),
                probes.top_neurons(neuron_scores[PRIMARY_VIEW, "gen0", cett_layer], k),
            )
            for k in det["overlap_ks"]
        },
    }

    top = probes.top_neurons(neuron_scores[PRIMARY_VIEW, "gen0", cett_layer], gen0_best["k"])
    results["section_F"] = dimensionality(X_gen0, top, y, folds, seed)
    results["section_G"] = calibration(probes.out_of_fold_probabilities(X_gen0[:, top], y, folds), y)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.model}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
