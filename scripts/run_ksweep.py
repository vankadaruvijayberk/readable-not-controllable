"""Probe AUROC as a function of neuron count at the best gen0 layer: top-k, random-k and random projections.

Reads the best layer from <detection>/<model>.json and writes <out>/<model>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from actground import probes
from actground.activations import build_layer_store, open_activations, read_labels, reliable_mask
from actground.config import add_config_arg, load_config


def summarize(aurocs):
    return {
        "n_draws": len(aurocs),
        "mean": float(aurocs.mean()),
        "std": float(aurocs.std()),
        "p05": float(np.percentile(aurocs, 5)),
        "p50": float(np.percentile(aurocs, 50)),
        "p95": float(np.percentile(aurocs, 95)),
        "min": float(aurocs.min()),
        "max": float(aurocs.max()),
    }


def neuron_sweep(X, y, cfg, folds, seed):
    ranked = probes.top_neurons(probes.neuron_aurocs(X, y), X.shape[1])
    rng = np.random.default_rng(seed)
    random_k, top_k = {}, {}
    for k in (k for k in cfg["k_grid"] if k <= X.shape[1]):
        n_draws = cfg["draws_small_k"] if k <= cfg["small_k_max"] else cfg["draws_large_k"]
        random_k[str(k)] = summarize(probes.random_subset_aurocs(X, y, k, n_draws, folds, rng))
        top_k[str(k)] = {"auroc": float(probes.probe_scores(X[:, ranked[:k]], y, folds).mean())}
    return random_k, top_k


def projection_sweep(X, y, cfg, folds, seed):
    Xz = StandardScaler().fit_transform(X).astype(np.float32)
    rng = np.random.default_rng(seed + 1)
    sweep = {}
    for dim in (d for d in cfg["projection_dims"] if d <= X.shape[1]):
        aurocs = np.empty(cfg["projection_draws"])
        for i in range(len(aurocs)):
            projection = rng.standard_normal((X.shape[1], dim)).astype(np.float32) / np.sqrt(dim)
            aurocs[i] = probes.probe_scores(Xz @ projection, y, folds).mean()
        sweep[str(dim)] = {key: value for key, value in summarize(aurocs).items() if key not in ("p50", "min", "max")}
    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--activations-dir", help="local activations; default streams from the Hub repo in the config")
    parser.add_argument("--work-dir", default="work")
    parser.add_argument("--detection", default="results/detection")
    parser.add_argument("--out", default="results/ksweep")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]

    with open(Path(args.detection) / f"{args.model}.json") as f:
        best = json.load(f)["section_B"]["best_conditions"]["raw_gen0"]

    source = open_activations(args.model, args.activations_dir, cfg["activations_repo"])
    work_dir = Path(args.work_dir) / args.model
    labels = read_labels(source, work_dir)
    mask = reliable_mask(labels, cfg["datasets"])
    y = labels.loc[mask, "hallucinated"].astype(int).to_numpy()
    store, _, _ = build_layer_store(source, mask, work_dir / "ksweep", ["raw"], ["gen0"], layers=[best["layer"]])
    X = store.matrix("raw", "gen0", best["layer"])

    folds = probes.make_folds(seed)
    full_layer = float(probes.probe_scores(X, y, folds, n_jobs=2).mean())
    random_k, top_k = neuron_sweep(X, y, cfg["ksweep"], folds, seed)
    results = {
        "model": args.model,
        "layer": best["layer"],
        "view": "raw",
        "position": "gen0",
        "n_samples": len(y),
        "d_inter": X.shape[1],
        "hall_rate": float(y.mean()),
        "full_layer_auroc": full_layer,
        "reported_auroc": best["auroc"],
        "k_grid": cfg["ksweep"]["k_grid"],
        "random_k": random_k,
        "top_k": top_k,
        "random_projection": projection_sweep(X, y, cfg["ksweep"], folds, seed),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.model}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
