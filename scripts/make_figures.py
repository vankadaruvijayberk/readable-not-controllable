"""Draw the k-sweep figures of the paper from results/ksweep.

Figure 1: one panel per model (top-k, random-k and the full-layer probe).
Figure 2: random-k curves of all models.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "biomistral-7b": "#2c7fb8",
    "meditron-7b": "#41ab5d",
    "pmc-llama-7b": "#d95f0e",
    "llama-3.1-8b-instruct": "#756bb1",
}


def random_curve(sweep):
    ks = [k for k in sweep["k_grid"] if str(k) in sweep["random_k"]]
    stats = [sweep["random_k"][str(k)] for k in ks]
    return ks, [s["mean"] for s in stats], [s["p05"] for s in stats], [s["p95"] for s in stats]


def style_axis(ax, xlabel):
    ax.axhline(0.5, color="#aaa", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def figure1(sweep, path):
    ks, mean, low, high = random_curve(sweep)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.fill_between(ks, low, high, alpha=0.25, color="#7fb3d5", label="random-k 5-95%")
    ax.plot(ks, mean, "o-", color="#2c7fb8", label="random-k mean")
    ax.plot(ks, [sweep["top_k"][str(k)]["auroc"] for k in ks], "s--", color="#d95f0e", label="top-k (ranked)")
    ax.axhline(sweep["full_layer_auroc"], color="#444", ls=":", label=f"full layer ({sweep['full_layer_auroc']:.3f})")
    ax.set_title(f"{sweep['model']}: random-k vs top-k (L{sweep['layer']}, {sweep['view']}/gen0)")
    style_axis(ax, "k (neurons)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure2(sweeps, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sweep in sweeps:
        ks, mean, low, high = random_curve(sweep)
        color = COLORS.get(sweep["model"])
        ax.fill_between(ks, low, high, alpha=0.12, color=color)
        ax.plot(ks, mean, "o-", ms=3, color=color, label=f"{sweep['model']} (full {sweep['full_layer_auroc']:.3f})")
    ax.set_title("Random-k dropoff across models (5-95% bands)")
    style_axis(ax, "k (number of random neurons)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ksweep", default="results/ksweep")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweeps = [json.loads(path.read_text()) for path in sorted(Path(args.ksweep).glob("*.json"))]
    for sweep in sweeps:
        figure1(sweep, out_dir / f"fig1_ksweep_{sweep['model']}.png")
    figure2(sweeps, out_dir / "fig2_ksweep_all_models.png")


if __name__ == "__main__":
    main()
