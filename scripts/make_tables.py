"""Build every table of the paper from the result files and write them as CSV to results/tables.

Values are printed at the paper's precision. docs/DIFFERENCES.md lists the cells that differ from the paper.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

MODELS = {
    "biomistral-7b": "BioMistral-7B",
    "meditron-7b": "Meditron-7B",
    "pmc-llama-7b": "PMC-LLaMA-7B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}
DATASETS = ["medmcqa", "medqa_usmle", "pubmedqa_artificial", "pubmedqa_labeled"]
CONDITIONS = {
    "raw / prompt": "raw_prompt",
    "raw / gen0": "raw_gen0",
    "exact_cett / prompt": "exact_cett_prompt",
    "exact_cett / gen0": "exact_cett_gen0",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def load(results_dir):
    results_dir = Path(results_dir)
    detection = {m: read_json(results_dir / "detection" / f"{m}.json") for m in MODELS}
    ksweep = {m: read_json(results_dir / "ksweep" / f"{m}.json") for m in MODELS}
    return detection, ksweep, read_json(results_dir / "interventions" / "analysis.json")


def by_model(rows, detection, decimals=None):
    """Table with one column per model. `rows` maps a row label to (function of a model's results, decimals)."""
    table = {}
    for label, spec in rows.items():
        get, digits = spec if isinstance(spec, tuple) else (spec, decimals)
        table[label] = {MODELS[m]: fmt(get(d), digits) for m, d in detection.items()}
    return pd.DataFrame(table).T


def fmt(value, decimals):
    return str(value) if decimals is None else f"{value:.{decimals}f}"


def detection_tables(det, ksweep):
    def best(cond):
        return lambda d: d["section_B"]["best_conditions"][cond]

    def detail(cond, key):
        return lambda d: d["section_B_detail"][cond][key]

    def sweep(get):
        return lambda d: get(ksweep[d["model_name"]])

    tables = {}
    tables["table01_best_layer"] = by_model({"Best layer": lambda d: f"L{best('raw_gen0')(d)['layer']}"}, det)
    tables["table02_detection"] = by_model(
        {
            "Best layer": lambda d: f"L{best('raw_gen0')(d)['layer']}",
            "Probe AUROC": (detail("raw_gen0", "best_auroc"), 3),
            "Full-layer AUROC": (sweep(lambda k: k["full_layer_auroc"]), 3),
            "Random k=500 AUROC": (sweep(lambda k: k["random_k"]["500"]["mean"]), 3),
            "Base hallucination rate": (lambda d: d["section_A"]["overall_hall_rate"], 3),
        },
        det,
    )
    tables["table03_baselines"] = by_model(
        {
            "linear probe on top-k neurons (ours)": (detail("raw_gen0", "best_auroc"), 3),
            **{
                name: (lambda d, name=name: d["baseline_table"][name]["auroc"], 4)
                for name in ("token_neg_prob", "token_entropy", "saplma_style_resid", "haloscope_style", "ccs_style")
            },
        },
        det,
    )
    tables["table04_cross_position"] = by_model(
        {
            "Jaccard top-50": lambda d: d["section_C"]["mean_jaccard_k50"],
            "Random Jaccard": lambda d: d["section_C"]["random_jaccard"]["50"],
            "Spearman (AUROC rank)": lambda d: d["section_C"]["mean_spearman"],
            "CKA top-500": lambda d: d["section_C"]["mean_cka"],
        },
        det,
        decimals=4,
    )
    tables["table06_models"] = by_model(
        {
            "d_inter": lambda d: d["d_inter"],
            "N (clean)": lambda d: d["section_A"]["n_samples_after_filter"],
            "Overall hall-rate": (lambda d: d["section_A"]["overall_hall_rate"], 3),
        },
        det,
    )
    tables["table07_datasets"] = pd.DataFrame(
        [
            {
                "dataset": ds,
                "model": MODELS[m],
                "n": d["section_A"]["per_dataset"][ds]["n"],
                "hall_rate": fmt(d["section_A"]["per_dataset"][ds]["hall_rate"], 3),
                "positives": d["section_A"]["per_dataset"][ds]["n_positives"],
            }
            for ds in DATASETS
            for m, d in det.items()
        ]
    ).set_index(["dataset", "model"])
    for number, (m, d) in zip((8, 9, 10, 11), det.items()):
        matrix = d["section_D"]["transfer_matrix"]
        tables[f"table{number:02d}_transfer_{m}"] = pd.DataFrame(
            {test: {train: fmt(matrix[train][test], 3) for train in DATASETS} for test in DATASETS}
        ).rename_axis("train \\ test")
    tables["table12_precision_at_recall"] = by_model(
        {f"{r / 100:.2f}": (lambda d, r=r: d["section_G"][f"precision_at_recall_{r}"]) for r in (50, 70, 80, 90)},
        det,
        decimals=4,
    )
    tables["table13_probe_by_view"] = by_model(
        {
            **{
                label: (lambda d, c=c: f"{best(c)(d)['auroc']:.4f} (L{best(c)(d)['layer']})")
                for label, c in CONDITIONS.items()
            },
            "Best": (lambda d: max(best(c)(d)["auroc"] for c in CONDITIONS.values()), 4),
        },
        det,
    )
    tables["table14_bootstrap_ci"] = by_model(
        {label: (lambda d, c=c: "[{:.4f}, {:.4f}]".format(*detail(c, "ci95")(d))) for label, c in CONDITIONS.items()},
        det,
    )
    tables["table15_topk_vs_random"] = by_model(
        {
            "best AUROC": detail("raw_gen0", "best_auroc"),
            "random-k AUROC": detail("raw_gen0", "random_mean"),
            "lift (raw)": detail("raw_gen0", "lift_over_random"),
            "lift (exact_cett)": detail("exact_cett_gen0", "lift_over_random"),
            "L1 nonzero neurons": (detail("raw_gen0", "l1_nonzero_neurons"), None),
            "L1 AUROC": detail("raw_gen0", "l1_sparse_auroc"),
        },
        det,
        decimals=4,
    )
    tables["table16_raw_cett_overlap"] = by_model(
        {
            "layer": (lambda d: f"L{best('exact_cett_gen0')(d)['layer']}", None),
            **{
                f"top-{k}": (lambda d, k=k: d["section_E"]["neuron_overlap_raw_vs_cett"][f"k{k}"])
                for k in (50, 100, 500)
            },
        },
        det,
        decimals=4,
    )
    tables["table17_position_fusion"] = by_model(
        {
            ds: (
                lambda d, ds=ds: " / ".join(
                    fmt(d["section_C"]["fusion"][ds][p], 3) for p in ("prompt", "gen0", "concat")
                )
            )
            for ds in ["all"] + DATASETS
        },
        det,
    )
    tables["table18_dimensionality"] = by_model(
        {
            "PC1 variance": (lambda d: d["section_F"]["pc1_variance_explained"], 3),
            "PC1-only AUROC": lambda d: d["section_F"]["pc1_only_auroc"],
            "non-linearity premium": lambda d: d["section_F"]["nonlinearity_premium"],
            "Cohen's d (best neuron)": lambda d: d["section_F"]["cohens_d_best_neuron"],
            "diff/correct norm ratio": lambda d: d["section_F"]["diff_correct_norm_ratio"],
            "Brier": lambda d: d["section_G"]["brier"],
            "ECE": lambda d: d["section_G"]["ece"],
        },
        det,
        decimals=4,
    )
    return tables


def intervention_table(analysis):
    rows = {  # label: (pooled key, decimals of estimate, CI low, CI high)
        "Selected-set swing": ("top_swing", 3, 3, 3),
        "Random-set swing": ("ctrl_swing", 3, 3, 3),
        "Delta (selected - random)": ("excess", 4, 4, 4),
        "Amplification effect": ("amp_effect", 4, 4, 3),
        "Logit slope, log P(correct) vs alpha": ("logit_slope", 4, 3, 3),
    }
    table = {}
    for label, (key, d_mean, d_low, d_high) in rows.items():
        pooled = analysis["pooled"][key]
        table[label] = {
            "estimate": fmt(pooled["mean"], d_mean),
            "ci_low": fmt(pooled["ci"][0], d_low),
            "ci_high": fmt(pooled["ci"][1], d_high),
        }
    return pd.DataFrame(table).T


def build_tables(results_dir):
    detection, ksweep, analysis = load(results_dir)
    tables = detection_tables(detection, ksweep)
    tables["table05_interventions"] = intervention_table(analysis)
    return dict(sorted(tables.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    out_dir = Path(args.results) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in build_tables(args.results).items():
        table.to_csv(out_dir / f"{name}.csv")
        print(out_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
