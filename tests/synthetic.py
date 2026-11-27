"""Fake activation chunks in the collection format, with a planted label signal."""

from pathlib import Path

import numpy as np

DATASETS = ["pubmedqa_labeled", "medqa_usmle", "medmcqa", "pubmedqa_artificial"]


def make_record(rng, dataset, index, n_layers, d_inter, d_model):
    hallucinated = bool(rng.random() < 0.4)
    record = {
        "sample_id": f"{dataset}_{index}",
        "source_dataset": dataset,
        "hallucinated": hallucinated,
        "label_confidence": "exact" if rng.random() < 0.95 else "unknown_unlabeled",
        "n_tokens_generated": 4,
        "first_tok_prob": float(rng.uniform(0.2, 1.0) - 0.15 * hallucinated),
        "first_tok_entropy": float(rng.uniform(0, 2) + 0.3 * hallucinated),
        "resid_post_gen0": rng.normal(size=(n_layers, d_model)).astype(np.float16),
    }
    record["resid_post_gen0"][:, :3] += 0.8 * hallucinated
    for offset, position in enumerate(("prompt", "gen0")):
        gate = rng.normal(size=(n_layers, d_inter))
        gate[:, 5 * offset : 5 * offset + 12] += 0.6 * hallucinated
        gate[:, 40 + len(dataset) % 7] += 0.8 * hallucinated
        record[f"mlp_post_{position}"] = gate.astype(np.float16)
        record[f"mlp_inter_{position}"] = (gate * rng.normal(1.0, 0.2, size=gate.shape)).astype(np.float16)
        record[f"mlp_ynorm_{position}"] = rng.uniform(1, 3, size=n_layers).astype(np.float32)
    return record


def write_activations(root, model="toy", per_dataset=160, n_layers=3, d_inter=96, d_model=24, chunk_size=50, seed=0):
    """Write chunks for four paper datasets plus one dataset the analysis must ignore."""
    rng = np.random.default_rng(seed)
    records = [
        make_record(rng, dataset, i, n_layers, d_inter, d_model)
        for dataset in DATASETS + ["bioasq"]
        for i in range(per_dataset)
    ]
    model_dir = Path(root) / model
    (model_dir / "chunks").mkdir(parents=True)
    np.save(model_dir / "wdown_col_norms.npy", rng.uniform(0.5, 2, size=(n_layers, d_inter)).astype(np.float32))
    for start in range(0, len(records), chunk_size):
        chunk = np.array(records[start : start + chunk_size], dtype=object)
        np.save(model_dir / "chunks" / f"chunk_{start // chunk_size + 1}.npy", chunk, allow_pickle=True)
    return model_dir
