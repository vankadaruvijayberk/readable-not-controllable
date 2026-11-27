"""Reading activation chunks and turning them into per-layer feature matrices.

A chunk is a pickled numpy array of per-example dicts. Each dict holds the labels and, for the
`prompt` (last prompt token) and `gen0` (first generated token) positions:
  mlp_post_<pos>   [n_layers, d_inter]  act_fn(gate_proj(x))
  mlp_inter_<pos>  [n_layers, d_inter]  act_fn(gate_proj(x)) * up_proj(x)
  mlp_ynorm_<pos>  [n_layers]           norm of the MLP output
  resid_post_gen0  [n_layers, d_model]  residual stream while predicting the first generated token
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm

POSITIONS = ["prompt", "gen0"]
VIEWS = ["raw", "exact_cett"]
# label_confidence of an answer that was parsed and compared with the key. Unparseable answers are dropped.
PARSED = "exact"
LABEL_COLUMNS = [
    "sample_id",
    "source_dataset",
    "hallucinated",
    "label_confidence",
    "first_tok_prob",
    "first_tok_entropy",
]


def _chunk_number(path):
    return int(re.search(r"chunk_(\d+)\.npy$", str(path)).group(1))


class LocalActivations:
    kind = "local"

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

    def chunk_names(self):
        return sorted((p.name for p in (self.model_dir / "chunks").glob("chunk_*.npy")), key=_chunk_number)

    def load_chunk(self, name):
        return np.load(self.model_dir / "chunks" / name, allow_pickle=True)

    def down_proj_norms(self):
        return np.load(self.model_dir / "wdown_col_norms.npy")


class HubActivations:
    """Streams chunks from the Hugging Face dataset repo and deletes each one after use.

    Chunks are pickled numpy object arrays, so only point this at a repo you trust.
    """

    kind = "hub"

    def __init__(self, repo, model):
        self.repo = repo
        self.prefix = f"models/{model}"

    def chunk_names(self):
        files = list_repo_files(self.repo, repo_type="dataset")
        names = [f.rsplit("/", 1)[1] for f in files if f.startswith(f"{self.prefix}/chunks/chunk_")]
        return sorted(names, key=_chunk_number)

    def load_chunk(self, name):
        path = Path(hf_hub_download(self.repo, f"{self.prefix}/chunks/{name}", repo_type="dataset"))
        chunk = np.load(path, allow_pickle=True)
        path.resolve().unlink()
        path.unlink(missing_ok=True)
        return chunk

    def down_proj_norms(self):
        return np.load(hf_hub_download(self.repo, f"{self.prefix}/wdown_col_norms.npy", repo_type="dataset"))


def open_activations(model, activations_dir, repo):
    if activations_dir:
        return LocalActivations(Path(activations_dir) / model)
    return HubActivations(repo, model)


def iter_records(source, desc):
    for name in tqdm(source.chunk_names(), desc=desc):
        yield from source.load_chunk(name)


def read_labels(source, work_dir):
    """One row of labels per collected example, in collection order. Cached because it reads every chunk."""
    cache_path = Path(work_dir) / f"labels_{source.kind}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    labels = pd.DataFrame([{k: r.get(k) for k in LABEL_COLUMNS} for r in iter_records(source, "labels")])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(cache_path, index=False)
    return labels


def reliable_mask(labels, datasets):
    return (
        (labels["label_confidence"] == PARSED)
        & labels["hallucinated"].notna()
        & labels["source_dataset"].isin(datasets)
    ).to_numpy()


def exact_cett(inter, ynorm, down_norms):
    """|neuron activation| * ||W_down[:, j]|| / ||MLP(x)||, per layer."""
    return np.abs(inter) * down_norms / np.maximum(ynorm.astype(np.float32), 1e-8)[:, None]


def record_features(record, view, position, down_norms):
    if view == "raw":
        return record[f"mlp_post_{position}"]
    return exact_cett(record[f"mlp_inter_{position}"], record[f"mlp_ynorm_{position}"], down_norms)


class LayerStore:
    """float16 memmaps of shape [n_examples, d_inter], one per (view, position, layer)."""

    def __init__(self, work_dir, n_examples, n_layers, d_inter):
        self.work_dir = Path(work_dir)
        self.n_layers = n_layers
        self.shape = (n_examples, d_inter)

    def path(self, view, position, layer):
        return self.work_dir / f"{view}_{position}_L{layer}.dat"

    def create(self, view, position, layer):
        return np.memmap(self.path(view, position, layer), dtype="float16", mode="w+", shape=self.shape)

    def matrix(self, view, position, layer):
        mm = np.memmap(self.path(view, position, layer), dtype="float16", mode="r", shape=self.shape)
        return np.asarray(mm, dtype=np.float32)


def build_layer_store(source, mask, work_dir, views, positions, layers=None, with_residual=False):
    """Write the kept examples into one memmap per (view, position, layer). `layers=None` keeps all layers.

    With `with_residual`, also returns the gen0 residual stream at 60% depth and that layer's index.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    down_norms = source.down_proj_norms() if "exact_cett" in views else None
    n_kept = int(mask.sum())
    store, maps, residual, residual_layer = None, {}, None, None
    row = 0
    for i, record in enumerate(iter_records(source, "activations")):
        if not mask[i]:
            continue
        if store is None:
            n_layers, d_inter = record["mlp_post_gen0"].shape
            store = LayerStore(work_dir, n_kept, n_layers, d_inter)
            layers = range(n_layers) if layers is None else layers
            maps = {(v, p, layer): store.create(v, p, layer) for v in views for p in positions for layer in layers}
            if with_residual:
                residual_layer = int(n_layers * 0.6)
                residual = np.zeros((n_kept, record["resid_post_gen0"].shape[1]), dtype=np.float32)
        for view in views:
            for position in positions:
                features = record_features(record, view, position, down_norms)
                for layer in layers:
                    maps[view, position, layer][row] = features[layer]
        if with_residual:
            residual[row] = record["resid_post_gen0"][residual_layer]
        row += 1
    for mm in maps.values():
        mm.flush()
    return store, residual, residual_layer
