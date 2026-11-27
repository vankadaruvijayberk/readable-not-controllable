"""Activation steering at one MLP layer: target selection, the forward hook, and batched generation."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from actground.probes import neuron_aurocs, top_neurons

ANSWER_LABELS = {"yesno": ["yes", "no", "maybe"], "mcq": ["A", "B", "C", "D"]}


def derive_targets(X, y, seed):
    """Steering targets from train activations: neuron ranking, PC1 and the difference-of-means direction."""
    std = X.std(0) + 1e-8
    pc1 = PCA(n_components=1, random_state=seed).fit((X - X.mean(0)) / std).components_[0]
    diff = X[y == 1].mean(0) - X[y == 0].mean(0)
    return {
        "ranked": top_neurons(neuron_aurocs(X, y), X.shape[1]),
        "pc1": pc1.astype(np.float32),
        "diffmean": (diff / (np.linalg.norm(diff) + 1e-8)).astype(np.float32),
        "mean": X.mean(0).astype(np.float32),
        "direction_scale": float(np.median(std)),
    }


def build_conditions(targets, cfg, seed):
    """The list of interventions run on every test example. alpha = 1 and strength = 0 are no-ops."""
    d_inter = len(targets["ranked"])
    rng = np.random.default_rng(seed)
    everything = list(range(d_inter))
    top_set = targets["ranked"][: cfg["dose_k"]].tolist()
    random_set = rng.choice(d_inter, cfg["dose_k"], replace=False).tolist()

    def scale(experiment, set_type, columns, alpha):
        return {
            "experiment": experiment,
            "set_type": set_type,
            "k": len(columns),
            "alpha": alpha,
            "strength": None,
            "mode": "scale",
            "columns": columns,
        }

    conditions = [scale("dose_scale", "top", top_set, a) for a in cfg["dose_alphas"]]
    conditions += [scale("dose_scale", "random", random_set, a) for a in cfg["dose_alphas"]]
    for k in cfg["sparse_ks"]:
        conditions.append(scale("sparse_vs_dist", "top", targets["ranked"][:k].tolist(), cfg["sparse_alpha"]))
        conditions.append(
            scale("sparse_vs_dist", "random", rng.choice(d_inter, k, replace=False).tolist(), cfg["sparse_alpha"])
        )
    conditions.append(scale("sparse_vs_dist", "full", everything, cfg["sparse_alpha"]))

    random_direction = rng.standard_normal(d_inter).astype(np.float32)
    random_direction /= np.linalg.norm(random_direction) + 1e-8
    directions = {"pc1": targets["pc1"], "diffmean": targets["diffmean"], "random": random_direction}
    for name, direction in directions.items():
        for strength in cfg["direction_strengths"]:
            conditions.append(
                {
                    "experiment": "direction",
                    "set_type": name,
                    "k": None,
                    "alpha": None,
                    "strength": strength,
                    "mode": "direction",
                    "direction": direction,
                }
            )

    for set_type, columns in (("top", top_set), ("random", random_set), ("full", everything)):
        conditions.append(
            {
                "experiment": "ablate",
                "set_type": set_type,
                "k": len(columns),
                "alpha": None,
                "strength": None,
                "mode": "ablate",
                "columns": columns,
            }
        )
    return conditions


class SteeringHook:
    """Forward hook on act_fn of one layer. Applies the current condition at every token position."""

    def __init__(self, model, layer, targets):
        self.condition = None
        self.mean = torch.tensor(targets["mean"], device=model.device)
        self.direction_scale = targets["direction_scale"]
        self.handle = model.model.layers[layer].mlp.act_fn.register_forward_hook(self)

    def __call__(self, module, inputs, output):
        c = self.condition
        if c is None:
            return output
        if c["mode"] == "scale":
            output[:, :, c["columns"]] *= c["alpha"]
        elif c["mode"] == "ablate":
            output[:, :, c["columns"]] = self.mean[c["columns"]].to(output.dtype)
        else:
            direction = torch.tensor(c["direction"], device=output.device, dtype=output.dtype)
            output = output + c["strength"] * self.direction_scale * direction
        return output

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def generate(model, tokenizer, prompts, generation):
    """Greedy generation for a batch. Returns the generated texts and first-step log-probabilities [batch, vocab]."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=generation["max_prompt_length"]
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=generation["max_new_tokens"],
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    texts = tokenizer.batch_decode(out.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return [t.strip() for t in texts], F.log_softmax(out.scores[0].float(), dim=-1).cpu().numpy()


def answer_token_ids(tokenizer, answer_type):
    return {label: tokenizer.encode(label, add_special_tokens=False)[0] for label in ANSWER_LABELS[answer_type]}


@torch.no_grad()
def last_token_activations(model, tokenizer, prompts, layer, generation):
    """act_fn output at the last prompt token of `layer`, one prompt at a time."""
    captured = {}
    handle = model.model.layers[layer].mlp.act_fn.register_forward_hook(
        lambda module, inputs, output: captured.update(out=output.detach())
    )
    rows = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=generation["max_prompt_length"]).to(
            model.device
        )
        model(**inputs)
        rows.append(captured["out"][0, -1].float().cpu().numpy())
    handle.remove()
    return np.stack(rows)
