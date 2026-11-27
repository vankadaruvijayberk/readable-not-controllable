"""Model loading and MLP hooks for Llama- and Mistral-family decoders.

Their MLP is SwiGLU: y = down_proj(act_fn(gate_proj(x)) * up_proj(x)).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# PMC-LLaMA ships a sentencepiece tokenizer that the fast tokenizer cannot load.
SLOW_TOKENIZER_REPOS = {"chaoyi-wu/PMC_LLAMA_7B"}


def load_model(repo, generation):
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=repo not in SLOW_TOKENIZER_REPOS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=getattr(torch, generation["dtype"]), device_map={"": generation["device"]}
    )
    return model.eval(), tokenizer


def mlp_layers(model):
    return [block.mlp for block in model.model.layers]


class MLPRecorder:
    """Keeps the latest act_fn and up_proj outputs of every layer."""

    def __init__(self, model):
        self.gate = {}
        self.up = {}
        self.handles = []
        for i, mlp in enumerate(mlp_layers(model)):
            self.handles.append(mlp.act_fn.register_forward_hook(self._save(self.gate, i)))
            self.handles.append(mlp.up_proj.register_forward_hook(self._save(self.up, i)))

    @staticmethod
    def _save(store, layer):
        def hook(module, inputs, output):
            store[layer] = output.detach()

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def down_proj_column_norms(model):
    """[n_layers, d_inter] norms of each neuron's write direction into the residual stream."""
    return torch.stack([mlp.down_proj.weight.detach().float().norm(dim=0) for mlp in mlp_layers(model)]).cpu().numpy()
