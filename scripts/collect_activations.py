"""Generate an answer for every question and store MLP activations at two token positions.

Writes <out>/<model>/chunks/chunk_<n>.npy and <out>/<model>/wdown_col_norms.npy.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from actground.config import add_config_arg, load_config
from actground.data import build_samples, label_answer
from actground.models import MLPRecorder, down_proj_column_norms, load_model


def mlp_signals(recorder, model, position):
    """Gate activations, neuron activations and MLP output norm of every layer at one token position."""
    post, inter, ynorm = [], [], []
    for layer, block in enumerate(model.model.layers):
        gate = recorder.gate[layer][0, position]
        neuron = gate * recorder.up[layer][0, position]
        weight = block.mlp.down_proj.weight
        post.append(gate.float().cpu().numpy().astype(np.float16))
        inter.append(neuron.float().cpu().numpy().astype(np.float16))
        ynorm.append(float((weight @ neuron.to(weight.dtype)).float().norm()))
    return np.stack(post), np.stack(inter), np.array(ynorm, dtype=np.float32)


def residual_stream(hidden_states, position):
    """[n_layers, d_model] residual stream after each block. Index 0 of hidden_states is the embedding."""
    return np.stack([h[0, position].float().cpu().numpy().astype(np.float16) for h in hidden_states[1:]])


@torch.no_grad()
def collect_example(sample, model, tokenizer, recorder, generation):
    inputs = tokenizer(
        sample["prompt"], return_tensors="pt", truncation=True, max_length=generation["max_prompt_length"]
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    model(**inputs)
    post_prompt, inter_prompt, ynorm_prompt = mlp_signals(recorder, model, prompt_len - 1)

    out = model.generate(
        **inputs,
        max_new_tokens=generation["max_new_tokens"],
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out.sequences[0, prompt_len:], skip_special_tokens=True).strip()
    extracted, hallucinated = label_answer(text, sample["correct_answer"], sample["answer_type"])
    first_token_probs = F.softmax(out.scores[0][0].float(), dim=-1)
    clamped = first_token_probs.clamp(min=1e-12)
    first_token = int(out.sequences[0, prompt_len])

    # Feed the prompt plus the first generated token to read the MLP at that token.
    model(input_ids=out.sequences[:, : prompt_len + 1])
    post_gen0, inter_gen0, ynorm_gen0 = mlp_signals(recorder, model, -1)

    return {
        "sample_id": sample["sample_id"],
        "source_dataset": sample["source_dataset"],
        "correct_answer": sample["correct_answer"],
        "generated_text": text,
        "extracted_answer": extracted,
        # An unparseable answer is stored as hallucinated with an "_unlabeled" tag and filtered out later.
        "hallucinated": True if hallucinated is None else hallucinated,
        "label_confidence": "unknown_unlabeled" if hallucinated is None else "exact",
        "n_tokens_generated": len(out.scores),
        "first_tok_prob": float(first_token_probs[first_token]),
        "first_tok_entropy": float(-(clamped * clamped.log()).sum()),
        "mlp_post_prompt": post_prompt,
        "mlp_inter_prompt": inter_prompt,
        "mlp_ynorm_prompt": ynorm_prompt,
        "mlp_post_gen0": post_gen0,
        "mlp_inter_gen0": inter_gen0,
        "mlp_ynorm_gen0": ynorm_gen0,
        "resid_post_gen0": residual_stream(out.hidden_states[0], -1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="activations")
    args = parser.parse_args()
    cfg = load_config(args.config)

    samples = [s for d in cfg["datasets"] for s in build_samples(d, cfg["collection"]["budgets"][d], cfg["seed"])]
    model, tokenizer = load_model(cfg["models"][args.model], cfg["generation"])
    recorder = MLPRecorder(model)

    chunk_dir = Path(args.out) / args.model / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    np.save(chunk_dir.parent / "wdown_col_norms.npy", down_proj_column_norms(model))

    chunk_size = cfg["collection"]["chunk_size"]
    for start in tqdm(range(0, len(samples), chunk_size), desc=args.model):
        chunk_path = chunk_dir / f"chunk_{start // chunk_size + 1}.npy"
        if chunk_path.exists():
            continue
        records = [
            collect_example(s, model, tokenizer, recorder, cfg["generation"])
            for s in samples[start : start + chunk_size]
        ]
        np.save(chunk_path, np.array(records, dtype=object), allow_pickle=True)
    recorder.remove()


if __name__ == "__main__":
    main()
