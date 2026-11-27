"""Steer one model on one dataset and record every test answer under every intervention.

Writes <out>/<model>__<dataset>.parquet with one row per test example and intervention.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from actground import steering
from actground.config import add_config_arg, load_config
from actground.data import build_samples, label_answer
from actground.models import load_model


def batches(items, size):
    return (items[i : i + size] for i in range(0, len(items), size))


def label_samples(samples, model, tokenizer, cfg):
    """Unsteered answers for every sample. Samples without a parseable answer are dropped."""
    labeled = []
    for batch in tqdm(list(batches(samples, cfg["intervention"]["batch_size"])), desc="labels"):
        texts, _ = steering.generate(model, tokenizer, [s["prompt"] for s in batch], cfg["generation"])
        for sample, text in zip(batch, texts):
            _, hallucinated = label_answer(text, sample["correct_answer"], sample["answer_type"])
            if hallucinated is not None:
                labeled.append({**sample, "hallucinated": hallucinated})
    return labeled


def train_test_split(samples, cfg, seed):
    order = np.arange(len(samples))
    np.random.default_rng(seed).shuffle(order)
    test_ids = set(order[: int(len(samples) * cfg["test_frac"])].tolist())
    train = [s for i, s in enumerate(samples) if i not in test_ids]
    test = [s for i, s in enumerate(samples) if i in test_ids]
    return train[: cfg["train_cap"]], test[: cfg["eval_cap"]]


def run_condition(condition, test, model, tokenizer, token_ids, cfg):
    label = "|".join(
        [
            condition["experiment"],
            str(condition["set_type"]),
            f"k={condition['k']}",
            f"a={condition['alpha']}",
            f"s={condition['strength']}",
        ]
    )
    rows = []
    for batch in batches(test, cfg["intervention"]["batch_size"]):
        texts, first_logprobs = steering.generate(model, tokenizer, [s["prompt"] for s in batch], cfg["generation"])
        for sample, text, logprobs in zip(batch, texts, first_logprobs):
            extracted, hallucinated = label_answer(text, sample["correct_answer"], sample["answer_type"])
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "condition": label,
                    "experiment": condition["experiment"],
                    "set_type": condition["set_type"],
                    "k": condition["k"],
                    "alpha": condition["alpha"],
                    "strength": condition["strength"],
                    "gen_text": text[:120],
                    "extracted": extracted,
                    "hallucinated": hallucinated,
                    "correct_answer": sample["correct_answer"],
                    "ca_logprob": float(logprobs[token_ids[sample["correct_answer"]]]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detection", default="results/detection")
    parser.add_argument("--out", default="outputs/interventions")
    args = parser.parse_args()
    cfg = load_config(args.config)
    icfg, seed = cfg["intervention"], cfg["seed"]

    with open(Path(args.detection) / f"{args.model}.json") as f:
        layer = json.load(f)["section_B"]["best_conditions"]["raw_gen0"]["layer"]

    samples = build_samples(args.dataset, icfg["budgets"][args.dataset], seed)
    if len(samples) > icfg["max_total_samples"]:
        keep = np.random.default_rng(seed).choice(len(samples), icfg["max_total_samples"], replace=False)
        samples = [samples[int(i)] for i in keep]

    model, tokenizer = load_model(cfg["models"][args.model], cfg["generation"])
    labeled = label_samples(samples, model, tokenizer, cfg)
    train, test = train_test_split(labeled, icfg, seed)

    X_train = steering.last_token_activations(model, tokenizer, [s["prompt"] for s in train], layer, cfg["generation"])
    y_train = np.array([int(s["hallucinated"]) for s in train])
    targets = steering.derive_targets(X_train, y_train, seed)
    conditions = steering.build_conditions(targets, icfg, seed)

    hook = steering.SteeringHook(model, layer, targets)
    token_ids = steering.answer_token_ids(tokenizer, test[0]["answer_type"])
    rows = []
    for condition in tqdm(conditions, desc="conditions"):
        hook.condition = condition
        rows += run_condition(condition, test, model, tokenizer, token_ids, cfg)
    hook.remove()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}__{args.dataset}"
    pd.DataFrame(rows).assign(model=args.model, dataset=args.dataset, layer=layer).to_parquet(
        out_dir / f"{stem}.parquet", index=False
    )


if __name__ == "__main__":
    main()
