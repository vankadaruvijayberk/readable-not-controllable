"""Prompt construction and answer labelling for the four medical QA datasets."""

import re

import numpy as np
from datasets import load_dataset

HF_SOURCES = {
    "pubmedqa_labeled": ("qiaojin/PubMedQA", "pqa_labeled"),
    "pubmedqa_artificial": ("qiaojin/PubMedQA", "pqa_artificial"),
    "medqa_usmle": ("GBaker/MedQA-USMLE-4-options", None),
    "medmcqa": ("openlifescienceai/medmcqa", None),
}
ID_PREFIX = {
    "pubmedqa_labeled": "pqa_lab",
    "pubmedqa_artificial": "pqa_art",
    "medqa_usmle": "medqa",
    "medmcqa": "medmcqa",
}
LETTERS = ["A", "B", "C", "D"]


def load_split(dataset):
    path, name = HF_SOURCES[dataset]
    return load_dataset(path, name, split="train")


def _pubmedqa_prompt(row):
    context = " ".join(row["context"]["contexts"])
    return f"Context: {context}\n\nQuestion: {row['question']}\n\nAnswer (yes/no/maybe):"


def _mcq_prompt(question, options):
    option_lines = "\n".join(f"{letter}) {text}" for letter, text in options.items())
    return f"Question: {question}\n\n{option_lines}\n\nAnswer (single letter A/B/C/D):"


def _format(dataset, row):
    if dataset.startswith("pubmedqa"):
        return _pubmedqa_prompt(row), row["final_decision"].lower(), "yesno"
    if dataset == "medqa_usmle":
        return _mcq_prompt(row["question"], row["options"]), str(row["answer_idx"]).upper().strip(), "mcq"
    options = dict(zip(LETTERS, [row["opa"], row["opb"], row["opc"], row["opd"]]))
    return _mcq_prompt(row["question"], options), LETTERS[int(row["cop"])], "mcq"


def _row_indices(n_rows, budget, dataset, seed):
    if budget is None or dataset == "pubmedqa_labeled":
        return range(n_rows if budget is None else min(budget, n_rows))
    return np.random.default_rng(seed).choice(n_rows, min(budget, n_rows), replace=False)


def build_samples(dataset, budget, seed):
    """Return prompt records for one dataset.

    A budget below the split size draws a seeded subsample. PubMedQA-labeled takes the first rows instead.
    """
    split = load_split(dataset)
    samples = []
    for i in _row_indices(len(split), budget, dataset, seed):
        prompt, answer, answer_type = _format(dataset, split[int(i)])
        samples.append(
            {
                "sample_id": f"{ID_PREFIX[dataset]}_{int(i)}",
                "source_dataset": dataset,
                "prompt": prompt,
                "correct_answer": answer,
                "answer_type": answer_type,
            }
        )
    return samples


def extract_yesno(text):
    t = text.strip().lower()
    first = t.split()[0] if t else ""
    if first in ("yes", "no", "maybe"):
        return first
    t = re.sub(r"^(the answer is|answer:|i think|the answer would be)\s*", "", t)
    first = t.split()[0] if t else ""
    if first in ("yes", "no", "maybe"):
        return first
    head = text.strip().lower()[:30]
    for word in ("yes", "no", "maybe"):
        if re.search(rf"\b{word}\b", head):
            return word
    return "unknown"


def extract_letter(text):
    """First of A-D found at the start of the answer. Free text that starts with one of these letters also matches."""
    t = text.strip()
    if not t:
        return "unknown"
    if t[0].upper() in "ABCD":
        return t[0].upper()
    for pattern, span in (
        (r"([ABCD])[.):\s]", t[:30].upper()),
        (r"(?:ANSWER\s+IS|ANSWER:)\s*([ABCD])\b", t.upper()),
        (r"\b([ABCD])\b", t[:20].upper()),
    ):
        match = re.search(pattern, span)
        if match:
            return match.group(1)
    return "unknown"


def label_answer(generated_text, correct_answer, answer_type):
    """Return (extracted answer, hallucinated). hallucinated is None when no answer can be parsed."""
    extracted = extract_yesno(generated_text) if answer_type == "yesno" else extract_letter(generated_text)
    if extracted == "unknown":
        return extracted, None
    return extracted, extracted != correct_answer
