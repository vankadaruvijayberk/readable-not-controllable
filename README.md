# Readable but Not Controllable

Code and result files for "Readable but Not Controllable: Neuron-Level Evidence for Medical LLM Hallucination"
(V. Vankadaru, A. Matthews, T. Roosta, P. Passban), DAIH workshop at COLM 2026, [arXiv:2607.00158](https://arxiv.org/abs/2607.00158).

## What the paper found

We ask whether the MLP activations that reveal a wrong answer can also be used to prevent it.
Four open models (BioMistral-7B, Meditron-7B, PMC-LLaMA-7B, Llama-3.1-8B-Instruct) answer medical questions from four datasets, and an answer that disagrees with the answer key counts as a hallucination.

1. A linear probe on the activations of one MLP layer detects hallucinated answers with AUROC 0.77 to 0.86.
2. The signal is spread over many neurons. A random set of a few hundred neurons does as well as the best-ranked set, and both do better than a probe on the whole layer.
3. Scaling the best-ranked neurons changes the hallucination rate about as much as scaling random neurons. For each model and dataset we take the largest change in hallucination rate across scaling factors. Averaged over the 16 pairs, the best-ranked set moves it by 0.0042 more than a random set (95% CI -0.0002 to 0.0096), and no pair is significant after Holm correction.

Terms used below: `prompt` is the last prompt token and `gen0` is the first generated token. `raw` features are the gate activations `act_fn(gate_proj(x))` of one MLP layer. `exact_cett` features are each neuron's contribution to the MLP output (CETT, Cumulative Errors of Tail Truncation, from Zhang et al., 2024). `actground` is the working name of the project and of the package.

## Install

Python 3.11.

```bash
git clone https://github.com/vankadaruvijayberk/readable-not-controllable.git
cd readable-not-controllable
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # or the build for your GPU
pip install -r requirements.txt
pip install -e .
```

Check the install:

```bash
pytest                       # under a minute, no GPU and no downloads
bash scripts/smoke_test.sh   # about five minutes on CPU, downloads a 135M model and two datasets
```

Run all commands from the repository root. `pytest` builds every table from `results/` and compares it with the 343 table cells printed in the paper (three known last-digit differences are listed in docs/DIFFERENCES.md).
The smoke test runs every experiment end to end with a small model. Its numbers are not meaningful.

## Reproduce the paper

There are three levels, from cheapest to most expensive.

### 1. Tables and figures from the saved results (seconds)

```bash
python scripts/make_tables.py     # results/tables/table01 ... table18
python scripts/make_figures.py    # figures/fig1_*.png, figures/fig2_ksweep_all_models.png
```

`results/` holds the result files the paper was written from. [docs/PAPER_MAP.md](docs/PAPER_MAP.md) lists, for each table and figure, the command, the result file and the field it reads.

### 2. Table 5 from the per-example intervention outputs (a few minutes, CPU)

```bash
python scripts/download_interventions.py   # 16 parquet files, 4.6 MB, into outputs/interventions
python scripts/analyze_interventions.py --out outputs/analysis
```

This recomputes the intervention statistics from every steered answer. The output matches `results/interventions/analysis.json`, including the bootstrap intervals. Once the files are downloaded, `pytest` checks the Table 5 values and the Holm-adjusted p-values against the saved file.

### 3. Everything from scratch (GPU)

Meditron-7B and Llama-3.1-8B-Instruct are gated on the Hugging Face Hub. Request access and run `huggingface-cli login` first.

```bash
python scripts/download_data.py

# per model: biomistral-7b, meditron-7b, pmc-llama-7b, llama-3.1-8b-instruct
python scripts/collect_activations.py --model biomistral-7b --out activations
python scripts/run_detection.py       --model biomistral-7b --activations-dir activations
python scripts/run_ksweep.py          --model biomistral-7b --activations-dir activations

# per model and dataset: pubmedqa_labeled, medqa_usmle, medmcqa, pubmedqa_artificial
python scripts/run_interventions.py   --model biomistral-7b --dataset medqa_usmle

python scripts/analyze_interventions.py
python scripts/make_tables.py
python scripts/make_figures.py
```

To skip collection, leave out `--activations-dir`. `run_detection.py` and `run_ksweep.py` then stream the released activations from the Hub, keeping one chunk on disk at a time. The release is 200 to 245 GB per model because it holds more arrays and datasets than these scripts read. `run_detection.py` passes over it twice and `run_ksweep.py` once.

These commands write to `results/` and replace the saved files, after which `results/MANIFEST.sha256` and the paper-number test no longer apply. Pass `--out` to write elsewhere. `cd results && sha256sum -c MANIFEST.sha256` checks that the saved files are untouched.

| Script | Produces | Needs |
|---|---|---|
| `collect_activations.py` | activation chunks | one GPU, about 80 GB of disk per model |
| `run_detection.py` | Tables 1 to 4 and 6 to 18 (with `run_ksweep.py` for part of Table 2) | CPU, 60 to 80 GB of scratch disk per model |
| `run_ksweep.py` | Table 2 (full-layer and random k=500 AUROC), Figures 1 and 2 | CPU, 1 GB of scratch disk |
| `run_interventions.py` | per-example steering outputs | one GPU |
| `analyze_interventions.py` | Table 5 and the significance tests | CPU |

The paper's runs used a single A100 GPU on Google Colab with the models in float16.
All settings are in `configs/paper.yaml`. The random seed is 42, except for the random neuron subsets in `run_detection.py` (123) and the random projections in `run_ksweep.py` (43).
Logistic regression and greedy decoding are deterministic, but GPU arithmetic is not bit-exact across hardware, so a rerun from scratch can differ from the saved results in the last digits.

## Data and models

No dataset, model weight or activation file is stored in this repository.

| Dataset | Source on the Hugging Face Hub | License on the Hub | Used |
|---|---|---|---|
| PubMedQA, labeled | `qiaojin/PubMedQA` (`pqa_labeled`) | MIT | all 1,000 questions |
| PubMedQA, artificial | `qiaojin/PubMedQA` (`pqa_artificial`) | MIT | 5,000 sampled questions |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | CC BY 4.0 | train split, 10,178 questions |
| MedMCQA | `openlifescienceai/medmcqa` | Apache-2.0 | 5,000 sampled questions from the train split |

The intervention experiments use at most 4,000 questions per dataset. 20% of the questions with a parseable answer are held out for testing (about 800, or 170 to 200 for PubMedQA-labeled) and up to 2,500 of the rest are used to choose the steering targets. Prompts are plain text without a chat template for all four models.

Models: `BioMistral/BioMistral-7B`, `epfl-llm/meditron-7b`, `chaoyi-wu/PMC_LLAMA_7B`, `meta-llama/Meta-Llama-3.1-8B-Instruct`.

Released files:

- Activations: [`vjdeara/actground-multimodel-public`](https://huggingface.co/datasets/vjdeara/actground-multimodel-public), 880 GB. It also contains three datasets that the paper does not use. The scripts ignore them.
- Intervention outputs: [`vjdeara/actground-interventions-public`](https://huggingface.co/datasets/vjdeara/actground-interventions-public), 4.6 MB.

## Layout

```
configs/    paper.yaml (all settings), smoke.yaml (CPU smoke test)
src/actground/   data, models, activations, probes, baselines, steering, stats
scripts/    one entry point per experiment, plus tables, figures and downloads
results/    result files behind the paper, and the tables built from them
figures/    Figures 1 and 2
docs/       PAPER_MAP.md, RESULTS.md, DIFFERENCES.md
tests/      unit tests, a pipeline test on fake activations, the paper-number check
```

## Notes for readers of the paper

[docs/DIFFERENCES.md](docs/DIFFERENCES.md) lists three table cells that differ from the paper in the last digit, describes where the code is more specific than the text, and records properties of the labels and of the log-probability measure that matter when reading the results.

## Citation

```bibtex
@inproceedings{vankadaru2026readable,
  title     = {Readable but Not Controllable: Neuron-Level Evidence for Medical {LLM} Hallucination},
  author    = {Vankadaru, Vijay and Matthews, Asha and Roosta, Tanya and Passban, Peyman},
  booktitle = {DAIH Workshop at the Conference on Language Modeling (COLM)},
  year      = {2026},
  eprint    = {2607.00158},
  archivePrefix = {arXiv}
}
```

Contact: Vijay Vankadaru, vankadaruvijay@berkeley.edu
