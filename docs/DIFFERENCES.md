# Differences from the paper

## Table cells

`tests/test_paper_numbers.py` compares all 343 table cells printed in the camera-ready paper with the tables built from `results/`. Three cells of Table 17 (position fusion) differ in the last digit. The paper rounded them to four decimals and then to three.

| Cell (G is the gen0 position) | Paper | Result file | Table in this repository |
|---|---|---|---|
| Llama-3.1, medmcqa, G | 0.851 | 0.850460 | 0.850 |
| Llama-3.1, medqa_usmle, concat | 0.799 | 0.798477 | 0.798 |
| Meditron, pubmedqa_labeled, G | 0.804 | 0.803484 | 0.803 |

The random k=500 AUROC of Llama-3.1 is 0.859 in Table 2 and 0.8584 in Table 15. The two tables come from different runs with different numbers of random draws (`run_ksweep.py`, 50 draws at k=500, and `run_detection.py`, 30 draws).

## What the code does, where the text is brief

- **Selected neurons.** Everywhere, including the interventions, neurons are ranked by their individual AUROC, folded to `max(AUC, 1 - AUC)`. The interventions rank on raw `act_fn` activations at the last prompt token of the train examples. CETT is used as a second feature view in the detection experiments only.
- **Selection and cross-validation.** Neurons are ranked on all examples and the probe is then scored with 5-fold cross-validation. The best layer and k are the ones with the highest cross-validated AUROC.
- **Baselines in Table 3.** `saplma_style_resid`, `haloscope_style` and `ccs_style` are short versions written for this project, all on the residual stream at 60% depth. The CCS version has no contrast pairs. The HaloScope version picks one of the top ten singular directions by its AUROC. The token baselines are not sign-corrected, which is why some are below 0.5.
- **Cross-dataset transfer, Tables 8 to 11.** Each probe is fit on all examples of its training dataset. The diagonal is therefore scored on the examples the probe was fit on.
- **Table 4.** Each number is a mean over all layers. Neurons are ranked on the CETT view. CKA is computed on raw activations of the 500 top-ranked neurons at each position.
- **Tables 8 to 12 and 16 to 18** use the CETT view at its best layer and k. For PMC-LLaMA that is k=100, for the other models k=500.
- **Residual baselines.** The residual stream is read at the last prompt token, while the gen0 probe reads the MLP at the first generated token.
- **Detection strength vs effect.** The detection AUROC in this correlation is one value per model (top-k probe at k=300 from the k-sweep), so the 16 points have four distinct x values.
- **Significance tests.** For each model and dataset, the smallest McNemar p-value over the seven scaling factors is Holm-adjusted across the 16 pairs.
- **Data.** Questions come from the train splits listed in the README, with seed 42 for the subsamples.
- **Positions.** `prompt` is the last prompt token and `gen0` is the first generated token. The detection results in Tables 1 and 2 use `gen0`.

## Properties of the labels and measures

These can be read off the released intervention outputs and the result files.

- **Answer distribution.** On the unsteered test examples, PMC-LLaMA answers A on more than 99.8% of MedMCQA and MedQA questions and yes on more than 99.7% of PubMedQA questions. Meditron answers A on 95 to 99% and yes on 89 to 97%. For these two models the label mostly records whether the answer key is A or yes. BioMistral and Llama-3.1 spread their answers over the options on the multiple-choice datasets and answer yes on 61 to 93% of PubMedQA questions.
- **Dataset identity.** Hallucination rates differ a lot between datasets (Table 7). A score equal to each dataset's hallucination rate reaches AUROC 0.719 (BioMistral), 0.770 (Meditron), 0.758 (PMC-LLaMA) and 0.667 (Llama-3.1) on the pooled data, against probe AUROCs of 0.808, 0.796, 0.767 and 0.863. Table 17 gives the per-dataset probe AUROCs.
- **Answer parsing.** A generation whose first character is A, B, C or D is read as that letter, including free text.
- **log P(correct answer).** The log-probability is read at the token id of the bare answer string (`A`, `yes`). Several tokenizers emit a different token for the answer the model actually writes (a leading space or a capital). On test examples the model answers correctly, the mean probability of the scored token is 0.4 to 0.5 for BioMistral on the multiple-choice datasets, 0.10 to 0.12 for Meditron and PMC-LLaMA on those datasets, and below 0.05 everywhere else, down to 0.001 for Llama-3.1. The logit-slope row of Table 5 and the `reliability` fields rest on this quantity. The hallucination rates and McNemar tests do not.

## Changes from the code that produced the saved results

The experiments were first run as Colab notebooks. The scripts here are a rewrite of those notebooks. On the same inputs they give the same outputs: the detection and k-sweep scripts were compared with the notebook code on synthetic activations, the intervention script on a small model, and the analysis script on the released intervention outputs. Two details are intentionally different:

- The L1 probe and the PCA in `run_detection.py` now take a fixed random seed. In the notebooks they were unseeded, so the L1 AUROC could move in the fourth decimal between runs.
- `collect_activations.py` stores only the four datasets and the arrays that the analysis reads.
- `run_detection.py` no longer writes fields that the paper does not use.
