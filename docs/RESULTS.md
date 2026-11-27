# Result files

The files in `results/detection`, `results/ksweep` and `results/interventions` are the ones the paper was written from. They are unchanged apart from their names. `results/MANIFEST.sha256` records their checksums. `results/tables` is built from them by `scripts/make_tables.py`.

## results/detection/<model>.json

Written by `scripts/run_detection.py`. Examples with an unparseable answer are excluded before any analysis.

| Key | Content |
|---|---|
| `section_A` | number of examples, hallucination rate, per-dataset counts |
| `baseline_table` | AUROC of the token-probability and residual-stream reference detectors |
| `section_B.layer_scan` | top-k probe AUROC for every view, position, layer and k |
| `section_B.best_conditions` | best layer and k for each view and position |
| `section_B_detail` | at each best condition: bootstrap CI, random-subset AUROC, L1 probe |
| `section_C` | prompt vs gen0 position: neuron overlap, rank correlation, CKA (each a mean over layers), and position fusion |
| `section_D` | cross-dataset transfer matrix |
| `section_E` | overlap of the neurons selected from the raw and the CETT view |
| `section_F` | PCA of the selected neurons, effect size of the best neuron |
| `section_G` | calibration and precision at fixed recall |

Views: `raw` is `act_fn(gate_proj(x))`. `exact_cett` is `|act_fn(gate_proj(x)) * up_proj(x)| * ||W_down[:, j]|| / ||MLP(x)||`.
Positions: `prompt` is the last prompt token. `gen0` is the first generated token.

The saved files carry fields that the paper does not use and that `run_detection.py` no longer writes: run bookkeeping (`feature_views`, `label_filter`, `datasets_present` and similar), `section_E.per_view_best` (a copy of `section_B.best_conditions`), and `length_confound_p`, `z_over_random`, `beats_random_p95`, `mi_top3_pcs`, `mean_correct_vec`, `mean_hall_vec`, `top_neurons_for_intervention`, `cross_dataset_neuron_jaccard`. Their `config` block uses the notebook's names for the settings in `configs/paper.yaml`.

`nonlinearity_premium` is the top-k probe AUROC minus the PC1-only AUROC. Both are linear.

## results/ksweep/<model>.json

Written by `scripts/run_ksweep.py` for the raw view at the best gen0 layer.

| Key | Content |
|---|---|
| `full_layer_auroc` | probe on all neurons of the layer |
| `top_k.<k>.auroc` | probe on the k neurons with the highest individual AUROC |
| `random_k.<k>` | mean, spread and percentiles over random sets of k neurons (200 draws up to k=50, 50 draws above) |
| `random_projection.<d>` | the same for Gaussian random projections to d dimensions (50 draws) |

## results/interventions

Written by `scripts/analyze_interventions.py` from the per-example outputs of `scripts/run_interventions.py`.

`analysis.json`:

| Key | Content |
|---|---|
| `results.<model>__<dataset>.baseline` | unsteered hallucination rate on the test examples |
| `results.*.dose_response` | hallucination rate and log P(correct answer) when the selected (`top`) or random set of 300 neurons is scaled by alpha, with McNemar tests of selected vs random |
| `results.*.sparse_vs_dist` | scaling 10, 300, 2,000 or all neurons by 0.5 |
| `results.*.direction` | adding the PC1, difference-of-means or a random direction |
| `results.*.ablation` | setting neurons to their train mean |
| `verdicts` | per combination: swing of the hallucination rate across alpha for the selected and random sets, the smallest McNemar p over alpha, and its Holm-adjusted value across the 16 combinations |
| `pooled` | means over the 16 combinations with bootstrap intervals (Table 5) |
| `detection_vs_causation` | correlation between detection AUROC at k=300 and the selected-set swing |

`summary.csv` has one row per combination with the quantities that are pooled.

Names: `top` is the selected set and `random`, `ctrl` or `rand` the random control set. A swing is the largest minus the smallest hallucination rate across the scaling factors. `ca_logprob` is the first-step log-probability of the answer-key token (see docs/DIFFERENCES.md), `logit_slope` its slope against alpha, and `reliability` the share of examples where it moved in the intended direction.

The `verdict` label is a rule of thumb that the paper does not use. It is POSITIVE when the selected-set dose-response is monotone (Spearman p < 0.05), its swing exceeds the random swing by more than 0.02, `reliability` is above 0.55 and at least 75% as many answers stay parseable. It is NULL when the trend is not monotone and the excess is below 0.02, and WEAK-MIXED otherwise. The paper's significance statement is based on `min_mcnemar_p_holm`.
