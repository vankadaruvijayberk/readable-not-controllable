# Paper results, scripts and files

Every table is written to `results/tables/` by `scripts/make_tables.py`. The columns below say which experiment script produced the numbers and where they are stored. `<model>` is one of `biomistral-7b`, `meditron-7b`, `pmc-llama-7b`, `llama-3.1-8b-instruct`.

| Paper item | Experiment script | Result file | Field |
|---|---|---|---|
| Table 1, best layer | `run_detection.py` | `results/detection/<model>.json` | `section_B.best_conditions.raw_gen0.layer` |
| Table 2, probe AUROC | `run_detection.py` | `results/detection/<model>.json` | `section_B_detail.raw_gen0.best_auroc` |
| Table 2, full-layer AUROC | `run_ksweep.py` | `results/ksweep/<model>.json` | `full_layer_auroc` |
| Table 2, random k=500 AUROC | `run_ksweep.py` | `results/ksweep/<model>.json` | `random_k.500.mean` |
| Table 2, base rate | `run_detection.py` | `results/detection/<model>.json` | `section_A.overall_hall_rate` |
| Figure 1 | `run_ksweep.py`, then `make_figures.py` | `results/ksweep/<model>.json` | `top_k`, `random_k`, `full_layer_auroc` |
| Figure 2 | `run_ksweep.py`, then `make_figures.py` | `results/ksweep/<model>.json` | `random_k` |
| Table 3, baselines | `run_detection.py` | `results/detection/<model>.json` | `baseline_table.*.auroc` |
| Table 4, cross-position structure | `run_detection.py` | `results/detection/<model>.json` | `section_C.mean_jaccard_k50`, `random_jaccard.50`, `mean_spearman`, `mean_cka` |
| Table 5, pooled intervention effects | `run_interventions.py`, then `analyze_interventions.py` | `results/interventions/analysis.json` | `pooled` |
| Holm-corrected McNemar tests (Section 5.2) | same | `results/interventions/analysis.json` | `verdicts.*.min_mcnemar_p_holm` |
| Detection strength vs effect (Section 5.2) | same | `results/interventions/analysis.json` | `detection_vs_causation` |
| Table 6, model details | `run_detection.py` | `results/detection/<model>.json` | `d_inter`, `section_A` |
| Table 7, dataset breakdown | `run_detection.py` | `results/detection/<model>.json` | `section_A.per_dataset` |
| Tables 8 to 11, cross-dataset transfer | `run_detection.py` | `results/detection/<model>.json` | `section_D.transfer_matrix` |
| Table 12, precision at fixed recall | `run_detection.py` | `results/detection/<model>.json` | `section_G` |
| Table 13, AUROC by view and position | `run_detection.py` | `results/detection/<model>.json` | `section_B.best_conditions` |
| Table 14, bootstrap intervals | `run_detection.py` | `results/detection/<model>.json` | `section_B_detail.*.ci95` |
| Table 15, top-k vs random-k, L1 probe | `run_detection.py` | `results/detection/<model>.json` | `section_B_detail.raw_gen0`, `section_B_detail.exact_cett_gen0.lift_over_random` |
| Table 16, raw vs CETT neuron overlap | `run_detection.py` | `results/detection/<model>.json` | `section_E.neuron_overlap_raw_vs_cett` |
| Table 17, position fusion | `run_detection.py` | `results/detection/<model>.json` | `section_C.fusion` |
| Table 18, dimensionality and calibration | `run_detection.py` | `results/detection/<model>.json` | `section_F`, `section_G` |
| Random projections (abstract) | `run_ksweep.py` | `results/ksweep/<model>.json` | `random_projection` |

The activations that `run_detection.py` and `run_ksweep.py` read come from `collect_activations.py`.
