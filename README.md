# CBASM

Official implementation of **A Cross-Paradigm Bidirectional Auxiliary Supervision Model for Semi-supervised Medical Image Segmentation by Mitigating Endogenous Loops**.

## Reviewer Quick Access

The files below are provided to make the revised experimental results easy to verify.

| Reviewer item | Fast link | What to check |
|---|---|---|
| Raw three-run values for Tables I-III and X | [`evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv) | Dice, Jaccard, 95HD, and ASD for each run |
| Mean and standard deviation for Tables I-III and X | [`evaluation_records/tables_I_III_X_three_run_statistics/mean_std_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/mean_std_values.csv) | Mean/std corresponding to the values discussed in the response |
| Per-case metrics for Tables I-III and X | [`evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv) | Index of all per-case CSV files |
| Empty-prediction check for final checkpoints | [`evaluation_records/final_3seed_metrics_empty_policy/all_case_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/all_case_metrics.csv) | Per-case records used to verify that no empty foreground predictions occurred |
| Revision artifacts from `code0915` | [`revision_artifacts`](revision_artifacts) | Fig. 5 passive-similarity files, final metric checks, and review-facing scripts |

Key folders:

- [`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics): raw run-level and case-level records for the CBASM entries in Tables I-III and X.
- [`evaluation_records/final_3seed_metrics_empty_policy`](evaluation_records/final_3seed_metrics_empty_policy): final-checkpoint per-case evaluation records used for the empty-prediction analysis.
- [`revision_artifacts`](revision_artifacts): lightweight materials merged from the local `code0915` revision workspace.

### Tables I-III and X: Three-run Summary

| Dataset | Table | Setting | Dice (%) | Jaccard (%) | 95HD | ASD |
|---|---|---:|---:|---:|---:|---:|
| LA | Table I | 4 labeled | 89.94 +/- 0.17 | 81.78 +/- 0.28 | 6.26 +/- 0.22 | 1.71 +/- 0.05 |
| LA | Table I | 8 labeled | 91.23 +/- 0.03 | 83.93 +/- 0.05 | 5.74 +/- 0.10 | 1.51 +/- 0.04 |
| Pancreas-NIH | Table II | 6 labeled | 82.32 +/- 0.16 | 70.27 +/- 0.19 | 5.06 +/- 0.16 | 1.30 +/- 0.10 |
| Pancreas-NIH | Table II | 12 labeled | 83.63 +/- 0.10 | 72.17 +/- 0.14 | 4.58 +/- 0.04 | 1.24 +/- 0.05 |
| PROMISE12 | Table III | 4 labeled | 83.66 +/- 0.58 | 72.26 +/- 0.80 | 4.82 +/- 0.65 | 1.83 +/- 0.20 |
| PROMISE12 | Table III | 7 labeled | 84.78 +/- 0.37 | 74.02 +/- 0.56 | 3.99 +/- 0.89 | 1.51 +/- 0.22 |
| ACDC | Table X | 3 labeled | 87.83 +/- 0.56 | 79.06 +/- 0.82 | 2.38 +/- 0.51 | 0.68 +/- 0.13 |
| ACDC | Table X | 7 labeled | 89.23 +/- 0.43 | 81.15 +/- 0.64 | 1.71 +/- 0.25 | 0.52 +/- 0.08 |

For the PROMISE12 4-labeled setting in Table III, the original single-run 95HD value `5.28` is included in [`per_run_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv) as `original_single_run`.

### Tables I-III and X: Raw Per-run Values

The table below displays the raw values for each run. The same values with full precision are stored in [`evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv). Per-case CSV files for each run are indexed in [`evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv).

| Dataset | Table | Setting | Run | Dice (%) | Jaccard (%) | 95HD | ASD |
|---|---|---:|---|---:|---:|---:|---:|
| LA | Table I | 4 labeled | seed2021 | 90.13 | 82.10 | 6.02 | 1.73 |
| LA | Table I | 4 labeled | seed2022 | 89.88 | 81.67 | 6.46 | 1.75 |
| LA | Table I | 4 labeled | seed2023 | 89.81 | 81.57 | 6.30 | 1.65 |
| LA | Table I | 8 labeled | seed2021 | 91.20 | 83.87 | 5.68 | 1.54 |
| LA | Table I | 8 labeled | seed2023 | 91.23 | 83.93 | 5.69 | 1.46 |
| LA | Table I | 8 labeled | seed2022 | 91.26 | 83.98 | 5.86 | 1.52 |
| Pancreas-NIH | Table II | 6 labeled | seed2021 | 82.31 | 70.20 | 5.06 | 1.21 |
| Pancreas-NIH | Table II | 6 labeled | seed2022 | 82.49 | 70.48 | 4.91 | 1.29 |
| Pancreas-NIH | Table II | 6 labeled | seed2023 | 82.16 | 70.12 | 5.22 | 1.41 |
| Pancreas-NIH | Table II | 12 labeled | seed2021 | 83.52 | 72.01 | 4.54 | 1.18 |
| Pancreas-NIH | Table II | 12 labeled | seed2022 | 83.69 | 72.25 | 4.61 | 1.28 |
| Pancreas-NIH | Table II | 12 labeled | seed2023 | 83.68 | 72.24 | 4.60 | 1.25 |
| PROMISE12 | Table III | 4 labeled | seed2021 | 83.60 | 72.15 | 5.11 | 1.61 |
| PROMISE12 | Table III | 4 labeled | seed2023_rerun | 84.28 | 73.10 | 4.08 | 1.90 |
| PROMISE12 | Table III | 4 labeled | original_single_run | 83.11 | 71.52 | 5.28 | 1.99 |
| PROMISE12 | Table III | 7 labeled | seed2021 | 84.74 | 73.97 | 3.59 | 1.61 |
| PROMISE12 | Table III | 7 labeled | seed2022 | 84.43 | 73.50 | 5.01 | 1.66 |
| PROMISE12 | Table III | 7 labeled | seed2023 | 85.17 | 74.60 | 3.36 | 1.27 |
| ACDC | Table X | 3 labeled | seed2021 | 88.07 | 79.38 | 2.79 | 0.70 |
| ACDC | Table X | 3 labeled | seed2022 | 87.18 | 78.13 | 2.53 | 0.80 |
| ACDC | Table X | 3 labeled | seed2023 | 88.23 | 79.68 | 1.81 | 0.54 |
| ACDC | Table X | 7 labeled | seed2021 | 89.57 | 81.68 | 1.43 | 0.43 |
| ACDC | Table X | 7 labeled | seed2022 | 88.75 | 80.43 | 1.80 | 0.54 |
| ACDC | Table X | 7 labeled | seed2023 | 89.37 | 81.33 | 1.90 | 0.58 |

## Final Evaluation Records

To improve reproducibility and transparency, we provide the per-case evaluation records for the final checkpoints used in the repeated-run evaluation.

The records are available in:

[`evaluation_records/final_3seed_metrics_empty_policy`](evaluation_records/final_3seed_metrics_empty_policy)

This folder contains:

- [`mean_std_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/mean_std_metrics.csv): mean and standard deviation over three seeds.
- [`per_seed_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/per_seed_metrics.csv): metrics for each seed.
- [`all_case_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/all_case_metrics.csv): merged per-case metrics for every test case in every evaluated dataset and seed.
- `case_metrics_ema.csv` files under each dataset/seed folder: per-case metrics for the final EMA checkpoints.

The merged per-case file contains 150 rows in total:

| Dataset | Setting | Seeds | Number of per-case records |
|---|---:|---:|---:|
| LA | 4 labeled | 2021, 2022, 2023 | 60 |
| Pancreas | 12 labeled | 2021, 2022, 2023 | 60 |
| PROMISE12 | 7 labeled | 2021, 2022, 2023 | 30 |

### Three-seed Results

| Dataset | Setting | Dice (%) | Jaccard (%) | 95HD | ASD | Empty-prediction cases |
|---|---:|---:|---:|---:|---:|---:|
| LA | 4 labeled | 89.94 +/- 0.17 | 81.78 +/- 0.28 | 6.26 +/- 0.22 | 1.71 +/- 0.05 | 0 |
| Pancreas | 12 labeled | 83.63 +/- 0.10 | 72.17 +/- 0.14 | 4.58 +/- 0.04 | 1.24 +/- 0.05 | 0 |
| PROMISE12 | 7 labeled | 84.78 +/- 0.37 | 74.02 +/- 0.56 | 3.99 +/- 0.89 | 1.51 +/- 0.22 | 0 |
| Average | - | 86.12 +/- 0.21 | 75.99 +/- 0.33 | 4.94 +/- 0.38 | 1.49 +/- 0.11 | 0 |

The `Average` row is the dataset-level macro average over LA, Pancreas, and PROMISE12.

We checked the per-case records of the final checkpoints for all three repeated runs on LA, Pancreas, and PROMISE12. No empty foreground predictions were observed in these final evaluations. Therefore, the reported 95HD and ASD values are not affected by the empty-prediction boundary-metric issue.

## Tables I-III and X Three-run Statistics

For the main comparisons in Tables I-III and X, the CBASM entries are reported as the mean values over three runs. The raw per-run values, standard deviations, and per-case metrics are provided in:

[`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics)

This folder contains:

- [`per_run_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv): raw metric values for each run.
- [`mean_std_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/mean_std_values.csv): mean and standard deviation over the three runs for each dataset/label setting.
- [`case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv): index of the per-case metric files.
- `*_case_metrics.csv`: per-case Dice, Jaccard, 95HD, and ASD values for each run.
