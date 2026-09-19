# CBASM

Official implementation of **A Cross-Paradigm Bidirectional Auxiliary Supervision Model for Semi-supervised Medical Image Segmentation by Mitigating Endogenous Loops**.

## Reviewer Quick Links

| Item | File or folder |
|---|---|
| Three-run raw values for Tables I-III and X | [`evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv) |
| Three-run mean/std for Tables I-III and X | [`evaluation_records/tables_I_III_X_three_run_statistics/mean_std_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/mean_std_values.csv) |
| Checkpoint path for each seed/run | [`evaluation_records/tables_I_III_X_three_run_statistics/checkpoint_paths.csv`](evaluation_records/tables_I_III_X_three_run_statistics/checkpoint_paths.csv) |
| Per-case metric file index | [`evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv) |
| Empty-prediction verification records | [`evaluation_records/final_3seed_metrics_empty_policy`](evaluation_records/final_3seed_metrics_empty_policy) |
| Fig. 5 and revision-related artifacts | [`revision_artifacts`](revision_artifacts) |

## Repeated-run Records for Tables I-III and X

The manuscript reports CBASM as a single value in Tables I-III and X. For CBASM, the manuscript value is the mean over three runs. The sections below show the corresponding mean/std, raw per-run values, and checkpoint paths table by table.

Full-precision CSV files:

- [`mean_std_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/mean_std_values.csv)
- [`per_run_values.csv`](evaluation_records/tables_I_III_X_three_run_statistics/per_run_values.csv)
- [`checkpoint_paths.csv`](evaluation_records/tables_I_III_X_three_run_statistics/checkpoint_paths.csv)
- [`case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv)

### Table I: LA

Summary:

| Setting | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD |
|---:|---:|---:|---:|---:|
| 4&nbsp;labeled | 89.94&nbsp;±&nbsp;0.17 | 81.78&nbsp;±&nbsp;0.28 | 6.26&nbsp;±&nbsp;0.22 | 1.71&nbsp;±&nbsp;0.05 |
| 8&nbsp;labeled | 91.23&nbsp;±&nbsp;0.03 | 83.93&nbsp;±&nbsp;0.05 | 5.74&nbsp;±&nbsp;0.10 | 1.51&nbsp;±&nbsp;0.04 |

Raw runs:

| Setting | Run | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD | Checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 4&nbsp;labeled | seed2021 | 90.13 | 82.10 | 6.02 | 1.73 | `LA/4_labeled_std_seed2021/vnet/vnet_best_ema_model.pth` |
| 4&nbsp;labeled | seed2022 | 89.88 | 81.67 | 6.46 | 1.75 | `LA/4_labeled_std_seed2022/vnet/vnet_best_ema_model.pth` |
| 4&nbsp;labeled | seed2023 | 89.81 | 81.57 | 6.30 | 1.65 | `LA/4_labeled_std_seed2023/vnet/vnet_best_ema_model.pth` |
| 8&nbsp;labeled | seed2021 | 91.20 | 83.87 | 5.68 | 1.54 | `LA/8_labeled_r15_la10_gpu1_seed2021/vnet/vnet_best_ema_model.pth` |
| 8&nbsp;labeled | seed2023 | 91.23 | 83.93 | 5.69 | 1.46 | `LA/8_labeled_r15_la10_gpu1_seed2023/vnet/vnet_best_ema_model.pth` |
| 8&nbsp;labeled | seed2022 | 91.26 | 83.98 | 5.86 | 1.52 | `LA/8_labeled_std_seed2022/vnet/vnet_best_ema_model.pth` |

Per-case files are listed in [`case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv) and stored as `LA_*_case_metrics.csv` in [`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics).

### Table II: Pancreas-NIH

Summary:

| Setting | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD |
|---:|---:|---:|---:|---:|
| 6&nbsp;labeled | 82.32&nbsp;±&nbsp;0.16 | 70.27&nbsp;±&nbsp;0.19 | 5.06&nbsp;±&nbsp;0.16 | 1.30&nbsp;±&nbsp;0.10 |
| 12&nbsp;labeled | 83.63&nbsp;±&nbsp;0.10 | 72.17&nbsp;±&nbsp;0.14 | 4.58&nbsp;±&nbsp;0.04 | 1.24&nbsp;±&nbsp;0.05 |

Raw runs:

| Setting | Run | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD | Checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 6&nbsp;labeled | seed2021 | 82.31 | 70.20 | 5.06 | 1.21 | `Pancreas/6_labeled_std_seed2021/vnet/vnet_best_ema_model.pth` |
| 6&nbsp;labeled | seed2022 | 82.49 | 70.48 | 4.91 | 1.29 | `Pancreas/6_labeled_std_seed2022/vnet/vnet_best_ema_model.pth` |
| 6&nbsp;labeled | seed2023 | 82.16 | 70.12 | 5.22 | 1.41 | `Pancreas/6_labeled_std_seed2023/vnet/vnet_best_ema_model.pth` |
| 12&nbsp;labeled | seed2021 | 83.52 | 72.01 | 4.54 | 1.18 | `Pancreas/12_labeled_std_seed2021/vnet/vnet_best_ema_model.pth` |
| 12&nbsp;labeled | seed2022 | 83.69 | 72.25 | 4.61 | 1.28 | `Pancreas/12_labeled_std_seed2022/vnet/vnet_best_ema_model.pth` |
| 12&nbsp;labeled | seed2023 | 83.68 | 72.24 | 4.60 | 1.25 | `Pancreas/12_labeled_std_seed2023/vnet/vnet_best_ema_model.pth` |

Per-case files are listed in [`case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv) and stored as `Pancreas-NIH_*_case_metrics.csv` in [`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics).

### Table III: PROMISE12

Summary:

| Setting | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD |
|---:|---:|---:|---:|---:|
| 4&nbsp;labeled | 83.66&nbsp;±&nbsp;0.58 | 72.26&nbsp;±&nbsp;0.80 | 4.82&nbsp;±&nbsp;0.65 | 1.83&nbsp;±&nbsp;0.20 |
| 7&nbsp;labeled | 84.78&nbsp;±&nbsp;0.37 | 74.02&nbsp;±&nbsp;0.56 | 3.99&nbsp;±&nbsp;0.89 | 1.51&nbsp;±&nbsp;0.22 |

Raw runs:

| Setting | Run | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD | Checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 4&nbsp;labeled | seed2021 | 83.60 | 72.15 | 5.11 | 1.61 | `Prostate/4_labeled_r15_prostate10_gpu1_seed2021/unet/unet_best_ema_model.pth` |
| 4&nbsp;labeled | seed2023 | 84.28 | 73.10 | 4.08 | 1.90 | `Prostate/4_labeled_r15_prostate10_gpu1_seed2023_rerun/unet/unet_best_ema_model.pth` |
| 4&nbsp;labeled | seed2022 | 83.11 | 71.52 | 5.28 | 1.99 | `Prostate/4_labeled_aut_addema_mar_60k/unet/unet_best_tea2_model.pth` |
| 7&nbsp;labeled | seed2021 | 84.74 | 73.97 | 3.59 | 1.61 | `Prostate/7_labeled_std_seed2021/unet/unet_best_ema_model.pth` |
| 7&nbsp;labeled | seed2022 | 84.43 | 73.50 | 5.01 | 1.66 | `Prostate/7_labeled_std_seed2022/unet/unet_best_ema_model.pth` |
| 7&nbsp;labeled | seed2023 | 85.17 | 74.60 | 3.36 | 1.27 | `Prostate/7_labeled_std_seed2023/unet/unet_best_ema_model.pth` |



Per-case files are listed in [`case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv) and stored as `PROMISE12_*_case_metrics.csv` in [`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics).

### Table X: ACDC

Summary:

| Setting | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD |
|---:|---:|---:|---:|---:|
| 3&nbsp;labeled | 87.83&nbsp;±&nbsp;0.56 | 79.06&nbsp;±&nbsp;0.82 | 2.38&nbsp;±&nbsp;0.51 | 0.68&nbsp;±&nbsp;0.13 |
| 7&nbsp;labeled | 89.23&nbsp;±&nbsp;0.43 | 81.15&nbsp;±&nbsp;0.64 | 1.71&nbsp;±&nbsp;0.25 | 0.52&nbsp;±&nbsp;0.08 |

Raw runs:

| Setting | Run | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD | Checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 3&nbsp;labeled | seed2021 | 88.07 | 79.38 | 2.79 | 0.70 | `ACDC/3_labeled_std_seed2021/unet/unet_best_ema_model.pth` |
| 3&nbsp;labeled | seed2022 | 87.18 | 78.13 | 2.53 | 0.80 | `ACDC/3_labeled_std_seed2022/unet/unet_best_ema_model.pth` |
| 3&nbsp;labeled | seed2023 | 88.23 | 79.68 | 1.81 | 0.54 | `ACDC/3_labeled_std_seed2023/unet/unet_best_ema_model.pth` |
| 7&nbsp;labeled | seed2021 | 89.57 | 81.68 | 1.43 | 0.43 | `ACDC/7_labeled_std_seed2021/unet/unet_best_ema_model.pth` |
| 7&nbsp;labeled | seed2022 | 88.75 | 80.43 | 1.80 | 0.54 | `ACDC/7_labeled_std_seed2022/unet/unet_best_ema_model.pth` |
| 7&nbsp;labeled | seed2023 | 89.37 | 81.33 | 1.90 | 0.58 | `ACDC/7_labeled_std_seed2023/unet/unet_best_ema_model.pth` |

Per-case files are listed in [`case_metric_sources.csv`](evaluation_records/tables_I_III_X_three_run_statistics/case_metric_sources.csv) and stored as `ACDC_*_case_metrics.csv` in [`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics).

## Empty-prediction Check

This section corresponds to the reviewer concern about empty foreground predictions and boundary metrics.

Records:

- [`evaluation_records/final_3seed_metrics_empty_policy/mean_std_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/mean_std_metrics.csv)
- [`evaluation_records/final_3seed_metrics_empty_policy/per_seed_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/per_seed_metrics.csv)
- [`evaluation_records/final_3seed_metrics_empty_policy/all_case_metrics.csv`](evaluation_records/final_3seed_metrics_empty_policy/all_case_metrics.csv)

Summary:

| Dataset | Setting | Dice&nbsp;(%) | Jaccard&nbsp;(%) | 95HD | ASD | Empty-prediction&nbsp;cases |
|---|---:|---:|---:|---:|---:|---:|
| LA | 4&nbsp;labeled | 89.94&nbsp;±&nbsp;0.17 | 81.78&nbsp;±&nbsp;0.28 | 6.26&nbsp;±&nbsp;0.22 | 1.71&nbsp;±&nbsp;0.05 | 0 |
| Pancreas | 12&nbsp;labeled | 83.63&nbsp;±&nbsp;0.10 | 72.17&nbsp;±&nbsp;0.14 | 4.58&nbsp;±&nbsp;0.04 | 1.24&nbsp;±&nbsp;0.05 | 0 |
| PROMISE12 | 7&nbsp;labeled | 84.78&nbsp;±&nbsp;0.37 | 74.02&nbsp;±&nbsp;0.56 | 3.99&nbsp;±&nbsp;0.89 | 1.51&nbsp;±&nbsp;0.22 | 0 |
| Average | - | 86.12&nbsp;±&nbsp;0.21 | 75.99&nbsp;±&nbsp;0.33 | 4.94&nbsp;±&nbsp;0.38 | 1.49&nbsp;±&nbsp;0.11 | 0 |

No empty foreground predictions were observed in these final evaluations. Therefore, the reported 95HD and ASD values are not affected by the empty-prediction boundary-metric issue.

The merged per-case file contains 150 rows:

| Dataset | Setting | Seeds | Number of per-case records |
|---|---:|---:|---:|
| LA | 4&nbsp;labeled | 2021, 2022, 2023 | 60 |
| Pancreas | 12&nbsp;labeled | 2021, 2022, 2023 | 60 |
| PROMISE12 | 7&nbsp;labeled | 2021, 2022, 2023 | 30 |

## Revision Artifacts

Additional reviewer-facing materials are stored in [`revision_artifacts`](revision_artifacts).

| Folder/File | Purpose |
|---|---|
| [`revision_artifacts/fig5_passive_similarity`](revision_artifacts/fig5_passive_similarity) | Passive feature-similarity curves, source data, and generated Fig. 5 files. |
| [`revision_artifacts/final_empty_prediction_check`](revision_artifacts/final_empty_prediction_check) | Final three-seed metric records used for the empty-prediction check. |
| [`revision_artifacts/scripts`](revision_artifacts/scripts) | Scripts used for passive Fig. 5 and final metric checks. |
| [`revision_artifacts/code_snapshot`](revision_artifacts/code_snapshot) | Lightweight helper-code snapshot for revision utilities and tools. |

## File Index

| Path | Description |
|---|---|
| [`evaluation_records/tables_I_III_X_three_run_statistics`](evaluation_records/tables_I_III_X_three_run_statistics) | Tables I-III and X repeated-run records. |
| [`evaluation_records/final_3seed_metrics_empty_policy`](evaluation_records/final_3seed_metrics_empty_policy) | Empty-prediction check records. |
| [`revision_artifacts`](revision_artifacts) | Fig. 5 and revision-related reproducibility materials. |
