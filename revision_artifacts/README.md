# Revision Artifacts

This folder contains reproducibility materials for the revised analyses and reviewer-requested checks.

## Contents

| Folder/File | Purpose |
|---|---|
| [`fig5_passive_similarity`](fig5_passive_similarity) | Passive feature-similarity curves, source data, and generated Fig. 5 files. |
| [`final_empty_prediction_check`](final_empty_prediction_check) | Final three-seed evaluation records used to verify the empty-prediction issue. |
| [`scripts`](scripts) | Scripts used to run the passive Fig. 5 analysis and final metric checks. |
| [`code_snapshot`](code_snapshot) | Lightweight helper-code snapshot for revision utilities and tools. |
| [`MODIFICATION_REPORT.md`](MODIFICATION_REPORT.md) | Summary of code and experiment updates. |
| [`REVISION_GUIDE.md`](REVISION_GUIDE.md) | Guide for running the revision-related checks. |

## Fig. 5 Passive Similarity

The main passive feature-similarity figure is:

- [`fig5_passive_similarity/fig5_passive_similarity.png`](fig5_passive_similarity/fig5_passive_similarity.png)

The corresponding source data are:

- [`fig5_passive_similarity/fig5_passive_similarity_source_data.csv`](fig5_passive_similarity/fig5_passive_similarity_source_data.csv)
- [`fig5_passive_similarity/fig5_passive_similarity_iteration_binned_source_data.csv`](fig5_passive_similarity/fig5_passive_similarity_iteration_binned_source_data.csv)

## Empty-prediction Check

The final-checkpoint evaluation records are in:

- [`final_empty_prediction_check/mean_std_metrics.csv`](final_empty_prediction_check/mean_std_metrics.csv)
- [`final_empty_prediction_check/per_seed_metrics.csv`](final_empty_prediction_check/per_seed_metrics.csv)

Per-case metrics are kept inside the dataset/seed subfolders under [`final_empty_prediction_check`](final_empty_prediction_check).
