#!/usr/bin/env python3
"""Evaluate final EMA checkpoints for the three main datasets.

The evaluation uses the corrected empty-set policy in this code copy:
- pred empty, gt present: Dice/Jaccard = 0, HD95/ASD = image diagonal penalty.
- pred present, gt empty: Dice/Jaccard = 0, HD95/ASD = image diagonal penalty.
- both empty: Dice/Jaccard = 1, HD95/ASD = 0.
"""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ORIG_RESULTS = Path("/home/ygq/cq/code/AD-MT-revision/code_revised/results_revised")
OUT_ROOT = Path("/home/ygq/cq/code0915/final_3seed_metrics_empty_policy")
DATA_SPLIT = Path("/home/ygq/cq/code/AD-MT-revision/data_split")
PYTHON = Path(os.environ.get("EVAL_PYTHON", sys.executable))


RUNS = [
    # Headline/final setting for each of the three main datasets.
    # LA, 5% labels. Three final repeated runs are available.
    {
        "dataset": "LA",
        "setting": "4_labeled",
        "label_num": 4,
        "model": "vnet",
        "script": "test_performance_3d.py",
        "root_path": DATA_SPLIT / "LA",
        "data_path": Path("/home/ygq/cq/dataset/LA/Left_Atrium/data"),
        "num_classes": 2,
        "seeds": {
            2021: ORIG_RESULTS / "LA/4_labeled_std_seed2021/vnet/vnet_best_ema_model.pth",
            2022: ORIG_RESULTS / "LA/4_labeled_std_seed2022/vnet/vnet_best_ema_model.pth",
            2023: ORIG_RESULTS / "LA/4_labeled_std_seed2023/vnet/vnet_best_ema_model.pth",
        },
    },
    # Pancreas, 20% labels.
    {
        "dataset": "Pancreas",
        "setting": "12_labeled",
        "label_num": 12,
        "model": "vnet",
        "script": "test_performance_3d.py",
        "root_path": DATA_SPLIT / "Pancreas",
        "data_path": Path("/home/ygq/cq/dataset/Pancreas/h5file"),
        "num_classes": 2,
        "seeds": {
            2021: ORIG_RESULTS / "Pancreas/12_labeled_std_seed2021/vnet/vnet_best_ema_model.pth",
            2022: ORIG_RESULTS / "Pancreas/12_labeled_std_seed2022/vnet/vnet_best_ema_model.pth",
            2023: ORIG_RESULTS / "Pancreas/12_labeled_std_seed2023/vnet/vnet_best_ema_model.pth",
        },
    },
    # PROMISE12 is stored as Prostate in the codebase, 20% labels.
    {
        "dataset": "PROMISE12",
        "setting": "7_labeled",
        "label_num": 7,
        "model": "unet",
        "script": "test_performance_2d.py",
        "root_path": DATA_SPLIT / "Prostate",
        "data_path": Path("/home/ygq/cq/dataset/Prostate/h5file"),
        "num_classes": 2,
        "seeds": {
            2021: ORIG_RESULTS / "Prostate/7_labeled_std_seed2021/unet/unet_best_ema_model.pth",
            2022: ORIG_RESULTS / "Prostate/7_labeled_std_seed2022/unet/unet_best_ema_model.pth",
            2023: ORIG_RESULTS / "Prostate/7_labeled_std_seed2023/unet/unet_best_ema_model.pth",
        },
    },
]


def run_eval(run: dict, seed: int, ckpt: Path, gpu: int) -> Path:
    out_dir = OUT_ROOT / run["dataset"] / run["setting"] / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = out_dir / "case_metrics_ema.csv"
    summary_csv = out_dir / "summary_metrics_ema.csv"
    log_path = out_dir / "eval.log"

    if case_csv.exists() and summary_csv.exists():
        print(f"[skip] existing metrics: {run['dataset']} {run['setting']} seed{seed}")
        return summary_csv

    if run["script"] == "test_performance_3d.py":
        cmd = [
            str(PYTHON), run["script"],
            "--root_path", str(run["root_path"]),
            "--data_path", str(run["data_path"]),
            "--dataset", "LA" if run["dataset"] == "LA" else "Pancreas",
            "--res_path", str(out_dir / "unused_res"),
            "--exp", f"{run['dataset']}_{run['setting']}_seed{seed}",
            "--model", run["model"],
            "--num_classes", str(run["num_classes"]),
            "--gpu", str(gpu),
            "--labeled_num", str(run["label_num"]),
            "--model_type", "ema",
            "--checkpoint_path", str(ckpt),
            "--case_metrics_csv", str(case_csv),
            "--summary_csv", str(summary_csv),
        ]
    else:
        cmd = [
            str(PYTHON), run["script"],
            "--root_path", str(run["root_path"]),
            "--data_path", str(run["data_path"]),
            "--res_path", str(out_dir / "unused_res"),
            "--exp", f"{run['dataset']}_{run['setting']}_seed{seed}",
            "--model", run["model"],
            "--num_classes", str(run["num_classes"]),
            "--gpu_id", str(gpu),
            "--labeled_num", str(run["label_num"]),
            "--model_type", "ema",
            "--checkpoint_path", str(ckpt),
            "--case_metrics_csv", str(case_csv),
            "--summary_csv", str(summary_csv),
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=CODE_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return summary_csv


def read_summary(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if len(rows) != 1:
        # Binary datasets should have one foreground class.  If this ever becomes
        # multi-class, average classes for the dataset-level record.
        out = {}
        for metric in ["dice", "jaccard", "hd95", "asd"]:
            vals = [float(r[f"{metric}_mean"]) for r in rows]
            out[f"{metric}_mean"] = sum(vals) / len(vals)
        return out
    return rows[0]


def read_empty_counts(case_csv: Path) -> tuple[int, int]:
    rows = list(csv.DictReader(case_csv.open(newline="", encoding="utf-8")))
    suspicious = 0
    penalized = 0
    for row in rows:
        dice = float(row["dice"])
        hd95 = float(row["hd95"])
        asd = float(row["asd"])
        if dice == 0.0 and hd95 == 0.0 and asd == 0.0:
            suspicious += 1
        if dice == 0.0 and (hd95 > 0.0 or asd > 0.0):
            penalized += 1
    return suspicious, penalized


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    avg = sum(values) / len(values)
    if len(values) < 2:
        return avg, 0.0
    var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return avg, math.sqrt(var)


def write_final_summary(records: list[dict]) -> None:
    per_seed_path = OUT_ROOT / "per_seed_metrics.csv"
    fields = ["dataset", "setting", "seed", "dice", "jaccard", "hd95", "asd", "zero_boundary_rows", "empty_penalty_rows", "checkpoint"]
    with per_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    grouped = {}
    for rec in records:
        grouped.setdefault((rec["dataset"], rec["setting"]), []).append(rec)

    summary_path = OUT_ROOT / "mean_std_metrics.csv"
    summary_fields = [
        "dataset", "setting", "n_seeds", "seeds",
        "dice_mean", "dice_std", "jaccard_mean", "jaccard_std",
        "hd95_mean", "hd95_std", "asd_mean", "asd_std",
        "empty_penalty_rows_total", "zero_boundary_rows_total",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for (dataset, setting), group in sorted(grouped.items()):
            row = {
                "dataset": dataset,
                "setting": setting,
                "n_seeds": len(group),
                "seeds": " ".join(str(g["seed"]) for g in group),
                "empty_penalty_rows_total": sum(int(g["empty_penalty_rows"]) for g in group),
                "zero_boundary_rows_total": sum(int(g["zero_boundary_rows"]) for g in group),
            }
            for metric in ["dice", "jaccard", "hd95", "asd"]:
                avg, std = mean_std([float(g[metric]) for g in group])
                row[f"{metric}_mean"] = avg
                row[f"{metric}_std"] = std
            writer.writerow(row)

    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    gpu_cycle = [int(x) for x in os.environ.get("EVAL_GPUS", "0").split(",")]
    job_idx = 0
    for run in RUNS:
        for seed, ckpt in sorted(run["seeds"].items()):
            if not ckpt.exists():
                print(f"[skip] missing checkpoint: {ckpt}")
                continue
            gpu = gpu_cycle[job_idx % len(gpu_cycle)]
            job_idx += 1
            print(f"[eval] {run['dataset']} {run['setting']} seed{seed} on GPU {gpu}")
            summary_csv = run_eval(run, seed, ckpt, gpu)
            summary = read_summary(summary_csv)
            case_csv = summary_csv.with_name("case_metrics_ema.csv")
            suspicious, penalized = read_empty_counts(case_csv)
            records.append({
                "dataset": run["dataset"],
                "setting": run["setting"],
                "seed": seed,
                "dice": float(summary["dice_mean"]) * 100,
                "jaccard": float(summary["jaccard_mean"]) * 100,
                "hd95": float(summary["hd95_mean"]),
                "asd": float(summary["asd_mean"]),
                "zero_boundary_rows": suspicious,
                "empty_penalty_rows": penalized,
                "checkpoint": str(ckpt),
            })
    write_final_summary(records)


if __name__ == "__main__":
    main()
