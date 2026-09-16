#!/usr/bin/env python3
"""Passively recompute feature cosine similarity from saved LA checkpoints.

This script is intentionally separate from the training code.  It sets every
compared network to eval mode, uses torch.inference_mode(), and checks that no
model buffers are changed by the measurement pass.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

REPO = Path("/home/ygq/cq/code/AD-MT-revision/code_revised")
sys.path.insert(0, str(REPO))

from networks.vnet import VNet  # noqa: E402


PATCH_SIZE = (112, 112, 80)
EPOCH_RE = re.compile(r"ep_(\d+)")


EXPERIMENTS = {
    "CBASM-4L-seed2021": {
        "root": Path("/home/ygq/cq/code/AD-MT-revision/code_revised/results_revised/LA/4_labeled_std_seed2021/vnet"),
        "roles": {"student": "student", "teacher1": "teacher1", "teacher2": "teacher2", "ema": "ema"},
    },
    "CBASM-4L-seed2023": {
        "root": Path("/home/ygq/cq/code/AD-MT-revision/code_revised/results_revised/LA/4_labeled_std_seed2023/vnet"),
        "roles": {"student": "student", "teacher1": "teacher1", "teacher2": "teacher2", "ema": "ema"},
    },
    "ADMT-4L-endogenous": {
        "root": Path("/home/ygq/cq/code/AD-MT-revision/code_revised/results_revised/LA/4_labeled_r22_admt_endogenous/vnet"),
        "roles": {"student": "student", "teacher1": "tea1", "teacher2": "tea2"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, default=Path("/home/ygq/cq/code/AD-MT-revision/data_split/LA/train.list"))
    parser.add_argument("--data-dir", type=Path, default=Path("/home/ygq/cq/dataset/LA/Left_Atrium/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ygq/cq/code0915/passive_similarity_out"))
    parser.add_argument("--num-volumes", type=int, default=2, help="Number of unlabeled training volumes used for the passive measurement.")
    parser.add_argument("--labeled-num", type=int, default=4, help="Skip this many labeled volumes and measure on unlabeled volumes.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def checkpoint_epochs(exp: dict) -> Dict[str, Dict[int, Path]]:
    by_role: Dict[str, Dict[int, Path]] = {}
    for canonical_role, folder in exp["roles"].items():
        role_dir = exp["root"] / folder
        epochs: Dict[int, Path] = {}
        for path in role_dir.glob("ep_*.pth"):
            match = EPOCH_RE.search(path.name)
            if match:
                epochs[int(match.group(1))] = path
        by_role[canonical_role] = epochs
    return by_role


def common_epochs(by_role: Dict[str, Dict[int, Path]]) -> List[int]:
    role_sets = [set(v) for v in by_role.values()]
    if not role_sets:
        return []
    return sorted(set.intersection(*role_sets))


def center_crop_or_pad(image: np.ndarray, output_size: Tuple[int, int, int] = PATCH_SIZE) -> np.ndarray:
    if any(image.shape[i] <= output_size[i] for i in range(3)):
        pad = [max((output_size[i] - image.shape[i]) // 2 + 3, 0) for i in range(3)]
        image = np.pad(image, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])], mode="constant")

    starts = [int(round((image.shape[i] - output_size[i]) / 2.0)) for i in range(3)]
    return image[
        starts[0] : starts[0] + output_size[0],
        starts[1] : starts[1] + output_size[1],
        starts[2] : starts[2] + output_size[2],
    ]


def load_volumes(train_list: Path, data_dir: Path, labeled_num: int, num_volumes: int, device: torch.device) -> List[torch.Tensor]:
    names = [line.strip() for line in train_list.read_text().splitlines() if line.strip()]
    selected = names[labeled_num : labeled_num + num_volumes]
    volumes: List[torch.Tensor] = []
    for name in selected:
        h5_path = data_dir / name / "mri_norm2.h5"
        with h5py.File(h5_path, "r") as handle:
            image = center_crop_or_pad(handle["image"][:].astype(np.float32))
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        volumes.append(tensor)
    return volumes


def make_model(ckpt: Path, device: torch.device) -> VNet:
    model = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.endswith("total_ops") or key.endswith("total_params") or key in {"total_ops", "total_params"}:
            continue
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[warn] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def clone_buffers(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: buf.detach().clone() for name, buf in model.named_buffers()}


def buffers_unchanged(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> bool:
    current = dict(model.named_buffers())
    return all(torch.equal(before[name].to(current[name].device), current[name]) for name in before)


def feature(model: torch.nn.Module, volume: torch.Tensor) -> torch.Tensor:
    model.eval()
    output = model(volume)
    return output[1].reshape(-1)


def cosine(model_a: torch.nn.Module, model_b: torch.nn.Module, volumes: Iterable[torch.Tensor]) -> Tuple[float, bool]:
    before_a = clone_buffers(model_a)
    before_b = clone_buffers(model_b)
    values = []
    with torch.inference_mode():
        for volume in volumes:
            values.append(F.cosine_similarity(feature(model_a, volume), feature(model_b, volume), dim=0).item())
    unchanged = buffers_unchanged(before_a, model_a) and buffers_unchanged(before_b, model_b)
    return float(np.mean(values)), unchanged


def write_plot(csv_path: Path, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional.
        print(f"[warn] matplotlib unavailable, skipped plot: {exc}")
        return

    rows = []
    with csv_path.open() as handle:
        rows.extend(csv.DictReader(handle))

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    for method in sorted({row["method"] for row in rows}):
        points = [row for row in rows if row["method"] == method]
        xs = [int(row["epoch"]) for row in points]
        ys = [float(row["student_teacher_mean"]) for row in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Teacher-student feature cosine similarity")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    volumes = load_volumes(args.train_list, args.data_dir, args.labeled_num, args.num_volumes, device)

    rows = []
    availability_rows = []
    for name, exp in EXPERIMENTS.items():
        by_role = checkpoint_epochs(exp)
        epochs = common_epochs(by_role)
        availability_rows.append({
            "method": name,
            "roles": ",".join(by_role),
            "common_epoch_count": len(epochs),
            "common_epochs": " ".join(map(str, epochs)),
        })
        for epoch in epochs:
            models = {role: make_model(paths[epoch], device) for role, paths in by_role.items()}
            t12, ok_t12 = cosine(models["teacher1"], models["teacher2"], volumes)
            st1, ok_st1 = cosine(models["student"], models["teacher1"], volumes)
            st2, ok_st2 = cosine(models["student"], models["teacher2"], volumes)
            row = {
                "method": name,
                "epoch": epoch,
                "teacher_teacher": t12,
                "student_teacher_1": st1,
                "student_teacher_2": st2,
                "student_teacher_mean": (st1 + st2) / 2,
                "buffers_unchanged": ok_t12 and ok_st1 and ok_st2,
            }
            if "ema" in models:
                sema, ok_sema = cosine(models["student"], models["ema"], volumes)
                row["student_ema"] = sema
                row["buffers_unchanged"] = row["buffers_unchanged"] and ok_sema
            rows.append(row)
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_path = args.out_dir / "passive_similarity.csv"
    fieldnames = [
        "method", "epoch", "teacher_teacher", "student_teacher_1", "student_teacher_2",
        "student_teacher_mean", "student_ema", "buffers_unchanged",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    availability_path = args.out_dir / "checkpoint_availability.csv"
    with availability_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "roles", "common_epoch_count", "common_epochs"])
        writer.writeheader()
        writer.writerows(availability_rows)

    write_plot(csv_path, args.out_dir / "passive_similarity.png")
    print(f"Wrote {csv_path}")
    print(f"Wrote {availability_path}")


if __name__ == "__main__":
    main()
