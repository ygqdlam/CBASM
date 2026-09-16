#!/usr/bin/env python3
"""Build Fig. 5 passive similarity CSV/PNG from corrected training logs."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


OUT_ROOT = Path("/home/ygq/cq/code0915/results_fig5_passive")
LA_ROOT = OUT_ROOT / "LA"
FIG_DIR = OUT_ROOT / "fig5"
ITER_RE = re.compile(
    r"iter_num:(?P<iter>\d+),cosine_sim_t12:(?P<t12>tensor\()? (?P<t12v>[-+0-9.eE]+)"
)


def parse_value(text: str, key: str) -> float | None:
    match = re.search(key + r":(?:tensor\()?([-+0-9.eE]+)", text)
    return float(match.group(1)) if match else None


def parse_log(method: str, log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "iter_num:" not in line or "cosine_sim_t12:" not in line:
            continue
        iter_match = re.search(r"iter_num:(\d+)", line)
        if not iter_match:
            continue
        st1 = parse_value(line, "cosine_sim_st1")
        st2 = parse_value(line, "cosine_sim_st2")
        if st1 is None or st2 is None:
            continue
        row = {
            "method": method,
            "iteration": int(iter_match.group(1)),
            "teacher_teacher": parse_value(line, "cosine_sim_t12"),
            "student_teacher_1": st1,
            "student_teacher_2": st2,
            "student_teacher_mean": (st1 + st2) / 2,
            "student_ema": parse_value(line, "cosine_sim_sema"),
            "buffers_unchanged": "cosine_sim_buffers_unchanged:True" in line,
        }
        rows.append(row)
    return rows


def parse_epoch_log(method: str, log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "epoch:" not in line or "cosine_sim_t12_avg:" not in line:
            continue
        epoch_match = re.search(r"epoch:(\d+)", line)
        if not epoch_match:
            continue
        st1 = parse_value(line, "cosine_sim_st1_avg")
        st2 = parse_value(line, "cosine_sim_st2_avg")
        if st1 is None or st2 is None:
            continue
        row = {
            "method": method,
            "epoch": int(epoch_match.group(1)),
            "teacher_teacher": parse_value(line, "cosine_sim_t12_avg"),
            "student_teacher_1": st1,
            "student_teacher_2": st2,
            "student_teacher_mean": (st1 + st2) / 2,
            "student_ema": parse_value(line, "cosine_sim_sema_avg")
            or parse_value(line, "cosine_sim_seam_avg"),
        }
        rows.append(row)
    return rows


def find_logs() -> dict[str, Path]:
    logs = {}
    for log_path in LA_ROOT.glob("4_labeled_fig5_passive_*_seed*/vnet/log.txt"):
        text = str(log_path)
        if "cbasm" in text:
            logs["CBASM"] = log_path
        elif "admt" in text:
            logs["AD-MT"] = log_path
    return logs


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "method",
        "iteration",
        "teacher_teacher",
        "student_teacher_1",
        "student_teacher_2",
        "student_teacher_mean",
        "student_ema",
        "buffers_unchanged",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_epoch_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "method",
        "epoch",
        "teacher_teacher",
        "student_teacher_1",
        "student_teacher_2",
        "student_teacher_mean",
        "student_ema",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bin_iteration_rows(rows: list[dict], bin_size: int = 100) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        bucket = (row["iteration"] - 1) // bin_size
        buckets[(row["method"], bucket)].append(row)

    binned = []
    for (method, bucket), points in sorted(buckets.items()):
        out = {
            "method": method,
            "iteration": int(round(sum(p["iteration"] for p in points) / len(points))),
            "buffers_unchanged": all(p["buffers_unchanged"] for p in points),
        }
        for key in [
            "teacher_teacher",
            "student_teacher_1",
            "student_teacher_2",
            "student_teacher_mean",
            "student_ema",
        ]:
            vals = [p[key] for p in points if p[key] is not None]
            out[key] = sum(vals) / len(vals) if vals else None
        binned.append(out)
    return binned


def write_plot(rows: list[dict], path: Path, x_key: str = "iteration") -> None:
    import matplotlib.pyplot as plt

    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    for method, points in sorted(by_method.items()):
        points = sorted(points, key=lambda row: row[x_key])
        ax.plot(
            [row[x_key] for row in points],
            [row["student_teacher_mean"] for row in points],
            label=method,
            linewidth=1.8,
        )
    ax.set_xlabel("Epoch" if x_key == "epoch" else "Iteration")
    ax.set_ylabel("Teacher-student feature cosine similarity")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logs = find_logs()
    rows = []
    for method, log_path in sorted(logs.items()):
        rows.extend(parse_log(method, log_path))
    epoch_rows = []
    for method, log_path in sorted(logs.items()):
        epoch_rows.extend(parse_epoch_log(method, log_path))
    if not rows:
        raise SystemExit(f"No cosine-similarity rows found under {LA_ROOT}")
    rows = sorted(rows, key=lambda row: (row["method"], row["iteration"]))
    binned_rows = bin_iteration_rows(rows, bin_size=100)
    epoch_rows = sorted(epoch_rows, key=lambda row: (row["method"], row["epoch"]))
    write_csv(rows, FIG_DIR / "fig5_passive_similarity_source_data.csv")
    write_csv(binned_rows, FIG_DIR / "fig5_passive_similarity_iteration_binned_source_data.csv")
    write_plot(binned_rows, FIG_DIR / "fig5_passive_similarity.png", x_key="iteration")
    if epoch_rows:
        write_epoch_csv(epoch_rows, FIG_DIR / "fig5_passive_similarity_epoch_source_data.csv")
        write_plot(epoch_rows, FIG_DIR / "fig5_passive_similarity_epoch.png", x_key="epoch")

    bad = [row for row in rows if not row["buffers_unchanged"]]
    print(f"Wrote {FIG_DIR / 'fig5_passive_similarity_source_data.csv'}")
    print(f"Wrote {FIG_DIR / 'fig5_passive_similarity_iteration_binned_source_data.csv'}")
    print(f"Wrote {FIG_DIR / 'fig5_passive_similarity.png'}")
    if epoch_rows:
        print(f"Wrote {FIG_DIR / 'fig5_passive_similarity_epoch_source_data.csv'}")
        print(f"Wrote {FIG_DIR / 'fig5_passive_similarity_epoch.png'}")
        print(f"Epoch rows: {len(epoch_rows)}")
    print(f"Rows: {len(rows)}; binned rows: {len(binned_rows)}; buffers unchanged for all rows: {not bad}")


if __name__ == "__main__":
    main()
