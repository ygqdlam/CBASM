#!/usr/bin/env python3
"""Redraw the R3-8 teacher-student similarity figure from epoch-level logs."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = CODE_DIR.parents[2] / "Thesis-Major-Revision-main" / "cq-paper-1"
LOG_DIR = CODE_DIR / "logs_r22"
OUT_STEM = PAPER_DIR / "chen4"
CSV_OUT = PAPER_DIR / "chen4_epoch_similarity_source_data.csv"

CBASM_LOG = LOG_DIR / "r22_cbasm_gpu1.log"
ADMT_LOG = LOG_DIR / "r22_admt_endogenous_gpu1.log"

EPOCH_PATTERN = re.compile(
    r"epoch:(?P<epoch>\d+).*?"
    r"cosine_sim_t12_avg:(?P<t12>[0-9.eE+-]+).*?"
    r"cosine_sim_st1_avg:(?P<st1>[0-9.eE+-]+).*?"
    r"cosine_sim_st2_avg:(?P<st2>[0-9.eE+-]+)"
)


def parse_epoch_similarity(log_path: Path, method: str) -> pd.DataFrame:
    rows = []
    with log_path.open("r", errors="ignore") as handle:
        for line in handle:
            match = EPOCH_PATTERN.search(line)
            if not match:
                continue
            row = {
                "method": method,
                "epoch": int(match.group("epoch")),
                "teacher_teacher": float(match.group("t12")),
                "student_teacher_1": float(match.group("st1")),
                "student_teacher_2": float(match.group("st2")),
            }
            row["student_teacher_mean"] = (
                row["student_teacher_1"] + row["student_teacher_2"]
            ) / 2.0
            rows.append(row)
    if not rows:
        raise ValueError(f"No epoch-level cosine similarity records found in {log_path}")
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def draw_similarity_figure(df: pd.DataFrame) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.55, 2.55))

    styles = {
        "CBASM": {"color": "#2F6B4F", "marker": "o"},
        "AD-MT": {"color": "#B55A30", "marker": "s"},
    }
    for method, method_df in df.groupby("method", sort=False):
        method_df = method_df.sort_values("epoch")
        ax.plot(
            method_df["epoch"],
            method_df["student_teacher_mean"],
            label=method,
            color=styles[method]["color"],
            marker=styles[method]["marker"],
            markevery=50,
            markersize=2.4,
            linewidth=1.7,
            markeredgewidth=0,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Feature-space cosine similarity")
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 0.75)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper left", handlelength=2.4)

    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT_STEM}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path in (CBASM_LOG, ADMT_LOG):
        if not path.exists():
            raise FileNotFoundError(path)
    if not PAPER_DIR.exists():
        raise FileNotFoundError(PAPER_DIR)

    df = pd.concat(
        [
            parse_epoch_similarity(CBASM_LOG, "CBASM"),
            parse_epoch_similarity(ADMT_LOG, "AD-MT"),
        ],
        ignore_index=True,
    )
    df.to_csv(CSV_OUT, index=False)
    draw_similarity_figure(df)

    late = (
        df[df["epoch"] >= int(df["epoch"].max() * 0.7)]
        .groupby("method")["student_teacher_mean"]
        .mean()
        .round(3)
    )
    print(f"Wrote {OUT_STEM}.png/.svg/.pdf")
    print(f"Wrote {CSV_OUT}")
    print("Late-stage mean feature-space cosine similarity:")
    print(late.to_string())


if __name__ == "__main__":
    main()
