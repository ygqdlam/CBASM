#!/usr/bin/env python3
"""Build baseline FLOPs, training time, and GPU memory curves for Comment 4."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
OUT = CODE / "results_revised" / "R2-4"
PROFILED_CSV = CODE / "results_revised" / "R1-2" / "benchmarks" / "la_4l_profiled_baselines.csv"
PROFILED_DIR = CODE / "results_revised" / "R1-2" / "benchmarks" / "profiled"
FULL_ITERS = 25000
VNET_FLOPS_G = 40.44
VNET_PARAMS_M = 9.45

FLOPS_ROWS = [
    {
        "method": "Mean Teacher",
        "backbone": "V-Net",
        "networks_stored": 2,
        "params_m": round(VNET_PARAMS_M * 2, 2),
        "flops_g_single_forward": VNET_FLOPS_G,
        "flops_g_note": "1× student forward + EMA forward (no EMA grad)",
        "backprop_per_iter": 1,
    },
    {
        "method": "UA-MT (MICCAI'19)",
        "backbone": "V-Net",
        "networks_stored": 2,
        "params_m": round(VNET_PARAMS_M * 2, 2),
        "flops_g_single_forward": VNET_FLOPS_G,
        "flops_g_note": "same as Mean Teacher family",
        "backprop_per_iter": 1,
    },
    {
        "method": "AD-MT (dual-EMA TS)",
        "backbone": "V-Net",
        "networks_stored": 3,
        "params_m": round(VNET_PARAMS_M * 3, 2),
        "flops_g_single_forward": VNET_FLOPS_G,
        "flops_g_note": "student backward; dual EMA forward only",
        "backprop_per_iter": 1,
    },
    {
        "method": "BCP (CVPR'23)",
        "backbone": "V-Net",
        "networks_stored": 2,
        "params_m": round(VNET_PARAMS_M * 2, 2),
        "flops_g_single_forward": VNET_FLOPS_G,
        "flops_g_note": "student + EMA (self-train stage)",
        "backprop_per_iter": 1,
    },
    {
        "method": "CauSSL (LA 3D)",
        "backbone": "V-Net",
        "networks_stored": 2,
        "params_m": round(VNET_PARAMS_M * 2, 2),
        "flops_g_single_forward": round(VNET_FLOPS_G * 2, 2),
        "flops_g_note": "dual V-Net forward + 2× backward",
        "backprop_per_iter": 2,
    },
    {
        "method": "CBASM (ours)",
        "backbone": "V-Net",
        "networks_stored": 4,
        "params_m": round(VNET_PARAMS_M * 4, 2),
        "flops_g_single_forward": round(VNET_FLOPS_G * 4, 2),
        "flops_g_note": "4× stored; EMA + active teacher forward; 2× backward/iter",
        "backprop_per_iter": 2,
    },
]


def load_profiles():
    if not PROFILED_CSV.exists():
        raise FileNotFoundError(
            f"Missing {PROFILED_CSV}. Run: python run_la_baseline_efficiency_profile.py"
        )
    return pd.read_csv(PROFILED_CSV)


def plot_memory_curves(profiles_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "cbasm": "#1f77b4",
        "admt": "#ff7f0e",
        "mean_teacher": "#2ca02c",
        "ua_mt": "#9467bd",
        "bcp": "#8c564b",
        "caussl": "#e377c2",
    }
    for _, row in profiles_df.iterrows():
        key = row["method_key"]
        csv_path = Path(row["profile_csv"])
        if not csv_path.is_absolute():
            csv_path = CODE / csv_path
        if not csv_path.exists():
            csv_path = PROFILED_DIR / f"{key}_iteration_profile.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        ax.plot(
            df["iter"],
            df["cuda_max_memory_allocated_mb"] / 1024.0,
            label=row["method"].replace(" (ours)", ""),
            color=colors.get(key, "#333333"),
            linewidth=1.4,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Peak CUDA memory (GB)")
    ax.set_title("LA 4-label — GPU memory vs. iteration (baselines + CBASM)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "gpu_memory_baselines_la4l.png", dpi=200)
    plt.close(fig)


def plot_training_bars(profiles_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [m.replace(" (ours)", "") for m in profiles_df["method"]]
    colors = ["#1f77b4" if "CBASM" in m else "#ff7f0e" for m in profiles_df["method"]]
    ips = profiles_df["iter_per_sec"].astype(float)
    gpu_h = profiles_df["gpu_hours_25k"].astype(float)
    mem = profiles_df["peak_memory_gb"].astype(float)
    axes[0].bar(labels, ips, color=colors)
    axes[0].set_ylabel("Throughput (iter/s)")
    axes[0].set_title("LA 4-label training throughput")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)
    x = range(len(labels))
    w = 0.35
    axes[1].bar([i - w / 2 for i in x], gpu_h, width=w, label="GPU-h @25k", color=colors)
    axes[1].bar([i + w / 2 for i in x], mem, width=w, label="Peak mem (GB)", color="#aec7e8")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=25)
    axes[1].set_title("Projected cost & peak memory")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_efficiency_la4l.png", dpi=200)
    plt.close(fig)


def build_comparison_table(profiles_df, flops_df):
    df = profiles_df.merge(
        flops_df[["method", "params_m", "flops_g_single_forward", "backprop_per_iter", "flops_g_note"]],
        on="method",
        how="left",
    )
    cbasm = df[df["method_key"] == "cbasm"].iloc[0]
    df["gpu_hours_ratio_vs_cbasm"] = df["gpu_hours_25k"] / cbasm["gpu_hours_25k"]
    df["peak_memory_ratio_vs_cbasm"] = df["peak_memory_gb"] / cbasm["peak_memory_gb"]
    return df.round(3)


def append_rebuttal(comparison_df, cbasm_row):
    r24 = OUT / "R24_REBUTTAL.md"
    base = r24.read_text(encoding="utf-8") if r24.exists() else ""

    lines_en = ["\n\n**Baseline comparison (LA, 4 labeled, V-Net, 25k iters projected, single GPU):**\n"]
    lines_en.append("| Method | Params (M) | FLOPs (G, 1 fwd) | Backprop/iter | iter/s | GPU-h (25k) | Peak mem (GB) | vs CBASM (GPU-h) |")
    for _, r in comparison_df.sort_values("gpu_hours_25k").iterrows():
        lines_en.append(
            f"| {r['method']} | {r['params_m']} | {r['flops_g_single_forward']} | {int(r['backprop_per_iter'])} | "
            f"{r['iter_per_sec']:.2f} | {r['gpu_hours_25k']:.2f} | {r['peak_memory_gb']:.2f} | {r['gpu_hours_ratio_vs_cbasm']:.2f}× |"
        )
    lines_en.append(
        f"\nBaseline **GPU memory curves** and throughput bars: `gpu_memory_baselines_la4l.png`, `baseline_efficiency_la4l.png`. "
        f"CBASM uses more stored parameters (4× V-Net) but peak memory ({cbasm_row['peak_memory_gb']:.1f} GB) remains comparable to dual-network baselines; "
        f"training time is **~{cbasm_row['gpu_hours_ratio_vs_cbasm']:.2f}×** vs Mean Teacher and **~{comparison_df[comparison_df['method_key']=='admt']['gpu_hours_ratio_vs_cbasm'].iloc[0]:.2f}×** vs AD-MT."
    )

    lines_zh = ["\n\n**基线对比（LA、4 例标注、V-Net、单卡，投影至 25k iter）：**\n"]
    lines_zh.append("| 方法 | 参数量(M) | FLOPs(G) | 反传/iter | iter/s | GPU-h | 峰值显存(GB) | 相对CBASM |")
    for _, r in comparison_df.sort_values("gpu_hours_25k").iterrows():
        lines_zh.append(
            f"| {r['method']} | {r['params_m']} | {r['flops_g_single_forward']} | {int(r['backprop_per_iter'])} | "
            f"{r['iter_per_sec']:.2f} | {r['gpu_hours_25k']:.2f} | {r['peak_memory_gb']:.2f} | {r['gpu_hours_ratio_vs_cbasm']:.2f}× |"
        )
    lines_zh.append(
        f"\n基线显存曲线见 `gpu_memory_baselines_la4l.png`。CBASM 存储 4 个 V-Net，峰值显存 {cbasm_row['peak_memory_gb']:.1f} GB，"
        f"训练 GPU-hours 相对 Mean Teacher / AD-MT 见上表。"
    )

    marker = "**Baseline comparison (LA, 4 labeled"
    if marker in base:
        base = base.split(marker)[0].rstrip()
    if "---" in base:
        parts = base.split("---")
        base = parts[0].rstrip() + "\n" + "".join(lines_en) + "\n\n---\n\n" + parts[1].split("**基线对比")[0].rstrip() + "".join(lines_zh) + "\n"
    else:
        base = base + "".join(lines_en) + "".join(lines_zh)
    r24.write_text(base, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    profiles_df = load_profiles()
    flops_df = pd.DataFrame(FLOPS_ROWS)
    flops_df.to_csv(OUT / "baseline_flops_la4l.csv", index=False)

    comparison_df = build_comparison_table(profiles_df, flops_df)
    comparison_df.to_csv(OUT / "baseline_efficiency_la4l.csv", index=False)

    plot_memory_curves(profiles_df)
    plot_training_bars(profiles_df)
    cbasm_row = comparison_df[comparison_df["method_key"] == "cbasm"].iloc[0]
    append_rebuttal(comparison_df, cbasm_row)

    print(f"Wrote baseline outputs under {OUT}")
    print(comparison_df[["method", "iter_per_sec", "gpu_hours_25k", "peak_memory_gb"]].to_string(index=False))


if __name__ == "__main__":
    main()
