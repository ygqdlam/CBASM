#!/usr/bin/env python3
"""Comment 2: wall-clock training time and overhead vs baselines."""
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_DIR / "results_revised" / "R1-2"
R24_SUMMARY = CODE_DIR / "results_revised" / "R2-4" / "efficiency_summary.csv"
BENCH_RAW = OUT_DIR / "benchmarks" / "la_4l_benchmark_raw.csv"
BENCH_EXT = OUT_DIR / "benchmarks" / "la_4l_extended_benchmark.csv"
BENCH_CAUBETA = OUT_DIR / "benchmarks" / "caussl_beta_benchmark.csv"
BENCH_INV = OUT_DIR / "benchmarks" / "method_inventory.md"
FULL_ITERS = 25000


def load_cbasm_full():
    df = pd.read_csv(R24_SUMMARY)
    la = df[df["dataset"] == "LA"].iloc[0]
    return {
        "method": "CBASM (ours, full 25k×3 seeds)",
        "method_key": "cbasm_full",
        "backprop_networks": 2,
        "iter_per_sec": float(la["iter_per_sec_mean"]),
        "iter_per_sec_std": float(la["iter_per_sec_std"]),
        "wall_hours_25k": float(la["wall_hours_mean"]),
        "wall_hours_std": float(la["wall_hours_std"]),
        "gpu_hours_25k": float(la["gpu_hours_mean"]),
        "gpu_hours_std": float(la["gpu_hours_std"]),
        "peak_memory_gb": float(la["peak_memory_gb_mean"]),
        "source": "std_seed_profiling",
    }


def load_benchmarks():
    rows = []
    for csv_path, name_key in [
        (BENCH_RAW, "method"),
        (BENCH_EXT, "paper_name"),
        (BENCH_CAUBETA, "method"),
    ]:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            ips = float(r["iter_per_sec"])
            label = r.get(name_key) or r.get("method") or r.get("paper_name")
            rows.append(
                {
                    "method": label,
                    "method_key": r["method_key"],
                    "backprop_networks": int(r["backprop_networks"]),
                    "iter_per_sec": ips,
                    "iter_per_sec_std": 0.0,
                    "wall_hours_25k": FULL_ITERS / ips / 3600.0,
                    "wall_hours_std": 0.0,
                    "gpu_hours_25k": FULL_ITERS / ips / 3600.0,
                    "gpu_hours_std": 0.0,
                    "peak_memory_gb": float(r["peak_memory_gb"]) if pd.notna(r.get("peak_memory_gb")) else None,
                    "source": r.get("source", "benchmark"),
                    "note": r.get("note", ""),
                    "dataset": r.get("dataset", "LA_4L_3D"),
                }
            )
    return rows


def plot_overhead(rows, cbasm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [r["method"].replace(" (ours)", "") for r in rows]
    colors = []
    for r in rows:
        if "CBASM" in r["method"]:
            colors.append("#1f77b4")
        elif "AD-MT" in r["method"]:
            colors.append("#ff7f0e")
        elif "Mean Teacher" in r["method"] or "UA-MT" in r["method"]:
            colors.append("#2ca02c")
        else:
            colors.append("#9467bd")
    ips = [r["iter_per_sec"] for r in rows]
    gpu_h = [r["gpu_hours_25k"] for r in rows]
    axes[0].bar(labels, ips, color=colors)
    axes[0].set_ylabel("Throughput (iter/s)")
    axes[0].set_title("LA 4-label training throughput")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, gpu_h, color=colors)
    axes[1].set_ylabel("Projected GPU-hours @ 25k iters")
    axes[1].set_title("Wall-clock / GPU-hour cost (single GPU)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_overhead_la4l.png", dpi=200)
    plt.close(fig)


def build_table(rows, cbasm):
    ref_ips = cbasm["iter_per_sec"]
    ref_gpu = cbasm["gpu_hours_25k"]
    table = []
    for r in rows:
        table.append(
            {
                "method": r["method"],
                "backprop_networks_per_iter": r["backprop_networks"],
                "iter_per_sec": round(r["iter_per_sec"], 3),
                "wall_hours_25k": round(r["wall_hours_25k"], 3),
                "gpu_hours_25k": round(r["gpu_hours_25k"], 3),
                "peak_memory_gb": round(r["peak_memory_gb"], 2) if r["peak_memory_gb"] else "",
                "iter_per_sec_ratio_vs_cbasm": round(ref_ips / r["iter_per_sec"], 3),
                "gpu_hours_ratio_vs_cbasm": round(r["gpu_hours_25k"] / ref_gpu, 3),
                "source": r["source"],
            }
        )
    return pd.DataFrame(table)


def build_rebuttal(cbasm, table_df):
    cbasm_row = table_df[table_df["method"].str.contains("CBASM")].iloc[0]
    mt = table_df[table_df["method"].str.contains("Mean Teacher")].iloc[0]
    admt = table_df[table_df["method"].str.contains("AD-MT")].iloc[0]

    extra_lines_en = []
    extra_lines_zh = []
    for key, label in [
        ("ua_mt", "UA-MT"),
        ("bcp", "BCP"),
    ]:
        sub = table_df[table_df["method"].str.contains(label, case=False, regex=False)]
        if len(sub):
            row = sub.iloc[0]
            extra_lines_en.append(
                f"| {row['method']} | {int(row['backprop_networks_per_iter'])} | {row['iter_per_sec']:.2f} | {row['gpu_hours_25k']:.2f} | **{row['gpu_hours_ratio_vs_cbasm']:.2f}×** |"
            )
            extra_lines_zh.append(
                f"| {row['method']} | {int(row['backprop_networks_per_iter'])} | {row['iter_per_sec']:.2f} | {row['gpu_hours_25k']:.2f} | {row['gpu_hours_ratio_vs_cbasm']:.2f}× |"
            )
    extra_en = "\n".join(extra_lines_en)
    extra_zh = "\n".join(extra_lines_zh)

    missing_note_en = ""
    missing_note_zh = ""
    if BENCH_INV.exists():
        missing_note_en = (
            "\n\n**Other Table-I baselines (SASSNet, DTC, MC-Net+, SS-NET, MCF):** "
            "their official LA trainers were **not available** in our local codebase; "
            "see `benchmarks/method_inventory.md`. CauSSL and beta-FFT target other datasets/modalities."
        )
        missing_note_zh = (
            "\n\n**Table I 其余基线（SASSNet、DTC、MC-Net+、SS-NET、MCF）：** "
            "本地未找到可直接在 LA 上运行的官方训练脚本，详见 `benchmarks/method_inventory.md`。"
            "CauSSL、beta-FFT 面向其他数据集/维度。"
        )

    en = f"""**Response to Comment 2 (wall-clock time & computational overhead)**

We clarify the training cost of CBASM and quantify overhead relative to representative SSL baselines on **LA, 4 labeled (Table I setting, V-Net, 25k iterations, single GPU)**.

**Backpropagation per iteration (as stated in the paper):**
- **CBASM:** **2 networks** receive gradients each iteration — the **student** and **one active DRSCM co-teacher** (alternating). The EMA teacher is updated by exponential moving average **without** backpropagation.
- **Mean Teacher baseline:** **1 network** (student); EMA updated without backward.
- **AD-MT (dual-EMA teacher–student):** **1 network** (student); two EMA teachers updated without backward.

**Measured wall-clock & throughput (CBASM, 3 seeds, `--enable_profile`):**
| Metric | CBASM (ours) |
| Throughput | **{cbasm['iter_per_sec']:.2f} ± {cbasm['iter_per_sec_std']:.2f} iter/s** |
| Wall time (25k) | **{cbasm['wall_hours_25k']:.2f} ± {cbasm['wall_hours_std']:.2f} h** |
| GPU-hours (25k) | **{cbasm['gpu_hours_25k']:.2f} ± {cbasm['gpu_hours_std']:.2f}** |
| Peak CUDA memory | **{cbasm['peak_memory_gb']:.1f} GB** (≤8 GB on RTX 3070 class GPUs) |

**Relative overhead vs baselines (same LA split, projected to 25k iters from benchmarked steady-state throughput):**
| Method | Backprop / iter | iter/s | GPU-hours (25k) | vs CBASM (GPU-h ratio) |
| Mean Teacher | 1 | {mt['iter_per_sec']:.2f} | {mt['gpu_hours_25k']:.2f} | **{mt['gpu_hours_ratio_vs_cbasm']:.2f}×** |
| AD-MT (dual-EMA) | 1 | {admt['iter_per_sec']:.2f} | {admt['gpu_hours_25k']:.2f} | **{admt['gpu_hours_ratio_vs_cbasm']:.2f}×** |
{extra_en}
| **CBASM** | **2** | **{cbasm_row['iter_per_sec']:.2f}** | **{cbasm_row['gpu_hours_25k']:.2f}** | **1.00×** |

CBASM is **~{(cbasm_row['gpu_hours_25k']/mt['gpu_hours_25k']):.2f}×** the GPU-hours of Mean Teacher and **~{(cbasm_row['gpu_hours_25k']/admt['gpu_hours_25k']):.2f}×** vs AD-MT on LA, reflecting one extra backward pass plus additional forward passes for cross-paradigm pseudo-labeling. This overhead is modest in absolute terms (**≈{cbasm_row['gpu_hours_25k']:.1f} GPU-hours** for full LA training) and remains practical on a single consumer GPU (**peak memory {cbasm['peak_memory_gb']:.1f} GB**).{missing_note_en}

See `training_overhead_la4l.png` and Tables I–III for accuracy gains that justify this cost.
"""

    zh = f"""**回复意见 2（墙钟训练时间与算力开销）**

我们在 **LA、4 例标注、25k iterations、单卡** 设置下补充了墙钟时间/GPU-hours，并与 Mean Teacher、AD-MT 对比。

**每 iter 反传网络数（与正文一致）：**
- **CBASM：2 个**（student + 当前激活 co-teacher）；EMA **无反传**更新。
- **Mean Teacher：1 个**（student）。
- **AD-MT：1 个**（student）；双 EMA **无反传**。

**CBASM 实测（3 seeds，`--enable_profile`）：**
- 吞吐：**{cbasm['iter_per_sec']:.2f} ± {cbasm['iter_per_sec_std']:.2f} iter/s**
- 墙钟（25k）：**{cbasm['wall_hours_25k']:.2f} ± {cbasm['wall_hours_std']:.2f} h**
- GPU-hours：**{cbasm['gpu_hours_25k']:.2f} ± {cbasm['gpu_hours_std']:.2f}**
- 峰值显存：**{cbasm['peak_memory_gb']:.1f} GB**（RTX 3070 级 ≤8 GB 可训）

**相对开销（投影至 25k iter）：**
| 方法 | 反传/iter | iter/s | GPU-hours | 相对 CBASM |
| Mean Teacher | 1 | {mt['iter_per_sec']:.2f} | {mt['gpu_hours_25k']:.2f} | {mt['gpu_hours_ratio_vs_cbasm']:.2f}× |
| AD-MT | 1 | {admt['iter_per_sec']:.2f} | {admt['gpu_hours_25k']:.2f} | {admt['gpu_hours_ratio_vs_cbasm']:.2f}× |
{extra_zh}
| CBASM | 2 | {cbasm_row['iter_per_sec']:.2f} | {cbasm_row['gpu_hours_25k']:.2f} | 1.00× |

CBASM 相对 Mean Teacher 约 **{(cbasm_row['gpu_hours_25k']/mt['gpu_hours_25k']):.2f}×** GPU-hours，绝对成本约 **{cbasm_row['gpu_hours_25k']:.1f} h/卡**，在单卡消费级 GPU 上仍实用。{missing_note_zh}
"""
    return en, zh


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cbasm = load_cbasm_full()
    bench = load_benchmarks()
    if not bench:
        raise FileNotFoundError(
            f"Missing {BENCH_RAW}. Run: python run_r12_efficiency_benchmark_la.py"
        )
    # Prefer full CBASM stats; replace short benchmark CBASM row if present
    seen = set()
    deduped = []
    for r in bench:
        if r["method_key"] in seen:
            continue
        seen.add(r["method_key"])
        deduped.append(r)
    rows = [r for r in deduped if r["method_key"] != "cbasm"]
    rows.insert(
        0,
        {
            **cbasm,
            "method": "CBASM (ours)",
            "method_key": "cbasm",
        },
    )
    table_df = build_table(rows, cbasm)
    table_df.to_csv(OUT_DIR / "training_overhead_comparison.csv", index=False)
    plot_overhead(rows, cbasm)
    en, zh = build_rebuttal(cbasm, table_df)
    (OUT_DIR / "R12_REBUTTAL.md").write_text(
        "# Comment 2 — Wall-clock Training Time & Overhead\n\n" + en + "\n\n---\n\n" + zh + "\n",
        encoding="utf-8",
    )
    print(f"Wrote outputs under {OUT_DIR}")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
