"""Sample GPU memory during external baseline training runs."""
import csv
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


class GpuSidecar:
    def __init__(self, gpu_index=0, interval_sec=1.0):
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(self.gpu_index),
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                mem_mb = float(out.strip().splitlines()[0])
                self.samples.append((datetime.now(), mem_mb))
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def write_samples(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "gpu_memory_used_mb"])
            for ts, mem in self.samples:
                w.writerow([ts.strftime("%H:%M:%S.%f")[:-3], f"{mem:.1f}"])


def _parse_iter_records(log_path):
    pat = re.compile(r"^\[(\d{2}:\d{2}:\d{2}\.\d{3})\].*iteration[:\s]+(\d+)")
    records = []
    with Path(log_path).open("r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%H:%M:%S.%f")
            records.append((t, int(m.group(2))))
    return records


def _parse_gpu_samples(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = datetime.strptime(row["timestamp"], "%H:%M:%S.%f")
            rows.append((t, float(row["gpu_memory_used_mb"])))
    return rows


def _nearest_mem(ts, gpu_rows):
    if not gpu_rows:
        return 0.0
    return min(gpu_rows, key=lambda r: abs((r[0] - ts).total_seconds()))[1]


def build_iteration_profile_from_sidecar(log_path, gpu_sample_csv, out_csv):
    iter_rows = _parse_iter_records(log_path)
    gpu_rows = _parse_gpu_samples(gpu_sample_csv)
    if len(iter_rows) < 2:
        raise RuntimeError(f"Not enough iteration logs in {log_path}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total_sec = 0.0
    prev_t = iter_rows[0][0]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "iter_time_sec",
                "iter_per_sec",
                "total_time_sec",
                "gpu_hours",
                "cumulative_gpu_hours",
                "cuda_memory_allocated_mb",
                "cuda_memory_reserved_mb",
                "cuda_max_memory_allocated_mb",
                "cuda_max_memory_reserved_mb",
            ],
        )
        w.writeheader()
        peak = 0.0
        for ts, it in iter_rows:
            dt = max(0.0, (ts - prev_t).total_seconds()) if it > iter_rows[0][1] else 0.0
            if it > iter_rows[0][1]:
                total_sec += dt
            prev_t = ts
            mem = _nearest_mem(ts, gpu_rows)
            peak = max(peak, mem)
            w.writerow(
                {
                    "iter": it,
                    "iter_time_sec": dt,
                    "iter_per_sec": (1.0 / dt) if dt > 0 else 0.0,
                    "total_time_sec": total_sec,
                    "gpu_hours": dt / 3600.0,
                    "cumulative_gpu_hours": total_sec / 3600.0,
                    "cuda_memory_allocated_mb": mem,
                    "cuda_memory_reserved_mb": mem,
                    "cuda_max_memory_allocated_mb": peak,
                    "cuda_max_memory_reserved_mb": peak,
                }
            )
    return out_csv
