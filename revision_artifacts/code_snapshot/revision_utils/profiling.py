import csv
import os
import time


class IterationProfiler:
    def __init__(self, enabled=False, num_gpus=1):
        self.enabled = enabled
        self.num_gpus = max(1, int(num_gpus or 1))
        self._start = None
        self.total_time_sec = 0.0

    def start(self):
        if self.enabled:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            self._start = time.perf_counter()

    def stop(self, iteration):
        elapsed = 0.0
        if self.enabled and self._start is not None:
            elapsed = time.perf_counter() - self._start
            self.total_time_sec += elapsed
        record = {
            "iter": iteration,
            "iter_time_sec": elapsed,
            "iter_per_sec": (1.0 / elapsed) if elapsed > 0 else 0.0,
            "total_time_sec": self.total_time_sec,
            "gpu_hours": elapsed * self.num_gpus / 3600.0,
            "cumulative_gpu_hours": self.total_time_sec * self.num_gpus / 3600.0,
            "cuda_memory_allocated_mb": 0.0,
            "cuda_memory_reserved_mb": 0.0,
            "cuda_max_memory_allocated_mb": 0.0,
            "cuda_max_memory_reserved_mb": 0.0,
        }
        try:
            import torch

            if torch.cuda.is_available():
                record.update(
                    {
                        "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                        "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                        "cuda_max_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
                    }
                )
        except Exception:
            pass
        return record


def append_profile_csv(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    fieldnames = [
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
    ]
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fieldnames})


def profile_model_complexity(model, input_shape):
    """Return FLOPs/params when thop is available; otherwise params only."""
    params = sum(p.numel() for p in model.parameters())
    result = {"params": params, "flops": ""}
    try:
        import torch
        from thop import profile

        device = next(model.parameters()).device
        dummy = torch.randn(*input_shape, device=device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        result = {"params": int(params), "flops": int(flops)}
    except Exception as exc:
        result["flops_error"] = str(exc)
    return result


def write_complexity_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["model_name", "input_shape", "params", "flops", "flops_error"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
