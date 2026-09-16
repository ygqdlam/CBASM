import csv
import math
import os
from collections import defaultdict


CASE_METRIC_FIELDS = ["case_id", "class_id", "dice", "jaccard", "hd95", "asd"]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _to_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def _mean(values):
    valid = [float(v) for v in values if not math.isnan(float(v))]
    if not valid:
        return math.nan
    return sum(valid) / len(valid)


def _sample_std(values):
    valid = [float(v) for v in values if not math.isnan(float(v))]
    if len(valid) < 2:
        return 0.0
    avg = _mean(valid)
    return math.sqrt(sum((v - avg) ** 2 for v in valid) / (len(valid) - 1))


def write_case_metrics_csv(path, rows, extra_fields=None):
    ensure_dir(os.path.dirname(path))
    extra_fields = extra_fields or []
    fieldnames = CASE_METRIC_FIELDS + [field for field in extra_fields if field not in CASE_METRIC_FIELDS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_case_metrics_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def summarize_case_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(row)

    summary = []
    for class_id in sorted(grouped):
        class_rows = grouped[class_id]
        record = {"class_id": class_id, "n_cases": len(class_rows)}
        for metric_name in ["dice", "jaccard", "hd95", "asd"]:
            values = [_to_float(row.get(metric_name)) for row in class_rows]
            record[f"{metric_name}_mean"] = _mean(values)
            record[f"{metric_name}_std"] = _sample_std(values)
        summary.append(record)
    return summary


def write_summary_csv(path, summary_rows):
    ensure_dir(os.path.dirname(path))
    fieldnames = [
        "class_id",
        "n_cases",
        "dice_mean",
        "dice_std",
        "jaccard_mean",
        "jaccard_std",
        "hd95_mean",
        "hd95_std",
        "asd_mean",
        "asd_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rows_from_metric_array(metric_array, case_ids=None, class_offset=1, extra=None):
    """Convert [case, class, metric] arrays/lists into CSV-ready rows."""
    extra = extra or {}
    rows = []
    for case_idx, case_metrics in enumerate(metric_array):
        case_id = case_ids[case_idx] if case_ids is not None else str(case_idx)
        for class_idx, metrics in enumerate(case_metrics):
            rows.append(
                {
                    "case_id": case_id,
                    "class_id": class_idx + class_offset,
                    "dice": float(metrics[0]),
                    "jaccard": float(metrics[1]) if len(metrics) > 1 else "",
                    "hd95": float(metrics[2]) if len(metrics) > 2 else "",
                    "asd": float(metrics[3]) if len(metrics) > 3 else "",
                    **extra,
                }
            )
    return rows

