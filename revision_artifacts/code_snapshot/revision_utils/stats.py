import csv
import math
import os
from collections import defaultdict

from revision_utils.metrics_export import read_case_metrics_csv


def _safe_float(value):
    if value in (None, ""):
        return math.nan
    return float(value)


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    return sum(values) / len(values)


def _sample_std(values):
    values = [value for value in values if not math.isnan(value)]
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def _paired_ttest(values_a, values_b):
    diffs = [b - a for a, b in zip(values_a, values_b) if not math.isnan(a) and not math.isnan(b)]
    n_pairs = len(diffs)
    if n_pairs == 0:
        return n_pairs, math.nan, math.nan, math.nan, "none"

    mean_diff = _mean(diffs)
    std_diff = _sample_std(diffs)
    if n_pairs < 2 or std_diff == 0:
        t_stat = 0.0 if mean_diff == 0 else math.inf
        p_value = 1.0 if mean_diff == 0 else 0.0
        return n_pairs, mean_diff, t_stat, p_value, "degenerate"

    t_stat = mean_diff / (std_diff / math.sqrt(n_pairs))
    try:
        from scipy import stats

        scipy_result = stats.ttest_rel(values_b, values_a, nan_policy="omit")
        return n_pairs, mean_diff, float(scipy_result.statistic), float(scipy_result.pvalue), "scipy.ttest_rel"
    except Exception:
        p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
        return n_pairs, mean_diff, t_stat, p_value, "normal_approx"


def paired_significance_rows(baseline_csv, candidate_csv, metrics=("dice", "hd95")):
    baseline_rows = read_case_metrics_csv(baseline_csv)
    candidate_rows = read_case_metrics_csv(candidate_csv)

    baseline = {(row["case_id"], row["class_id"]): row for row in baseline_rows}
    candidate = {(row["case_id"], row["class_id"]): row for row in candidate_rows}
    keys_by_class = defaultdict(list)
    for key in sorted(set(baseline).intersection(candidate)):
        keys_by_class[key[1]].append(key)

    results = []
    for class_id, keys in sorted(keys_by_class.items(), key=lambda item: int(item[0])):
        for metric_name in metrics:
            values_a = [_safe_float(baseline[key].get(metric_name)) for key in keys]
            values_b = [_safe_float(candidate[key].get(metric_name)) for key in keys]
            n_pairs, mean_diff, t_stat, p_value, method = _paired_ttest(values_a, values_b)
            results.append(
                {
                    "class_id": class_id,
                    "metric": metric_name,
                    "n_pairs": n_pairs,
                    "baseline_mean": _mean(values_a),
                    "candidate_mean": _mean(values_b),
                    "mean_diff_candidate_minus_baseline": mean_diff,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "method": method,
                }
            )
    return results


def write_significance_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "class_id",
        "metric",
        "n_pairs",
        "baseline_mean",
        "candidate_mean",
        "mean_diff_candidate_minus_baseline",
        "t_stat",
        "p_value",
        "method",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
