import math


EPS = 1e-8


def _flatten_nested(values):
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy().tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()

    flattened = []

    def visit(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            flattened.append(float(item))

    visit(values)
    return flattened


def _normalize_rows(probabilities):
    rows = probabilities
    if hasattr(rows, "detach"):
        rows = rows.detach().float().cpu().numpy().tolist()
    if hasattr(rows, "tolist"):
        rows = rows.tolist()

    if not rows:
        return []

    if isinstance(rows[0], (int, float)):
        rows = [rows]

    normalized = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        total = sum(max(float(v), 0.0) for v in row)
        if total <= EPS:
            normalized.append([1.0 / len(row) for _ in row])
        else:
            normalized.append([max(float(v), EPS) / total for v in row])
    return normalized


def _kl_rows(p_rows, q_rows):
    total = 0.0
    count = 0
    for p_row, q_row in zip(p_rows, q_rows):
        total += sum(p * math.log((p + EPS) / (q + EPS)) for p, q in zip(p_row, q_row))
        count += 1
    return total / max(count, 1)


def _argmax(row):
    return max(range(len(row)), key=lambda idx: row[idx])


def diversity_from_probabilities_np(p, q):
    p_rows = _normalize_rows(p)
    q_rows = _normalize_rows(q)
    if len(p_rows) != len(q_rows):
        raise ValueError("p and q must contain the same number of probability rows")

    kl_pq = _kl_rows(p_rows, q_rows)
    kl_qp = _kl_rows(q_rows, p_rows)
    mid = [[(p_v + q_v) * 0.5 for p_v, q_v in zip(p_row, q_row)] for p_row, q_row in zip(p_rows, q_rows)]
    js = 0.5 * _kl_rows(p_rows, mid) + 0.5 * _kl_rows(q_rows, mid)
    disagreement = sum(_argmax(p_row) != _argmax(q_row) for p_row, q_row in zip(p_rows, q_rows)) / max(len(p_rows), 1)
    return {
        "kl_pq": kl_pq,
        "kl_qp": kl_qp,
        "js": js,
        "disagreement": float(disagreement),
    }


DIVERSITY_METRIC_KEYS = ("kl_pq", "kl_qp", "js", "disagreement", "entropy_p", "entropy_q")


def diversity_csv_prefix(prefix):
    return prefix.replace("/", "_")


def empty_diversity_csv_row(prefix):
    base = diversity_csv_prefix(prefix)
    return {f"{base}_{key}": float("nan") for key in DIVERSITY_METRIC_KEYS}


def diversity_metrics_to_csv_row(prefix, metrics):
    base = diversity_csv_prefix(prefix)
    return {f"{base}_{key}": metrics[key] for key in DIVERSITY_METRIC_KEYS}


def maybe_log_diversity(writer, iter_num, prefix, prob_a, prob_b, args):
    """Log KL/JS/disagreement to TensorBoard and return fixed-schema CSV fields."""
    if not args.get("enable_diversity_metrics", False):
        return {}
    if iter_num % max(1, args.get("diversity_interval", 1)) != 0:
        return empty_diversity_csv_row(prefix)
    metrics_div = diversity_from_torch_probs(prob_a, prob_b)
    for name, value in metrics_div.items():
        writer.add_scalar("{}/{}".format(prefix, name), value, iter_num)
    return diversity_metrics_to_csv_row(prefix, metrics_div)


def diversity_from_torch_probs(p, q):
    """Compute prediction diversity from torch tensors shaped B,C,*."""
    import torch
    import torch.nn.functional as F

    p = p.detach().float().clamp_min(EPS)
    q = q.detach().float().clamp_min(EPS)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(EPS)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(EPS)
    spatial_dims = tuple(range(1, p.dim()))
    kl_pq = F.kl_div(torch.log(p), q, reduction="batchmean").detach()
    kl_qp = F.kl_div(torch.log(q), p, reduction="batchmean").detach()
    mid = (p + q) * 0.5
    js = 0.5 * F.kl_div(torch.log(p), mid, reduction="batchmean") + 0.5 * F.kl_div(torch.log(q), mid, reduction="batchmean")
    disagreement = (torch.argmax(p, dim=1) != torch.argmax(q, dim=1)).float().mean()
    entropy_p = -(p * torch.log(p)).sum(dim=1).mean()
    entropy_q = -(q * torch.log(q)).sum(dim=1).mean()
    return {
        "kl_pq": float(kl_pq.item()),
        "kl_qp": float(kl_qp.item()),
        "js": float(js.item()),
        "disagreement": float(disagreement.item()),
        "entropy_p": float(entropy_p.item()),
        "entropy_q": float(entropy_q.item()),
    }

