"""Data export API for analysis results.

Exports RunAnalysis data as structured dicts suitable for pandas
DataFrame construction or other downstream analysis tools.
"""

from __future__ import annotations

from typing import Any

from .core import RunAnalysis


def analyze_to_dict(analysis: RunAnalysis) -> dict[str, list[Any]]:
    """Export analysis as dict-of-lists (columnar format).

    Returns a dict where each key is a column name and each value is a
    list of equal length. Can be passed directly to pandas.DataFrame().

    Columns: item_id, all_passed, metrics_passed, metrics_failed,
    plus one column per metric (score) and one per metric (passed).
    """
    item_ids = sorted(analysis.item_stats.keys())
    metric_ids = sorted(analysis.metric_stats.keys())

    result: dict[str, list[Any]] = {
        "item_id": [],
        "all_passed": [],
        "metrics_passed": [],
        "metrics_failed": [],
    }
    for mid in metric_ids:
        result[f"{mid}_score"] = []
        result[f"{mid}_passed"] = []

    for item_id in item_ids:
        item = analysis.item_stats[item_id]
        result["item_id"].append(item_id)
        result["all_passed"].append(item.all_passed)
        result["metrics_passed"].append(item.metrics_passed)
        result["metrics_failed"].append(item.metrics_failed)

        for mid in metric_ids:
            mr = item.metric_results.get(mid, {})
            result[f"{mid}_score"].append(mr.get("score"))
            result[f"{mid}_passed"].append(mr.get("passed"))

    return result


def metric_summary_to_dict(analysis: RunAnalysis) -> dict[str, list[Any]]:
    """Export metric-level summary as dict-of-lists.

    Columns: metric_id, metric_type, count, passed, failed,
    pass_rate, avg_score, min_score, max_score, std_dev.
    """
    result: dict[str, list[Any]] = {
        "metric_id": [],
        "metric_type": [],
        "count": [],
        "passed": [],
        "failed": [],
        "pass_rate": [],
        "avg_score": [],
        "min_score": [],
        "max_score": [],
        "std_dev": [],
    }

    for mid in sorted(analysis.metric_stats.keys()):
        ms = analysis.metric_stats[mid]
        result["metric_id"].append(ms.metric_id)
        result["metric_type"].append(ms.metric_type)
        result["count"].append(ms.count)
        result["passed"].append(ms.passed)
        result["failed"].append(ms.failed)
        result["pass_rate"].append(ms.pass_rate)
        result["avg_score"].append(ms.avg_score)
        result["min_score"].append(ms.min_score)
        result["max_score"].append(ms.max_score)
        result["std_dev"].append(ms.std_dev)

    return result


def scores_matrix(analysis: RunAnalysis) -> dict[str, list[float | None]]:
    """Export item x metric score matrix.

    Returns dict with "item_ids" key (list of item IDs) and one key
    per metric_id (list of scores, None where missing).
    """
    item_ids = sorted(analysis.item_stats.keys())
    metric_ids = sorted(analysis.metric_stats.keys())

    result: dict[str, list[float | None]] = {"item_id": item_ids}
    for mid in metric_ids:
        col = []
        for item_id in item_ids:
            mr = analysis.item_stats[item_id].metric_results.get(mid, {})
            col.append(mr.get("score"))
        result[mid] = col

    return result


def export_flat_records(analysis: RunAnalysis) -> list[dict[str, Any]]:
    """Export as list of flat records (one per item-metric pair).

    Each record has: item_id, metric_id, score, passed.
    Suitable for JSON export or loading into columnar formats.
    """
    records = []
    for item_id in sorted(analysis.item_stats.keys()):
        item = analysis.item_stats[item_id]
        for mid in sorted(item.metric_results.keys()):
            mr = item.metric_results[mid]
            records.append({
                "item_id": item_id,
                "metric_id": mid,
                "score": mr.get("score"),
                "passed": mr.get("passed"),
            })
    return records
