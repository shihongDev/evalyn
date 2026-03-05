"""
Insights engine: diagnostic, prescriptive, and proactive analysis.

Provides pure functions operating on RunAnalysis data to surface
actionable intelligence about evaluation results.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .core import RunAnalysis


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CorrelationResult:
    """Correlation between two metrics' scores across items."""
    metric_a: str
    metric_b: str
    pearson: float
    relationship: Literal["redundant", "tradeoff", "independent"]


@dataclass
class RegressionAlert:
    """Alert for a metric whose pass rate dropped between runs."""
    metric_id: str
    previous_pass_rate: float
    current_pass_rate: float
    delta: float
    severity: Literal["critical", "warning", "info"]


@dataclass
class FeatureInsight:
    """Insight linking an input/output feature to pass/fail rates."""
    feature_name: str
    finding: str
    affected_items: int
    pass_rate_low: float
    pass_rate_high: float


@dataclass
class DistributionInsight:
    """Insight about unusual score distribution shape for a metric."""
    metric_id: str
    shape: Literal["normal", "bimodal", "skewed_low", "skewed_high", "cliff", "uniform"]
    finding: str


@dataclass
class Recommendation:
    """Prioritized actionable recommendation."""
    priority: int
    category: str
    message: str
    action: str


@dataclass
class InsightsReport:
    """Complete insights report combining all analysis types."""
    correlations: List[CorrelationResult] = field(default_factory=list)
    regressions: List[RegressionAlert] = field(default_factory=list)
    feature_insights: List[FeatureInsight] = field(default_factory=list)
    distribution_insights: List[DistributionInsight] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric Correlations
# ---------------------------------------------------------------------------

MIN_ITEMS_FOR_CORRELATION = 5
REDUNDANT_THRESHOLD = 0.7
TRADEOFF_THRESHOLD = -0.5


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient. Returns None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def _classify_correlation(r: float) -> Literal["redundant", "tradeoff", "independent"]:
    if r >= REDUNDANT_THRESHOLD:
        return "redundant"
    if r <= TRADEOFF_THRESHOLD:
        return "tradeoff"
    return "independent"


def compute_metric_correlations(run_analysis: RunAnalysis) -> List[CorrelationResult]:
    """Compute pairwise Pearson correlation between metric scores.

    For each pair of metrics, collects items that have scores for both,
    computes Pearson r, and classifies the relationship.

    Returns only non-independent pairs (|r| > 0.5) sorted by |r| descending.
    Requires at least MIN_ITEMS_FOR_CORRELATION items with scores for both metrics.
    """
    # Build per-item score vectors: {metric_id: {item_id: score}}
    metric_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for item_id, item_stats in run_analysis.item_stats.items():
        for metric_id, result in item_stats.metric_results.items():
            score = result.get("score")
            if score is not None:
                metric_scores[metric_id][item_id] = score

    metric_ids = sorted(metric_scores.keys())
    results: List[CorrelationResult] = []

    for i, m_a in enumerate(metric_ids):
        for m_b in metric_ids[i + 1:]:
            # Find common items
            common_items = set(metric_scores[m_a].keys()) & set(metric_scores[m_b].keys())
            if len(common_items) < MIN_ITEMS_FOR_CORRELATION:
                continue

            xs = [metric_scores[m_a][item] for item in common_items]
            ys = [metric_scores[m_b][item] for item in common_items]

            r = _pearson_r(xs, ys)
            if r is None:
                continue

            rel = _classify_correlation(r)
            if rel != "independent":
                results.append(CorrelationResult(
                    metric_a=m_a,
                    metric_b=m_b,
                    pearson=round(r, 3),
                    relationship=rel,
                ))

    results.sort(key=lambda c: abs(c.pearson), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Regression Detection
# ---------------------------------------------------------------------------

CRITICAL_THRESHOLD = 0.15  # 15 percentage point drop
WARNING_THRESHOLD = 0.05   # 5 percentage point drop

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def detect_regressions(
    current: RunAnalysis,
    previous: RunAnalysis,
) -> List[RegressionAlert]:
    """Compare two runs and detect metric pass rate drops.

    For each metric present in both runs, compares pass rates.
    Only reports metrics where pass rate decreased.

    Severity thresholds:
    - critical: dropped > 15 percentage points
    - warning: dropped > 5 percentage points
    - info: any drop > 0

    Returns alerts sorted by severity (critical first), then by delta magnitude.
    Metrics only in one run are ignored.
    """
    alerts: List[RegressionAlert] = []

    for metric_id, current_stats in current.metric_stats.items():
        if metric_id not in previous.metric_stats:
            continue

        prev_stats = previous.metric_stats[metric_id]
        current_rate = current_stats.pass_rate
        prev_rate = prev_stats.pass_rate

        if current_rate is None or prev_rate is None:
            continue

        delta = current_rate - prev_rate
        if delta >= 0:
            continue

        abs_delta = abs(delta)
        if abs_delta > CRITICAL_THRESHOLD:
            severity: Literal["critical", "warning", "info"] = "critical"
        elif abs_delta > WARNING_THRESHOLD:
            severity = "warning"
        else:
            severity = "info"

        alerts.append(RegressionAlert(
            metric_id=metric_id,
            previous_pass_rate=round(prev_rate, 4),
            current_pass_rate=round(current_rate, 4),
            delta=round(delta, 4),
            severity=severity,
        ))

    alerts.sort(key=lambda a: (_SEVERITY_ORDER[a.severity], a.delta))
    return alerts


# ---------------------------------------------------------------------------
# Input Feature Analysis
# ---------------------------------------------------------------------------

FEATURE_DIFF_THRESHOLD = 0.10  # 10 percentage point difference to report


def _extract_text_length(item: Dict[str, Any], key: str) -> Optional[int]:
    """Extract text length from a dataset item field.

    Handles both string values and dict values (flattens to JSON string length).
    """
    val = item.get(key)
    if val is None:
        if key == "input_length":
            val = item.get("inputs") or item.get("input")
        elif key == "output_length":
            val = item.get("output")
    if val is None:
        return None
    if isinstance(val, str):
        return len(val)
    if isinstance(val, dict):
        return len(json.dumps(val))
    return len(str(val))


def analyze_input_features(
    dataset_items: List[Dict[str, Any]],
    run_analysis: RunAnalysis,
) -> List[FeatureInsight]:
    """Correlate input/output characteristics with pass/fail rates.

    Analyzes input and output text length. Splits items at median,
    compares pass rates between low and high groups. Reports features
    where the difference exceeds FEATURE_DIFF_THRESHOLD.

    Args:
        dataset_items: Raw dataset items (dicts with id, inputs, output).
        run_analysis: Analyzed run with per-item stats.
    """
    if not dataset_items or not run_analysis.item_stats:
        return []

    item_passed: Dict[str, bool] = {
        item_id: ist.all_passed
        for item_id, ist in run_analysis.item_stats.items()
    }

    item_lookup: Dict[str, Dict[str, Any]] = {}
    for item in dataset_items:
        item_id = item.get("id")
        if item_id:
            item_lookup[item_id] = item

    results: List[FeatureInsight] = []
    features = [
        ("input_length", lambda item: _extract_text_length(item, "input_length")),
        ("output_length", lambda item: _extract_text_length(item, "output_length")),
    ]

    for feature_name, extractor in features:
        data_points: List[Tuple[int, bool]] = []
        for item_id in run_analysis.item_stats:
            if item_id not in item_lookup:
                continue
            val = extractor(item_lookup[item_id])
            if val is None:
                continue
            data_points.append((val, item_passed.get(item_id, True)))

        if len(data_points) < 4:
            continue

        data_points.sort(key=lambda x: x[0])
        mid = len(data_points) // 2
        low_group = data_points[:mid]
        high_group = data_points[mid:]

        if not low_group or not high_group:
            continue

        low_pass = sum(1 for _, p in low_group if p) / len(low_group)
        high_pass = sum(1 for _, p in high_group if p) / len(high_group)

        diff = abs(high_pass - low_pass)
        if diff < FEATURE_DIFF_THRESHOLD:
            continue

        median_val = data_points[mid][0]
        if high_pass < low_pass:
            direction = f"Items with {feature_name.replace('_', ' ')} > {median_val} chars fail more"
        else:
            direction = f"Items with {feature_name.replace('_', ' ')} > {median_val} chars pass more"

        results.append(FeatureInsight(
            feature_name=feature_name,
            finding=f"{direction} ({low_pass:.0%} vs {high_pass:.0%} pass rate)",
            affected_items=len(data_points),
            pass_rate_low=round(low_pass, 4),
            pass_rate_high=round(high_pass, 4),
        ))

    return results


# ---------------------------------------------------------------------------
# Score Distribution Analysis
# ---------------------------------------------------------------------------

MIN_SCORES_FOR_DISTRIBUTION = 5


def _bucket_scores(scores: List[float]) -> List[float]:
    """Bucket scores into 5 bins. Returns list of 5 fractions."""
    n = len(scores)
    if n == 0:
        return [0.0] * 5
    buckets = [0] * 5
    for s in scores:
        idx = max(0, min(int(s * 5), 4))
        buckets[idx] += 1
    return [b / n for b in buckets]


def analyze_score_distributions(
    run_analysis: RunAnalysis,
) -> List[DistributionInsight]:
    """Detect unusual score distribution shapes per metric.

    Shapes detected (checked in order - first match wins):
    cliff, bimodal, skewed_low, skewed_high, uniform.
    Normal distributions are not reported.

    Requires at least MIN_SCORES_FOR_DISTRIBUTION scores per metric.
    """
    results: List[DistributionInsight] = []

    for metric_id, stats in sorted(run_analysis.metric_stats.items()):
        scores = stats.scores
        if len(scores) < MIN_SCORES_FOR_DISTRIBUTION:
            continue

        n = len(scores)
        mean = stats.avg_score
        std = stats.std_dev
        buckets = _bucket_scores(scores)

        # cliff: >70% at exactly 0.0 or 1.0
        at_zero = sum(1 for s in scores if s == 0.0) / n
        at_one = sum(1 for s in scores if s == 1.0) / n
        if at_zero > 0.7:
            results.append(DistributionInsight(
                metric_id=metric_id, shape="cliff",
                finding=f"{at_zero:.0%} of scores at 0.0 - most items fail completely",
            ))
            continue
        if at_one > 0.7:
            results.append(DistributionInsight(
                metric_id=metric_id, shape="cliff",
                finding=f"{at_one:.0%} of scores at 1.0 - metric may be too lenient",
            ))
            continue

        # bimodal: >30% low AND >30% high with <20% middle
        low_frac = sum(1 for s in scores if s <= 0.3) / n
        high_frac = sum(1 for s in scores if s >= 0.7) / n
        mid_frac = sum(1 for s in scores if 0.3 < s < 0.7) / n
        if low_frac > 0.3 and high_frac > 0.3 and mid_frac < 0.2:
            results.append(DistributionInsight(
                metric_id=metric_id, shape="bimodal",
                finding=f"Split outcomes: {low_frac:.0%} low, {high_frac:.0%} high - review rubric",
            ))
            continue

        # skewed_low
        if mean < 0.3 and std > 0.1:
            results.append(DistributionInsight(
                metric_id=metric_id, shape="skewed_low",
                finding=f"Mean score {mean:.2f} - most items score poorly",
            ))
            continue

        # skewed_high
        if mean > 0.8 and std < 0.15:
            results.append(DistributionInsight(
                metric_id=metric_id, shape="skewed_high",
                finding=f"Mean score {mean:.2f} with low variance - may be too lenient",
            ))
            continue

        # uniform
        if std > 0.25 and all(b <= 0.35 for b in buckets):
            results.append(DistributionInsight(
                metric_id=metric_id, shape="uniform",
                finding="Scores spread uniformly - metric may lack discriminative power",
            ))
            continue

    return results


# ---------------------------------------------------------------------------
# Recommendations Engine
# ---------------------------------------------------------------------------


def generate_recommendations(
    run_analysis: RunAnalysis,
    report: InsightsReport,
    dataset_path: Optional[str] = None,
) -> List[Recommendation]:
    """Generate prioritized recommendations from all insight sources.

    Combines findings from correlations, regressions, distributions,
    and feature analysis into actionable, prioritized suggestions.
    """
    recs: List[Recommendation] = []
    priority = 1
    dataset_flag = f"--dataset {dataset_path}" if dataset_path else "--latest"

    correlations = report.correlations
    regressions = report.regressions
    feature_insights = report.feature_insights
    distribution_insights = report.distribution_insights

    # 1. Critical regressions
    for reg in regressions:
        if reg.severity == "critical":
            recs.append(Recommendation(
                priority=priority,
                category="regression",
                message=f"Investigate {reg.metric_id} regression: {abs(reg.delta) * 100:.0f}% drop ({reg.previous_pass_rate * 100:.0f}% -> {reg.current_pass_rate * 100:.0f}%)",
                action=f"evalyn analyze {dataset_flag}",
            ))
            priority += 1

    # 2. Redundant metrics
    for corr in correlations:
        if corr.relationship == "redundant":
            recs.append(Recommendation(
                priority=priority,
                category="metric_config",
                message=f"Consider removing one of: {corr.metric_a}, {corr.metric_b} (r={corr.pearson})",
                action=f"evalyn list-metrics {dataset_flag}",
            ))
            priority += 1

    # 3. Cliff distributions at 1.0 (too lenient)
    for dist in distribution_insights:
        if dist.shape == "cliff" and "lenient" in dist.finding:
            recs.append(Recommendation(
                priority=priority,
                category="metric_config",
                message=f"{dist.metric_id}: {dist.finding}",
                action=f"evalyn calibrate --metric-id {dist.metric_id} {dataset_flag}",
            ))
            priority += 1

    # 4. Bimodal distributions
    for dist in distribution_insights:
        if dist.shape == "bimodal":
            recs.append(Recommendation(
                priority=priority,
                category="calibration",
                message=f"{dist.metric_id} has split outcomes - review rubric or calibrate",
                action=f"evalyn calibrate --metric-id {dist.metric_id} {dataset_flag}",
            ))
            priority += 1

    # 5. Feature insights
    for feat in feature_insights:
        recs.append(Recommendation(
            priority=priority,
            category="data_quality",
            message=feat.finding,
            action=f"evalyn cluster-failures {dataset_flag}",
        ))
        priority += 1

    # 6. Tradeoff metrics
    for corr in correlations:
        if corr.relationship == "tradeoff":
            recs.append(Recommendation(
                priority=priority,
                category="metric_config",
                message=f"{corr.metric_a} and {corr.metric_b} trade off (r={corr.pearson}) - decide which matters more",
                action=f"evalyn list-metrics {dataset_flag}",
            ))
            priority += 1

    # 7. Problem metrics needing calibration
    problem_metrics = [
        mid for mid, ms in run_analysis.metric_stats.items()
        if ms.pass_rate is not None and ms.pass_rate < 0.8
    ]
    already_mentioned = {r.metric_id for r in regressions if r.severity == "critical"}
    for mid in problem_metrics:
        if mid not in already_mentioned:
            recs.append(Recommendation(
                priority=priority,
                category="calibration",
                message=f"{mid} has low pass rate ({run_analysis.metric_stats[mid].pass_rate * 100:.0f}%)",
                action=f"evalyn calibrate --metric-id {mid} {dataset_flag}",
            ))
            priority += 1
            break  # Only suggest one

    # 8. Warning regressions
    for reg in regressions:
        if reg.severity == "warning":
            recs.append(Recommendation(
                priority=priority,
                category="regression",
                message=f"{reg.metric_id} dropped {abs(reg.delta) * 100:.0f}%",
                action=f"evalyn compare {dataset_flag}",
            ))
            priority += 1

    return recs
