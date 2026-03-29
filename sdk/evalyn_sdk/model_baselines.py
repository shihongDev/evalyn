"""
Local model performance baselines: benchmarking framework for comparing
local judge models against API models for evaluation quality.

Computes agreement rates, Pearson correlation, and generates baseline
reports - all pure Python with no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class JudgeScore:
    """A single score assigned by a judge model to an item-metric pair."""

    item_id: str
    metric_id: str
    judge_model: str
    score: float
    latency_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "judge_model": self.judge_model,
            "score": self.score,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JudgeScore:
        return cls(
            item_id=data["item_id"],
            metric_id=data["metric_id"],
            judge_model=data["judge_model"],
            score=data["score"],
            latency_ms=data.get("latency_ms", 0.0),
        )


@dataclass
class AgreementResult:
    """Agreement statistics between a local and API model for one metric."""

    metric_id: str
    local_model: str
    api_model: str
    agreement_rate: float
    correlation: float
    mean_score_diff: float
    n_items: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "local_model": self.local_model,
            "api_model": self.api_model,
            "agreement_rate": self.agreement_rate,
            "correlation": self.correlation,
            "mean_score_diff": self.mean_score_diff,
            "n_items": self.n_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgreementResult:
        return cls(
            metric_id=data["metric_id"],
            local_model=data["local_model"],
            api_model=data["api_model"],
            agreement_rate=data.get("agreement_rate", 0.0),
            correlation=data.get("correlation", 0.0),
            mean_score_diff=data.get("mean_score_diff", 0.0),
            n_items=data.get("n_items", 0),
        )


@dataclass
class BaselineReport:
    """Full baseline comparison report across all metrics."""

    agreements: List[AgreementResult]
    overall_agreement: float
    recommended_local_metrics: List[str]
    not_recommended_metrics: List[str]
    generated_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agreements": [a.as_dict() for a in self.agreements],
            "overall_agreement": self.overall_agreement,
            "recommended_local_metrics": list(self.recommended_local_metrics),
            "not_recommended_metrics": list(self.not_recommended_metrics),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaselineReport:
        return cls(
            agreements=[
                AgreementResult.from_dict(a) for a in data.get("agreements", [])
            ],
            overall_agreement=data.get("overall_agreement", 0.0),
            recommended_local_metrics=list(
                data.get("recommended_local_metrics", [])
            ),
            not_recommended_metrics=list(
                data.get("not_recommended_metrics", [])
            ),
            generated_at=data.get("generated_at", ""),
        )


# ---------------------------------------------------------------------------
# Core Computation
# ---------------------------------------------------------------------------


def compute_correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient - pure Python, no numpy.

    Returns 0.0 when the inputs have zero variance or are empty.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-15:
        return 0.0

    return cov / denom


def compute_agreement(
    local_scores: List[JudgeScore],
    api_scores: List[JudgeScore],
    threshold: float = 0.5,
) -> AgreementResult:
    """Compute agreement between local and API scores for matched items.

    Matches scores by (item_id, metric_id). Agreement rate is the fraction
    of pairs whose absolute score difference is within *threshold*.
    """
    # Build lookup for API scores
    api_map: Dict[Tuple[str, str], JudgeScore] = {}
    for s in api_scores:
        api_map[(s.item_id, s.metric_id)] = s

    local_vals: List[float] = []
    api_vals: List[float] = []
    diffs: List[float] = []
    agreed = 0
    local_model = ""
    api_model = ""
    metric_id = ""

    for ls in local_scores:
        key = (ls.item_id, ls.metric_id)
        if key not in api_map:
            continue
        api_s = api_map[key]

        if not local_model:
            local_model = ls.judge_model
        if not api_model:
            api_model = api_s.judge_model
        if not metric_id:
            metric_id = ls.metric_id

        local_vals.append(ls.score)
        api_vals.append(api_s.score)
        diff = ls.score - api_s.score
        diffs.append(diff)
        if abs(diff) <= threshold:
            agreed += 1

    n = len(local_vals)
    if n == 0:
        return AgreementResult(
            metric_id=metric_id or "unknown",
            local_model=local_model or "unknown",
            api_model=api_model or "unknown",
            agreement_rate=0.0,
            correlation=0.0,
            mean_score_diff=0.0,
            n_items=0,
        )

    agreement_rate = agreed / n
    correlation = compute_correlation(local_vals, api_vals)
    mean_diff = sum(diffs) / n

    return AgreementResult(
        metric_id=metric_id,
        local_model=local_model,
        api_model=api_model,
        agreement_rate=agreement_rate,
        correlation=correlation,
        mean_score_diff=mean_diff,
        n_items=n,
    )


def build_baseline_report(
    local_scores: List[JudgeScore],
    api_scores: List[JudgeScore],
    agreement_threshold: float = 0.8,
) -> BaselineReport:
    """Build a full baseline report grouped by metric.

    Metrics with agreement_rate >= *agreement_threshold* are recommended
    for local use; others are not recommended.
    """
    # Group scores by metric_id
    local_by_metric: Dict[str, List[JudgeScore]] = {}
    api_by_metric: Dict[str, List[JudgeScore]] = {}

    for s in local_scores:
        local_by_metric.setdefault(s.metric_id, []).append(s)
    for s in api_scores:
        api_by_metric.setdefault(s.metric_id, []).append(s)

    all_metrics = sorted(set(local_by_metric) | set(api_by_metric))

    agreements: List[AgreementResult] = []
    for metric in all_metrics:
        local_group = local_by_metric.get(metric, [])
        api_group = api_by_metric.get(metric, [])
        if local_group and api_group:
            agreement = compute_agreement(local_group, api_group)
            agreements.append(agreement)

    recommended: List[str] = []
    not_recommended: List[str] = []
    total_rate = 0.0

    for a in agreements:
        if a.agreement_rate >= agreement_threshold:
            recommended.append(a.metric_id)
        else:
            not_recommended.append(a.metric_id)
        total_rate += a.agreement_rate

    overall = total_rate / len(agreements) if agreements else 0.0

    return BaselineReport(
        agreements=agreements,
        overall_agreement=overall,
        recommended_local_metrics=recommended,
        not_recommended_metrics=not_recommended,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_baseline_report(report: BaselineReport) -> str:
    """Format a BaselineReport as a human-readable string with agreement table."""
    lines: List[str] = []
    lines.append("Model Baseline Report")
    lines.append("=" * 60)
    lines.append(f"Generated: {report.generated_at}")
    lines.append(f"Overall agreement: {report.overall_agreement:.1%}")
    lines.append("")

    if report.agreements:
        # Header
        lines.append(
            f"{'Metric':<20} {'Agreement':>10} {'Correlation':>12} "
            f"{'Mean Diff':>10} {'N':>5}"
        )
        lines.append("-" * 60)
        for a in report.agreements:
            lines.append(
                f"{a.metric_id:<20} {a.agreement_rate:>10.1%} "
                f"{a.correlation:>12.3f} {a.mean_score_diff:>10.3f} "
                f"{a.n_items:>5d}"
            )
        lines.append("")

    if report.recommended_local_metrics:
        lines.append("Recommended for local use:")
        for m in report.recommended_local_metrics:
            lines.append(f"  + {m}")
        lines.append("")

    if report.not_recommended_metrics:
        lines.append("Not recommended for local use:")
        for m in report.not_recommended_metrics:
            lines.append(f"  - {m}")
        lines.append("")

    return "\n".join(lines)


def rank_metrics_by_agreement(
    report: BaselineReport,
) -> List[Tuple[str, float]]:
    """Return (metric_id, agreement_rate) pairs sorted descending."""
    pairs = [(a.metric_id, a.agreement_rate) for a in report.agreements]
    pairs.sort(key=lambda x: -x[1])
    return pairs
