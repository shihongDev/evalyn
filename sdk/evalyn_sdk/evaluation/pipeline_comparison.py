from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunSummary:
    """Summary of a single pipeline run."""

    run_id: str
    pipeline_name: str
    pass_rate: float = 0.0
    total_items: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    metrics_scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "pass_rate": self.pass_rate,
            "total_items": self.total_items,
            "total_cost_usd": self.total_cost_usd,
            "duration_ms": self.duration_ms,
            "metrics_scores": dict(self.metrics_scores),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunSummary:
        return cls(
            run_id=data["run_id"],
            pipeline_name=data["pipeline_name"],
            pass_rate=data.get("pass_rate", 0.0),
            total_items=data.get("total_items", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            metrics_scores=data.get("metrics_scores", {}),
        )


@dataclass
class MetricDelta:
    """Difference between two runs for a single metric."""

    metric_id: str
    score_a: float
    score_b: float
    delta: float
    improved: bool

    def as_dict(self) -> dict:
        return {
            "metric_id": self.metric_id,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "delta": self.delta,
            "improved": self.improved,
        }


@dataclass
class PipelineComparisonReport:
    """Full comparison between two pipeline runs."""

    run_a: RunSummary
    run_b: RunSummary
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    pass_rate_delta: float = 0.0
    cost_delta: float = 0.0
    duration_delta: float = 0.0
    overall_improved: bool = False

    def as_dict(self) -> dict:
        return {
            "run_a": self.run_a.as_dict(),
            "run_b": self.run_b.as_dict(),
            "metric_deltas": [d.as_dict() for d in self.metric_deltas],
            "pass_rate_delta": self.pass_rate_delta,
            "cost_delta": self.cost_delta,
            "duration_delta": self.duration_delta,
            "overall_improved": self.overall_improved,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineComparisonReport:
        run_a = RunSummary.from_dict(data["run_a"])
        run_b = RunSummary.from_dict(data["run_b"])
        metric_deltas = [MetricDelta(**d) for d in data.get("metric_deltas", [])]
        return cls(
            run_a=run_a,
            run_b=run_b,
            metric_deltas=metric_deltas,
            pass_rate_delta=data.get("pass_rate_delta", 0.0),
            cost_delta=data.get("cost_delta", 0.0),
            duration_delta=data.get("duration_delta", 0.0),
            overall_improved=data.get("overall_improved", False),
        )

    def format_text(self) -> str:
        """Human-readable text summary of the comparison."""
        lines = [
            f"Pipeline Comparison: {self.run_a.pipeline_name} vs {self.run_b.pipeline_name}",
            f"  Run A: {self.run_a.run_id} | Run B: {self.run_b.run_id}",
            f"  Pass rate: {self.run_a.pass_rate:.2%} -> {self.run_b.pass_rate:.2%} (delta: {self.pass_rate_delta:+.2%})",
            f"  Cost: ${self.run_a.total_cost_usd:.4f} -> ${self.run_b.total_cost_usd:.4f} (delta: ${self.cost_delta:+.4f})",
            f"  Duration: {self.run_a.duration_ms:.0f}ms -> {self.run_b.duration_ms:.0f}ms (delta: {self.duration_delta:+.0f}ms)",
        ]
        if self.metric_deltas:
            lines.append("  Metric deltas:")
            for d in self.metric_deltas:
                marker = "+" if d.improved else "-"
                lines.append(
                    f"    [{marker}] {d.metric_id}: {d.score_a:.3f} -> {d.score_b:.3f} (delta: {d.delta:+.3f})"
                )
        status = "IMPROVED" if self.overall_improved else "NOT IMPROVED"
        lines.append(f"  Overall: {status}")
        return "\n".join(lines)


def compute_metric_deltas(
    scores_a: dict[str, float], scores_b: dict[str, float]
) -> list[MetricDelta]:
    """Compute per-metric deltas between two score dictionaries."""
    all_keys = sorted(set(scores_a) | set(scores_b))
    deltas = []
    for key in all_keys:
        sa = scores_a.get(key, 0.0)
        sb = scores_b.get(key, 0.0)
        delta = sb - sa
        deltas.append(MetricDelta(key, sa, sb, delta, delta > 0))
    return deltas


def compare_runs(run_a: RunSummary, run_b: RunSummary) -> PipelineComparisonReport:
    """Compare two pipeline runs and produce a full report."""
    metric_deltas = compute_metric_deltas(run_a.metrics_scores, run_b.metrics_scores)
    pass_rate_delta = run_b.pass_rate - run_a.pass_rate
    cost_delta = run_b.total_cost_usd - run_a.total_cost_usd
    duration_delta = run_b.duration_ms - run_a.duration_ms
    overall_improved = pass_rate_delta > 0
    return PipelineComparisonReport(
        run_a=run_a,
        run_b=run_b,
        metric_deltas=metric_deltas,
        pass_rate_delta=pass_rate_delta,
        cost_delta=cost_delta,
        duration_delta=duration_delta,
        overall_improved=overall_improved,
    )


def is_regression(report: PipelineComparisonReport, threshold: float = 0.05) -> bool:
    """Return True if pass rate dropped more than threshold."""
    return report.pass_rate_delta < -threshold


def format_comparison_table(report: PipelineComparisonReport) -> str:
    """Format an ASCII table of metric deltas."""
    header = f"{'Metric':<25} {'Run A':>8} {'Run B':>8} {'Delta':>8} {'Status':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for d in report.metric_deltas:
        status = "better" if d.improved else "worse"
        if d.delta == 0:
            status = "same"
        lines.append(
            f"{d.metric_id:<25} {d.score_a:>8.3f} {d.score_b:>8.3f} {d.delta:>+8.3f} {status:>8}"
        )
    lines.append(sep)
    lines.append(
        f"{'Pass rate':<25} {report.run_a.pass_rate:>8.3f} {report.run_b.pass_rate:>8.3f} {report.pass_rate_delta:>+8.3f}"
    )
    return "\n".join(lines)


def rank_by_improvement(deltas: list[MetricDelta]) -> list[MetricDelta]:
    """Sort metric deltas by delta descending (most improved first)."""
    return sorted(deltas, key=lambda d: d.delta, reverse=True)
