from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class SpanScore:
    span_id: str
    span_name: str
    span_type: str
    score: float
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "span_name": self.span_name,
            "span_type": self.span_type,
            "score": self.score,
            "passed": self.passed,
            "checks": self.checks,
        }


@dataclass
class SpanEvalConfig:
    check_llm_quality: bool = True
    check_tool_success: bool = True
    check_latency: bool = True
    max_latency_ms: float = 5000.0
    min_output_length: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_llm_quality": self.check_llm_quality,
            "check_tool_success": self.check_tool_success,
            "check_latency": self.check_latency,
            "max_latency_ms": self.max_latency_ms,
            "min_output_length": self.min_output_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanEvalConfig:
        return cls(
            check_llm_quality=data.get("check_llm_quality", True),
            check_tool_success=data.get("check_tool_success", True),
            check_latency=data.get("check_latency", True),
            max_latency_ms=data.get("max_latency_ms", 5000.0),
            min_output_length=data.get("min_output_length", 10),
        )


@dataclass
class SpanEvalReport:
    span_scores: list[SpanScore]
    total_spans: int
    passed_count: int
    failed_count: int
    pass_rate: float
    by_type: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_scores": [s.as_dict() for s in self.span_scores],
            "total_spans": self.total_spans,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": self.pass_rate,
            "by_type": self.by_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanEvalReport:
        return cls(
            span_scores=[
                SpanScore(
                    span_id=s["span_id"],
                    span_name=s["span_name"],
                    span_type=s["span_type"],
                    score=s["score"],
                    passed=s["passed"],
                    checks=s.get("checks", {}),
                )
                for s in data.get("span_scores", [])
            ],
            total_spans=data["total_spans"],
            passed_count=data["passed_count"],
            failed_count=data["failed_count"],
            pass_rate=data["pass_rate"],
            by_type=data.get("by_type", {}),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Span Evaluation: {self.total_spans} spans")
        lines.append(
            f"Passed: {self.passed_count}  Failed: {self.failed_count}  Rate: {self.pass_rate:.1%}"
        )
        lines.append("")
        for type_name, breakdown in self.by_type.items():
            lines.append(
                f"  {type_name}: {breakdown.get('count', 0)} spans, "
                f"pass_rate={breakdown.get('pass_rate', 0.0):.1%}, "
                f"avg_score={breakdown.get('avg_score', 0.0):.2f}"
            )
        lines.append("")
        for ss in self.span_scores:
            status = "PASS" if ss.passed else "FAIL"
            lines.append(f"  {ss.span_name} ({ss.span_type}): {ss.score:.2f} [{status}]")
        return "\n".join(lines)


def evaluate_span(span: Span, config: SpanEvalConfig) -> SpanScore:
    """Evaluate a single span based on its type.

    - llm_call: check output exists and length >= min_output_length,
      check latency, check no error status
    - tool_call: check status != error, check has output
    - retrieval: check has output
    - Default: check status != error
    Score = fraction of checks passed.
    """
    checks: dict[str, bool] = {}
    attrs = span.attributes

    if span.span_type == "llm_call":
        # Output quality check
        output = attrs.get("output", "")
        if config.check_llm_quality:
            checks["output_exists"] = bool(output)
            checks["output_length"] = len(str(output)) >= config.min_output_length

        # Latency check
        if config.check_latency:
            duration = span.duration_ms
            checks["latency_ok"] = duration is not None and duration <= config.max_latency_ms

        # Error check
        checks["no_error"] = span.status != "error"

    elif span.span_type == "tool_call":
        if config.check_tool_success:
            checks["no_error"] = span.status != "error"
            checks["has_output"] = bool(attrs.get("output", ""))

    elif span.span_type == "retrieval":
        checks["has_output"] = bool(attrs.get("output", ""))

    else:
        checks["no_error"] = span.status != "error"

    if not checks:
        score = 1.0
        passed = True
    else:
        passed_checks = sum(1 for v in checks.values() if v)
        score = passed_checks / len(checks)
        passed = score >= 0.5

    return SpanScore(
        span_id=span.id,
        span_name=span.name,
        span_type=span.span_type,
        score=score,
        passed=passed,
        checks=checks,
    )


def evaluate_spans(spans: list[Span], config: SpanEvalConfig | None = None) -> SpanEvalReport:
    """Evaluate all spans and compute per-type breakdown."""
    if config is None:
        config = SpanEvalConfig()

    scores = [evaluate_span(s, config) for s in spans]
    total = len(scores)
    passed = sum(1 for s in scores if s.passed)
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0.0
    by_type = compute_type_breakdown(scores)

    return SpanEvalReport(
        span_scores=scores,
        total_spans=total,
        passed_count=passed,
        failed_count=failed,
        pass_rate=pass_rate,
        by_type=by_type,
    )


def find_failing_spans(report: SpanEvalReport) -> list[SpanScore]:
    """Return spans with passed=False."""
    return [s for s in report.span_scores if not s.passed]


def compute_type_breakdown(
    scores: list[SpanScore],
) -> dict[str, dict[str, Any]]:
    """Group by span_type, compute pass_rate and avg_score per type."""
    groups: dict[str, list[SpanScore]] = {}
    for s in scores:
        groups.setdefault(s.span_type, []).append(s)

    result: dict[str, dict[str, Any]] = {}
    for type_name, group in groups.items():
        count = len(group)
        type_passed = sum(1 for s in group if s.passed)
        avg_score = sum(s.score for s in group) / count if count > 0 else 0.0
        result[type_name] = {
            "count": count,
            "passed": type_passed,
            "failed": count - type_passed,
            "pass_rate": type_passed / count if count > 0 else 0.0,
            "avg_score": avg_score,
        }

    return result
