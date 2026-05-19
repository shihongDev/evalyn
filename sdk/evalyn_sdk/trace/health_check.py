from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class HealthCheckItem:
    """Single health check result."""

    name: str
    passed: bool
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class HealthCheckReport:
    """Aggregated health check report."""

    checks: list[HealthCheckItem] = field(default_factory=list)
    all_passed: bool = True
    passed_count: int = 0
    failed_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "all_passed": self.all_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthCheckReport:
        checks = [
            HealthCheckItem(
                name=c["name"],
                passed=c["passed"],
                details=c.get("details", ""),
            )
            for c in data.get("checks", [])
        ]
        return cls(
            checks=checks,
            all_passed=data.get("all_passed", True),
            passed_count=data.get("passed_count", 0),
            failed_count=data.get("failed_count", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Instrumentation Health Check")
        lines.append("-" * 40)
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            line = f"  [{status}] {c.name}"
            if c.details:
                line += f" - {c.details}"
            lines.append(line)
        lines.append("-" * 40)
        lines.append(
            f"Passed: {self.passed_count}/{self.passed_count + self.failed_count}"
        )
        if self.all_passed:
            lines.append("All checks passed.")
        else:
            lines.append(f"{self.failed_count} check(s) failed.")
        return "\n".join(lines)


def check_spans_captured(
    spans: list[Span], min_expected: int = 1
) -> HealthCheckItem:
    """Check that at least min_expected spans were captured."""
    count = len(spans)
    if count >= min_expected:
        return HealthCheckItem(
            name="spans_captured",
            passed=True,
            details=f"{count} span(s) captured (min: {min_expected})",
        )
    return HealthCheckItem(
        name="spans_captured",
        passed=False,
        details=f"{count} span(s) captured, expected at least {min_expected}",
    )


def check_span_types_present(
    spans: list[Span], required_types: list[str] | None = None
) -> HealthCheckItem:
    """Check that required span types exist."""
    if required_types is None:
        required_types = ["llm_call"]
    present_types = {s.span_type for s in spans}
    missing = [t for t in required_types if t not in present_types]
    if not missing:
        return HealthCheckItem(
            name="span_types_present",
            passed=True,
            details=f"All required types found: {required_types}",
        )
    return HealthCheckItem(
        name="span_types_present",
        passed=False,
        details=f"Missing span types: {missing}",
    )


def check_no_orphans(spans: list[Span]) -> HealthCheckItem:
    """Check that all spans with parent_id reference an existing span."""
    span_ids = {s.id for s in spans}
    orphans = [
        s.id
        for s in spans
        if s.parent_id is not None and s.parent_id not in span_ids
    ]
    if not orphans:
        return HealthCheckItem(
            name="no_orphans",
            passed=True,
            details="No orphan spans detected",
        )
    return HealthCheckItem(
        name="no_orphans",
        passed=False,
        details=f"{len(orphans)} orphan span(s) found: {orphans}",
    )


def check_timestamps_valid(spans: list[Span]) -> HealthCheckItem:
    """Check that all spans have start_time and none have end_time before start_time."""
    invalid: list[str] = []
    for s in spans:
        if s.start_time is None:
            invalid.append(s.id)
            continue
        if s.end_time is not None and s.end_time < s.start_time:
            invalid.append(s.id)
    if not invalid:
        return HealthCheckItem(
            name="timestamps_valid",
            passed=True,
            details="All span timestamps are valid",
        )
    return HealthCheckItem(
        name="timestamps_valid",
        passed=False,
        details=f"{len(invalid)} span(s) with invalid timestamps: {invalid}",
    )


def check_attributes_present(
    spans: list[Span], required_keys: list[str] | None = None
) -> HealthCheckItem:
    """Check that spans have required attribute keys."""
    if required_keys is None:
        required_keys = []
    if not required_keys:
        return HealthCheckItem(
            name="attributes_present",
            passed=True,
            details="No required attributes specified",
        )
    missing_map: dict[str, list[str]] = {}
    for s in spans:
        missing = [k for k in required_keys if k not in s.attributes]
        if missing:
            missing_map[s.id] = missing
    if not missing_map:
        return HealthCheckItem(
            name="attributes_present",
            passed=True,
            details=f"All spans have required attributes: {required_keys}",
        )
    return HealthCheckItem(
        name="attributes_present",
        passed=False,
        details=f"{len(missing_map)} span(s) missing required attributes",
    )


def run_health_check(
    spans: list[Span],
    required_types: list[str] | None = None,
    required_attributes: list[str] | None = None,
) -> HealthCheckReport:
    """Run all health checks and return aggregated report."""
    if required_types is None:
        required_types = ["llm_call"]
    if required_attributes is None:
        required_attributes = []

    checks = [
        check_spans_captured(spans),
        check_span_types_present(spans, required_types),
        check_no_orphans(spans),
        check_timestamps_valid(spans),
        check_attributes_present(spans, required_attributes),
    ]
    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count
    return HealthCheckReport(
        checks=checks,
        all_passed=failed_count == 0,
        passed_count=passed_count,
        failed_count=failed_count,
    )
