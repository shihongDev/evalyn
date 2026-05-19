"""Dataset quality gate: block evaluation start if dataset fails quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityCheck:
    """Result of a single quality check."""

    name: str
    passed: bool
    severity: str = "warning"  # "info" / "warning" / "error"
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class QualityGateConfig:
    """Configuration for quality gate checks."""

    min_items: int = 1
    max_items: int = 100000
    require_outputs: bool = False
    max_empty_inputs_pct: float = 0.05
    min_avg_input_length: int = 10
    fail_on_error: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_items": self.min_items,
            "max_items": self.max_items,
            "require_outputs": self.require_outputs,
            "max_empty_inputs_pct": self.max_empty_inputs_pct,
            "min_avg_input_length": self.min_avg_input_length,
            "fail_on_error": self.fail_on_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityGateConfig:
        return cls(
            min_items=data.get("min_items", 1),
            max_items=data.get("max_items", 100000),
            require_outputs=data.get("require_outputs", False),
            max_empty_inputs_pct=data.get("max_empty_inputs_pct", 0.05),
            min_avg_input_length=data.get("min_avg_input_length", 10),
            fail_on_error=data.get("fail_on_error", True),
        )


@dataclass
class QualityGateResult:
    """Result of running the quality gate."""

    passed: bool = True
    checks: list[QualityCheck] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    blocked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "errors": self.errors,
            "warnings": self.warnings,
            "blocked": self.blocked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityGateResult:
        checks = [
            QualityCheck(
                name=c["name"],
                passed=c["passed"],
                severity=c.get("severity", "warning"),
                message=c.get("message", ""),
            )
            for c in data.get("checks", [])
        ]
        return cls(
            passed=data.get("passed", True),
            checks=checks,
            errors=data.get("errors", 0),
            warnings=data.get("warnings", 0),
            blocked=data.get("blocked", False),
        )

    def format_text(self) -> str:
        status = "PASSED" if self.passed else "BLOCKED" if self.blocked else "WARNINGS"
        lines = [f"Quality Gate: {status}"]
        lines.append(f"  Checks: {len(self.checks)}, Errors: {self.errors}, Warnings: {self.warnings}")
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name} ({check.severity}): {check.message}")
        return "\n".join(lines)


def check_min_items(items: list[dict], config: QualityGateConfig) -> QualityCheck:
    """Check >= min_items."""
    count = len(items)
    passed = count >= config.min_items
    return QualityCheck(
        name="min_items",
        passed=passed,
        severity="error",
        message=f"{count} items (minimum: {config.min_items})",
    )


def check_max_items(items: list[dict], config: QualityGateConfig) -> QualityCheck:
    """Check <= max_items."""
    count = len(items)
    passed = count <= config.max_items
    return QualityCheck(
        name="max_items",
        passed=passed,
        severity="error",
        message=f"{count} items (maximum: {config.max_items})",
    )


def check_empty_inputs(items: list[dict], config: QualityGateConfig) -> QualityCheck:
    """Check empty input rate <= max_empty_inputs_pct."""
    if not items:
        return QualityCheck(
            name="empty_inputs",
            passed=True,
            severity="warning",
            message="No items to check",
        )
    empty = sum(1 for item in items if not str(item.get("input", "")).strip())
    rate = empty / len(items)
    passed = rate <= config.max_empty_inputs_pct
    return QualityCheck(
        name="empty_inputs",
        passed=passed,
        severity="warning",
        message=f"{empty}/{len(items)} empty ({rate:.1%}, max: {config.max_empty_inputs_pct:.1%})",
    )


def check_avg_input_length(items: list[dict], config: QualityGateConfig) -> QualityCheck:
    """Check avg input length >= min_avg_input_length."""
    if not items:
        return QualityCheck(
            name="avg_input_length",
            passed=True,
            severity="warning",
            message="No items to check",
        )
    lengths = [len(str(item.get("input", ""))) for item in items]
    avg = sum(lengths) / len(lengths)
    passed = avg >= config.min_avg_input_length
    return QualityCheck(
        name="avg_input_length",
        passed=passed,
        severity="warning",
        message=f"Average: {avg:.1f} chars (minimum: {config.min_avg_input_length})",
    )


def check_has_outputs(items: list[dict], config: QualityGateConfig) -> QualityCheck:
    """Check all items have 'output' field (only if require_outputs)."""
    if not config.require_outputs:
        return QualityCheck(
            name="has_outputs",
            passed=True,
            severity="info",
            message="Output check not required",
        )
    missing = sum(1 for item in items if "output" not in item)
    passed = missing == 0
    return QualityCheck(
        name="has_outputs",
        passed=passed,
        severity="error",
        message=f"{missing}/{len(items)} items missing 'output' field",
    )


def check_duplicate_ids(items: list[dict]) -> QualityCheck:
    """Check no duplicate 'id' fields."""
    ids = [item.get("id") for item in items if "id" in item]
    unique = set(ids)
    dupes = len(ids) - len(unique)
    passed = dupes == 0
    return QualityCheck(
        name="duplicate_ids",
        passed=passed,
        severity="error" if dupes > 0 else "info",
        message=f"{dupes} duplicate IDs found" if dupes else "No duplicate IDs",
    )


def run_quality_gate(
    items: list[dict[str, Any]],
    config: QualityGateConfig | None = None,
) -> QualityGateResult:
    """Run all checks. Blocked if any error check fails and fail_on_error is True."""
    if config is None:
        config = QualityGateConfig()
    checks = [
        check_min_items(items, config),
        check_max_items(items, config),
        check_empty_inputs(items, config),
        check_avg_input_length(items, config),
        check_has_outputs(items, config),
        check_duplicate_ids(items),
    ]
    errors = sum(1 for c in checks if not c.passed and c.severity == "error")
    warnings = sum(1 for c in checks if not c.passed and c.severity == "warning")
    all_passed = all(c.passed for c in checks)
    blocked = errors > 0 and config.fail_on_error
    return QualityGateResult(
        passed=all_passed and not blocked,
        checks=checks,
        errors=errors,
        warnings=warnings,
        blocked=blocked,
    )


def should_proceed(result: QualityGateResult) -> bool:
    """True if not blocked."""
    return not result.blocked
