"""Graceful item-level failure handling.

Continue evaluation when individual items fail. Collects errors,
enforces failure limits, and provides recovery suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ItemFailure:
    """Record of a single item-level failure."""

    item_id: str
    metric_id: str = ""
    error: str = ""
    error_type: str = ""
    recoverable: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "error": self.error,
            "error_type": self.error_type,
            "recoverable": self.recoverable,
        }


@dataclass
class GracefulConfig:
    """Configuration for graceful failure handling."""

    max_failures: int = 0  # 0 means unlimited
    skip_on_timeout: bool = True
    retry_count: int = 0
    collect_errors: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_failures": self.max_failures,
            "skip_on_timeout": self.skip_on_timeout,
            "retry_count": self.retry_count,
            "collect_errors": self.collect_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GracefulConfig:
        return cls(
            max_failures=data.get("max_failures", 0),
            skip_on_timeout=data.get("skip_on_timeout", True),
            retry_count=data.get("retry_count", 0),
            collect_errors=data.get("collect_errors", True),
        )


@dataclass
class GracefulReport:
    """Summary report of evaluation failures."""

    total_items: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    failures: List[ItemFailure] = field(default_factory=list)
    failure_rate: float = 0.0
    should_abort: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "failures": [f.as_dict() for f in self.failures],
            "failure_rate": self.failure_rate,
            "should_abort": self.should_abort,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GracefulReport:
        failures = [
            ItemFailure(**f) for f in data.get("failures", [])
        ]
        return cls(
            total_items=data.get("total_items", 0),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
            skipped=data.get("skipped", 0),
            failures=failures,
            failure_rate=data.get("failure_rate", 0.0),
            should_abort=data.get("should_abort", False),
        )

    def format_text(self) -> str:
        lines = [
            f"Total items: {self.total_items}",
            f"Succeeded: {self.succeeded}",
            f"Failed: {self.failed}",
            f"Skipped: {self.skipped}",
            f"Failure rate: {self.failure_rate:.1%}",
        ]
        if self.should_abort:
            lines.append("Status: ABORTED (failure limit reached)")
        if self.failures:
            lines.append("Failures:")
            for f in self.failures:
                parts = [f"  - {f.item_id}"]
                if f.metric_id:
                    parts.append(f"[{f.metric_id}]")
                if f.error_type:
                    parts.append(f"({f.error_type})")
                if f.error:
                    parts.append(f": {f.error}")
                lines.append(" ".join(parts))
        return "\n".join(lines)


class GracefulEvaluator:
    """Tracks item-level successes and failures during evaluation.

    Usage:
        evaluator = GracefulEvaluator(GracefulConfig(max_failures=10))

        for item in items:
            try:
                result = evaluate(item)
                evaluator.record_success(item.id)
            except Exception as e:
                evaluator.record_failure(item.id, str(e))
                if not evaluator.should_continue():
                    break

        report = evaluator.get_report()
    """

    def __init__(self, config: GracefulConfig) -> None:
        self.config = config
        self._succeeded: int = 0
        self._failed: int = 0
        self._skipped: int = 0
        self._failures: List[ItemFailure] = []

    def record_success(self, item_id: str) -> None:
        """Record a successful item evaluation."""
        self._succeeded += 1

    def record_failure(
        self,
        item_id: str,
        error: str,
        metric_id: str = "",
        recoverable: bool = True,
    ) -> None:
        """Record a failed item evaluation."""
        error_type, _ = classify_error(error)
        failure = ItemFailure(
            item_id=item_id,
            metric_id=metric_id,
            error=error,
            error_type=error_type,
            recoverable=recoverable,
        )
        self._failed += 1
        if self.config.collect_errors:
            self._failures.append(failure)

    def should_continue(self) -> bool:
        """Check whether evaluation should continue.

        Returns True if under the max_failures limit (0 = unlimited).
        """
        if self.config.max_failures == 0:
            return True
        return self._failed < self.config.max_failures

    def get_report(self) -> GracefulReport:
        """Build a summary report of the current state."""
        total = self._succeeded + self._failed + self._skipped
        failure_rate = self._failed / total if total > 0 else 0.0
        should_abort = not self.should_continue()
        return GracefulReport(
            total_items=total,
            succeeded=self._succeeded,
            failed=self._failed,
            skipped=self._skipped,
            failures=list(self._failures),
            failure_rate=failure_rate,
            should_abort=should_abort,
        )

    def reset(self) -> None:
        """Clear all recorded state."""
        self._succeeded = 0
        self._failed = 0
        self._skipped = 0
        self._failures.clear()


def create_graceful_evaluator(max_failures: int = 0) -> GracefulEvaluator:
    """Factory: create a GracefulEvaluator with a given failure limit."""
    config = GracefulConfig(max_failures=max_failures)
    return GracefulEvaluator(config)


def classify_error(error: str) -> Tuple[str, bool]:
    """Classify an error string into a type and recoverability.

    Returns:
        (error_type, recoverable) tuple.
    """
    lower = error.lower()
    if "timeout" in lower or "timed out" in lower:
        return ("timeout", True)
    if "parse" in lower or "json" in lower or "decode" in lower:
        return ("parse_error", True)
    if "api" in lower or "rate" in lower or "429" in lower or "500" in lower:
        return ("api_error", True)
    if "valid" in lower or "schema" in lower or "required" in lower:
        return ("validation_error", False)
    return ("unknown", False)


def suggest_recovery(failure: ItemFailure) -> str:
    """Suggest a recovery action based on error type.

    Args:
        failure: The item failure to suggest recovery for.

    Returns:
        A human-readable suggestion string.
    """
    suggestions = {
        "timeout": "Retry with a longer timeout or reduce input size",
        "parse_error": "Check the prompt template for output format instructions",
        "api_error": "Retry after a backoff period or switch providers",
        "validation_error": "Fix the input data - check required fields and schema",
        "unknown": "Inspect the error manually and add specific handling",
    }
    return suggestions.get(failure.error_type, suggestions["unknown"])
