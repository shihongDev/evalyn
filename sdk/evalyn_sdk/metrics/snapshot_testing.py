"""Snapshot testing for metrics.

Detects unintended changes to metric scoring behavior by comparing
actual scores against a saved snapshot of expected results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricSnapshot:
    """A saved snapshot of expected metric scores for a set of test inputs."""

    metric_id: str
    test_inputs: list[dict[str, str]] = field(default_factory=list)
    expected_scores: list[float] = field(default_factory=list)
    snapshot_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "test_inputs": list(self.test_inputs),
            "expected_scores": list(self.expected_scores),
            "snapshot_hash": self.snapshot_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricSnapshot:
        return cls(
            metric_id=data["metric_id"],
            test_inputs=data.get("test_inputs", []),
            expected_scores=data.get("expected_scores", []),
            snapshot_hash=data.get("snapshot_hash", ""),
        )


@dataclass
class SnapshotTestResult:
    """Result of verifying actual scores against a snapshot."""

    metric_id: str
    passed: bool
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    total_tests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "total_tests": self.total_tests,
        }

    def format_text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{self.metric_id}: {status} ({self.total_tests} tests)"]
        for m in self.mismatches:
            lines.append(
                f"  index {m['index']}: expected={m['expected']}, actual={m['actual']}, delta={m['delta']}"
            )
        return "\n".join(lines)


@dataclass
class SnapshotSuite:
    """Collection of metric snapshots."""

    snapshots: list[MetricSnapshot] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshots": [s.as_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotSuite:
        return cls(
            snapshots=[MetricSnapshot.from_dict(s) for s in data.get("snapshots", [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    @classmethod
    def from_json(cls, data: str) -> SnapshotSuite:
        return cls.from_dict(json.loads(data))


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _compute_snapshot_hash(inputs: list[dict[str, str]], scores: list[float]) -> str:
    """Deterministic hash of inputs + expected scores."""
    payload = json.dumps({"inputs": inputs, "scores": scores}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def create_snapshot(
    metric_id: str,
    test_cases: list[tuple[dict[str, str], float]],
) -> MetricSnapshot:
    """Build a MetricSnapshot from (input, expected_score) pairs."""
    inputs = [tc[0] for tc in test_cases]
    scores = [tc[1] for tc in test_cases]
    return MetricSnapshot(
        metric_id=metric_id,
        test_inputs=inputs,
        expected_scores=scores,
        snapshot_hash=_compute_snapshot_hash(inputs, scores),
    )


def verify_snapshot(
    snapshot: MetricSnapshot,
    actual_scores: list[float],
    tolerance: float = 0.01,
) -> SnapshotTestResult:
    """Compare *actual_scores* against the snapshot's expected scores."""
    mismatches: list[dict[str, Any]] = []
    total = len(snapshot.expected_scores)
    for i, (expected, actual) in enumerate(zip(snapshot.expected_scores, actual_scores)):
        delta = abs(expected - actual)
        if delta > tolerance:
            mismatches.append(
                {"index": i, "expected": expected, "actual": actual, "delta": round(delta, 6)}
            )
    return SnapshotTestResult(
        metric_id=snapshot.metric_id,
        passed=len(mismatches) == 0,
        mismatches=mismatches,
        total_tests=total,
    )


def update_snapshot(
    snapshot: MetricSnapshot,
    new_scores: list[float],
) -> MetricSnapshot:
    """Return a new snapshot with *new_scores* replacing the expected scores."""
    return MetricSnapshot(
        metric_id=snapshot.metric_id,
        test_inputs=list(snapshot.test_inputs),
        expected_scores=list(new_scores),
        snapshot_hash=_compute_snapshot_hash(snapshot.test_inputs, new_scores),
    )


def detect_behavior_change(
    old_scores: list[float],
    new_scores: list[float],
    threshold: float = 0.05,
) -> bool:
    """Return True if any score changed by more than *threshold*."""
    for old, new in zip(old_scores, new_scores):
        if abs(old - new) > threshold:
            return True
    return False


def build_snapshot_report(results: list[SnapshotTestResult]) -> dict[str, Any]:
    """Aggregate pass/fail counts from a list of snapshot test results."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "results": [r.as_dict() for r in results],
    }
