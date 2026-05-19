"""
Performance benchmark suite: track and prevent performance regressions in evalyn.

Provides dataclasses and pure functions for running benchmarks, comparing
results against baselines, detecting regressions/improvements, and rendering
ASCII tables of results.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str = ""
    duration_ms: float = 0.0
    memory_mb: float = 0.0
    items_per_second: float = 0.0
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "memory_mb": self.memory_mb,
            "items_per_second": self.items_per_second,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        return cls(
            name=data.get("name", ""),
            duration_ms=data.get("duration_ms", 0.0),
            memory_mb=data.get("memory_mb", 0.0),
            items_per_second=data.get("items_per_second", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class BenchmarkBaseline:
    """Expected performance baseline for comparison."""

    name: str = ""
    expected_duration_ms: float = 0.0
    tolerance_pct: float = 20.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_duration_ms": self.expected_duration_ms,
            "tolerance_pct": self.tolerance_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkBaseline:
        return cls(
            name=data.get("name", ""),
            expected_duration_ms=data.get("expected_duration_ms", 0.0),
            tolerance_pct=data.get("tolerance_pct", 20.0),
        )


@dataclass
class BenchmarkReport:
    """Aggregate report of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    total_benchmarks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "total_benchmarks": self.total_benchmarks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReport:
        return cls(
            results=[
                BenchmarkResult.from_dict(r) for r in data.get("results", [])
            ],
            regressions=list(data.get("regressions", [])),
            improvements=list(data.get("improvements", [])),
            total_benchmarks=data.get("total_benchmarks", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Benchmark Report",
            f"  total_benchmarks: {self.total_benchmarks}",
            f"  regressions: {len(self.regressions)}",
            f"  improvements: {len(self.improvements)}",
        ]
        for r in self.results:
            lines.append(
                f"  {r.name}: {r.duration_ms:.2f}ms "
                f"({r.items_per_second:.1f} items/s)"
            )
        if self.regressions:
            lines.append("  REGRESSIONS:")
            for name in self.regressions:
                lines.append(f"    - {name}")
        if self.improvements:
            lines.append("  IMPROVEMENTS:")
            for name in self.improvements:
                lines.append(f"    - {name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def run_benchmark(
    name: str,
    fn: Callable,
    iterations: int = 10,
) -> BenchmarkResult:
    """Time a function over N iterations and report average duration."""
    if iterations <= 0:
        return BenchmarkResult(
            name=name,
            duration_ms=0.0,
            items_per_second=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    items_per_sec = iterations / elapsed if elapsed > 0 else 0.0

    return BenchmarkResult(
        name=name,
        duration_ms=avg_ms,
        items_per_second=items_per_sec,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def compare_to_baseline(
    result: BenchmarkResult,
    baseline: BenchmarkBaseline,
) -> str:
    """Compare a result to its baseline.

    Returns "regression" if duration exceeds baseline + tolerance,
    "improvement" if duration is below baseline - tolerance,
    or "stable" otherwise.
    """
    upper = baseline.expected_duration_ms * (1 + baseline.tolerance_pct / 100)
    lower = baseline.expected_duration_ms * (1 - baseline.tolerance_pct / 100)

    if result.duration_ms > upper:
        return "regression"
    if result.duration_ms < lower:
        return "improvement"
    return "stable"


def build_benchmark_report(
    results: list[BenchmarkResult],
    baselines: list[BenchmarkBaseline] = None,
) -> BenchmarkReport:
    """Build a full benchmark report, comparing to baselines if provided."""
    if baselines is None:
        baselines = []

    baseline_map: dict[str, BenchmarkBaseline] = {b.name: b for b in baselines}
    regressions: list[str] = []
    improvements: list[str] = []

    for result in results:
        if result.name in baseline_map:
            status = compare_to_baseline(result, baseline_map[result.name])
            if status == "regression":
                regressions.append(result.name)
            elif status == "improvement":
                improvements.append(result.name)

    return BenchmarkReport(
        results=list(results),
        regressions=regressions,
        improvements=improvements,
        total_benchmarks=len(results),
    )


def save_baselines(
    results: list[BenchmarkResult],
    tolerance_pct: float = 20.0,
) -> list[BenchmarkBaseline]:
    """Convert benchmark results into baselines for future comparison."""
    return [
        BenchmarkBaseline(
            name=r.name,
            expected_duration_ms=r.duration_ms,
            tolerance_pct=tolerance_pct,
        )
        for r in results
    ]


def render_benchmark_table(report: BenchmarkReport) -> str:
    """ASCII table of benchmark results."""
    if not report.results:
        return "No benchmark results to display."

    # Determine column widths
    name_w = max(len(r.name) for r in report.results)
    name_w = max(name_w, len("Name"))
    dur_w = max(len(f"{r.duration_ms:.2f}") for r in report.results)
    dur_w = max(dur_w, len("Duration(ms)"))
    ips_w = max(len(f"{r.items_per_second:.1f}") for r in report.results)
    ips_w = max(ips_w, len("Items/s"))

    header = (
        f"{'Name':<{name_w}}  {'Duration(ms)':>{dur_w}}  {'Items/s':>{ips_w}}  Status"
    )
    sep = "-" * len(header)

    regression_set = set(report.regressions)
    improvement_set = set(report.improvements)

    lines = [header, sep]
    for r in report.results:
        if r.name in regression_set:
            status = "REGRESSION"
        elif r.name in improvement_set:
            status = "IMPROVEMENT"
        else:
            status = "stable"
        lines.append(
            f"{r.name:<{name_w}}  {r.duration_ms:>{dur_w}.2f}  "
            f"{r.items_per_second:>{ips_w}.1f}  {status}"
        )

    return "\n".join(lines)
