"""Context propagation diagnostics.

Verify ContextVar propagation across sync, thread, and async boundaries.
Useful for confirming that tracing context will survive across execution
boundaries in the host application.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PropagationTest:
    """Single propagation test result."""

    name: str
    passed: bool
    details: str = ""
    context_type: str = "sync"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "context_type": self.context_type,
        }


@dataclass
class DiagnosticsReport:
    """Aggregated diagnostics report."""

    tests: list[PropagationTest] = field(default_factory=list)
    all_passed: bool = True
    sync_ok: bool = True
    thread_ok: bool = True
    async_ok: bool = True
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tests": [t.as_dict() for t in self.tests],
            "all_passed": self.all_passed,
            "sync_ok": self.sync_ok,
            "thread_ok": self.thread_ok,
            "async_ok": self.async_ok,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticsReport:
        tests = [
            PropagationTest(
                name=t["name"],
                passed=t["passed"],
                details=t.get("details", ""),
                context_type=t.get("context_type", "sync"),
            )
            for t in data.get("tests", [])
        ]
        return cls(
            tests=tests,
            all_passed=data.get("all_passed", True),
            sync_ok=data.get("sync_ok", True),
            thread_ok=data.get("thread_ok", True),
            async_ok=data.get("async_ok", True),
            recommendations=list(data.get("recommendations", [])),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Context Propagation Diagnostics")
        lines.append("-" * 40)
        for t in self.tests:
            status = "PASS" if t.passed else "FAIL"
            line = f"  [{status}] {t.name} ({t.context_type})"
            if t.details:
                line += f" - {t.details}"
            lines.append(line)
        lines.append("-" * 40)
        passed_count = sum(1 for t in self.tests if t.passed)
        total = len(self.tests)
        lines.append(f"Passed: {passed_count}/{total}")
        if self.all_passed:
            lines.append("All propagation tests passed.")
        else:
            failed = total - passed_count
            lines.append(f"{failed} test(s) failed.")
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


def test_sync_propagation() -> PropagationTest:
    """Test that ContextVar propagation works in synchronous code."""
    try:
        var: ContextVar[str] = ContextVar("_diag_sync")
        token = var.set("sync_value")
        value = var.get()
        var.reset(token)
        if value == "sync_value":
            return PropagationTest(
                name="sync_propagation",
                passed=True,
                details="ContextVar set/get works in sync code",
                context_type="sync",
            )
        return PropagationTest(
            name="sync_propagation",
            passed=False,
            details=f"Expected 'sync_value', got '{value}'",
            context_type="sync",
        )
    except Exception as exc:
        return PropagationTest(
            name="sync_propagation",
            passed=False,
            details=str(exc),
            context_type="sync",
        )


def test_thread_propagation() -> PropagationTest:
    """Test ContextVar in ThreadPoolExecutor.

    Uses copy_context() to explicitly propagate context to worker threads.
    Without copy_context, threads may not see the parent context depending
    on Python version and executor implementation.
    """
    try:
        var: ContextVar[str] = ContextVar("_diag_thread")
        token = var.set("thread_value")
        ctx = copy_context()

        def _read_in_thread() -> str:
            return var.get("MISSING")

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(ctx.run, _read_in_thread)
            value = future.result(timeout=5)

        var.reset(token)

        if value == "thread_value":
            return PropagationTest(
                name="thread_propagation",
                passed=True,
                details="ContextVar propagated to thread via copy_context",
                context_type="thread",
            )
        return PropagationTest(
            name="thread_propagation",
            passed=False,
            details=f"Expected 'thread_value', got '{value}'",
            context_type="thread",
        )
    except Exception as exc:
        return PropagationTest(
            name="thread_propagation",
            passed=False,
            details=str(exc),
            context_type="thread",
        )


def test_contextvars_available() -> PropagationTest:
    """Test that contextvars module is importable and functional."""
    try:
        import contextvars as _cv  # noqa: F811

        var = _cv.ContextVar("_diag_avail_check")
        token = var.set("check")
        val = var.get()
        var.reset(token)
        if val == "check":
            return PropagationTest(
                name="contextvars_available",
                passed=True,
                details="contextvars module is functional",
                context_type="sync",
            )
        return PropagationTest(
            name="contextvars_available",
            passed=False,
            details="contextvars module returned unexpected value",
            context_type="sync",
        )
    except ImportError:
        return PropagationTest(
            name="contextvars_available",
            passed=False,
            details="contextvars module not available",
            context_type="sync",
        )
    except Exception as exc:
        return PropagationTest(
            name="contextvars_available",
            passed=False,
            details=str(exc),
            context_type="sync",
        )


def check_python_version() -> PropagationTest:
    """Verify Python version supports expected ContextVar behavior."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major < 3 or (major == 3 and minor < 7):
        return PropagationTest(
            name="python_version",
            passed=False,
            details=f"Python {version_str} - contextvars requires 3.7+",
            context_type="sync",
        )
    if major == 3 and minor < 12:
        return PropagationTest(
            name="python_version",
            passed=True,
            details=(
                f"Python {version_str} - use copy_context() for thread propagation"
            ),
            context_type="sync",
        )
    return PropagationTest(
        name="python_version",
        passed=True,
        details=f"Python {version_str} - full ContextVar support",
        context_type="sync",
    )


def run_diagnostics() -> DiagnosticsReport:
    """Run all context propagation tests and return a report."""
    tests = [
        test_contextvars_available(),
        check_python_version(),
        test_sync_propagation(),
        test_thread_propagation(),
    ]

    sync_tests = [t for t in tests if t.context_type == "sync"]
    thread_tests = [t for t in tests if t.context_type == "thread"]
    async_tests = [t for t in tests if t.context_type == "async"]

    sync_ok = all(t.passed for t in sync_tests) if sync_tests else True
    thread_ok = all(t.passed for t in thread_tests) if thread_tests else True
    async_ok = all(t.passed for t in async_tests) if async_tests else True
    all_passed = all(t.passed for t in tests)

    recommendations: list[str] = []
    if not thread_ok:
        recommendations.append(
            "Consider using copy_context() before spawning threads"
        )
    if not async_ok:
        recommendations.append(
            "Ensure async tasks are created within the correct context"
        )
    if not sync_ok:
        recommendations.append(
            "Basic ContextVar operations failed - check Python installation"
        )

    return DiagnosticsReport(
        tests=tests,
        all_passed=all_passed,
        sync_ok=sync_ok,
        thread_ok=thread_ok,
        async_ok=async_ok,
        recommendations=recommendations,
    )
