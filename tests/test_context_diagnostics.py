"""Tests for trace.context_diagnostics module."""

from __future__ import annotations

import sys

import pytest

from evalyn_sdk.trace.context_diagnostics import (
    DiagnosticsReport,
    PropagationTest,
    check_python_version,
    run_diagnostics,
    test_contextvars_available,
    test_sync_propagation,
    test_thread_propagation,
)


# ---------------------------------------------------------------------------
# PropagationTest
# ---------------------------------------------------------------------------


class TestPropagationTest:
    def test_defaults(self):
        t = PropagationTest(name="a", passed=True)
        assert t.details == ""
        assert t.context_type == "sync"

    def test_as_dict(self):
        t = PropagationTest(
            name="x", passed=False, details="oops", context_type="thread"
        )
        d = t.as_dict()
        assert d["name"] == "x"
        assert d["passed"] is False
        assert d["details"] == "oops"
        assert d["context_type"] == "thread"

    def test_as_dict_keys(self):
        d = PropagationTest(name="k", passed=True).as_dict()
        assert set(d.keys()) == {"name", "passed", "details", "context_type"}

    def test_context_type_values(self):
        for ct in ("sync", "thread", "async"):
            t = PropagationTest(name="n", passed=True, context_type=ct)
            assert t.context_type == ct


# ---------------------------------------------------------------------------
# DiagnosticsReport
# ---------------------------------------------------------------------------


class TestDiagnosticsReport:
    def test_defaults(self):
        r = DiagnosticsReport()
        assert r.tests == []
        assert r.all_passed is True
        assert r.sync_ok is True
        assert r.thread_ok is True
        assert r.async_ok is True
        assert r.recommendations == []

    def test_as_dict(self):
        t = PropagationTest(name="a", passed=True)
        r = DiagnosticsReport(tests=[t], all_passed=True)
        d = r.as_dict()
        assert len(d["tests"]) == 1
        assert d["all_passed"] is True
        assert d["sync_ok"] is True
        assert d["recommendations"] == []

    def test_from_dict_roundtrip(self):
        original = DiagnosticsReport(
            tests=[
                PropagationTest("a", True, "ok", "sync"),
                PropagationTest("b", False, "bad", "thread"),
            ],
            all_passed=False,
            sync_ok=True,
            thread_ok=False,
            async_ok=True,
            recommendations=["fix threads"],
        )
        d = original.as_dict()
        restored = DiagnosticsReport.from_dict(d)
        assert restored.all_passed is False
        assert restored.sync_ok is True
        assert restored.thread_ok is False
        assert restored.async_ok is True
        assert len(restored.tests) == 2
        assert restored.tests[0].name == "a"
        assert restored.tests[1].passed is False
        assert restored.recommendations == ["fix threads"]

    def test_from_dict_empty(self):
        r = DiagnosticsReport.from_dict({})
        assert r.tests == []
        assert r.all_passed is True
        assert r.recommendations == []

    def test_from_dict_preserves_context_type(self):
        d = {
            "tests": [
                {"name": "t", "passed": True, "context_type": "async"},
            ],
        }
        r = DiagnosticsReport.from_dict(d)
        assert r.tests[0].context_type == "async"

    def test_format_text_all_pass(self):
        r = DiagnosticsReport(
            tests=[PropagationTest("sync_test", True, "ok", "sync")],
            all_passed=True,
        )
        text = r.format_text()
        assert "Context Propagation Diagnostics" in text
        assert "[PASS]" in text
        assert "All propagation tests passed." in text

    def test_format_text_with_failures(self):
        r = DiagnosticsReport(
            tests=[PropagationTest("thread_test", False, "broken", "thread")],
            all_passed=False,
            recommendations=["Use copy_context()"],
        )
        text = r.format_text()
        assert "[FAIL]" in text
        assert "1 test(s) failed." in text
        assert "Recommendations:" in text
        assert "Use copy_context()" in text

    def test_format_text_shows_context_type(self):
        r = DiagnosticsReport(
            tests=[PropagationTest("x", True, "", "thread")],
            all_passed=True,
        )
        text = r.format_text()
        assert "(thread)" in text

    def test_format_text_no_recommendations_when_empty(self):
        r = DiagnosticsReport(
            tests=[PropagationTest("x", True)],
            all_passed=True,
            recommendations=[],
        )
        text = r.format_text()
        assert "Recommendations:" not in text


# ---------------------------------------------------------------------------
# test_sync_propagation
# ---------------------------------------------------------------------------


class TestSyncPropagation:
    def test_passes(self):
        result = test_sync_propagation()
        assert result.passed is True
        assert result.name == "sync_propagation"
        assert result.context_type == "sync"

    def test_details_not_empty(self):
        result = test_sync_propagation()
        assert result.details != ""


# ---------------------------------------------------------------------------
# test_thread_propagation
# ---------------------------------------------------------------------------


class TestThreadPropagation:
    def test_passes_with_copy_context(self):
        result = test_thread_propagation()
        assert result.passed is True
        assert result.name == "thread_propagation"
        assert result.context_type == "thread"

    def test_details_mention_copy_context(self):
        result = test_thread_propagation()
        assert "copy_context" in result.details


# ---------------------------------------------------------------------------
# test_contextvars_available
# ---------------------------------------------------------------------------


class TestContextvarsAvailable:
    def test_passes(self):
        result = test_contextvars_available()
        assert result.passed is True
        assert result.name == "contextvars_available"

    def test_context_type_is_sync(self):
        result = test_contextvars_available()
        assert result.context_type == "sync"


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    def test_passes_on_current_python(self):
        result = check_python_version()
        assert result.passed is True
        assert result.name == "python_version"

    def test_details_include_version(self):
        result = check_python_version()
        major, minor = sys.version_info[:2]
        assert f"{major}.{minor}" in result.details


# ---------------------------------------------------------------------------
# run_diagnostics
# ---------------------------------------------------------------------------


class TestRunDiagnostics:
    def test_returns_report(self):
        report = run_diagnostics()
        assert isinstance(report, DiagnosticsReport)
        assert len(report.tests) >= 4

    def test_all_passed_on_healthy_system(self):
        report = run_diagnostics()
        assert report.all_passed is True

    def test_sync_ok(self):
        report = run_diagnostics()
        assert report.sync_ok is True

    def test_thread_ok(self):
        report = run_diagnostics()
        assert report.thread_ok is True

    def test_no_recommendations_when_all_pass(self):
        report = run_diagnostics()
        assert report.recommendations == []

    def test_report_as_dict_roundtrip(self):
        report = run_diagnostics()
        d = report.as_dict()
        restored = DiagnosticsReport.from_dict(d)
        assert restored.all_passed == report.all_passed
        assert len(restored.tests) == len(report.tests)

    def test_format_text_output(self):
        report = run_diagnostics()
        text = report.format_text()
        assert "Context Propagation Diagnostics" in text
        assert "Passed:" in text


# ---------------------------------------------------------------------------
# Edge cases: manually constructed failure scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_report_with_all_failures(self):
        tests = [
            PropagationTest("a", False, "fail1", "sync"),
            PropagationTest("b", False, "fail2", "thread"),
            PropagationTest("c", False, "fail3", "async"),
        ]
        r = DiagnosticsReport(
            tests=tests,
            all_passed=False,
            sync_ok=False,
            thread_ok=False,
            async_ok=False,
            recommendations=["rec1", "rec2"],
        )
        d = r.as_dict()
        assert d["all_passed"] is False
        assert len(d["recommendations"]) == 2
        text = r.format_text()
        assert "3 test(s) failed." in text

    def test_report_with_mixed_results(self):
        tests = [
            PropagationTest("a", True, "ok", "sync"),
            PropagationTest("b", False, "fail", "thread"),
        ]
        r = DiagnosticsReport(
            tests=tests,
            all_passed=False,
            sync_ok=True,
            thread_ok=False,
        )
        text = r.format_text()
        assert "[PASS]" in text
        assert "[FAIL]" in text

    def test_empty_report_format(self):
        r = DiagnosticsReport()
        text = r.format_text()
        assert "Passed: 0/0" in text
        assert "All propagation tests passed." in text
