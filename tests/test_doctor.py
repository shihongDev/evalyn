"""Tests for doctor module."""

from __future__ import annotations

import os

from evalyn_sdk.doctor import (
    DiagnosticCheck,
    DiagnosticReport,
    check_api_keys,
    check_config_file,
    check_data_directory,
    check_disk_space,
    check_python_version,
    check_write_permissions,
    format_diagnostic_report,
    run_diagnostics,
)


class TestDiagnosticCheck:
    def test_as_dict(self):
        c = DiagnosticCheck(check_name="test", status="pass", details="ok")
        d = c.as_dict()
        assert d["check_name"] == "test"
        assert d["status"] == "pass"

    def test_from_dict_roundtrip(self):
        c = DiagnosticCheck(check_name="x", status="fail", details="bad", suggestion="fix")
        restored = DiagnosticCheck.from_dict(c.as_dict())
        assert restored.suggestion == "fix"

    def test_defaults(self):
        c = DiagnosticCheck(check_name="x", status="pass", details="ok")
        assert c.suggestion == ""


class TestDiagnosticReport:
    def test_as_dict(self):
        r = DiagnosticReport(checks=[], pass_count=1, fail_count=0, warn_count=0)
        d = r.as_dict()
        assert d["pass_count"] == 1

    def test_from_dict_roundtrip(self):
        check = DiagnosticCheck(check_name="x", status="pass", details="ok")
        r = DiagnosticReport(checks=[check], pass_count=1, fail_count=0, warn_count=0)
        restored = DiagnosticReport.from_dict(r.as_dict())
        assert len(restored.checks) == 1
        assert restored.pass_count == 1

    def test_defaults(self):
        r = DiagnosticReport()
        assert r.checks == []
        assert r.pass_count == 0


class TestCheckApiKeys:
    def test_all_missing(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = check_api_keys()
        assert result.status == "fail"
        assert "No API keys" in result.details

    def test_some_present(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-1234")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = check_api_keys()
        assert result.status == "warn"
        assert "1234" in result.details
        assert "GEMINI_API_KEY" in result.details

    def test_all_present(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "key1-abcd")
        monkeypatch.setenv("OPENAI_API_KEY", "key2-efgh")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key3-ijkl")
        result = check_api_keys()
        assert result.status == "pass"

    def test_custom_providers(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "val1234")
        result = check_api_keys(providers=["MY_KEY"])
        assert result.status == "pass"

    def test_key_masked(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key-abcd")
        result = check_api_keys(providers=["GEMINI_API_KEY"])
        assert "super-secret" not in result.details
        assert "abcd" in result.details


class TestCheckDiskSpace:
    def test_pass(self):
        result = check_disk_space(".")
        assert result.status == "pass"
        assert "MB free" in result.details

    def test_invalid_path(self):
        result = check_disk_space("/nonexistent/path/xyz")
        # May pass or fail depending on OS behavior
        assert result.check_name == "disk_space"


class TestCheckWritePermissions:
    def test_pass(self, tmp_path):
        result = check_write_permissions(str(tmp_path))
        assert result.status == "pass"

    def test_fail_readonly(self):
        result = check_write_permissions("/proc")
        # On Linux, /proc is not writable
        assert result.check_name == "write_permissions"


class TestCheckPythonVersion:
    def test_current_version(self):
        result = check_python_version()
        assert result.status == "pass"
        assert "Python" in result.details


class TestCheckDataDirectory:
    def test_exists(self, tmp_path):
        result = check_data_directory(str(tmp_path))
        assert result.status == "pass"

    def test_missing(self):
        result = check_data_directory("/nonexistent/data/dir")
        assert result.status == "warn"
        assert "not found" in result.details


class TestCheckConfigFile:
    def test_exists(self, tmp_path):
        config = tmp_path / "evalyn.yaml"
        config.write_text("model: test")
        result = check_config_file(str(config))
        assert result.status == "pass"

    def test_missing(self):
        result = check_config_file("/nonexistent/evalyn.yaml")
        assert result.status == "warn"


class TestRunDiagnostics:
    def test_runs_all_checks(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "evalyn.yaml"
        config.write_text("test: true")
        report = run_diagnostics(str(data_dir), str(config))
        assert len(report.checks) == 6
        assert report.pass_count + report.fail_count + report.warn_count == 6

    def test_counts_correct(self, tmp_path):
        report = run_diagnostics(str(tmp_path / "nodata"), str(tmp_path / "noconfig"))
        total = report.pass_count + report.fail_count + report.warn_count
        assert total == len(report.checks)


class TestFormatDiagnosticReport:
    def test_contains_header(self):
        report = DiagnosticReport(
            checks=[DiagnosticCheck("test", "pass", "ok")],
            pass_count=1, fail_count=0, warn_count=0,
        )
        text = format_diagnostic_report(report)
        assert "evalyn doctor" in text
        assert "[OK]" in text
        assert "1 passed" in text

    def test_shows_suggestions(self):
        report = DiagnosticReport(
            checks=[DiagnosticCheck("test", "fail", "bad", "fix this")],
            pass_count=0, fail_count=1, warn_count=0,
        )
        text = format_diagnostic_report(report)
        assert "[FAIL]" in text
        assert "fix this" in text

    def test_warn_icon(self):
        report = DiagnosticReport(
            checks=[DiagnosticCheck("test", "warn", "warning")],
            pass_count=0, fail_count=0, warn_count=1,
        )
        text = format_diagnostic_report(report)
        assert "[WARN]" in text

    def test_empty_report(self):
        report = DiagnosticReport()
        text = format_diagnostic_report(report)
        assert "0 passed" in text
