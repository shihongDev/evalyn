"""Diagnose common setup issues for evalyn.

Pure Python, no external dependencies. Does NOT make API calls.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticCheck:
    """Result of a single diagnostic check."""

    check_name: str
    status: str  # "pass", "fail", "warn"
    details: str
    suggestion: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status,
            "details": self.details,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticCheck:
        return cls(
            check_name=data.get("check_name", ""),
            status=data.get("status", "fail"),
            details=data.get("details", ""),
            suggestion=data.get("suggestion", ""),
        )


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""

    checks: list[DiagnosticCheck] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticReport:
        return cls(
            checks=[DiagnosticCheck.from_dict(c) for c in data.get("checks", [])],
            pass_count=data.get("pass_count", 0),
            fail_count=data.get("fail_count", 0),
            warn_count=data.get("warn_count", 0),
        )


def _mask_key(key: str) -> str:
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]


def check_api_keys(providers: list[str] | None = None) -> DiagnosticCheck:
    """Check if API key env vars exist for common providers."""
    if providers is None:
        providers = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

    found: list[str] = []
    missing: list[str] = []
    for var in providers:
        val = os.environ.get(var, "")
        if val:
            found.append(f"{var}={_mask_key(val)}")
        else:
            missing.append(var)

    if not missing:
        return DiagnosticCheck(
            check_name="api_keys",
            status="pass",
            details=f"All {len(found)} provider keys found: {', '.join(found)}",
        )
    if found:
        return DiagnosticCheck(
            check_name="api_keys",
            status="warn",
            details=f"Found: {', '.join(found)}. Missing: {', '.join(missing)}",
            suggestion=f"Set missing env vars: {', '.join(missing)}",
        )
    return DiagnosticCheck(
        check_name="api_keys",
        status="fail",
        details=f"No API keys found. Checked: {', '.join(missing)}",
        suggestion="Set at least one provider API key as an environment variable",
    )


def check_disk_space(path: str = ".") -> DiagnosticCheck:
    """Check available disk space (pass if > 100MB)."""
    try:
        usage = shutil.disk_usage(path)
        free_mb = usage.free / (1024 * 1024)
        if free_mb > 100:
            return DiagnosticCheck(
                check_name="disk_space",
                status="pass",
                details=f"{free_mb:.0f} MB free",
            )
        return DiagnosticCheck(
            check_name="disk_space",
            status="warn",
            details=f"Only {free_mb:.0f} MB free",
            suggestion="Free up disk space for evaluation data",
        )
    except OSError as e:
        return DiagnosticCheck(
            check_name="disk_space",
            status="fail",
            details=str(e),
            suggestion="Check path accessibility",
        )


def check_write_permissions(path: str = ".") -> DiagnosticCheck:
    """Check write permissions by creating a temp file."""
    try:
        fd, tmp = tempfile.mkstemp(dir=path, prefix=".evalyn_check_")
        os.close(fd)
        os.unlink(tmp)
        return DiagnosticCheck(
            check_name="write_permissions",
            status="pass",
            details=f"Write access confirmed at {path}",
        )
    except OSError as e:
        return DiagnosticCheck(
            check_name="write_permissions",
            status="fail",
            details=str(e),
            suggestion=f"Check write permissions for {path}",
        )


def check_python_version() -> DiagnosticCheck:
    """Check Python >= 3.9."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 9):
        return DiagnosticCheck(
            check_name="python_version",
            status="pass",
            details=f"Python {version_str}",
        )
    return DiagnosticCheck(
        check_name="python_version",
        status="fail",
        details=f"Python {version_str} (requires >= 3.9)",
        suggestion="Upgrade Python to 3.9 or later",
    )


def check_data_directory(path: str = "data") -> DiagnosticCheck:
    """Check if data directory exists."""
    if Path(path).is_dir():
        return DiagnosticCheck(
            check_name="data_directory",
            status="pass",
            details=f"Data directory found at {path}",
        )
    return DiagnosticCheck(
        check_name="data_directory",
        status="warn",
        details=f"Data directory not found at {path}",
        suggestion=f"Create {path}/ or run evalyn init",
    )


def check_config_file(path: str = "evalyn.yaml") -> DiagnosticCheck:
    """Check if config file exists."""
    if Path(path).is_file():
        return DiagnosticCheck(
            check_name="config_file",
            status="pass",
            details=f"Config file found at {path}",
        )
    return DiagnosticCheck(
        check_name="config_file",
        status="warn",
        details=f"Config file not found at {path}",
        suggestion="Run evalyn init to create evalyn.yaml",
    )


def run_diagnostics(
    data_dir: str = "data", config_path: str = "evalyn.yaml"
) -> DiagnosticReport:
    """Run all diagnostic checks."""
    checks = [
        check_python_version(),
        check_api_keys(),
        check_disk_space(),
        check_write_permissions(),
        check_data_directory(data_dir),
        check_config_file(config_path),
    ]
    pass_count = sum(1 for c in checks if c.status == "pass")
    fail_count = sum(1 for c in checks if c.status == "fail")
    warn_count = sum(1 for c in checks if c.status == "warn")
    return DiagnosticReport(
        checks=checks,
        pass_count=pass_count,
        fail_count=fail_count,
        warn_count=warn_count,
    )


def format_diagnostic_report(report: DiagnosticReport) -> str:
    """Human-readable diagnostic report."""
    icons = {"pass": "[OK]", "fail": "[FAIL]", "warn": "[WARN]"}
    lines: list[str] = []
    lines.append("evalyn doctor")
    lines.append("=" * 40)
    for check in report.checks:
        icon = icons.get(check.status, "[??]")
        lines.append(f"  {icon} {check.check_name}: {check.details}")
        if check.suggestion:
            lines.append(f"       -> {check.suggestion}")
    lines.append("")
    lines.append(
        f"Summary: {report.pass_count} passed, "
        f"{report.warn_count} warnings, {report.fail_count} failed"
    )
    return "\n".join(lines)
