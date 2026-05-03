"""Tests for entry-point based plugin discovery in the core CLI."""

from __future__ import annotations

import importlib.metadata as md
import subprocess
import sys

import pytest


def test_dashboard_plugin_registered() -> None:
    """The evalyn-dashboard package must register a 'dashboard' entry point."""
    eps = list(md.entry_points(group="evalyn.commands"))
    names = [ep.name for ep in eps]
    assert "dashboard" in names, f"expected 'dashboard' entry point, got {names}"


def test_evalyn_dashboard_help_works(capsys: pytest.CaptureFixture[str]) -> None:
    """`evalyn dashboard --help` must reach the plugin's argparse parser."""
    from evalyn_sdk.cli.main import main

    with pytest.raises(SystemExit) as exc_info:
        main(["dashboard", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "dashboard" in out.lower()


def test_evalyn_dashboard_runs_placeholder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Calling `evalyn dashboard` should hit the placeholder cmd from the plugin."""
    from evalyn_sdk.cli.main import main

    main(["dashboard"])
    out = capsys.readouterr().out
    assert "Phase 1" in out or "not yet implemented" in out.lower()


def test_evalyn_dashboard_via_subprocess() -> None:
    """End-to-end: invoke the installed `evalyn` script and confirm exit code 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "evalyn_sdk.cli.main", "dashboard", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0
    assert "dashboard" in proc.stdout.lower()
