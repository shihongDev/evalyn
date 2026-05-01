"""Localhost binding guard tests (A1.3).

Spec ref: design §10 (Network exposure). The dashboard binds 127.0.0.1
only by default; non-loopback hosts require the explicit ``--unsafe-bind``
flag. Without it, ``cmd_dashboard`` exits with a non-zero status and
prints a stderr explanation.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest import mock

import pytest

from evalyn_dashboard import cli_command


def _ns(**kwargs: Any) -> argparse.Namespace:
    """Build a Namespace with the canonical default cmd_dashboard args."""

    base = dict(
        host="127.0.0.1",
        port=7401,
        no_browser=True,
        unsafe_bind=False,
        dev=False,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_loopback_default_passes_guard(monkeypatch) -> None:
    """The default 127.0.0.1 host should not trip the guard."""

    called: dict[str, Any] = {}

    def fake_run(**kw: Any) -> int:
        called.update(kw)
        return 0

    monkeypatch.setattr(cli_command, "_run_server", fake_run)
    rc = cli_command.cmd_dashboard(_ns())
    assert rc == 0
    assert called["host"] == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::"])
def test_non_loopback_rejected_without_unsafe(host: str, capsys) -> None:
    """Public hosts without ``--unsafe-bind`` exit non-zero with stderr msg."""

    rc = cli_command.cmd_dashboard(_ns(host=host))
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "127.0.0.1" in err or "loopback" in err
    assert "--unsafe-bind" in err


def test_localhost_string_is_treated_as_loopback(monkeypatch) -> None:
    """``localhost`` resolves to a loopback address; allow it."""

    monkeypatch.setattr(cli_command, "_run_server", lambda **kw: 0)
    rc = cli_command.cmd_dashboard(_ns(host="localhost"))
    assert rc == 0


def test_unsafe_bind_allows_public_host_with_warning(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli_command, "_run_server", lambda **kw: 0)
    rc = cli_command.cmd_dashboard(_ns(host="0.0.0.0", unsafe_bind=True))
    assert rc == 0
    err = capsys.readouterr().err.lower()
    # Visible warning so the operator knows what they did.
    assert "unsafe" in err or "warning" in err


def test_argparse_exposes_unsafe_bind_and_dev_flags() -> None:
    """The subparser must accept the new flags."""

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    cli_command.register_commands(sub)
    # Parse with the new flags - should not raise SystemExit.
    args = parser.parse_args(
        [
            "dashboard",
            "--host",
            "0.0.0.0",
            "--port",
            "7402",
            "--unsafe-bind",
            "--no-browser",
            "--dev",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 7402
    assert args.unsafe_bind is True
    assert args.no_browser is True
    assert args.dev is True


def test_no_browser_default_false() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    cli_command.register_commands(sub)
    args = parser.parse_args(["dashboard"])
    assert args.host == "127.0.0.1"
    assert args.port == 7401
    assert args.no_browser is False
    assert args.unsafe_bind is False
    assert args.dev is False


def test_cmd_dashboard_forwards_args_to_run_server(monkeypatch) -> None:
    """All CLI args make it through to the server launcher."""

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli_command, "_run_server", lambda **kw: captured.update(kw) or 0
    )
    rc = cli_command.cmd_dashboard(
        _ns(host="127.0.0.1", port=7777, no_browser=True, dev=True)
    )
    assert rc == 0
    assert captured == {
        "host": "127.0.0.1",
        "port": 7777,
        "no_browser": True,
        "dev": True,
    }
