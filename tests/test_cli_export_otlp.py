"""Tests for `evalyn export-otlp` command."""

from __future__ import annotations

import argparse
import io
from unittest.mock import patch

import pytest

from evalyn_sdk.cli.commands import export_otlp


def _make_args(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        endpoint="http://localhost:4317",
        protocol="grpc",
        dry_run=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ----------------------------------------------------------------------
# CLI behavior
# ----------------------------------------------------------------------


def test_dry_run_skips_send_and_prints_config():
    args = _make_args(dry_run=True, endpoint="https://otel.example:4317")
    buf = io.StringIO()
    with patch.object(export_otlp, "_try_send_test_span") as mock_send, \
         patch("sys.stdout", new=buf):
        export_otlp.cmd_export_otlp(args)
    assert not mock_send.called
    output = buf.getvalue()
    assert "Dry run" in output
    assert "https://otel.example:4317" in output
    assert "EVALYN_OTEL_EXPORTER=otlp" in output


def test_successful_send_prints_config():
    args = _make_args(endpoint="https://otel.example:4317")
    buf = io.StringIO()
    with patch.object(export_otlp, "_try_send_test_span", return_value=(True, "ok")), \
         patch("sys.stdout", new=buf):
        export_otlp.cmd_export_otlp(args)
    output = buf.getvalue()
    assert "OK" in output
    assert "EVALYN_OTEL_EXPORTER=otlp" in output
    assert "https://otel.example:4317" in output


def test_failure_exits_nonzero_with_help():
    args = _make_args()
    stderr = io.StringIO()
    with patch.object(export_otlp, "_try_send_test_span", return_value=(False, "boom")), \
         patch("sys.stderr", new=stderr), pytest.raises(SystemExit) as exc:
        export_otlp.cmd_export_otlp(args)
    assert exc.value.code == 1
    err = stderr.getvalue()
    assert "FAIL" in err
    assert "boom" in err
    assert "Common causes" in err


def test_http_protocol_includes_protocol_env_var():
    args = _make_args(protocol="http", endpoint="http://localhost:4318/v1/traces")
    buf = io.StringIO()
    with patch.object(export_otlp, "_try_send_test_span", return_value=(True, "ok")), \
         patch("sys.stdout", new=buf):
        export_otlp.cmd_export_otlp(args)
    output = buf.getvalue()
    assert "EVALYN_OTEL_PROTOCOL=http" in output


def test_grpc_protocol_omits_protocol_env_var():
    args = _make_args(protocol="grpc")
    buf = io.StringIO()
    with patch.object(export_otlp, "_try_send_test_span", return_value=(True, "ok")), \
         patch("sys.stdout", new=buf):
        export_otlp.cmd_export_otlp(args)
    output = buf.getvalue()
    assert "EVALYN_OTEL_PROTOCOL" not in output


# ----------------------------------------------------------------------
# Argparse wiring
# ----------------------------------------------------------------------


def test_register_commands_wires_subparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    export_otlp.register_commands(subparsers)

    args = parser.parse_args(["export-otlp", "--endpoint", "x", "--dry-run"])
    assert args.endpoint == "x"
    assert args.protocol == "grpc"
    assert args.dry_run is True
    assert hasattr(args, "func")


def test_register_commands_requires_endpoint():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    export_otlp.register_commands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["export-otlp"])


def test_register_commands_validates_protocol():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    export_otlp.register_commands(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["export-otlp", "--endpoint", "x", "--protocol", "bogus"])
