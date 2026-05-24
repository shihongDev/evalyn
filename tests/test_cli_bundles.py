"""Tests for `evalyn list-bundles` and `evalyn show-bundle`."""

from __future__ import annotations

import argparse
import io
import json
from unittest.mock import patch

import pytest

from evalyn_sdk.cli.commands import bundles
from evalyn_sdk.cli.constants import BUNDLES, BUNDLE_DESCRIPTIONS


def _make_args(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(format="table", name=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ----------------------------------------------------------------------
# list-bundles
# ----------------------------------------------------------------------


def test_list_bundles_table_includes_all_bundles():
    args = _make_args()
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_list_bundles(args)
    output = buf.getvalue()
    for name in BUNDLES:
        assert name in output


def test_list_bundles_table_shows_metric_counts():
    args = _make_args()
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_list_bundles(args)
    output = buf.getvalue()
    sample = next(iter(BUNDLES))
    assert str(len(BUNDLES[sample])) in output


def test_list_bundles_json_format():
    args = _make_args(format="json")
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_list_bundles(args)
    payload = json.loads(buf.getvalue())
    assert payload["count"] == len(BUNDLES)
    names = {b["name"] for b in payload["bundles"]}
    assert names == set(BUNDLES.keys())


def test_list_bundles_json_each_record_has_metric_count():
    args = _make_args(format="json")
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_list_bundles(args)
    payload = json.loads(buf.getvalue())
    for record in payload["bundles"]:
        assert record["metric_count"] == len(record["metrics"])
        assert record["metric_count"] > 0


# ----------------------------------------------------------------------
# show-bundle
# ----------------------------------------------------------------------


def test_show_bundle_lists_all_metrics_in_bundle():
    name = "rag-qa"
    args = _make_args(name=name)
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_show_bundle(args)
    output = buf.getvalue()
    assert name in output
    for metric_id in BUNDLES[name]:
        assert metric_id in output


def test_show_bundle_json_format():
    name = "chatbot"
    args = _make_args(name=name, format="json")
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_show_bundle(args)
    payload = json.loads(buf.getvalue())
    assert payload["name"] == name
    assert payload["metric_count"] == len(BUNDLES[name])
    assert payload["metrics"] == BUNDLES[name]
    assert payload["description"] == BUNDLE_DESCRIPTIONS.get(name, "")


def test_show_bundle_unknown_name_exits_nonzero():
    args = _make_args(name="bogus-bundle-name")
    stdout = io.StringIO()
    with patch("sys.stdout", new=stdout), pytest.raises(SystemExit) as exc:
        bundles.cmd_show_bundle(args)
    assert exc.value.code == 2
    output = stdout.getvalue()
    assert "not found" in output
    assert "Available bundles" in output


def test_show_bundle_case_insensitive():
    # Regression for review finding: `evalyn show-bundle RAG-QA` used to
    # fail because lookup was case-sensitive on lowercase-with-hyphens keys.
    args = _make_args(name="RAG-QA")
    buf = io.StringIO()
    with patch("sys.stdout", new=buf):
        bundles.cmd_show_bundle(args)
    output = buf.getvalue()
    assert "rag-qa" in output  # printed using the canonical lowercase name


# ----------------------------------------------------------------------
# Argparse wiring
# ----------------------------------------------------------------------


def test_register_commands_wires_both():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    bundles.register_commands(subparsers)

    list_args = parser.parse_args(["list-bundles", "--format", "json"])
    assert list_args.format == "json"
    assert hasattr(list_args, "func")

    show_args = parser.parse_args(["show-bundle", "rag-qa"])
    assert show_args.name == "rag-qa"
    assert show_args.format == "table"
    assert hasattr(show_args, "func")
