"""Tests for ``evalyn_dashboard.introspect``.

A2.2 covers the per-kind classification path. A2.4 covers the
``build_catalog`` walker over every evalyn command module and validates the
output against ``dashboard/schema/catalog.schema.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evalyn_dashboard.introspect import (
    CliSchema,
    ParamSchema,
    _classify_kind,
    build_catalog,
    catalog_to_payload,
    introspect_parser,
)

from tests.fixtures.sample_parsers import (
    make_bool_parser,
    make_long_text_parser,
    make_multiselect_parser,
    make_number_parser,
    make_path_parser,
    make_select_parser,
    make_string_parser,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schema" / "catalog.schema.json"
)


# ---------------------------------------------------------------------------
# Per-kind classification (A2.2)
# ---------------------------------------------------------------------------


def test_bool_kind() -> None:
    schema = introspect_parser(make_bool_parser())
    flag = next(p for p in schema.params if p.name == "flag")
    assert flag.kind == "bool"
    assert flag.default is False


def test_select_kind() -> None:
    schema = introspect_parser(make_select_parser())
    mode = next(p for p in schema.params if p.name == "mode")
    assert mode.kind == "select"
    assert mode.options == ["a", "b", "c"]
    assert mode.default == "a"


def test_multiselect_kind() -> None:
    schema = introspect_parser(make_multiselect_parser())
    tags = next(p for p in schema.params if p.name == "tags")
    assert tags.kind == "multiselect"
    assert tags.options == ["x", "y", "z"]


def test_number_kind() -> None:
    schema = introspect_parser(make_number_parser())
    workers = next(p for p in schema.params if p.name == "workers")
    threshold = next(p for p in schema.params if p.name == "threshold")
    assert workers.kind == "number"
    assert threshold.kind == "number"
    assert workers.default == 8
    assert threshold.default == 0.5


def test_path_kind_by_name() -> None:
    schema = introspect_parser(make_path_parser())
    out = next(p for p in schema.params if p.name == "output")
    inp = next(p for p in schema.params if p.name == "input_file")
    assert out.kind == "path"
    assert inp.kind == "path"
    assert inp.required is True


def test_long_text_kind_by_name() -> None:
    schema = introspect_parser(make_long_text_parser())
    prompt = next(p for p in schema.params if p.name == "prompt")
    sys_prompt = next(p for p in schema.params if p.name == "system_prompt")
    assert prompt.kind == "long-text"
    assert sys_prompt.kind == "long-text"


def test_string_default_kind() -> None:
    schema = introspect_parser(make_string_parser())
    name = next(p for p in schema.params if p.name == "name")
    assert name.kind == "string"
    assert name.required is True


def test_help_and_version_actions_skipped() -> None:
    """Auto-generated --help (and --version) entries are filtered out."""
    p = argparse.ArgumentParser(prog="x", description="")
    p.add_argument("--keep", default="ok")
    schema = introspect_parser(p)
    names = {param.name for param in schema.params}
    assert "help" not in names
    assert "keep" in names


def test_classify_order_bool_beats_choices() -> None:
    """``store_true`` actions must classify as bool even though argparse sets
    their ``choices`` to ``None`` (defensive: explicit bool check first)."""
    p = argparse.ArgumentParser(prog="x")
    p.add_argument("--flag", action="store_true")
    bool_action = next(a for a in p._actions if a.dest == "flag")
    assert _classify_kind(bool_action) == "bool"


def test_classify_choices_with_nargs_plus_is_multiselect() -> None:
    p = argparse.ArgumentParser(prog="x")
    p.add_argument("--xs", choices=["a", "b"], nargs="+")
    a = next(a for a in p._actions if a.dest == "xs")
    assert _classify_kind(a) == "multiselect"


def test_classify_int_type_is_number() -> None:
    p = argparse.ArgumentParser(prog="x")
    p.add_argument("--n", type=int)
    a = next(a for a in p._actions if a.dest == "n")
    assert _classify_kind(a) == "number"


def test_classify_path_dest_dash_normalised() -> None:
    """``--input-file`` becomes dest ``input_file``; the path regex matches
    when underscores are normalised back to dashes."""
    p = argparse.ArgumentParser(prog="x")
    p.add_argument("--input-file")
    a = next(a for a in p._actions if a.dest == "input_file")
    assert _classify_kind(a) == "path"


# ---------------------------------------------------------------------------
# Catalog builder (A2.4)
# ---------------------------------------------------------------------------


def test_build_catalog_has_at_least_35_entries() -> None:
    catalog = build_catalog()
    ids = [c.id for c in catalog]
    assert len(catalog) >= 35, f"expected >= 35 commands, got {len(catalog)}: {ids}"


def test_build_catalog_ids_are_unique() -> None:
    catalog = build_catalog()
    ids = [c.id for c in catalog]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


def test_build_catalog_includes_known_commands() -> None:
    catalog = build_catalog()
    ids = {c.id for c in catalog}
    for expected in ("run-eval", "analyze", "compare", "list-runs", "build-dataset"):
        assert expected in ids, f"missing {expected!r}"


def test_build_catalog_groups_assigned() -> None:
    """Every catalog entry advertises a non-empty group string."""
    catalog = build_catalog()
    for entry in catalog:
        assert entry.group, f"{entry.id} has empty group"
        assert entry.group != "Misc", f"{entry.id} fell through to 'Misc'"


def test_build_catalog_validates_against_schema() -> None:
    """``build_catalog()`` output must validate against the JSON schema."""
    jsonschema = pytest.importorskip("jsonschema")
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    catalog = build_catalog()
    payload = catalog_to_payload(catalog)
    jsonschema.validate(instance=payload, schema=schema)


def test_build_catalog_param_kinds_are_in_enum() -> None:
    """Every classified ``kind`` must be one of the 7 enum members."""
    valid_kinds = {
        "bool",
        "string",
        "number",
        "select",
        "multiselect",
        "path",
        "long-text",
    }
    catalog = build_catalog()
    for entry in catalog:
        for param in entry.params:
            assert param.kind in valid_kinds, f"{entry.id}.{param.name} bad kind {param.kind!r}"


def test_dataclasses_are_exported() -> None:
    """Public dataclass shape used by api/cli to serialise."""
    assert ParamSchema.__dataclass_fields__.keys() >= {
        "name",
        "kind",
        "default",
        "options",
        "help",
        "required",
        "advanced",
        "essential",
    }
    assert CliSchema.__dataclass_fields__.keys() >= {
        "id",
        "name",
        "group",
        "blurb",
        "params",
    }


# ---------------------------------------------------------------------------
# essential flag (Approach D)
# ---------------------------------------------------------------------------


def test_essential_defaults_false() -> None:
    """Without an explicit `essential=` set, every param is essential=False."""
    schema = introspect_parser(make_string_parser())
    for p in schema.params:
        assert p.essential is False, f"{p.name} unexpectedly essential"


def test_introspect_parser_respects_essential_arg() -> None:
    """Names listed in `essential=` are flagged on the resulting params."""
    p = argparse.ArgumentParser(prog="x")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--project", default=None)
    schema = introspect_parser(p, essential={"limit"})
    by_name = {param.name: param for param in schema.params}
    assert by_name["limit"].essential is True
    assert by_name["project"].essential is False


def test_build_catalog_reads_module_essential_constant() -> None:
    """`build_catalog` honours the per-module ``ESSENTIAL`` constant.

    `traces.py` declares ``ESSENTIAL = {"limit"}``; the resulting
    ``list-calls`` schema must mark that param essential.
    """
    catalog = build_catalog()
    list_calls = next(c for c in catalog if c.id == "list-calls")
    by_name = {p.name: p for p in list_calls.params}
    assert "limit" in by_name, "list-calls is expected to expose --limit"
    assert by_name["limit"].essential is True
    # Sanity: at least one other non-essential param exists in the same CLI so
    # the form's "default visible" partition is meaningful.
    assert any(not p.essential and not p.required for p in list_calls.params)


def test_build_catalog_essential_payload_round_trip() -> None:
    """``catalog_to_payload`` carries the `essential` flag through ``asdict``."""
    catalog = build_catalog()
    payload = catalog_to_payload(catalog)
    list_calls = next(c for c in payload if c["id"] == "list-calls")
    limit_param = next(p for p in list_calls["params"] if p["name"] == "limit")
    assert limit_param["essential"] is True


def test_build_catalog_logs_when_plugin_import_fails(monkeypatch, caplog) -> None:
    """A broken plugin should not crash the catalog walk - the existing
    swallow-and-continue behavior is correct - but a WARN-level log
    line should give operators a breadcrumb when the plugin's
    commands silently vanish.
    """
    import importlib as _importlib
    from evalyn_dashboard import introspect as _introspect

    real_import = _importlib.import_module

    def boom(name, *args, **kwargs):
        # Synthesize a fake "broken plugin" module path. Re-raise
        # ImportError so the existing except-clause path is hit.
        if name == "evalyn.commands.synthesized_broken_plugin":
            raise ImportError("synthetic plugin import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_importlib, "import_module", boom)

    # Inject our synthetic plugin into the discovered list so the
    # walker will try to import it. _collect_command_modules is
    # the seam the catalog walker iterates over.
    real_collect = _introspect._collect_command_modules

    def collect_with_synthetic():
        return list(real_collect()) + ["evalyn.commands.synthesized_broken_plugin"]

    monkeypatch.setattr(
        _introspect, "_collect_command_modules", collect_with_synthetic
    )

    with caplog.at_level("WARNING", logger="evalyn_dashboard.introspect"):
        # Should not raise; the broken plugin is skipped.
        catalog = build_catalog()
        # Real plugins still loaded.
        assert len(catalog) > 0
        # The breadcrumb is in the log: module path + exc type +
        # message, so postmortems can find the culprit.
        warn_records = [
            r for r in caplog.records if r.levelname == "WARNING"
        ]
        assert any(
            "synthesized_broken_plugin" in r.message for r in warn_records
        ), f"expected a WARN log mentioning the broken plugin; got {[r.message for r in warn_records]}"
