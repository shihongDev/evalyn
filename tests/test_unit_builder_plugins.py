"""Tests for unit_builder_plugins module."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.evaluation.unit_builder_plugins import (
    BuilderRegistry,
    EvalUnit,
    PerCallBuilder,
    PerSessionBuilder,
    PerToolBuilder,
    auto_select_builder,
    build_units,
    get_default_registry,
)


def _span(sid, span_type="llm_call", parent_id=None, **attrs):
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- PerCallBuilder ---


def test_per_call_builder_creates_units():
    spans = [_span("s1"), _span("s2"), _span("s3", span_type="tool_call")]
    builder = PerCallBuilder()
    units = builder.build_units(spans)
    assert len(units) == 2
    assert all(u["unit_type"] == "per_call" for u in units)


def test_per_call_builder_empty():
    builder = PerCallBuilder()
    assert builder.build_units([]) == []


def test_per_call_builder_no_llm_calls():
    spans = [_span("s1", span_type="tool_call"), _span("s2", span_type="retrieval")]
    builder = PerCallBuilder()
    assert builder.build_units(spans) == []


def test_per_call_builder_unit_ids():
    spans = [_span("abc"), _span("def")]
    builder = PerCallBuilder()
    units = builder.build_units(spans)
    ids = [u["unit_id"] for u in units]
    assert "abc" in ids
    assert "def" in ids


# --- PerSessionBuilder ---


def test_per_session_builder_groups():
    spans = [
        _span("s1", session_id="sess_a"),
        _span("s2", session_id="sess_a"),
        _span("s3", session_id="sess_b"),
    ]
    builder = PerSessionBuilder()
    units = builder.build_units(spans)
    assert len(units) == 2
    ids = {u["unit_id"] for u in units}
    assert ids == {"sess_a", "sess_b"}


def test_per_session_builder_no_sessions():
    spans = [_span("s1"), _span("s2")]
    builder = PerSessionBuilder()
    assert builder.build_units(spans) == []


def test_per_session_builder_empty():
    builder = PerSessionBuilder()
    assert builder.build_units([]) == []


def test_per_session_builder_span_counts():
    spans = [
        _span("s1", session_id="sess"),
        _span("s2", session_id="sess"),
        _span("s3", session_id="sess"),
    ]
    builder = PerSessionBuilder()
    units = builder.build_units(spans)
    assert len(units) == 1
    assert len(units[0]["spans"]) == 3


# --- PerToolBuilder ---


def test_per_tool_builder_includes_parent():
    parent = _span("llm1", span_type="llm_call")
    tool = _span("tool1", span_type="tool_call", parent_id="llm1")
    builder = PerToolBuilder()
    units = builder.build_units([parent, tool])
    assert len(units) == 1
    span_ids = [s.id for s in units[0]["spans"]]
    assert "llm1" in span_ids
    assert "tool1" in span_ids


def test_per_tool_builder_no_parent():
    tool = _span("tool1", span_type="tool_call")
    builder = PerToolBuilder()
    units = builder.build_units([tool])
    assert len(units) == 1
    assert len(units[0]["spans"]) == 1


def test_per_tool_builder_parent_not_llm():
    parent = _span("node1", span_type="node")
    tool = _span("tool1", span_type="tool_call", parent_id="node1")
    builder = PerToolBuilder()
    units = builder.build_units([parent, tool])
    assert len(units) == 1
    # Parent is not llm_call so should not be included
    assert len(units[0]["spans"]) == 1


def test_per_tool_builder_empty():
    builder = PerToolBuilder()
    assert builder.build_units([]) == []


def test_per_tool_builder_no_tools():
    spans = [_span("s1"), _span("s2")]
    builder = PerToolBuilder()
    assert builder.build_units(spans) == []


# --- build_units function ---


def test_build_units_with_per_call():
    spans = [_span("s1"), _span("s2")]
    units = build_units(spans, builder_name="per_call")
    assert len(units) == 2
    assert all(isinstance(u, EvalUnit) for u in units)


def test_build_units_with_per_session():
    spans = [_span("s1", session_id="sess_a"), _span("s2", session_id="sess_a")]
    units = build_units(spans, builder_name="per_session")
    assert len(units) == 1
    assert units[0].unit_type == "per_session"


def test_build_units_with_per_tool():
    parent = _span("llm1", span_type="llm_call")
    tool = _span("tool1", span_type="tool_call", parent_id="llm1")
    units = build_units([parent, tool], builder_name="per_tool")
    assert len(units) == 1
    assert "llm1" in units[0].span_ids
    assert "tool1" in units[0].span_ids


def test_build_units_unknown_builder():
    with pytest.raises(ValueError, match="Unknown builder"):
        build_units([], builder_name="nonexistent")


def test_build_units_custom_registry():
    registry = BuilderRegistry()
    registry.register("per_call", PerCallBuilder())
    spans = [_span("s1")]
    units = build_units(spans, builder_name="per_call", registry=registry)
    assert len(units) == 1


# --- auto_select_builder ---


def test_auto_select_session():
    spans = [_span("s1", session_id="sess")]
    assert auto_select_builder(spans) == "per_session"


def test_auto_select_tool():
    spans = [_span("s1", span_type="tool_call")]
    assert auto_select_builder(spans) == "per_tool"


def test_auto_select_default():
    spans = [_span("s1")]
    assert auto_select_builder(spans) == "per_call"


def test_auto_select_session_over_tool():
    spans = [
        _span("s1", span_type="tool_call", session_id="sess"),
        _span("s2", span_type="llm_call"),
    ]
    assert auto_select_builder(spans) == "per_session"


def test_auto_select_empty():
    assert auto_select_builder([]) == "per_call"


# --- BuilderRegistry ---


def test_registry_register_get():
    reg = BuilderRegistry()
    builder = PerCallBuilder()
    reg.register("my_builder", builder)
    assert reg.get("my_builder") is builder


def test_registry_get_missing():
    reg = BuilderRegistry()
    assert reg.get("nope") is None


def test_registry_list_builders():
    reg = BuilderRegistry()
    reg.register("a", PerCallBuilder())
    reg.register("b", PerSessionBuilder())
    names = reg.list_builders()
    assert "a" in names
    assert "b" in names


def test_registry_remove():
    reg = BuilderRegistry()
    reg.register("a", PerCallBuilder())
    assert reg.remove("a") is True
    assert reg.get("a") is None


def test_registry_remove_missing():
    reg = BuilderRegistry()
    assert reg.remove("nonexistent") is False


def test_registry_reset():
    reg = get_default_registry()
    reg.register("custom", PerCallBuilder())
    assert "custom" in reg.list_builders()
    reg.reset()
    assert "custom" not in reg.list_builders()
    assert "per_call" in reg.list_builders()


# --- get_default_registry ---


def test_get_default_registry_has_three():
    reg = get_default_registry()
    names = reg.list_builders()
    assert "per_call" in names
    assert "per_session" in names
    assert "per_tool" in names
    assert len(names) == 3


# --- EvalUnit as_dict / from_dict ---


def test_eval_unit_as_dict():
    unit = EvalUnit(
        unit_id="u1", unit_type="per_call",
        span_ids=["s1", "s2"], metadata={"key": "val"},
    )
    d = unit.as_dict()
    assert d["unit_id"] == "u1"
    assert d["span_ids"] == ["s1", "s2"]
    assert d["metadata"] == {"key": "val"}


def test_eval_unit_from_dict():
    d = {
        "unit_id": "u2",
        "unit_type": "per_session",
        "span_ids": ["s3"],
        "metadata": {},
    }
    unit = EvalUnit.from_dict(d)
    assert unit.unit_id == "u2"
    assert unit.unit_type == "per_session"


def test_eval_unit_roundtrip():
    unit = EvalUnit(unit_id="u3", unit_type="per_tool", span_ids=["a", "b"])
    restored = EvalUnit.from_dict(unit.as_dict())
    assert restored.unit_id == unit.unit_id
    assert restored.unit_type == unit.unit_type
    assert restored.span_ids == unit.span_ids


def test_eval_unit_defaults():
    unit = EvalUnit(unit_id="u4", unit_type="custom")
    assert unit.span_ids == []
    assert unit.metadata == {}
