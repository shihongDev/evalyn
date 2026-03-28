"""Tests for agent tool call evaluation metrics."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.metrics.objective import (
    tool_call_accuracy_metric,
    tool_call_sequence_metric,
    _lcs_length,
    _get_tool_spans,
    _get_tool_name,
)
from evalyn_sdk.metrics.factory import build_objective_metric
from evalyn_sdk.models import DatasetItem, FunctionCall, Span


def _now():
    return datetime.now(timezone.utc)


def _make_span(name: str, span_type: str = "tool_call", attrs: dict = None):
    return Span(
        id=f"span-{name}",
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=_now(),
        attributes=attrs or {"tool.name": name},
    )


def _make_call(spans: list = None, output="done"):
    return FunctionCall(
        id="call-1",
        function_name="agent",
        inputs={"query": "test"},
        output=output,
        error=None,
        started_at=_now(),
        ended_at=_now(),
        duration_ms=100,
        session_id=None,
        spans=spans or [],
    )


def _make_item(expected_tools=None, expected_sequence=None):
    meta = {}
    if expected_tools is not None:
        meta["expected_tools"] = expected_tools
    if expected_sequence is not None:
        meta["expected_tool_sequence"] = expected_sequence
    return DatasetItem(id="item-1", input={"query": "test"}, output="done", metadata=meta)


class TestToolCallAccuracy:
    """Test tool_call_accuracy metric."""

    def test_all_tools_correct(self):
        spans = [_make_span("search"), _make_span("summarize")]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search"},
            {"name": "summarize"},
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 1.0
        assert result.passed is True
        assert result.details["matched"] == 2

    def test_one_missing(self):
        spans = [_make_span("search")]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search"},
            {"name": "summarize"},
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 0.5
        assert result.details["matched"] == 1
        assert result.details["total_expected"] == 2

    def test_no_matches(self):
        spans = [_make_span("wrong_tool")]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search"},
            {"name": "summarize"},
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 0.0
        assert result.passed is False

    def test_with_args_matching(self):
        spans = [_make_span("search", attrs={
            "tool.name": "search",
            "tool.parameters": {"query": "weather in tokyo"},
        })]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search", "args": {"query": "weather in tokyo"}},
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 1.0

    def test_args_mismatch(self):
        spans = [_make_span("search", attrs={
            "tool.name": "search",
            "tool.parameters": {"query": "wrong query"},
        })]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search", "args": {"query": "weather in tokyo"}},
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 0.0

    def test_name_only_matching_ignores_args(self):
        spans = [_make_span("search", attrs={
            "tool.name": "search",
            "tool.parameters": {"query": "anything"},
        })]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search"},  # no args specified = name-only match
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 1.0

    def test_no_expected_tools_returns_none(self):
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(_make_call(), _make_item())
        assert result.score is None
        assert result.passed is None

    def test_no_spans_all_fail(self):
        call = _make_call(spans=[])
        item = _make_item(expected_tools=[{"name": "search"}])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 0.0

    def test_extra_actual_tools_dont_penalize(self):
        spans = [_make_span("search"), _make_span("extra"), _make_span("summarize")]
        call = _make_call(spans)
        item = _make_item(expected_tools=[
            {"name": "search"},
            {"name": "summarize"},
        ])
        metric = tool_call_accuracy_metric()
        result = metric.evaluate(call, item)
        assert result.score == 1.0


class TestToolCallSequence:
    """Test tool_call_sequence metric."""

    def test_correct_order(self):
        spans = [_make_span("search"), _make_span("summarize"), _make_span("respond")]
        call = _make_call(spans)
        item = _make_item(expected_sequence=["search", "summarize", "respond"])
        metric = tool_call_sequence_metric()
        result = metric.evaluate(call, item)
        assert result.score == 1.0
        assert result.passed is True

    def test_reversed_order(self):
        spans = [_make_span("respond"), _make_span("summarize"), _make_span("search")]
        call = _make_call(spans)
        item = _make_item(expected_sequence=["search", "summarize", "respond"])
        metric = tool_call_sequence_metric()
        result = metric.evaluate(call, item)
        # LCS of reversed = 1 (any single element), score = 1/3
        assert result.score == pytest.approx(1/3, abs=0.01)

    def test_partial_match(self):
        spans = [_make_span("search"), _make_span("respond")]
        call = _make_call(spans)
        item = _make_item(expected_sequence=["search", "summarize", "respond"])
        metric = tool_call_sequence_metric()
        result = metric.evaluate(call, item)
        # LCS = ["search", "respond"] = 2, score = 2/3
        assert result.score == pytest.approx(2/3, abs=0.01)

    def test_no_expected_sequence_returns_none(self):
        metric = tool_call_sequence_metric()
        result = metric.evaluate(_make_call(), _make_item())
        assert result.score is None

    def test_empty_actual(self):
        call = _make_call(spans=[])
        item = _make_item(expected_sequence=["search"])
        metric = tool_call_sequence_metric()
        result = metric.evaluate(call, item)
        assert result.score == 0.0


class TestLCSLength:
    """Test the LCS helper."""

    def test_identical(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_empty(self):
        assert _lcs_length([], ["a", "b"]) == 0
        assert _lcs_length(["a"], []) == 0

    def test_no_common(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial(self):
        assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_with_extras(self):
        assert _lcs_length(["a", "b", "c"], ["x", "a", "y", "b", "z", "c"]) == 3


class TestHelpers:
    """Test helper functions."""

    def test_get_tool_spans_filters_correctly(self):
        spans = [
            _make_span("search", "tool_call"),
            _make_span("llm", "llm_call"),
            _make_span("action", "tool_use"),
        ]
        call = _make_call(spans)
        tool_spans = _get_tool_spans(call)
        assert len(tool_spans) == 2
        names = [s.name for s in tool_spans]
        assert "search" in names
        assert "action" in names
        assert "llm" not in names

    def test_get_tool_name_from_attributes(self):
        span = _make_span("raw_name", attrs={"tool.name": "proper_name"})
        assert _get_tool_name(span) == "proper_name"

    def test_get_tool_name_fallback_to_span_name(self):
        span = _make_span("fallback_name", attrs={})
        assert _get_tool_name(span) == "fallback_name"


class TestFactoryIntegration:
    """Test factory registration."""

    def test_build_tool_call_accuracy(self):
        m = build_objective_metric("tool_call_accuracy", {})
        assert m is not None
        assert m.spec.id == "tool_call_accuracy"

    def test_build_tool_call_sequence(self):
        m = build_objective_metric("tool_call_sequence", {})
        assert m is not None
        assert m.spec.id == "tool_call_sequence"
