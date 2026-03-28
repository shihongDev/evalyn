from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.query_language import (
    Query,
    QueryPredicate,
    parse_query,
    match_span,
    filter_spans,
    filter_spans_by_string,
    _get_span_value,
    _compare,
)


def _span(sid, name="s", span_type="custom", status="ok", start_time=None, **attrs):
    t = start_time or datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=t,
        status=status,
        attributes=attrs,
    )


# --- Parsing tests ---


class TestParseQuery:
    def test_simple_exact_match(self):
        q = parse_query("span_type=llm_call")
        assert len(q.predicates) == 1
        assert q.predicates[0].field == "span_type"
        assert q.predicates[0].operator == "="
        assert q.predicates[0].value == "llm_call"
        assert q.combine == "and"

    def test_numeric_greater_than(self):
        q = parse_query("duration_ms>1000")
        assert q.predicates[0].operator == ">"
        assert q.predicates[0].value == "1000"

    def test_contains_operator(self):
        q = parse_query("name~generate")
        assert q.predicates[0].operator == "~"
        assert q.predicates[0].value == "generate"

    def test_multiple_predicates_and(self):
        q = parse_query("span_type=llm_call status=error")
        assert len(q.predicates) == 2
        assert q.combine == "and"

    def test_or_keyword(self):
        q = parse_query("status=error or status=running")
        assert len(q.predicates) == 2
        assert q.combine == "or"

    def test_gte_operator(self):
        q = parse_query("cost>=0.01")
        assert q.predicates[0].operator == ">="
        assert q.predicates[0].value == "0.01"

    def test_lte_operator(self):
        q = parse_query("duration_ms<=500")
        assert q.predicates[0].operator == "<="

    def test_lt_operator(self):
        q = parse_query("duration_ms<100")
        assert q.predicates[0].operator == "<"

    def test_ne_operator(self):
        q = parse_query("status!=ok")
        assert q.predicates[0].operator == "!="
        assert q.predicates[0].value == "ok"

    def test_nested_attribute(self):
        q = parse_query("attributes.model=gpt-4")
        assert q.predicates[0].field == "attributes.model"
        assert q.predicates[0].value == "gpt-4"

    def test_empty_query(self):
        q = parse_query("")
        assert len(q.predicates) == 0
        assert q.combine == "and"

    def test_whitespace_only(self):
        q = parse_query("   ")
        assert len(q.predicates) == 0

    def test_invalid_token_skipped(self):
        q = parse_query("!!!invalid span_type=llm_call")
        assert len(q.predicates) == 1
        assert q.predicates[0].field == "span_type"


# --- match_span tests ---


class TestMatchSpan:
    def test_exact_match(self):
        s = _span("1", span_type="llm_call")
        q = parse_query("span_type=llm_call")
        assert match_span(s, q) is True

    def test_exact_no_match(self):
        s = _span("1", span_type="custom")
        q = parse_query("span_type=llm_call")
        assert match_span(s, q) is False

    def test_numeric_gt(self):
        s = _span("1")
        s.end_time = s.start_time + timedelta(seconds=2)  # 2000ms
        q = parse_query("duration_ms>1000")
        assert match_span(s, q) is True

    def test_numeric_gt_no_match(self):
        s = _span("1")
        s.end_time = s.start_time + timedelta(milliseconds=500)
        q = parse_query("duration_ms>1000")
        assert match_span(s, q) is False

    def test_contains(self):
        s = _span("1", name="generate_response")
        q = parse_query("name~generate")
        assert match_span(s, q) is True

    def test_contains_no_match(self):
        s = _span("1", name="summarize")
        q = parse_query("name~generate")
        assert match_span(s, q) is False

    def test_nested_attribute(self):
        s = _span("1", model="gpt-4")
        q = parse_query("attributes.model=gpt-4")
        assert match_span(s, q) is True

    def test_nested_attribute_no_match(self):
        s = _span("1", model="gpt-3.5")
        q = parse_query("attributes.model=gpt-4")
        assert match_span(s, q) is False

    def test_and_both_match(self):
        s = _span("1", span_type="llm_call", status="error")
        q = parse_query("span_type=llm_call status=error")
        assert match_span(s, q) is True

    def test_and_one_fails(self):
        s = _span("1", span_type="llm_call", status="ok")
        q = parse_query("span_type=llm_call status=error")
        assert match_span(s, q) is False

    def test_or_one_matches(self):
        s = _span("1", status="running")
        q = parse_query("status=error or status=running")
        assert match_span(s, q) is True

    def test_or_none_match(self):
        s = _span("1", status="ok")
        q = parse_query("status=error or status=running")
        assert match_span(s, q) is False

    def test_empty_query_matches_all(self):
        s = _span("1")
        q = parse_query("")
        assert match_span(s, q) is True

    def test_ne_operator(self):
        s = _span("1", status="error")
        q = parse_query("status!=ok")
        assert match_span(s, q) is True

    def test_gte_operator(self):
        s = _span("1")
        s.end_time = s.start_time + timedelta(seconds=1)  # 1000ms exactly
        q = parse_query("duration_ms>=1000")
        assert match_span(s, q) is True

    def test_lte_operator(self):
        s = _span("1")
        s.end_time = s.start_time + timedelta(seconds=1)
        q = parse_query("duration_ms<=1000")
        assert match_span(s, q) is True

    def test_lt_operator(self):
        s = _span("1")
        s.end_time = s.start_time + timedelta(milliseconds=500)
        q = parse_query("duration_ms<1000")
        assert match_span(s, q) is True

    def test_none_value_does_not_match(self):
        s = _span("1")  # no end_time, duration_ms is None
        q = parse_query("duration_ms>0")
        assert match_span(s, q) is False


# --- filter_spans tests ---


class TestFilterSpans:
    def test_returns_matching(self):
        spans = [
            _span("1", span_type="llm_call"),
            _span("2", span_type="custom"),
            _span("3", span_type="llm_call"),
        ]
        q = parse_query("span_type=llm_call")
        result = filter_spans(spans, q)
        assert len(result) == 2
        assert all(s.span_type == "llm_call" for s in result)

    def test_empty_spans(self):
        result = filter_spans([], parse_query("span_type=llm_call"))
        assert result == []

    def test_by_string_convenience(self):
        spans = [
            _span("1", status="error"),
            _span("2", status="ok"),
        ]
        result = filter_spans_by_string(spans, "status=error")
        assert len(result) == 1
        assert result[0].id == "1"


# --- as_dict / from_dict roundtrip ---


class TestSerialization:
    def test_query_predicate_as_dict(self):
        p = QueryPredicate(field="name", operator="=", value="test")
        d = p.as_dict()
        assert d == {"field": "name", "operator": "=", "value": "test"}

    def test_query_roundtrip(self):
        q = parse_query("span_type=llm_call duration_ms>1000")
        d = q.as_dict()
        q2 = Query.from_dict(d)
        assert len(q2.predicates) == 2
        assert q2.combine == "and"
        assert q2.predicates[0].field == "span_type"
        assert q2.predicates[1].operator == ">"

    def test_query_or_roundtrip(self):
        q = parse_query("status=error or status=running")
        d = q.as_dict()
        q2 = Query.from_dict(d)
        assert q2.combine == "or"
        assert len(q2.predicates) == 2


# --- Helper tests ---


class TestHelpers:
    def test_get_span_value_id(self):
        s = _span("abc")
        assert _get_span_value(s, "id") == "abc"

    def test_get_span_value_parent_id(self):
        s = _span("1")
        assert _get_span_value(s, "parent_id") is None

    def test_get_span_value_unknown_field(self):
        s = _span("1")
        assert _get_span_value(s, "nonexistent") is None

    def test_compare_string_eq(self):
        assert _compare("hello", "=", "hello") is True
        assert _compare("hello", "=", "world") is False

    def test_compare_numeric_float(self):
        assert _compare(1.5, ">", "1.0") is True
        assert _compare(0.5, ">", "1.0") is False

    def test_compare_contains(self):
        assert _compare("generate_response", "~", "generate") is True
        assert _compare("summarize", "~", "generate") is False

    def test_compare_none(self):
        assert _compare(None, "=", "x") is False
