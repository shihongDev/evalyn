from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.trace_decorator import (
    SpanStack,
    SpanRecord,
    DecoratorConfig,
    trace_span,
    get_recorded_spans,
    clear_recorded_spans,
    _STACK,
)


def _span(sid: str = "s1", **attrs) -> Span:
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# -- SpanStack -----------------------------------------------------------------

class TestSpanStack:
    def test_push_pop(self):
        stack = SpanStack()
        stack.push("a")
        stack.push("b")
        assert stack.pop() == "b"
        assert stack.pop() == "a"

    def test_pop_empty(self):
        stack = SpanStack()
        assert stack.pop() is None

    def test_current(self):
        stack = SpanStack()
        assert stack.current() is None
        stack.push("x")
        assert stack.current() == "x"
        # current does not remove
        assert stack.current() == "x"

    def test_depth(self):
        stack = SpanStack()
        assert stack.depth() == 0
        stack.push("a")
        assert stack.depth() == 1
        stack.push("b")
        assert stack.depth() == 2
        stack.pop()
        assert stack.depth() == 1

    def test_clear(self):
        stack = SpanStack()
        stack.push("a")
        stack.push("b")
        stack.clear()
        assert stack.depth() == 0
        assert stack.current() is None

    def test_push_pop_sequence(self):
        stack = SpanStack()
        stack.push("1")
        stack.push("2")
        stack.push("3")
        assert stack.pop() == "3"
        assert stack.pop() == "2"
        assert stack.pop() == "1"
        assert stack.pop() is None


# -- SpanRecord ----------------------------------------------------------------

class TestSpanRecord:
    def test_basic(self):
        s = _span()
        rec = SpanRecord(span=s, function_name="my_func", args_summary="a=1, b=2")
        assert rec.function_name == "my_func"
        assert rec.args_summary == "a=1, b=2"

    def test_defaults(self):
        s = _span()
        rec = SpanRecord(span=s)
        assert rec.function_name == ""
        assert rec.args_summary == ""

    def test_as_dict(self):
        s = _span()
        rec = SpanRecord(span=s, function_name="fn", args_summary="x=1")
        d = rec.as_dict()
        assert d["function_name"] == "fn"
        assert d["args_summary"] == "x=1"
        assert "span" in d
        assert d["span"]["id"] == "s1"


# -- DecoratorConfig -----------------------------------------------------------

class TestDecoratorConfig:
    def test_defaults(self):
        cfg = DecoratorConfig()
        assert cfg.capture_args is True
        assert cfg.capture_return is False
        assert cfg.max_arg_length == 200

    def test_as_dict(self):
        cfg = DecoratorConfig(capture_args=False, capture_return=True, max_arg_length=50)
        d = cfg.as_dict()
        assert d == {"capture_args": False, "capture_return": True, "max_arg_length": 50}

    def test_from_dict(self):
        cfg = DecoratorConfig.from_dict({"capture_args": False, "max_arg_length": 100})
        assert cfg.capture_args is False
        assert cfg.capture_return is False
        assert cfg.max_arg_length == 100

    def test_roundtrip(self):
        original = DecoratorConfig(capture_args=True, capture_return=True, max_arg_length=300)
        rebuilt = DecoratorConfig.from_dict(original.as_dict())
        assert rebuilt.capture_args == original.capture_args
        assert rebuilt.capture_return == original.capture_return
        assert rebuilt.max_arg_length == original.max_arg_length


# -- trace_span as decorator ---------------------------------------------------

class TestTraceSpanDecorator:
    def setup_method(self):
        clear_recorded_spans()
        _STACK.clear()

    def test_creates_span(self):
        @trace_span("my_op")
        def do_work():
            return 42

        result = do_work()
        assert result == 42
        records = get_recorded_spans()
        assert len(records) == 1
        assert records[0].span.name == "my_op"
        assert records[0].span.status == "ok"

    def test_uses_function_name_when_no_name(self):
        @trace_span()
        def some_function():
            return 1

        some_function()
        records = get_recorded_spans()
        assert records[0].span.name == "some_function"

    def test_captures_args(self):
        @trace_span("op")
        def add(a, b):
            return a + b

        add(1, 2)
        records = get_recorded_spans()
        assert "args" in records[0].span.attributes
        assert "1" in records[0].span.attributes["args"]
        assert "2" in records[0].span.attributes["args"]

    def test_records_function_name(self):
        @trace_span("op")
        def my_func():
            pass

        my_func()
        records = get_recorded_spans()
        assert records[0].function_name == "my_func"

    def test_sets_parent_from_stack(self):
        @trace_span("outer")
        def outer():
            inner()

        @trace_span("inner")
        def inner():
            pass

        outer()
        records = get_recorded_spans()
        # inner finishes first, outer finishes second
        inner_rec = next(r for r in records if r.span.name == "inner")
        outer_rec = next(r for r in records if r.span.name == "outer")
        assert inner_rec.span.parent_id == outer_rec.span.id

    def test_exception_sets_error_status(self):
        @trace_span("failing")
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

        records = get_recorded_spans()
        assert len(records) == 1
        assert records[0].span.status == "error"
        assert "boom" in records[0].span.attributes.get("error", "")

    def test_exception_still_propagates(self):
        @trace_span("failing")
        def fail():
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            fail()

    def test_end_time_set(self):
        @trace_span("timed")
        def work():
            return 1

        work()
        records = get_recorded_spans()
        assert records[0].span.end_time is not None
        assert records[0].span.duration_ms is not None
        assert records[0].span.duration_ms >= 0

    def test_span_type(self):
        @trace_span("op", span_type="llm_call")
        def call_llm():
            pass

        call_llm()
        records = get_recorded_spans()
        assert records[0].span.span_type == "llm_call"

    def test_nested_decorators_hierarchy(self):
        @trace_span("level1")
        def level1():
            level2()

        @trace_span("level2")
        def level2():
            level3()

        @trace_span("level3")
        def level3():
            pass

        level1()
        records = get_recorded_spans()
        assert len(records) == 3

        by_name = {r.span.name: r for r in records}
        assert by_name["level1"].span.parent_id is None
        assert by_name["level2"].span.parent_id == by_name["level1"].span.id
        assert by_name["level3"].span.parent_id == by_name["level2"].span.id

    def test_no_args_capture(self):
        cfg = DecoratorConfig(capture_args=False)

        @trace_span("op", config=cfg)
        def work(secret):
            pass

        work("password123")
        records = get_recorded_spans()
        assert "args" not in records[0].span.attributes

    def test_preserves_function_metadata(self):
        @trace_span("op")
        def documented_func():
            """My docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "My docstring."


# -- trace_span as context manager ---------------------------------------------

class TestTraceSpanContextManager:
    def setup_method(self):
        clear_recorded_spans()
        _STACK.clear()

    def test_creates_span(self):
        with trace_span("block") as s:
            assert isinstance(s, Span)
            assert s.name == "block"
            assert s.status == "running"

        records = get_recorded_spans()
        assert len(records) == 1
        assert records[0].span.status == "ok"

    def test_end_time_set(self):
        with trace_span("block") as s:
            pass
        assert s.end_time is not None

    def test_exception_sets_error(self):
        with pytest.raises(ValueError):
            with trace_span("block") as s:
                raise ValueError("fail")

        assert s.status == "error"
        assert "fail" in s.attributes.get("error", "")

    def test_nested_context_managers(self):
        with trace_span("outer") as outer:
            with trace_span("inner") as inner:
                pass

        assert inner.parent_id == outer.id
        assert outer.parent_id is None

    def test_stack_cleaned_on_exit(self):
        with trace_span("block"):
            assert _STACK.depth() == 1
        assert _STACK.depth() == 0

    def test_stack_cleaned_on_exception(self):
        try:
            with trace_span("block"):
                assert _STACK.depth() == 1
                raise RuntimeError("x")
        except RuntimeError:
            pass
        assert _STACK.depth() == 0

    def test_default_name(self):
        with trace_span() as s:
            pass
        assert s.name == "context_span"


# -- get_recorded_spans / clear_recorded_spans ---------------------------------

class TestRecordedSpans:
    def setup_method(self):
        clear_recorded_spans()
        _STACK.clear()

    def test_get_returns_copy(self):
        @trace_span("op")
        def work():
            pass

        work()
        a = get_recorded_spans()
        b = get_recorded_spans()
        assert a is not b
        assert len(a) == len(b) == 1

    def test_clear_empties(self):
        @trace_span("op")
        def work():
            pass

        work()
        assert len(get_recorded_spans()) == 1
        clear_recorded_spans()
        assert len(get_recorded_spans()) == 0

    def test_multiple_calls_accumulate(self):
        @trace_span("op")
        def work():
            pass

        work()
        work()
        work()
        assert len(get_recorded_spans()) == 3
