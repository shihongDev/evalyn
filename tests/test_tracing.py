"""
Tests for tracing functionality.

Run with: pytest tests/test_tracing.py -v
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest


class TestEvalDecorator:
    """Tests for @eval decorator."""

    def test_sync_function_decoration(self):
        """Test decorating a sync function."""
        from evalyn_sdk import eval

        @eval(project="test", version="v1")
        def simple_func(x: int) -> int:
            return x * 2

        result = simple_func(5)
        assert result == 10

    def test_async_function_decoration(self):
        """Test decorating an async function."""
        from evalyn_sdk import eval

        @eval(project="test", version="v1")
        async def async_func(x: int) -> int:
            return x * 2

        result = asyncio.run(async_func(5))
        assert result == 10

    def test_function_with_kwargs(self):
        """Test decorated function with keyword arguments."""
        from evalyn_sdk import eval

        @eval(project="test", version="v1")
        def func_with_kwargs(a: int, b: int = 10) -> int:
            return a + b

        assert func_with_kwargs(5) == 15
        assert func_with_kwargs(5, b=20) == 25
        assert func_with_kwargs(a=3, b=7) == 10

    def test_function_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        from evalyn_sdk import eval

        @eval(project="test", version="v1")
        def documented_func(x: int) -> int:
            """This is the docstring."""
            return x

        assert documented_func.__name__ == "documented_func"
        assert "docstring" in documented_func.__doc__

    def test_function_with_exception(self):
        """Test decorated function that raises exception."""
        from evalyn_sdk import eval

        @eval(project="test", version="v1")
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_custom_name(self):
        """Test decorator with custom name."""
        from evalyn_sdk import eval

        @eval(project="test", version="v1", name="custom_name")
        def my_func():
            return "result"

        result = my_func()
        assert result == "result"


class TestSpanContext:
    """Tests for span context management."""

    def test_span_creation(self):
        """Test creating a span."""
        from evalyn_sdk.trace import span, get_current_span_id

        with span("test_span", "test") as s:
            # span returns a Span object
            assert s is not None
            assert hasattr(s, "id")

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        from evalyn_sdk.trace import span

        with span("outer", "test") as outer:
            assert outer is not None
            with span("inner", "test") as inner:
                assert inner is not None
                # Inner should have outer as parent
                assert inner.parent_id == outer.id

    def test_span_attributes(self):
        """Test span with attributes."""
        from evalyn_sdk.trace import span

        with span("test", "llm_call", model="gpt-4", tokens=100) as s:
            assert s.attributes.get("model") == "gpt-4"
            assert s.attributes.get("tokens") == 100


class TestEvalTracer:
    """Tests for EvalTracer class."""

    def test_tracer_creation(self):
        """Test creating an EvalTracer."""
        from evalyn_sdk import EvalTracer

        tracer = EvalTracer()
        assert tracer is not None

    def test_tracer_with_storage(self, temp_db):
        """Test EvalTracer with explicit storage."""
        from evalyn_sdk import EvalTracer
        from evalyn_sdk.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(str(temp_db))
        tracer = EvalTracer(storage=storage)
        assert tracer is not None


class TestEvalSession:
    """Tests for eval_session context manager."""

    def test_session_context(self):
        """Test eval_session creates session context."""
        from evalyn_sdk import eval_session, eval

        with eval_session() as session_id:
            assert session_id is not None

            @eval(project="test", version="v1")
            def inner_func():
                return "result"

            result = inner_func()
            assert result == "result"

    def test_nested_sessions(self):
        """Test nested eval_session contexts."""
        from evalyn_sdk import eval_session

        with eval_session() as outer:
            assert outer is not None

            with eval_session() as inner:
                assert inner is not None
                assert inner != outer


class TestAutoInstrumentation:
    """Tests for auto-instrumentation."""

    def test_patch_all_callable(self):
        """Test patch_all is callable."""
        from evalyn_sdk import patch_all
        assert callable(patch_all)

    def test_is_patched(self):
        """Test is_patched function."""
        from evalyn_sdk import is_patched
        # Should return boolean for any library
        result = is_patched("openai")
        assert isinstance(result, bool)

    def test_calculate_cost(self):
        """Test calculate_cost function."""
        from evalyn_sdk import calculate_cost

        # calculate_cost takes model, input_tokens, output_tokens
        cost = calculate_cost("gpt-4", 100, 50)
        assert isinstance(cost, (int, float))
        assert cost >= 0


class TestSpanModel:
    """Tests for Span model."""

    def test_span_creation(self):
        """Test creating a Span model."""
        from evalyn_sdk.models import Span

        s = Span(
            id="test-span-1",
            name="test_operation",
            span_type="llm_call",
            parent_id=None,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            status="ok",
            attributes={"model": "gpt-4"},
        )

        assert s.id == "test-span-1"
        assert s.name == "test_operation"
        assert s.span_type == "llm_call"
        assert s.parent_id is None
        assert s.status == "ok"
        assert s.attributes["model"] == "gpt-4"

    def test_span_with_parent(self):
        """Test creating a child span."""
        from evalyn_sdk.models import Span

        parent = Span(
            id="parent-1",
            name="parent",
            span_type="session",
            parent_id=None,
            start_time=datetime.now(timezone.utc),
            status="ok",
            attributes={},
        )

        child = Span(
            id="child-1",
            name="child",
            span_type="llm_call",
            parent_id=parent.id,
            start_time=datetime.now(timezone.utc),
            status="ok",
            attributes={},
        )

        assert child.parent_id == parent.id

    def test_span_duration(self):
        """Test span duration calculation."""
        from evalyn_sdk.models import Span
        from datetime import timedelta

        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=500)

        s = Span(
            id="dur-test",
            name="test",
            span_type="test",
            parent_id=None,
            start_time=start,
            end_time=end,
            status="ok",
            attributes={},
        )

        assert s.duration_ms == pytest.approx(500, rel=0.01)


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_trace_event_creation(self):
        """Test creating a TraceEvent."""
        from evalyn_sdk.models import TraceEvent

        event = TraceEvent(
            kind="llm.request",
            timestamp=datetime.now(timezone.utc),
            detail={"model": "gpt-4", "input_tokens": 100},
        )

        assert event.kind == "llm.request"
        assert event.detail["model"] == "gpt-4"

    def test_trace_event_with_span_id(self):
        """Test TraceEvent with span_id."""
        from evalyn_sdk.models import TraceEvent

        event = TraceEvent(
            kind="llm.response",
            timestamp=datetime.now(timezone.utc),
            detail={},
            span_id="span-123",
        )

        assert event.span_id == "span-123"


class TestFunctionCall:
    """Tests for FunctionCall model."""

    def test_function_call_creation(self):
        """Test creating a FunctionCall using factory method."""
        from evalyn_sdk.models import FunctionCall

        call = FunctionCall.new(
            function_name="my_func",
            inputs={"x": 1, "y": 2},
            session_id=None,
        )

        assert call.function_name == "my_func"
        assert call.inputs == {"x": 1, "y": 2}

    def test_function_call_with_metadata(self):
        """Test FunctionCall with metadata."""
        from evalyn_sdk.models import FunctionCall

        call = FunctionCall.new(
            function_name="my_func",
            inputs={},
            session_id="session-1",
            metadata={"project": "test", "version": "v1"},
        )

        assert call.session_id == "session-1"
        assert call.metadata["project"] == "test"

    def test_function_call_serialization(self):
        """Test FunctionCall serialization."""
        from evalyn_sdk.models import FunctionCall

        call = FunctionCall.new(
            function_name="test_func",
            inputs={"key": "value"},
            session_id=None,
        )
        call.output = "result"

        data = call.as_dict()
        assert data["function_name"] == "test_func"
        assert data["inputs"] == {"key": "value"}

        # Roundtrip
        restored = FunctionCall.from_dict(data)
        assert restored.function_name == call.function_name

    def test_function_call_spans(self):
        """Test adding spans to FunctionCall."""
        from evalyn_sdk.models import FunctionCall, Span

        call = FunctionCall.new(
            function_name="test",
            inputs={},
            session_id=None,
        )

        span = Span(
            id="span-1",
            name="test_span",
            span_type="llm_call",
            parent_id=None,
            start_time=datetime.now(timezone.utc),
            status="ok",
            attributes={},
        )

        call.add_span(span)
        assert len(call.spans) == 1
        assert call.spans[0].id == "span-1"


class TestXAIInstrumentation:
    """Tests for xAI instrumentation."""

    def test_detect_provider_xai(self):
        """Test _detect_provider returns 'xai' for x.ai base_url."""
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider

        # Create mock completions instance with x.ai base_url
        class MockClient:
            base_url = "https://api.x.ai/v1"

        class MockCompletions:
            _client = MockClient()

        result = _detect_provider(MockCompletions())
        assert result == "xai"

    def test_detect_provider_openai(self):
        """Test _detect_provider returns 'openai' for OpenAI base_url."""
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider

        # Create mock completions instance with OpenAI base_url
        class MockClient:
            base_url = "https://api.openai.com/v1"

        class MockCompletions:
            _client = MockClient()

        result = _detect_provider(MockCompletions())
        assert result == "openai"

    def test_detect_provider_no_client(self):
        """Test _detect_provider handles missing _client gracefully."""
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider

        class MockCompletions:
            pass

        result = _detect_provider(MockCompletions())
        assert result == "openai"

    def test_xai_model_pricing(self):
        """Test xAI model cost calculation."""
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost

        # Test grok-4-0709 pricing: $3/1M input, $15/1M output
        cost = calculate_cost("grok-4-0709", 1000, 500)
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == pytest.approx(expected)

    def test_xai_model_pricing_fast(self):
        """Test grok-4-fast-reasoning cost calculation."""
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost

        # Test grok-4-fast-reasoning pricing: $0.20/1M input, $0.50/1M output
        cost = calculate_cost("grok-4-fast-reasoning", 10000, 5000)
        expected = (10000 / 1_000_000) * 0.20 + (5000 / 1_000_000) * 0.50
        assert cost == pytest.approx(expected)

    def test_xai_instrumentor_is_available(self):
        """Test XAIInstrumentor.is_available() behavior."""
        from evalyn_sdk.trace.instrumentation import XAIInstrumentor

        instrumentor = XAIInstrumentor()
        # is_available returns boolean based on xai_sdk presence
        result = instrumentor.is_available()
        assert isinstance(result, bool)

    def test_xai_instrumentor_name(self):
        """Test XAIInstrumentor.name property."""
        from evalyn_sdk.trace.instrumentation import XAIInstrumentor

        instrumentor = XAIInstrumentor()
        assert instrumentor.name == "xai"

    def test_xai_model_pricing_known(self):
        """Test that xAI models are recognized in pricing registry."""
        from evalyn_sdk.trace.instrumentation.providers._shared import is_model_pricing_known

        assert is_model_pricing_known("grok-4") is True
        assert is_model_pricing_known("grok-4-0709") is True
        assert is_model_pricing_known("grok-4-fast-reasoning") is True
        assert is_model_pricing_known("grok-4-1-fast-reasoning") is True
        assert is_model_pricing_known("grok-code-fast-1") is True
