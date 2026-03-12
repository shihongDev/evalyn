"""
Tests for LangChain and LangGraph instrumentors.

These tests mock the framework imports to test instrumentation logic
without requiring the actual frameworks to be installed.

Run with: pytest tests/test_langchain_instrumentors.py -v
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace import context as span_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_span_context():
    """Reset span context for test isolation."""
    span_context._span_stack.set([])
    span_context._span_collector.set([])
    span_context._active_call.set(None)


def _get_collected_spans() -> List[Span]:
    """Get spans from the collector."""
    return span_context._span_collector.get()


# ---------------------------------------------------------------------------
# Mock LangChain classes
# ---------------------------------------------------------------------------

class MockBaseCallbackHandler:
    """Mock langchain_core.callbacks.BaseCallbackHandler."""
    pass


class MockLLMResult:
    """Mock LangChain LLMResult with llm_output dict."""

    def __init__(self, llm_output: Optional[Dict[str, Any]] = None):
        self.llm_output = llm_output


# ---------------------------------------------------------------------------
# Mock LangGraph classes
# ---------------------------------------------------------------------------

class MockCompiledGraph:
    """Mock langgraph.graph.graph.CompiledGraph."""

    def __init__(self):
        self.name = "test_graph"
        self.nodes = {}

    def invoke(self, input_data, config=None, **kwargs):
        return {"result": "ok"}

    async def ainvoke(self, input_data, config=None, **kwargs):
        return {"result": "ok_async"}


class MockStateGraph:
    """Mock langgraph.graph.StateGraph."""

    def __init__(self, schema=None):
        self.schema = schema

    def compile(self, *args, **kwargs):
        return MockCompiledGraph()


# ===========================================================================
# LangChain Instrumentor Tests
# ===========================================================================

class TestLangChainInstrumentorProperties:
    """Test basic LangChainInstrumentor properties."""

    def setup_method(self):
        _reset_span_context()

    def test_name(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        assert inst.name == "langchain"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = LangChainInstrumentor()
        assert inst.instrumentor_type == InstrumentorType.MONKEY_PATCH

    def test_is_available_when_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert inst.is_available() is True

    def test_is_available_when_not_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_is_instrumented_default(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        assert inst.is_instrumented() is False

    def test_get_handler_before_instrument(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        assert inst.get_handler() is None


class TestLangChainInstrumentation:
    """Test LangChain instrumentation and callback handler."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        return inst

    def test_instrument_success(self):
        inst = self._create_instrumentor()
        result = inst.instrument()
        assert result is True
        assert inst.is_instrumented() is True
        assert inst.get_handler() is not None

    def test_instrument_idempotent(self):
        inst = self._create_instrumentor()
        inst.instrument()
        handler1 = inst.get_handler()
        inst.instrument()
        handler2 = inst.get_handler()
        assert handler1 is handler2

    def test_instrument_fails_without_langchain(self):
        inst = self._create_instrumentor()
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.callbacks": None}):
            result = inst.instrument()
        assert result is False
        assert inst.is_instrumented() is False

    def test_handler_has_callback_methods(self):
        inst = self._create_instrumentor()
        inst.instrument()
        handler = inst.get_handler()
        assert hasattr(handler, "on_llm_start")
        assert hasattr(handler, "on_llm_end")
        assert hasattr(handler, "on_llm_error")
        assert hasattr(handler, "on_tool_start")
        assert hasattr(handler, "on_tool_end")
        assert hasattr(handler, "on_tool_error")


class TestLangChainCallbackHandlerLLM:
    """Test LangChain callback handler LLM call tracking."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_start_records_start_time(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-001"
        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        assert run_id in handler._start_times

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_creates_span(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-002"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        response = MockLLMResult(llm_output={
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model_name": "gpt-4o",
        })
        handler.on_llm_end(response, run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.span_type == "llm_call"
        assert span.attributes["provider"] == "langchain"
        assert span.attributes["model"] == "gpt-4o"
        assert span.attributes["input_tokens"] == 10
        assert span.attributes["output_tokens"] == 20

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_clears_start_time(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-003"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id=run_id)

        assert run_id not in handler._start_times

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_with_no_llm_output(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-004"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        # Response with no llm_output attribute
        response = MockLLMResult(llm_output=None)
        handler.on_llm_end(response, run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].attributes["model"] == "unknown"
        assert spans[0].attributes["input_tokens"] == 0
        assert spans[0].attributes["output_tokens"] == 0

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_with_missing_token_usage(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-005"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        response = MockLLMResult(llm_output={"model_name": "gpt-3.5-turbo"})
        handler.on_llm_end(response, run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].attributes["model"] == "gpt-3.5-turbo"
        assert spans[0].attributes["input_tokens"] == 0
        assert spans[0].attributes["output_tokens"] == 0

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_without_prior_start(self, mock_tracer_fn):
        """LLM end without start should still create a span (uses fallback time)."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="run-orphan")

        spans = _get_collected_spans()
        assert len(spans) == 1

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_duration_is_positive(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-dur"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        time.sleep(0.01)  # Small delay to get measurable duration
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        # The span should have a positive duration via start_time manipulation
        assert spans[0].duration_ms is not None
        assert spans[0].duration_ms >= 0

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_logs_tracer_event(self, mock_tracer_fn):
        mock_tracer = MagicMock()
        mock_tracer_fn.return_value = mock_tracer
        handler = self._get_handler()
        run_id = "run-event"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        response = MockLLMResult(llm_output={"model_name": "gpt-4o"})
        handler.on_llm_end(response, run_id=run_id)

        mock_tracer.log_event.assert_called()
        call_args = mock_tracer.log_event.call_args
        assert call_args[0][0] == "langchain.completion"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_span_name_format(self, mock_tracer_fn):
        """Span name should be 'provider:model' format."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["prompt"], run_id="name-test")
        response = MockLLMResult(llm_output={"model_name": "claude-3-5-sonnet"})
        handler.on_llm_end(response, run_id="name-test")

        spans = _get_collected_spans()
        assert spans[0].name == "langchain:claude-3-5-sonnet"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_span_status_ok(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["prompt"], run_id="ok-test")
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="ok-test")

        spans = _get_collected_spans()
        assert spans[0].status == "ok"


class TestLangChainCallbackHandlerErrors:
    """Test LangChain callback handler error handling."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_creates_error_span(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-err"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_error(ValueError("rate limit exceeded"), run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status == "error"
        assert span.attributes["error"] == "rate limit exceeded"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_clears_start_time(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "run-err-clear"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        handler.on_llm_error(ValueError("fail"), run_id=run_id)

        assert run_id not in handler._start_times

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_without_start(self, mock_tracer_fn):
        """LLM error without start should still create a span."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_llm_error(RuntimeError("unexpected"), run_id="run-orphan-err")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_uses_unknown_model(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_llm_start({}, ["prompt"], run_id="run-err-model")
        handler.on_llm_error(Exception("fail"), run_id="run-err-model")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "unknown"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_span_name_format(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_llm_start({}, ["prompt"], run_id="err-name")
        handler.on_llm_error(Exception("fail"), run_id="err-name")

        spans = _get_collected_spans()
        assert spans[0].name == "langchain:unknown"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_propagates_model_from_start(self, mock_tracer_fn):
        """Model name from on_llm_start should propagate to on_llm_error."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}},
            ["prompt"],
            run_id="err-model-prop",
        )
        handler.on_llm_error(Exception("rate limit"), run_id="err-model-prop")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "gpt-4o"
        assert spans[0].name == "langchain:gpt-4o"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_error_propagates_model_from_invocation_params(self, mock_tracer_fn):
        """Model from invocation_params should propagate to on_llm_error."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_llm_start(
            {},
            ["prompt"],
            run_id="err-inv-params",
            invocation_params={"model_name": "claude-3-5-sonnet"},
        )
        handler.on_llm_error(Exception("timeout"), run_id="err-inv-params")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "claude-3-5-sonnet"


class TestLangChainCallbackHandlerTools:
    """Test LangChain callback handler tool call tracking."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_start_records_time(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "tool-run-001"
        handler.on_tool_start({}, "search query", run_id=run_id)
        assert f"tool_{run_id}" in handler._start_times

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_end_creates_span(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "tool-run-002"

        handler.on_tool_start({}, "search query", run_id=run_id)
        handler.on_tool_end("search results here", run_id=run_id, name="web_search")

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.span_type == "tool_call"
        assert span.attributes["tool_name"] == "web_search"
        assert span.status == "ok"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_end_clears_start_time(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "tool-run-003"

        handler.on_tool_start({}, "input", run_id=run_id)
        handler.on_tool_end("output", run_id=run_id, name="calc")

        assert f"tool_{run_id}" not in handler._start_times

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_end_with_unknown_name(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "tool-run-004"

        handler.on_tool_start({}, "input", run_id=run_id)
        # No 'name' kwarg
        handler.on_tool_end("output", run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].attributes["tool_name"] == "unknown"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_error_creates_error_span(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "tool-run-err"

        handler.on_tool_start({}, "input", run_id=run_id)
        handler.on_tool_error(
            TimeoutError("tool timeout"),
            run_id=run_id,
            name="slow_tool",
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status == "error"
        assert span.attributes["error"] == "tool timeout"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_error_without_start(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_tool_error(
            RuntimeError("no start"),
            run_id="tool-orphan",
            name="missing",
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_output_truncation(self, mock_tracer_fn):
        """Tool output should be truncated in the span detail."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "tool-trunc"

        handler.on_tool_start({}, "input", run_id=run_id)
        long_output = "x" * 5000
        handler.on_tool_end(long_output, run_id=run_id, name="verbose_tool")

        # The tracer event should have truncated output via log_tool_call
        mock_tracer = mock_tracer_fn.return_value
        mock_tracer.log_event.assert_called()
        call_args = mock_tracer.log_event.call_args
        detail = call_args[0][1]
        # log_tool_call truncates to 1000 chars
        assert len(detail.get("output", "")) <= 1000

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_multiple_concurrent_tools(self, mock_tracer_fn):
        """Multiple tools with different run_ids should be tracked independently."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_tool_start({}, "query1", run_id="t1")
        handler.on_tool_start({}, "query2", run_id="t2")
        handler.on_tool_end("result2", run_id="t2", name="tool_b")
        handler.on_tool_end("result1", run_id="t1", name="tool_a")

        spans = _get_collected_spans()
        assert len(spans) == 2
        tool_names = {s.attributes["tool_name"] for s in spans}
        assert tool_names == {"tool_a", "tool_b"}

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_span_name_is_tool_name(self, mock_tracer_fn):
        """Tool span name should be the tool name."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_tool_start({}, "input", run_id="tn")
        handler.on_tool_end("output", run_id="tn", name="calculator")

        spans = _get_collected_spans()
        assert spans[0].name == "calculator"


class TestLangChainCallbackHandlerSpanHierarchy:
    """Test that LangChain handler creates spans with correct parent-child relationships."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_span_has_parent_when_stack_set(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        # Push a parent span onto the stack
        parent_id = "parent-span-123"
        span_context._span_stack.set([parent_id])

        handler.on_llm_start({}, ["prompt"], run_id="child-llm")
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="child-llm")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].parent_id == parent_id

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_span_has_parent_when_stack_set(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        parent_id = "parent-span-456"
        span_context._span_stack.set([parent_id])

        handler.on_tool_start({}, "input", run_id="child-tool")
        handler.on_tool_end("output", run_id="child-tool", name="my_tool")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].parent_id == parent_id

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_spans_are_root_when_stack_empty(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["prompt"], run_id="root-llm")
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="root-llm")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].parent_id is None


class TestLangChainUninstrumentation:
    """Test LangChain uninstrumentation."""

    def setup_method(self):
        _reset_span_context()

    def test_uninstrument_clears_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        assert inst.get_handler() is not None

        inst.uninstrument()
        assert inst.get_handler() is None
        assert inst.is_instrumented() is False

    def test_uninstrument_idempotent(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        result = inst.uninstrument()
        assert result is True

    def test_reinstrument_after_uninstrument(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        inst.uninstrument()
        result = inst.instrument()
        assert result is True
        assert inst.get_handler() is not None


class TestLangChainLLMCostCalculation:
    """Test that LangChain LLM calls use cost calculation from _shared."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_known_model_gets_cost(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        run_id = "cost-test"

        handler.on_llm_start({}, ["prompt"], run_id=run_id)
        response = MockLLMResult(llm_output={
            "token_usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "model_name": "gpt-4o",
        })
        handler.on_llm_end(response, run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 1
        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(spans[0].attributes["cost_usd"] - expected_cost) < 0.0001

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_unknown_model_gets_default_cost(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["prompt"], run_id="cost-unknown")
        response = MockLLMResult(llm_output={
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model_name": "custom-model-v1",
        })
        handler.on_llm_end(response, run_id="cost-unknown")

        spans = _get_collected_spans()
        assert len(spans) == 1
        # Default: $1/1M for all tokens
        expected_cost = 150 / 1_000_000 * 1.0
        assert abs(spans[0].attributes["cost_usd"] - expected_cost) < 0.0001


class TestLangChainMultipleLLMCalls:
    """Test multiple sequential and interleaved LLM calls."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sequential_llm_calls(self, mock_tracer_fn):
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        for i in range(3):
            run_id = f"seq-{i}"
            handler.on_llm_start({}, [f"prompt-{i}"], run_id=run_id)
            response = MockLLMResult(llm_output={"model_name": f"model-{i}"})
            handler.on_llm_end(response, run_id=run_id)

        spans = _get_collected_spans()
        assert len(spans) == 3

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_interleaved_llm_calls(self, mock_tracer_fn):
        """Multiple LLM calls started before any finish."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["p1"], run_id="r1")
        handler.on_llm_start({}, ["p2"], run_id="r2")
        handler.on_llm_end(MockLLMResult(llm_output={"model_name": "m2"}), run_id="r2")
        handler.on_llm_end(MockLLMResult(llm_output={"model_name": "m1"}), run_id="r1")

        spans = _get_collected_spans()
        assert len(spans) == 2
        # Both run_ids should have been tracked
        assert "r1" not in handler._start_times
        assert "r2" not in handler._start_times

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_mixed_llm_and_tool_calls(self, mock_tracer_fn):
        """LLM calls and tool calls should coexist."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["prompt"], run_id="llm-1")
        handler.on_tool_start({}, "search", run_id="tool-1")
        handler.on_tool_end("results", run_id="tool-1", name="search_tool")
        handler.on_llm_end(MockLLMResult(llm_output={}), run_id="llm-1")

        spans = _get_collected_spans()
        assert len(spans) == 2
        span_types = {s.span_type for s in spans}
        assert span_types == {"llm_call", "tool_call"}


# ===========================================================================
# LangGraph Instrumentor Tests
# ===========================================================================

class TestLangGraphInstrumentorProperties:
    """Test basic LangGraphInstrumentor properties."""

    def setup_method(self):
        _reset_span_context()

    def test_name(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        assert inst.name == "langgraph"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = LangGraphInstrumentor()
        assert inst.instrumentor_type == InstrumentorType.MONKEY_PATCH

    def test_is_available_when_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert inst.is_available() is True

    def test_is_available_when_not_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_is_instrumented_default(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        assert inst.is_instrumented() is False


class TestLangGraphInstrumentation:
    """Test LangGraph instrumentation patching."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    def test_instrument_success(self):
        inst = self._create_instrumentor()
        result = inst.instrument()
        assert result is True
        assert inst.is_instrumented() is True
        # Cleanup
        inst.uninstrument()

    def test_instrument_idempotent(self):
        inst = self._create_instrumentor()
        inst.instrument()
        result = inst.instrument()
        assert result is True
        inst.uninstrument()

    def test_instrument_stores_original_compile(self):
        """After instrumentation, _original_compile should be saved."""
        inst = self._create_instrumentor()
        inst.instrument()
        assert inst._original_compile is not None
        inst.uninstrument()

    def test_instrument_fails_without_langgraph(self):
        inst = self._create_instrumentor()
        with patch.dict("sys.modules", {"langgraph": None, "langgraph.graph": None, "langgraph.graph.state": None}):
            result = inst.instrument()
        assert result is False
        assert inst.is_instrumented() is False


class TestLangGraphGraphExecution:
    """Test LangGraph graph execution span creation via _wrap_compiled_graph."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    def test_invoke_creates_graph_span(self):
        inst = self._create_instrumentor()

        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        result = wrapped.invoke({"input": "test"})
        assert result == {"result": "ok"}

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "graph:test_graph"
        assert span.span_type == "graph"
        assert span.status == "ok"

    def test_invoke_graph_span_has_no_parent_by_default(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)
        wrapped.invoke({"input": "test"})

        spans = _get_collected_spans()
        assert spans[0].parent_id is None

    def test_invoke_graph_span_has_parent_when_stack_set(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        parent_id = "outer-span-001"
        span_context._span_stack.set([parent_id])

        wrapped.invoke({"input": "test"})

        spans = _get_collected_spans()
        assert spans[0].parent_id == parent_id

    def test_invoke_restores_span_stack(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        original_stack = ["pre-existing-span"]
        span_context._span_stack.set(original_stack)

        wrapped.invoke({"input": "test"})

        # Stack should be restored
        assert span_context._span_stack.get() == original_stack

    def test_invoke_error_creates_error_span(self):
        inst = self._create_instrumentor()

        graph = MockCompiledGraph()
        # Make invoke raise an error
        def failing_invoke(input_data, config=None, **kwargs):
            raise RuntimeError("graph execution failed")
        graph.invoke = failing_invoke
        wrapped = inst._wrap_compiled_graph(graph)

        with pytest.raises(RuntimeError, match="graph execution failed"):
            wrapped.invoke({"input": "test"})

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status == "error"
        assert span.attributes.get("error") == "graph execution failed"

    def test_invoke_error_restores_stack(self):
        inst = self._create_instrumentor()

        graph = MockCompiledGraph()
        def failing_invoke(input_data, config=None, **kwargs):
            raise RuntimeError("fail")
        graph.invoke = failing_invoke
        wrapped = inst._wrap_compiled_graph(graph)

        original_stack = ["existing"]
        span_context._span_stack.set(original_stack)

        with pytest.raises(RuntimeError):
            wrapped.invoke({})

        assert span_context._span_stack.get() == original_stack

    def test_invoke_pushes_graph_span_onto_stack(self):
        """During graph execution, the graph span should be on the stack so children can attach."""
        inst = self._create_instrumentor()

        captured_stack = []
        graph = MockCompiledGraph()
        original_invoke = graph.invoke
        def capturing_invoke(input_data, config=None, **kwargs):
            captured_stack.extend(span_context._span_stack.get())
            return original_invoke(input_data, config=config, **kwargs)
        graph.invoke = capturing_invoke
        wrapped = inst._wrap_compiled_graph(graph)

        wrapped.invoke({"input": "test"})

        # During execution, the graph span ID should have been on the stack
        assert len(captured_stack) == 1
        # And it should match the graph span
        spans = _get_collected_spans()
        assert captured_stack[0] == spans[0].id

    def test_invoke_with_custom_graph_name(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        graph.name = "my_pipeline"
        wrapped = inst._wrap_compiled_graph(graph)

        wrapped.invoke({})

        spans = _get_collected_spans()
        assert spans[0].name == "graph:my_pipeline"

    def test_invoke_without_name_uses_default(self):
        """Graph without name attribute should use 'graph' as default."""
        inst = self._create_instrumentor()

        class NamelessGraph:
            def invoke(self, input_data, config=None, **kwargs):
                return {"ok": True}

        graph = NamelessGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        wrapped.invoke({})

        spans = _get_collected_spans()
        assert spans[0].name == "graph:graph"


class TestLangGraphAsyncExecution:
    """Test LangGraph async graph execution."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    def test_ainvoke_creates_graph_span(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(wrapped.ainvoke({"input": "test_async"}))
        finally:
            loop.close()
        assert result == {"result": "ok_async"}

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "graph:test_graph"
        assert span.span_type == "graph"
        assert span.status == "ok"

    def test_ainvoke_error_creates_error_span(self):
        inst = self._create_instrumentor()

        graph = MockCompiledGraph()
        async def failing_ainvoke(input_data, config=None, **kwargs):
            raise RuntimeError("async graph failed")
        graph.ainvoke = failing_ainvoke
        wrapped = inst._wrap_compiled_graph(graph)

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RuntimeError, match="async graph failed"):
                loop.run_until_complete(wrapped.ainvoke({"input": "test"}))
        finally:
            loop.close()

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert spans[0].attributes.get("error") == "async graph failed"

    def test_ainvoke_restores_stack(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        original_stack = ["async-parent"]
        span_context._span_stack.set(original_stack)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(wrapped.ainvoke({"input": "test"}))
        finally:
            loop.close()

        assert span_context._span_stack.get() == original_stack

    def test_ainvoke_has_parent_when_stack_set(self):
        inst = self._create_instrumentor()
        graph = MockCompiledGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        parent_id = "async-parent-id"
        span_context._span_stack.set([parent_id])

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(wrapped.ainvoke({"input": "test"}))
        finally:
            loop.close()

        spans = _get_collected_spans()
        assert spans[0].parent_id == parent_id


class TestLangGraphNodeExecution:
    """Test LangGraph node execution span creation."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    def test_wrap_sync_node_function(self):
        inst = self._create_instrumentor()

        def my_node(state):
            return {"processed": True}

        wrapped = inst._wrap_node_function("my_node", my_node)
        result = wrapped({"input": "data"})
        assert result == {"processed": True}

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "node:my_node"
        assert span.span_type == "node"
        assert span.attributes["node_name"] == "my_node"
        assert span.status == "ok"

    def test_wrap_async_node_function(self):
        inst = self._create_instrumentor()

        async def async_node(state):
            return {"async_processed": True}

        wrapped = inst._wrap_node_function("async_node", async_node)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(wrapped({"input": "data"}))
        finally:
            loop.close()
        assert result == {"async_processed": True}

        spans = _get_collected_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "node:async_node"
        assert span.span_type == "node"
        assert span.status == "ok"

    def test_node_error_creates_error_span(self):
        inst = self._create_instrumentor()

        def failing_node(state):
            raise ValueError("node processing failed")

        wrapped = inst._wrap_node_function("failing_node", failing_node)
        with pytest.raises(ValueError, match="node processing failed"):
            wrapped({})

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert spans[0].attributes.get("error") == "node processing failed"

    def test_async_node_error_creates_error_span(self):
        inst = self._create_instrumentor()

        async def failing_async_node(state):
            raise RuntimeError("async node failed")

        wrapped = inst._wrap_node_function("failing_async_node", failing_async_node)
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RuntimeError, match="async node failed"):
                loop.run_until_complete(wrapped({}))
        finally:
            loop.close()

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"

    def test_node_span_has_parent_from_stack(self):
        inst = self._create_instrumentor()

        parent_id = "graph-span-id"
        span_context._span_stack.set([parent_id])

        def my_node(state):
            return state

        wrapped = inst._wrap_node_function("child_node", my_node)
        wrapped({})

        spans = _get_collected_spans()
        assert spans[0].parent_id == parent_id

    def test_node_pushes_span_for_children(self):
        """During node execution, the node span should be on the stack so LLM calls become children."""
        inst = self._create_instrumentor()

        captured_stack = []
        def capturing_node(state):
            captured_stack.extend(span_context._span_stack.get())
            return state

        wrapped = inst._wrap_node_function("capturing_node", capturing_node)
        wrapped({})

        spans = _get_collected_spans()
        assert len(captured_stack) == 1
        assert captured_stack[0] == spans[0].id

    def test_node_restores_stack_on_success(self):
        inst = self._create_instrumentor()

        original_stack = ["pre-node"]
        span_context._span_stack.set(original_stack)

        wrapped = inst._wrap_node_function("test_node", lambda s: s)
        wrapped({})

        assert span_context._span_stack.get() == original_stack

    def test_node_restores_stack_on_error(self):
        inst = self._create_instrumentor()

        original_stack = ["pre-node-err"]
        span_context._span_stack.set(original_stack)

        def failing(state):
            raise RuntimeError("fail")

        wrapped = inst._wrap_node_function("fail_node", failing)
        with pytest.raises(RuntimeError):
            wrapped({})

        assert span_context._span_stack.get() == original_stack


class TestLangGraphNodePatching:
    """Test LangGraph node runner patching via _patch_node_runners."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    def test_patch_compiled_graph_class_with_execute_node(self):
        """Test that _execute_node gets patched if it exists."""
        inst = self._create_instrumentor()

        class GraphWithExecuteNode:
            def _execute_node(self, node_name, *args, **kwargs):
                return {"executed": node_name}
            nodes = {}

        original_fn = GraphWithExecuteNode._execute_node
        inst._patch_compiled_graph_class(GraphWithExecuteNode)

        # The _execute_node method should now be different (patched)
        assert GraphWithExecuteNode._execute_node is not original_fn

    def test_patched_execute_node_creates_span(self):
        inst = self._create_instrumentor()

        class GraphWithExecuteNode:
            def _execute_node(self, node_name, *args, **kwargs):
                return {"executed": node_name}
            nodes = {}

        inst._patch_compiled_graph_class(GraphWithExecuteNode)

        obj = GraphWithExecuteNode()
        result = obj._execute_node("process_data")
        assert result == {"executed": "process_data"}

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "node:process_data"
        assert spans[0].attributes["node_name"] == "process_data"

    def test_patched_execute_node_error_handling(self):
        inst = self._create_instrumentor()

        class GraphWithExecuteNode:
            def _execute_node(self, node_name, *args, **kwargs):
                raise ValueError("node error")
            nodes = {}

        inst._patch_compiled_graph_class(GraphWithExecuteNode)

        obj = GraphWithExecuteNode()
        with pytest.raises(ValueError, match="node error"):
            obj._execute_node("broken_node")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"

    def test_patch_node_runners_wraps_init(self):
        """_patch_node_runners should wrap __init__ to wrap node functions."""
        inst = self._create_instrumentor()

        class TestCompiledGraph:
            def __init__(self):
                self.nodes = {
                    "step_a": lambda state: {"a": True},
                    "step_b": lambda state: {"b": True},
                }

        original_init = TestCompiledGraph.__init__
        inst._patch_node_runners(TestCompiledGraph)

        # __init__ should now be wrapped
        assert TestCompiledGraph.__init__ is not original_init

        # Creating an instance should wrap the node functions
        obj = TestCompiledGraph()
        assert "step_a" in obj.nodes
        assert "step_b" in obj.nodes

        # Calling a node should produce spans
        result_a = obj.nodes["step_a"]({})
        assert result_a == {"a": True}

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "node:step_a"

    def test_patch_node_runners_wraps_all_nodes(self):
        """All nodes in the dict should be wrapped."""
        inst = self._create_instrumentor()

        class TestCompiledGraph:
            def __init__(self):
                self.nodes = {
                    "n1": lambda s: {"n": 1},
                    "n2": lambda s: {"n": 2},
                    "n3": lambda s: {"n": 3},
                }

        inst._patch_node_runners(TestCompiledGraph)
        obj = TestCompiledGraph()

        for key in ["n1", "n2", "n3"]:
            obj.nodes[key]({})

        spans = _get_collected_spans()
        assert len(spans) == 3
        names = {s.name for s in spans}
        assert names == {"node:n1", "node:n2", "node:n3"}


class TestLangGraphCompilePatching:
    """Test LangGraph StateGraph.compile patching via instrument()."""

    def setup_method(self):
        _reset_span_context()

    def test_compile_returns_instrumented_graph(self):
        """After instrument(), StateGraph.compile should be patched."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        from langgraph.graph import StateGraph

        original_compile = StateGraph.compile

        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        inst.instrument()

        try:
            # StateGraph.compile should now be patched
            assert StateGraph.compile is not original_compile
            # The original should be saved
            assert inst._original_compile is original_compile
        finally:
            inst.uninstrument()
            # After uninstrument, compile should be restored
            assert StateGraph.compile is original_compile


class TestLangGraphFullHierarchy:
    """Test full graph-node hierarchy span creation."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_graph_invoke_with_nested_node_spans(self, mock_tracer_fn):
        """Graph invoke should create a parent span, and node calls inside become children."""
        mock_tracer_fn.return_value = MagicMock()
        inst = self._create_instrumentor()

        # Simulate a compiled graph whose invoke runs nodes internally
        graph = MockCompiledGraph()
        graph.name = "test_pipeline"

        # Wrap a node function
        node_fn = inst._wrap_node_function("process_step", lambda state: state)

        original_invoke = graph.invoke
        def invoke_with_nodes(input_data, config=None, **kwargs):
            node_fn(input_data)  # Simulate node execution
            return original_invoke(input_data, config=config, **kwargs)
        graph.invoke = invoke_with_nodes

        wrapped = inst._wrap_compiled_graph(graph)
        wrapped.invoke({"data": "test"})

        spans = _get_collected_spans()
        # Should have node span + graph span
        assert len(spans) == 2

        # Find graph and node spans
        graph_span = next(s for s in spans if s.span_type == "graph")
        node_span = next(s for s in spans if s.span_type == "node")

        assert graph_span.name == "graph:test_pipeline"
        assert node_span.name == "node:process_step"
        # The node should be a child of the graph
        assert node_span.parent_id == graph_span.id

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_graph_with_multiple_nodes(self, mock_tracer_fn):
        """Graph with multiple nodes should create multiple node spans, all children of the graph."""
        mock_tracer_fn.return_value = MagicMock()
        inst = self._create_instrumentor()

        graph = MockCompiledGraph()
        graph.name = "multi_node_graph"

        node_a = inst._wrap_node_function("step_a", lambda s: s)
        node_b = inst._wrap_node_function("step_b", lambda s: s)

        original_invoke = graph.invoke
        def invoke_with_multi_nodes(input_data, config=None, **kwargs):
            node_a(input_data)
            node_b(input_data)
            return original_invoke(input_data, config=config, **kwargs)
        graph.invoke = invoke_with_multi_nodes

        wrapped = inst._wrap_compiled_graph(graph)
        wrapped.invoke({"data": "test"})

        spans = _get_collected_spans()
        assert len(spans) == 3  # graph + 2 nodes

        graph_span = next(s for s in spans if s.span_type == "graph")
        node_spans = [s for s in spans if s.span_type == "node"]

        assert len(node_spans) == 2
        for ns in node_spans:
            assert ns.parent_id == graph_span.id

        node_names = {ns.name for ns in node_spans}
        assert node_names == {"node:step_a", "node:step_b"}

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_node_with_llm_call_inside(self, mock_tracer_fn):
        """LLM calls inside a node should be children of the node span."""
        mock_tracer_fn.return_value = MagicMock()
        inst = self._create_instrumentor()

        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        def node_with_llm(state):
            # Simulate an LLM call inside the node
            log_llm_call(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                duration_ms=200,
                success=True,
            )
            return state

        wrapped_node = inst._wrap_node_function("llm_node", node_with_llm)
        wrapped_node({"input": "test"})

        spans = _get_collected_spans()
        assert len(spans) == 2

        node_span = next(s for s in spans if s.span_type == "node")
        llm_span = next(s for s in spans if s.span_type == "llm_call")

        assert node_span.name == "node:llm_node"
        assert llm_span.parent_id == node_span.id
        assert llm_span.attributes["model"] == "gpt-4o"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_full_graph_node_llm_hierarchy(self, mock_tracer_fn):
        """Full three-level hierarchy: graph -> node -> llm_call."""
        mock_tracer_fn.return_value = MagicMock()
        inst = self._create_instrumentor()

        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        def node_with_llm(state):
            log_llm_call(
                provider="anthropic",
                model="claude-3-5-sonnet",
                input_tokens=200,
                output_tokens=100,
                duration_ms=500,
                success=True,
            )
            return state

        wrapped_node = inst._wrap_node_function("agent_node", node_with_llm)

        graph = MockCompiledGraph()
        graph.name = "agent_graph"
        original_invoke = graph.invoke
        def invoke_with_node(input_data, config=None, **kwargs):
            wrapped_node(input_data)
            return original_invoke(input_data, config=config, **kwargs)
        graph.invoke = invoke_with_node

        wrapped = inst._wrap_compiled_graph(graph)
        wrapped.invoke({"query": "test"})

        spans = _get_collected_spans()
        assert len(spans) == 3

        graph_span = next(s for s in spans if s.span_type == "graph")
        node_span = next(s for s in spans if s.span_type == "node")
        llm_span = next(s for s in spans if s.span_type == "llm_call")

        # Verify hierarchy: graph -> node -> llm
        assert graph_span.parent_id is None
        assert node_span.parent_id == graph_span.id
        assert llm_span.parent_id == node_span.id


class TestLangGraphUninstrumentation:
    """Test LangGraph uninstrumentation."""

    def setup_method(self):
        _reset_span_context()

    def test_uninstrument_restores_compile(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        from langgraph.graph import StateGraph

        original_compile = StateGraph.compile

        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        inst.instrument()

        # compile should now be patched
        assert StateGraph.compile is not original_compile

        inst.uninstrument()
        # compile should be restored
        assert StateGraph.compile is original_compile
        assert inst.is_instrumented() is False

    def test_uninstrument_idempotent(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        result = inst.uninstrument()
        assert result is True

    def test_uninstrument_clears_originals(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        inst.instrument()
        inst.uninstrument()

        assert inst._original_compile is None
        assert inst._original_init is None

    def test_uninstrument_fails_without_langgraph(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        inst.instrument()

        with patch.dict("sys.modules", {"langgraph": None, "langgraph.graph": None, "langgraph.graph.state": None}):
            result = inst.uninstrument()
        assert result is False

    def test_reinstrument_after_uninstrument(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None

        inst.instrument()
        inst.uninstrument()

        result = inst.instrument()
        assert result is True
        assert inst.is_instrumented() is True
        inst.uninstrument()


class TestLangGraphGraphWithoutAinvoke:
    """Test LangGraph wrapping when graph has no ainvoke."""

    def setup_method(self):
        _reset_span_context()

    def test_wrap_graph_without_ainvoke(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None

        class SyncOnlyGraph:
            name = "sync_graph"
            def invoke(self, input_data, config=None, **kwargs):
                return {"sync": True}

        graph = SyncOnlyGraph()
        wrapped = inst._wrap_compiled_graph(graph)

        result = wrapped.invoke({"input": "test"})
        assert result == {"sync": True}

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "graph:sync_graph"


class TestLangChainEnhancedCapture:
    """Test enhanced LangChain capture: model from start, request/response data."""

    def setup_method(self):
        _reset_span_context()

    def _get_handler(self):
        from evalyn_sdk.trace.instrumentation.providers.langchain import LangChainInstrumentor
        inst = LangChainInstrumentor()
        inst._instrumented = False
        inst._handler = None
        inst.instrument()
        return inst.get_handler()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_model_from_invocation_params(self, mock_tracer_fn):
        """Model name should be captured from invocation_params at start time."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start(
            {},
            ["prompt"],
            run_id="model-inv",
            invocation_params={"model_name": "gpt-4-turbo"},
        )
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="model-inv")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "gpt-4-turbo"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_model_from_invocation_params_model_key(self, mock_tracer_fn):
        """Model should also be found via 'model' key in invocation_params."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start(
            {},
            ["prompt"],
            run_id="model-key",
            invocation_params={"model": "claude-3-5-sonnet"},
        )
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="model-key")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "claude-3-5-sonnet"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_model_from_serialized(self, mock_tracer_fn):
        """Model name should fall back to serialized kwargs."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o-mini"}},
            ["prompt"],
            run_id="model-ser",
        )
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="model-ser")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "gpt-4o-mini"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_model_from_response_overrides_start(self, mock_tracer_fn):
        """Model from llm_output should override the start-time captured model."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start(
            {},
            ["prompt"],
            run_id="model-override",
            invocation_params={"model_name": "gpt-4"},
        )
        response = MockLLMResult(llm_output={"model_name": "gpt-4-0125-preview"})
        handler.on_llm_end(response, run_id="model-override")

        spans = _get_collected_spans()
        # The response model_name should take precedence
        assert spans[0].attributes["model"] == "gpt-4-0125-preview"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_llm_end_falls_back_to_start_model(self, mock_tracer_fn):
        """If response has no model_name, fall back to the start-time captured model."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start(
            {},
            ["prompt"],
            run_id="model-fallback",
            invocation_params={"model_name": "gpt-4-turbo"},
        )
        # Response with no model_name in llm_output
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="model-fallback")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "gpt-4-turbo"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_request_messages_captured(self, mock_tracer_fn):
        """Request prompts should be captured and passed to log_llm_call."""
        mock_tracer = MagicMock()
        mock_tracer_fn.return_value = mock_tracer
        handler = self._get_handler()

        handler.on_llm_start({}, ["Hello, how are you?"], run_id="req-capture")
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="req-capture")

        # Check that log_event was called and request was passed
        spans = _get_collected_spans()
        assert len(spans) == 1
        # The request should be in the span attributes
        assert "request" in spans[0].attributes
        assert "Hello, how are you?" in str(spans[0].attributes["request"])

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_request_messages_truncated(self, mock_tracer_fn):
        """Long request prompts should be truncated to 500 chars each."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        long_prompt = "x" * 1000
        handler.on_llm_start({}, [long_prompt], run_id="req-trunc")
        response = MockLLMResult(llm_output={})
        handler.on_llm_end(response, run_id="req-trunc")

        spans = _get_collected_spans()
        # The request messages list should have a truncated prompt
        request = spans[0].attributes.get("request", {})
        if isinstance(request, dict) and "messages" in request:
            for msg in request["messages"]:
                assert len(msg) <= 500

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_response_content_captured(self, mock_tracer_fn):
        """Response content from generations should be captured."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start({}, ["prompt"], run_id="resp-capture")

        # Create a mock response with generations
        class MockGeneration:
            def __init__(self, text):
                self.text = text

        response = MockLLMResult(llm_output={})
        response.generations = [[MockGeneration("The answer is 42.")]]
        handler.on_llm_end(response, run_id="resp-capture")

        spans = _get_collected_spans()
        assert "response" in spans[0].attributes
        assert "42" in str(spans[0].attributes["response"])

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_error_handler_uses_captured_model(self, mock_tracer_fn):
        """Error handler should use the model captured at start time, not hardcoded 'unknown'."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()

        handler.on_llm_start(
            {},
            ["prompt"],
            run_id="err-model",
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_error(ValueError("quota exceeded"), run_id="err-model")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "gpt-4o"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_error_without_start_uses_unknown_model(self, mock_tracer_fn):
        """Error without a preceding start should still use 'unknown'."""
        mock_tracer_fn.return_value = MagicMock()
        handler = self._get_handler()
        handler.on_llm_error(RuntimeError("fail"), run_id="err-no-start")

        spans = _get_collected_spans()
        assert spans[0].attributes["model"] == "unknown"


class TestLangGraphDoubleSpanGuard:
    """Test that the double-span guard prevents duplicate node spans.

    When both _execute_node patching and node function wrapping fire for the
    same node execution, only one node span should be created.
    """

    def setup_method(self):
        _reset_span_context()
        # Reset the _active_node_spans ContextVar
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        _active_node_spans.set(frozenset())

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.langgraph import LangGraphInstrumentor
        inst = LangGraphInstrumentor()
        inst._instrumented = False
        inst._original_compile = None
        inst._original_init = None
        return inst

    def test_execute_node_skips_when_already_active(self):
        """_execute_node should skip span creation if node is already active."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        inst = self._create_instrumentor()

        class GraphWithExecuteNode:
            def _execute_node(self, node_name, *args, **kwargs):
                return {"executed": node_name}
            nodes = {}

        inst._patch_compiled_graph_class(GraphWithExecuteNode)

        # Simulate that "test_node" is already active
        _active_node_spans.set(frozenset({"test_node"}))

        obj = GraphWithExecuteNode()
        result = obj._execute_node("test_node")
        assert result == {"executed": "test_node"}

        # No span should be created since node was already active
        spans = _get_collected_spans()
        assert len(spans) == 0

    def test_execute_node_creates_span_when_not_active(self):
        """_execute_node should create a span when node is not already active."""
        inst = self._create_instrumentor()

        class GraphWithExecuteNode:
            def _execute_node(self, node_name, *args, **kwargs):
                return {"executed": node_name}
            nodes = {}

        inst._patch_compiled_graph_class(GraphWithExecuteNode)

        obj = GraphWithExecuteNode()
        result = obj._execute_node("new_node")
        assert result == {"executed": "new_node"}

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "node:new_node"

    def test_wrapped_node_skips_when_already_active(self):
        """Wrapped node function should skip span creation if node is already active."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        inst = self._create_instrumentor()

        def my_node(state):
            return {"done": True}

        wrapped = inst._wrap_node_function("my_node", my_node)

        # Simulate that "my_node" is already active (from _execute_node)
        _active_node_spans.set(frozenset({"my_node"}))

        result = wrapped({})
        assert result == {"done": True}

        # No span should be created since node was already active
        spans = _get_collected_spans()
        assert len(spans) == 0

    def test_wrapped_async_node_skips_when_already_active(self):
        """Wrapped async node function should skip span creation if node is already active."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        inst = self._create_instrumentor()

        async def my_async_node(state):
            return {"async_done": True}

        wrapped = inst._wrap_node_function("my_async_node", my_async_node)

        # Simulate that "my_async_node" is already active
        _active_node_spans.set(frozenset({"my_async_node"}))

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(wrapped({}))
        finally:
            loop.close()
        assert result == {"async_done": True}

        spans = _get_collected_spans()
        assert len(spans) == 0

    def test_guard_resets_after_execution(self):
        """The active node set should be restored after node execution completes."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        inst = self._create_instrumentor()

        def my_node(state):
            # During execution, check that our node is in the active set
            active = _active_node_spans.get()
            assert "my_node" in active
            return state

        wrapped = inst._wrap_node_function("my_node", my_node)
        wrapped({})

        # After execution, the active set should be restored to original
        active_after = _active_node_spans.get()
        assert "my_node" not in active_after

    def test_different_node_not_blocked_by_guard(self):
        """Guard for one node should not block a different node."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        inst = self._create_instrumentor()

        # "node_a" is already active
        _active_node_spans.set(frozenset({"node_a"}))

        def node_b(state):
            return {"b": True}

        wrapped = inst._wrap_node_function("node_b", node_b)
        wrapped({})

        # node_b should still create a span
        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "node:node_b"

    def test_execute_node_then_wrapped_node_no_duplicate(self):
        """Simulates both _execute_node and wrapped node firing - only one span should be created."""
        from evalyn_sdk.trace.instrumentation.providers.langgraph import _active_node_spans
        inst = self._create_instrumentor()

        inner_results = []

        class GraphWithExecuteNode:
            def _execute_node(self, node_name, *args, **kwargs):
                # After _execute_node creates the span and marks node active,
                # it calls the node function which is also wrapped
                node_fn = args[0] if args else lambda: None
                return node_fn()
            nodes = {}

        inst._patch_compiled_graph_class(GraphWithExecuteNode)

        # Also wrap the node function
        def real_node():
            inner_results.append("executed")
            return {"result": True}

        wrapped_fn = inst._wrap_node_function("dual_node", real_node)

        obj = GraphWithExecuteNode()
        result = obj._execute_node("dual_node", wrapped_fn)
        assert result == {"result": True}
        assert inner_results == ["executed"]

        # Should have exactly 1 span (from _execute_node), not 2
        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "node:dual_node"
