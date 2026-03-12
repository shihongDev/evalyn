"""
Tests for hook-based instrumentors (Google ADK, Claude Agent SDK).

These tests mock the framework imports to test instrumentation logic
without requiring the actual frameworks to be installed.

Run with: pytest tests/test_hook_based_instrumentors.py -v
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional
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


def _find_spans_by_type(span_type: str) -> List[Span]:
    """Find all collected spans of a given type."""
    return [s for s in _get_collected_spans() if s.span_type == span_type]


def _find_span_by_name(name: str) -> Optional[Span]:
    """Find the first collected span with a given name."""
    for s in _get_collected_spans():
        if s.name == name:
            return s
    return None


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Mock objects for Google ADK
# ---------------------------------------------------------------------------

class MockCallbackContext:
    """Mock google.adk CallbackContext."""

    def __init__(
        self,
        agent_name: str = "test_agent",
        invocation_id: str = "inv-001",
        session: Any = None,
    ):
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.session = session


class MockSession:
    """Mock ADK session."""

    def __init__(self, session_id: str = "session-001"):
        self.id = session_id


class MockToolContext(MockCallbackContext):
    """Mock google.adk ToolContext (extends CallbackContext)."""
    pass


class MockBaseTool:
    """Mock google.adk BaseTool."""

    def __init__(self, name: str = "search_tool"):
        self.name = name


class MockAgentTool:
    """Mock google.adk AgentTool."""

    def __init__(self, name: str = "agent_tool", agent: Any = None):
        self.name = name
        self.agent = agent

    @property
    def __class__(self):
        """Make type(tool).__name__ return 'AgentTool'."""
        return type("AgentTool", (), {"__name__": "AgentTool"})


class MockLlmRequest:
    """Mock google.adk LlmRequest."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        contents: Optional[List[Any]] = None,
        config: Any = None,
    ):
        self.model = model
        self.contents = contents or []
        self.config = config


class MockLlmConfig:
    """Mock LLM config object."""

    def __init__(self, temperature: float = 0.7, max_output_tokens: int = 2048):
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


class MockLlmResponse:
    """Mock google.adk LlmResponse."""

    def __init__(
        self,
        content: Any = None,
        usage_metadata: Any = None,
        model_version: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self.content = content
        self.usage_metadata = usage_metadata
        self.model_version = model_version
        self.error_code = error_code
        self.error_message = error_message


class MockContent:
    """Mock google.genai.types.Content."""

    def __init__(self, parts: Optional[List[Any]] = None):
        self.parts = parts or []


class MockTextPart:
    """Mock text part."""

    def __init__(self, text: str = "Hello world"):
        self.text = text
        self.function_call = None


class MockFunctionCallPart:
    """Mock function call part."""

    def __init__(self, name: str = "search", args: Optional[Dict] = None):
        self.text = None
        self.function_call = MockFunctionCall(name=name, args=args or {})


class MockFunctionCall:
    """Mock function call."""

    def __init__(self, name: str = "search", args: Optional[Dict] = None):
        self.name = name
        self.args = args or {}


class MockUsageMetadata:
    """Mock usage metadata."""

    def __init__(
        self,
        prompt_token_count: int = 100,
        candidates_token_count: int = 50,
        total_token_count: int = 150,
        cached_content_token_count: int = 0,
        thoughts_token_count: int = 0,
    ):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.cached_content_token_count = cached_content_token_count
        self.thoughts_token_count = thoughts_token_count


class MockADKEvent:
    """Mock ADK event for stream adapter."""

    def __init__(
        self,
        content: Any = None,
        usage_metadata: Any = None,
        is_final: bool = False,
    ):
        self.content = content
        self.usage_metadata = usage_metadata
        self._is_final = is_final

    def is_final_response(self) -> bool:
        return self._is_final


# ---------------------------------------------------------------------------
# Mock objects for Claude Agent SDK
# ---------------------------------------------------------------------------

class MockClaudeContext:
    """Mock claude_agent_sdk context object."""

    def __init__(
        self,
        turn: Optional[int] = None,
        model: Optional[str] = None,
    ):
        self.turn = turn
        self.model = model


def _make_named_class(name, init_fn):
    """Create a class with the exact __name__ needed for type(obj).__name__ checks."""
    cls = type(name, (), {"__init__": init_fn})
    return cls


def _assistant_message_init(
    self,
    model="claude-sonnet-4",
    content=None,
    parent_tool_use_id=None,
):
    self.model = model
    self.content = content or []
    self.parent_tool_use_id = parent_tool_use_id


# The MessageStreamAdapter checks type(msg).__name__, so class names must match exactly.
AssistantMessage = _make_named_class("AssistantMessage", _assistant_message_init)


def _text_block_init(self, text=""):
    self.text = text


TextBlock = _make_named_class("TextBlock", _text_block_init)


def _tool_use_block_init(self, name="search", id="tool-001", input=None):
    self.name = name
    self.id = id
    self.input = input or {}


ToolUseBlock = _make_named_class("ToolUseBlock", _tool_use_block_init)


def _thinking_block_init(self, thinking="", signature=None):
    self.thinking = thinking
    self.signature = signature


ThinkingBlock = _make_named_class("ThinkingBlock", _thinking_block_init)


def _result_message_init(
    self,
    duration_ms=None,
    duration_api_ms=None,
    total_cost_usd=None,
    usage=None,
    num_turns=None,
    is_error=False,
    result=None,
    structured_output=None,
    session_id=None,
):
    self.duration_ms = duration_ms
    self.duration_api_ms = duration_api_ms
    self.total_cost_usd = total_cost_usd
    self.usage = usage
    self.num_turns = num_turns
    self.is_error = is_error
    self.result = result
    self.structured_output = structured_output
    self.session_id = session_id


ResultMessage = _make_named_class("ResultMessage", _result_message_init)


class MockClaudeUsage:
    """Mock claude_agent_sdk usage object."""

    def __init__(
        self,
        input_tokens: int = 200,
        output_tokens: int = 100,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class MockUserHooks:
    """Mock user-provided hooks for composition."""

    def __init__(self):
        self.pre_tool_use_called = False
        self.post_tool_use_called = False

    async def pre_tool_use_hook(self, hook_input, tool_use_id, context):
        self.pre_tool_use_called = True

    async def post_tool_use_hook(self, hook_input, tool_use_id, context):
        self.post_tool_use_called = True


# ===========================================================================
# Google ADK Instrumentor Tests
# ===========================================================================

class TestEvalynADKCallbacks:
    """Tests for EvalynADKCallbacks (core callback logic)."""

    def setup_method(self):
        _reset_span_context()

    def _create_callbacks(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        return EvalynADKCallbacks()

    # -- Agent lifecycle ------------------------------------------------

    def test_before_agent_creates_span_and_pushes_stack(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext(agent_name="researcher", invocation_id="inv-1")

        result = cb.before_agent_callback(ctx)

        assert result is None  # should not intercept
        # Span stack should have one entry
        stack = span_context._span_stack.get()
        assert len(stack) == 1
        # Agent spans dict should have one entry
        assert len(cb._agent_spans) == 1
        assert len(cb._agent_contexts) == 1

    def test_after_agent_finishes_span_and_pops_stack(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext(agent_name="researcher", invocation_id="inv-1")

        cb.before_agent_callback(ctx)
        result = cb.after_agent_callback(ctx)

        assert result is None
        stack = span_context._span_stack.get()
        assert len(stack) == 0
        assert len(cb._agent_spans) == 0
        assert len(cb._agent_contexts) == 0

        spans = _find_spans_by_type("agent")
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "agent:researcher"
        assert span.status == "ok"
        assert span.attributes["agent_name"] == "researcher"
        assert span.attributes["invocation_id"] == "inv-1"
        assert "duration_ms" in span.attributes

    def test_agent_span_captures_session_id(self):
        cb = self._create_callbacks()
        session = MockSession(session_id="sess-42")
        ctx = MockCallbackContext(
            agent_name="agent1", invocation_id="inv-1", session=session
        )

        cb.before_agent_callback(ctx)
        cb.after_agent_callback(ctx)

        spans = _find_spans_by_type("agent")
        assert spans[0].attributes.get("session_id") == "sess-42"

    def test_agent_span_captures_output_text(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        cb.before_agent_callback(ctx)

        # Simulate LLM producing output
        cb._output_text.append("This is the agent output.")

        cb.after_agent_callback(ctx)

        spans = _find_spans_by_type("agent")
        assert "output_preview" in spans[0].attributes
        assert spans[0].attributes["output_preview"] == "This is the agent output."
        assert spans[0].attributes["output_size"] == 25

    def test_nested_agent_tracks_parent_agent(self):
        cb = self._create_callbacks()

        # Start parent agent
        parent_ctx = MockCallbackContext(
            agent_name="orchestrator", invocation_id="inv-parent"
        )
        cb.before_agent_callback(parent_ctx)

        # Start child agent (different invocation_id signals sub-agent)
        child_ctx = MockCallbackContext(
            agent_name="researcher", invocation_id="inv-child"
        )
        cb.before_agent_callback(child_ctx)

        # Finish child
        cb.after_agent_callback(child_ctx)
        child_spans = [
            s for s in _find_spans_by_type("agent") if s.name == "agent:researcher"
        ]
        assert len(child_spans) == 1
        assert child_spans[0].attributes.get("parent_agent") == "orchestrator"

        # Finish parent
        cb.after_agent_callback(parent_ctx)

    def test_run_start_time_set_once(self):
        cb = self._create_callbacks()
        assert cb._run_start_time is None

        ctx = MockCallbackContext()
        cb.before_agent_callback(ctx)
        first_time = cb._run_start_time

        # Second call should not change it
        ctx2 = MockCallbackContext(agent_name="agent2", invocation_id="inv-2")
        cb.before_agent_callback(ctx2)
        assert cb._run_start_time == first_time

    # -- LLM call lifecycle ---------------------------------------------

    def test_before_model_creates_llm_span(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext(agent_name="agent1", invocation_id="inv-1")
        req = MockLlmRequest(model="gemini-2.5-flash")

        result = cb.before_model_callback(ctx, req)

        assert result is None
        assert len(cb._llm_spans) == 1
        assert cb._llm_call_counts["agent1"] == 1

    def test_after_model_finishes_llm_span_with_response(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext(agent_name="agent1", invocation_id="inv-1")
        req = MockLlmRequest(model="gemini-2.5-flash")
        content = MockContent(parts=[MockTextPart("The answer is 42.")])
        usage = MockUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        )
        resp = MockLlmResponse(content=content, usage_metadata=usage)

        cb.before_model_callback(ctx, req)
        result = cb.after_model_callback(ctx, resp)

        assert result is None
        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "llm:gemini-2.5-flash"
        assert span.status == "ok"
        assert span.attributes["model"] == "gemini-2.5-flash"
        assert span.attributes["provider"] == "google"
        assert span.attributes["agent_name"] == "agent1"
        assert span.attributes["turn"] == 1
        assert span.attributes["output_preview"] == "The answer is 42."
        assert span.attributes["output"] == "The answer is 42."
        assert span.attributes["prompt_tokens"] == 100
        assert span.attributes["completion_tokens"] == 50
        assert span.attributes["total_tokens"] == 150

    def test_llm_span_captures_model_version(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        req = MockLlmRequest(model="gemini-2.5-flash")
        resp = MockLlmResponse(model_version="gemini-2.5-flash-001")

        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, resp)

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes.get("model_version") == "gemini-2.5-flash-001"

    def test_llm_span_captures_cache_and_thinking_tokens(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        req = MockLlmRequest(model="gemini-2.5-flash")
        usage = MockUsageMetadata(
            prompt_token_count=200,
            candidates_token_count=80,
            total_token_count=280,
            cached_content_token_count=50,
            thoughts_token_count=30,
        )
        resp = MockLlmResponse(usage_metadata=usage)

        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, resp)

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes["cached_tokens"] == 50
        assert spans[0].attributes["thoughts_tokens"] == 30

    def test_llm_span_captures_tool_calls(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        req = MockLlmRequest(model="gemini-2.5-flash")
        content = MockContent(
            parts=[
                MockFunctionCallPart(name="search", args={"q": "hello"}),
                MockFunctionCallPart(name="calculator", args={"expr": "1+1"}),
            ]
        )
        resp = MockLlmResponse(content=content)

        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, resp)

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes["tool_calls"] == ["search", "calculator"]
        assert spans[0].attributes["tool_call_count"] == 2

    def test_llm_span_captures_config(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        config = MockLlmConfig(temperature=0.3, max_output_tokens=4096)
        req = MockLlmRequest(model="gemini-2.5-flash", config=config)

        cb.before_model_callback(ctx, req)
        # Finish the span
        cb.after_model_callback(ctx, MockLlmResponse())

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes["temperature"] == 0.3
        assert spans[0].attributes["max_output_tokens"] == 4096

    def test_llm_span_captures_prompt_preview(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        content1 = MockContent(parts=[MockTextPart("What is 2+2?")])
        content2 = MockContent(parts=[MockTextPart("Please explain.")])
        req = MockLlmRequest(model="gemini-2.5-flash", contents=[content1, content2])

        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, MockLlmResponse())

        spans = _find_spans_by_type("llm_call")
        assert "What is 2+2?" in spans[0].attributes["prompt_preview"]
        assert "Please explain." in spans[0].attributes["prompt_preview"]

    def test_llm_span_error_response(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        req = MockLlmRequest(model="gemini-2.5-flash")
        resp = MockLlmResponse(error_code="RATE_LIMIT", error_message="Too many requests")

        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, resp)

        spans = _find_spans_by_type("llm_call")
        assert spans[0].status == "error"
        assert spans[0].attributes.get("error_code") == "RATE_LIMIT"
        assert spans[0].attributes.get("error_message") == "Too many requests"

    def test_multiple_llm_calls_increment_turn(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext(agent_name="agent1")

        for i in range(3):
            req = MockLlmRequest(model="gemini-2.5-flash")
            cb.before_model_callback(ctx, req)
            cb.after_model_callback(ctx, MockLlmResponse())

        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 3
        turns = [s.attributes["turn"] for s in spans]
        assert turns == [1, 2, 3]

    # -- Tool call lifecycle --------------------------------------------

    def test_before_tool_creates_tool_span(self):
        cb = self._create_callbacks()
        tool = MockBaseTool(name="search")
        ctx = MockToolContext(agent_name="agent1", invocation_id="inv-1")
        args = {"query": "test"}

        result = cb.before_tool_callback(tool, args, ctx)

        assert result is None
        assert len(cb._tool_spans) == 1

    def test_after_tool_finishes_tool_span(self):
        cb = self._create_callbacks()
        tool = MockBaseTool(name="web_search")
        ctx = MockToolContext(agent_name="agent1", invocation_id="inv-1")
        args = {"query": "test query"}
        tool_result = {"results": ["result1", "result2"]}

        cb.before_tool_callback(tool, args, ctx)
        cb.after_tool_callback(tool, args, ctx, tool_result)

        spans = _find_spans_by_type("tool_call")
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "web_search"
        assert span.status == "ok"
        assert span.attributes["tool_name"] == "web_search"
        assert span.attributes["agent_name"] == "agent1"
        assert "output_size" in span.attributes
        assert "duration_ms" in span.attributes

    def test_tool_span_captures_input(self):
        cb = self._create_callbacks()
        tool = MockBaseTool(name="calculator")
        ctx = MockToolContext()
        args = {"expression": "2 + 2"}

        cb.before_tool_callback(tool, args, ctx)
        cb.after_tool_callback(tool, args, ctx, {"result": 4})

        spans = _find_spans_by_type("tool_call")
        assert "expression" in spans[0].attributes["input"]
        assert spans[0].attributes["input_size"] > 0

    def test_tool_span_detects_agent_tool(self):
        """Test that AgentTool (sub-agent delegation) is properly tagged."""
        cb = self._create_callbacks()

        # Create a mock that returns 'AgentTool' for type().__name__
        tool = MagicMock()
        tool.name = "delegate_agent"
        type(tool).__name__ = "AgentTool"
        sub_agent = MagicMock()
        sub_agent.name = "research_agent"
        tool.agent = sub_agent

        ctx = MockToolContext()
        args = {"task": "research"}

        cb.before_tool_callback(tool, args, ctx)
        cb.after_tool_callback(tool, args, ctx, {"output": "done"})

        spans = _find_spans_by_type("tool_call")
        assert spans[0].attributes.get("is_agent_tool") is True
        assert spans[0].attributes.get("sub_agent_name") == "research_agent"

    def test_tool_span_truncates_large_input(self):
        cb = self._create_callbacks()
        tool = MockBaseTool(name="big_input_tool")
        ctx = MockToolContext()
        args = {"data": "x" * 5000}

        cb.before_tool_callback(tool, args, ctx)
        cb.after_tool_callback(tool, args, ctx, None)

        spans = _find_spans_by_type("tool_call")
        assert spans[0].attributes["input_truncated"] is True
        assert len(spans[0].attributes["input"]) <= 4000

    def test_tool_span_truncates_large_output(self):
        cb = self._create_callbacks()
        tool = MockBaseTool(name="big_output_tool")
        ctx = MockToolContext()
        args = {}

        cb.before_tool_callback(tool, args, ctx)
        large_result = {"data": "y" * 5000}
        cb.after_tool_callback(tool, args, ctx, large_result)

        spans = _find_spans_by_type("tool_call")
        assert spans[0].attributes["output_truncated"] is True

    # -- User input / agent output capture ------------------------------

    def test_capture_user_input(self):
        cb = self._create_callbacks()
        cb.capture_user_input("Hello, how are you?")

        spans = _find_spans_by_type("user_message")
        assert len(spans) == 1
        assert spans[0].attributes["content"] == "Hello, how are you?"
        assert spans[0].attributes["content_length"] == 19
        assert spans[0].status == "ok"

    def test_capture_user_input_long_message_truncated(self):
        cb = self._create_callbacks()
        long_msg = "A" * 600
        cb.capture_user_input(long_msg)

        spans = _find_spans_by_type("user_message")
        assert spans[0].attributes["content_preview"] == "A" * 500
        assert spans[0].attributes["content_truncated"] is True

    def test_capture_user_input_with_session_id(self):
        cb = self._create_callbacks()
        cb.capture_user_input("hello", session_id="sess-1")

        spans = _find_spans_by_type("user_message")
        assert spans[0].attributes["session_id"] == "sess-1"

    def test_capture_final_output(self):
        cb = self._create_callbacks()
        cb.capture_final_output("The answer is 42.")

        spans = _find_spans_by_type("agent_message")
        assert len(spans) == 1
        assert spans[0].attributes["content"] == "The answer is 42."
        assert spans[0].attributes["content_length"] == 17
        assert spans[0].status == "ok"

    def test_capture_final_output_long_content(self):
        cb = self._create_callbacks()
        long_output = "B" * 5000
        cb.capture_final_output(long_output)

        spans = _find_spans_by_type("agent_message")
        # Content truncated to 4000
        assert len(spans[0].attributes["content"]) == 4000
        assert spans[0].attributes["content_length"] == 5000
        assert spans[0].attributes["content_preview"] == "B" * 500
        assert spans[0].attributes["content_truncated"] is True

    # -- Extract helpers ------------------------------------------------

    def test_extract_usage_with_none(self):
        cb = self._create_callbacks()
        result = cb._extract_usage(None)
        assert result == {}

    def test_extract_usage_with_metadata(self):
        cb = self._create_callbacks()
        usage = MockUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=20,
            thoughts_token_count=10,
        )
        result = cb._extract_usage(usage)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["cached_tokens"] == 20
        assert result["thoughts_tokens"] == 10

    def test_extract_text_from_content_empty(self):
        cb = self._create_callbacks()
        assert cb._extract_text_from_content(None) == ""
        assert cb._extract_text_from_content(MockContent(parts=[])) == ""

    def test_extract_text_from_content(self):
        cb = self._create_callbacks()
        content = MockContent(
            parts=[MockTextPart("Hello"), MockTextPart("World")]
        )
        assert cb._extract_text_from_content(content) == "Hello\nWorld"

    def test_extract_tool_calls_from_content(self):
        cb = self._create_callbacks()
        content = MockContent(
            parts=[
                MockFunctionCallPart(name="search", args={"q": "test"}),
                MockTextPart("some text"),
            ]
        )
        tool_calls = cb._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"
        assert tool_calls[0]["args"] == {"q": "test"}

    def test_extract_tool_calls_from_none(self):
        cb = self._create_callbacks()
        assert cb._extract_tool_calls_from_content(None) == []

    # -- Reset ----------------------------------------------------------

    def test_reset_clears_all_state(self):
        cb = self._create_callbacks()
        ctx = MockCallbackContext()
        cb.before_agent_callback(ctx)
        cb._output_text.append("some output")
        cb._llm_call_counts["agent"] = 3
        cb._tool_call_counts["agent"] = 2

        cb.reset()

        assert len(cb._agent_spans) == 0
        assert len(cb._llm_spans) == 0
        assert len(cb._tool_spans) == 0
        assert len(cb._agent_contexts) == 0
        assert len(cb._llm_call_counts) == 0
        assert len(cb._tool_call_counts) == 0
        assert len(cb._output_text) == 0
        assert cb._run_start_time is None


class TestADKStreamAdapter:
    """Tests for ADKStreamAdapter."""

    def setup_method(self):
        _reset_span_context()

    def _create_adapter(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            ADKStreamAdapter,
            EvalynADKCallbacks,
        )
        callbacks = EvalynADKCallbacks()
        return ADKStreamAdapter(callbacks), callbacks

    @pytest.mark.asyncio
    async def test_wrap_stream_captures_text(self):
        adapter, cb = self._create_adapter()
        events = [
            MockADKEvent(content=MockContent(parts=[MockTextPart("Hello")])),
            MockADKEvent(content=MockContent(parts=[MockTextPart("World")])),
        ]

        collected = []
        async for event in adapter.wrap_stream(_async_iter(events)):
            collected.append(event)

        assert len(collected) == 2
        assert adapter.get_accumulated_output() == "Hello\nWorld"

    @pytest.mark.asyncio
    async def test_wrap_stream_captures_usage(self):
        adapter, cb = self._create_adapter()
        usage = MockUsageMetadata(prompt_token_count=50, candidates_token_count=25)
        events = [MockADKEvent(usage_metadata=usage)]

        async for _ in adapter.wrap_stream(_async_iter(events)):
            pass

        last_usage = adapter.get_last_usage()
        assert last_usage is not None
        assert last_usage["prompt_tokens"] == 50
        assert last_usage["completion_tokens"] == 25

    @pytest.mark.asyncio
    async def test_wrap_stream_captures_final_output(self):
        adapter, cb = self._create_adapter()
        events = [
            MockADKEvent(content=MockContent(parts=[MockTextPart("Final answer")])),
            MockADKEvent(
                content=MockContent(parts=[MockTextPart("More content")]),
                is_final=True,
            ),
        ]

        async for _ in adapter.wrap_stream(_async_iter(events)):
            pass

        # Final output should be captured as agent_message span
        spans = _find_spans_by_type("agent_message")
        assert len(spans) == 1
        assert "Final answer" in spans[0].attributes["content"]

    @pytest.mark.asyncio
    async def test_wrap_stream_yields_all_events(self):
        adapter, cb = self._create_adapter()
        events = [MockADKEvent(), MockADKEvent(), MockADKEvent()]

        collected = []
        async for event in adapter.wrap_stream(_async_iter(events)):
            collected.append(event)

        assert len(collected) == 3

    @pytest.mark.asyncio
    async def test_wrap_stream_no_content(self):
        adapter, cb = self._create_adapter()
        events = [MockADKEvent(content=None)]

        async for _ in adapter.wrap_stream(_async_iter(events)):
            pass

        assert adapter.get_accumulated_output() == ""

    @pytest.mark.asyncio
    async def test_get_last_usage_none_without_events(self):
        adapter, cb = self._create_adapter()
        assert adapter.get_last_usage() is None


class TestGoogleADKInstrumentor:
    """Tests for GoogleADKInstrumentor properties and behavior."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            GoogleADKInstrumentor,
        )
        return GoogleADKInstrumentor()

    def test_name(self):
        inst = self._create_instrumentor()
        assert inst.name == "google_adk"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = self._create_instrumentor()
        assert inst.instrumentor_type == InstrumentorType.OTEL_NATIVE

    def test_is_available_without_sdk(self):
        inst = self._create_instrumentor()
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_is_available_with_sdk(self):
        inst = self._create_instrumentor()
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec", return_value=mock_spec):
            assert inst.is_available() is True

    def test_is_instrumented_initially_false(self):
        inst = self._create_instrumentor()
        assert inst.is_instrumented() is False

    def test_instrument_fails_when_not_available(self):
        inst = self._create_instrumentor()
        with patch.object(inst, "is_available", return_value=False):
            assert inst.instrument() is False

    def test_get_callbacks_returns_callbacks(self):
        """get_callbacks() creates callbacks if not yet created, by calling instrument()."""
        inst = self._create_instrumentor()
        # Patch instrument to just create callbacks without importing real SDK
        inst._callbacks = None
        with patch.object(inst, "instrument") as mock_inst:
            def set_callbacks():
                from evalyn_sdk.trace.instrumentation.providers.google_adk import (
                    EvalynADKCallbacks,
                )
                inst._callbacks = EvalynADKCallbacks()
                return True
            mock_inst.side_effect = set_callbacks
            cb = inst.get_callbacks()
            assert cb is not None
            mock_inst.assert_called_once()

    def test_uninstrument_when_not_instrumented(self):
        inst = self._create_instrumentor()
        # Should return True even when not instrumented
        assert inst.uninstrument() is True


class TestCreateADKCallbacks:
    """Tests for the create_adk_callbacks factory function."""

    def test_create_adk_callbacks(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            create_adk_callbacks,
        )
        cb = create_adk_callbacks()
        assert cb is not None
        assert len(cb._agent_spans) == 0


class TestCreateADKStreamAdapter:
    """Tests for the create_stream_adapter factory function."""

    def test_create_stream_adapter(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            create_adk_callbacks,
            create_stream_adapter,
        )
        cb = create_adk_callbacks()
        adapter = create_stream_adapter(cb)
        assert adapter is not None
        assert adapter._callbacks is cb


# ===========================================================================
# Claude Agent SDK Instrumentor Tests
# ===========================================================================

class TestEvalynAgentHooks:
    """Tests for EvalynAgentHooks (core hook logic)."""

    def setup_method(self):
        _reset_span_context()

    def _create_hooks(self, user_hooks=None):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        return EvalynAgentHooks(user_hooks=user_hooks)

    # -- Tool use hooks -------------------------------------------------

    @pytest.mark.asyncio
    async def test_pre_tool_use_creates_span(self):
        hooks = self._create_hooks()
        hook_input = {"tool_name": "bash", "tool_input": {"command": "ls"}}
        context = MockClaudeContext()

        result = await hooks.pre_tool_use_hook(hook_input, "tool-001", context)

        assert result == {"continue_": True}
        assert "tool-001" in hooks._tool_spans

    @pytest.mark.asyncio
    async def test_post_tool_use_finishes_span(self):
        hooks = self._create_hooks()
        hook_input = {"tool_name": "bash", "tool_input": {"command": "ls"}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-001", context)
        result = await hooks.post_tool_use_hook(
            {"tool_response": "file1.py\nfile2.py"}, "tool-001", context
        )

        assert result == {"continue_": True}
        assert "tool-001" not in hooks._tool_spans

        spans = _find_spans_by_type("tool_call")
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "bash"
        assert span.status == "ok"
        assert span.attributes["tool_name"] == "bash"
        assert span.attributes["tool_use_id"] == "tool-001"

    @pytest.mark.asyncio
    async def test_tool_span_captures_input_size(self):
        hooks = self._create_hooks()
        hook_input = {"tool_name": "search", "tool_input": {"query": "test query"}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-002", context)
        await hooks.post_tool_use_hook({"tool_response": "results"}, "tool-002", context)

        spans = _find_spans_by_type("tool_call")
        assert spans[0].attributes["input_size"] > 0

    @pytest.mark.asyncio
    async def test_tool_span_captures_output_size(self):
        hooks = self._create_hooks()
        hook_input = {"tool_name": "read", "tool_input": {"path": "/tmp/test"}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-003", context)
        await hooks.post_tool_use_hook(
            {"tool_response": "file contents here"}, "tool-003", context
        )

        spans = _find_spans_by_type("tool_call")
        assert "output_size" in spans[0].attributes

    @pytest.mark.asyncio
    async def test_tool_span_error_response(self):
        hooks = self._create_hooks()
        hook_input = {"tool_name": "bash", "tool_input": {"command": "invalid"}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-err", context)
        await hooks.post_tool_use_hook(
            {"tool_response": {"error": "Command not found"}}, "tool-err", context
        )

        spans = _find_spans_by_type("tool_call")
        assert spans[0].status == "error"
        assert "Command not found" in spans[0].attributes.get("error", "")

    @pytest.mark.asyncio
    async def test_tool_span_task_tool_registers_subagent(self):
        hooks = self._create_hooks()
        hook_input = {
            "tool_name": "Task",
            "tool_input": {
                "subagent_type": "researcher",
                "description": "Find info",
            },
        }
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "task-001", context)

        assert "task-001" in hooks._subagent_contexts
        assert hooks._subagent_contexts["task-001"].subagent_type == "researcher"

        spans_in_progress = hooks._tool_spans
        span_state = spans_in_progress["task-001"]
        assert span_state.span.attributes["subagent_type"] == "researcher"
        assert span_state.span.attributes["description"] == "Find info"

    @pytest.mark.asyncio
    async def test_tool_span_with_parent_tool_use_id(self):
        hooks = self._create_hooks()
        # Register a subagent
        hooks.register_subagent("parent-task-001", {"subagent_type": "coder"})
        hooks.set_current_context("parent-task-001")

        hook_input = {"tool_name": "bash", "tool_input": {"command": "echo hi"}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-sub-001", context)
        await hooks.post_tool_use_hook(
            {"tool_response": "hi"}, "tool-sub-001", context
        )

        spans = _find_spans_by_type("tool_call")
        assert spans[0].attributes["parent_tool_use_id"] == "parent-task-001"
        assert spans[0].attributes["executing_subagent"] == "coder"

    @pytest.mark.asyncio
    async def test_tool_span_captures_session_id(self):
        hooks = self._create_hooks()
        hook_input = {
            "tool_name": "bash",
            "tool_input": {"command": "ls"},
            "session_id": "session-42",
        }
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-sess", context)
        await hooks.post_tool_use_hook(
            {"tool_response": "ok"}, "tool-sess", context
        )

        spans = _find_spans_by_type("tool_call")
        assert spans[0].attributes["session_id"] == "session-42"

    @pytest.mark.asyncio
    async def test_post_tool_use_no_matching_span(self):
        """PostToolUse with unknown tool_use_id should not crash."""
        hooks = self._create_hooks()
        context = MockClaudeContext()

        result = await hooks.post_tool_use_hook(
            {"tool_response": "ok"}, "nonexistent-id", context
        )

        assert result == {"continue_": True}
        # Should not produce any spans
        assert len(_find_spans_by_type("tool_call")) == 0

    # -- User hooks composition -----------------------------------------

    @pytest.mark.asyncio
    async def test_user_hooks_called_on_pre_tool_use(self):
        user_hooks = MockUserHooks()
        hooks = self._create_hooks(user_hooks=user_hooks)

        hook_input = {"tool_name": "test", "tool_input": {}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-user-1", context)

        assert user_hooks.pre_tool_use_called is True

    @pytest.mark.asyncio
    async def test_user_hooks_called_on_post_tool_use(self):
        user_hooks = MockUserHooks()
        hooks = self._create_hooks(user_hooks=user_hooks)

        hook_input = {"tool_name": "test", "tool_input": {}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-user-2", context)
        await hooks.post_tool_use_hook(
            {"tool_response": "ok"}, "tool-user-2", context
        )

        assert user_hooks.post_tool_use_called is True

    # -- LLM turn logging -----------------------------------------------

    def test_log_llm_turn(self):
        hooks = self._create_hooks()
        hooks.log_llm_turn(
            turn=1,
            model="claude-sonnet-4",
            output_text="Hello world",
            tool_calls=["bash", "read"],
        )

        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "llm_turn_1"
        assert span.attributes["turn"] == 1
        assert span.attributes["model"] == "claude-sonnet-4"
        assert span.attributes["provider"] == "anthropic"
        assert span.attributes["output_preview"] == "Hello world"
        assert span.attributes["tool_calls"] == ["bash", "read"]
        assert span.status == "ok"

    def test_log_llm_turn_with_parent_tool_use_id(self):
        hooks = self._create_hooks()
        hooks.log_llm_turn(
            turn=2,
            model="claude-sonnet-4",
            parent_tool_use_id="task-parent-1",
        )

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes["parent_tool_use_id"] == "task-parent-1"

    def test_log_llm_turn_no_output_text(self):
        hooks = self._create_hooks()
        hooks.log_llm_turn(turn=1, model="claude-sonnet-4")

        spans = _find_spans_by_type("llm_call")
        assert "output_preview" not in spans[0].attributes

    # -- User input capture ---------------------------------------------

    def test_capture_user_input(self):
        hooks = self._create_hooks()
        hooks.capture_user_input("What is the capital of France?")

        spans = _find_spans_by_type("user_message")
        assert len(spans) == 1
        assert spans[0].attributes["content"] == "What is the capital of France?"
        assert spans[0].attributes["content_length"] == 30
        assert spans[0].status == "ok"

    def test_capture_user_input_long_message(self):
        hooks = self._create_hooks()
        long_msg = "Z" * 600
        hooks.capture_user_input(long_msg)

        spans = _find_spans_by_type("user_message")
        assert spans[0].attributes["content_preview"] == "Z" * 500
        assert spans[0].attributes["content_truncated"] is True

    # -- Thinking capture -----------------------------------------------

    def test_capture_thinking_text_only(self):
        hooks = self._create_hooks()
        hooks.capture_thinking("Let me think about this...")

        assert len(hooks._thinking_blocks) == 1
        assert hooks._thinking_blocks[0] == "Let me think about this..."

    def test_capture_thinking_with_signature(self):
        hooks = self._create_hooks()
        hooks.capture_thinking("My analysis...", signature="sig-123")

        assert len(hooks._thinking_blocks) == 1
        assert hooks._thinking_blocks[0] == {
            "text": "My analysis...",
            "signature": "sig-123",
        }

    # -- Context management ---------------------------------------------

    def test_set_current_context(self):
        hooks = self._create_hooks()
        hooks.set_current_context("task-001")
        assert hooks._current_parent_tool_use_id == "task-001"

        hooks.set_current_context(None)
        assert hooks._current_parent_tool_use_id is None

    def test_register_subagent(self):
        hooks = self._create_hooks()
        hooks.register_subagent("task-001", {"subagent_type": "researcher"})

        assert "task-001" in hooks._subagent_contexts
        ctx = hooks._subagent_contexts["task-001"]
        assert ctx.subagent_type == "researcher"
        assert ctx.tool_use_id == "task-001"

    # -- Try log turn from context (fallback LLM span creation) ---------

    @pytest.mark.asyncio
    async def test_try_log_turn_from_context_with_turn_info(self):
        hooks = self._create_hooks()
        context = MagicMock()
        context.turn = 1
        context.model = "claude-sonnet-4"

        hooks._try_log_turn_from_context(context, "tool-new-1")

        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1
        assert spans[0].attributes["model"] == "claude-sonnet-4"
        assert hooks._last_logged_turn == 1

    @pytest.mark.asyncio
    async def test_try_log_turn_skips_duplicate_tool_use_id(self):
        hooks = self._create_hooks()
        context = MagicMock()
        context.turn = 1
        context.model = "claude-sonnet-4"

        hooks._try_log_turn_from_context(context, "tool-dup-1")
        hooks._try_log_turn_from_context(context, "tool-dup-1")

        # Should only create one span
        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1

    @pytest.mark.asyncio
    async def test_try_log_turn_fallback_creates_initial_span(self):
        hooks = self._create_hooks()
        # Context without turn info
        context = MagicMock(spec=[])

        hooks._try_log_turn_from_context(context, "tool-fallback-1")

        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1
        assert spans[0].attributes["model"] == "claude-agent-sdk"
        assert hooks._last_logged_turn == 1

    @pytest.mark.asyncio
    async def test_try_log_turn_fallback_only_once(self):
        hooks = self._create_hooks()
        context = MagicMock(spec=[])

        hooks._try_log_turn_from_context(context, "tool-fb-1")
        hooks._try_log_turn_from_context(context, "tool-fb-2")

        # Fallback should only fire once
        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1

    # -- Finalize run ---------------------------------------------------

    def test_finalize_run_stores_metrics_on_root_span(self):
        hooks = self._create_hooks()
        # Create a root span on the stack
        root_span = Span.new(name="root", span_type="session")
        hooks._span_stack.append(SpanState(span=root_span, start_time=time.time()))

        usage = MockClaudeUsage(
            input_tokens=500,
            output_tokens=200,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=100,
        )
        result = ResultMessage(
            duration_ms=5000,
            duration_api_ms=4500,
            total_cost_usd=0.05,
            usage=usage,
            num_turns=3,
            is_error=False,
            result="Task completed successfully",
            session_id="sess-final",
        )

        hooks.finalize_run(result)

        attrs = root_span.attributes
        assert attrs["result_duration_ms"] == 5000
        assert attrs["duration_api_ms"] == 4500
        assert attrs["total_cost_usd"] == 0.05
        assert attrs["num_turns"] == 3
        assert attrs["is_error"] is False
        assert attrs["session_id"] == "sess-final"
        assert attrs["result_preview"] == "Task completed successfully"
        assert attrs["total_input_tokens"] == 500
        assert attrs["total_output_tokens"] == 200
        assert attrs["total_cache_creation_tokens"] == 50
        assert attrs["total_cache_read_tokens"] == 100

    def test_finalize_run_with_error(self):
        hooks = self._create_hooks()
        root_span = Span.new(name="root", span_type="session")
        hooks._span_stack.append(SpanState(span=root_span, start_time=time.time()))

        result = ResultMessage(is_error=True, result="Something went wrong")
        hooks.finalize_run(result)

        assert root_span.attributes["is_error"] is True

    def test_finalize_run_with_structured_output(self):
        hooks = self._create_hooks()
        root_span = Span.new(name="root", span_type="session")
        hooks._span_stack.append(SpanState(span=root_span, start_time=time.time()))

        result = ResultMessage(structured_output={"key": "value"})
        hooks.finalize_run(result)

        assert root_span.attributes["has_structured_output"] is True

    def test_finalize_run_stores_thinking_blocks(self):
        hooks = self._create_hooks()
        root_span = Span.new(name="root", span_type="session")
        hooks._span_stack.append(SpanState(span=root_span, start_time=time.time()))

        hooks.capture_thinking("thinking 1")
        hooks.capture_thinking("thinking 2", signature="sig-1")

        result = ResultMessage()
        hooks.finalize_run(result)

        assert root_span.attributes["thinking_blocks"] == [
            "thinking 1",
            {"text": "thinking 2", "signature": "sig-1"},
        ]

    def test_finalize_run_with_none_result(self):
        hooks = self._create_hooks()
        # Should not crash
        hooks.finalize_run(None)

    def test_finalize_run_without_root_span(self):
        hooks = self._create_hooks()
        result = ResultMessage(duration_ms=1000)
        # Should not crash - just no span to store on
        hooks.finalize_run(result)

    # -- Hook matchers --------------------------------------------------

    def test_get_hook_matchers(self):
        hooks = self._create_hooks()
        matchers = hooks.get_hook_matchers()

        assert "PreToolUse" in matchers
        assert "PostToolUse" in matchers
        assert len(matchers["PreToolUse"]) == 1
        assert len(matchers["PostToolUse"]) == 1
        assert matchers["PreToolUse"][0]["matcher"] is None
        assert hooks.pre_tool_use_hook in matchers["PreToolUse"][0]["hooks"]
        assert hooks.post_tool_use_hook in matchers["PostToolUse"][0]["hooks"]

    # -- Token usage extraction -----------------------------------------

    def test_extract_usage(self):
        hooks = self._create_hooks()
        usage = MockClaudeUsage(
            input_tokens=300,
            output_tokens=150,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=80,
        )
        result = hooks._extract_usage(usage)
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 150
        assert result["cache_creation_input_tokens"] == 20
        assert result["cache_read_input_tokens"] == 80

    def test_extract_usage_missing_attrs(self):
        hooks = self._create_hooks()
        usage = MagicMock(spec=[])  # No attributes
        result = hooks._extract_usage(usage)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


# Need the SpanState for finalize_run tests
from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import SpanState


class TestMessageStreamAdapter:
    """Tests for MessageStreamAdapter."""

    def setup_method(self):
        _reset_span_context()

    def _create_adapter(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
            MessageStreamAdapter,
        )
        hooks = EvalynAgentHooks()
        return MessageStreamAdapter(hooks), hooks

    @pytest.mark.asyncio
    async def test_wrap_stream_assistant_message_creates_llm_span(self):
        adapter, hooks = self._create_adapter()

        msg = AssistantMessage(
            model="claude-sonnet-4",
            content=[TextBlock("Hello there")],
        )

        collected = []
        async for m in adapter.wrap_stream(_async_iter([msg])):
            collected.append(m)

        assert len(collected) == 1
        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1
        assert spans[0].attributes["model"] == "claude-sonnet-4"
        assert spans[0].attributes["turn"] == 1

    @pytest.mark.asyncio
    async def test_wrap_stream_multiple_turns(self):
        adapter, hooks = self._create_adapter()

        msgs = [
            AssistantMessage(model="claude-sonnet-4", content=[TextBlock("Turn 1")]),
            AssistantMessage(model="claude-sonnet-4", content=[TextBlock("Turn 2")]),
            AssistantMessage(model="claude-sonnet-4", content=[TextBlock("Turn 3")]),
        ]

        async for _ in adapter.wrap_stream(_async_iter(msgs)):
            pass

        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 3
        turns = [s.attributes["turn"] for s in spans]
        assert turns == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_wrap_stream_captures_tool_uses(self):
        adapter, hooks = self._create_adapter()

        msg = AssistantMessage(
            model="claude-sonnet-4",
            content=[
                TextBlock("Let me search"),
                ToolUseBlock(name="bash", id="tu-001", input={"command": "ls"}),
            ],
        )

        async for _ in adapter.wrap_stream(_async_iter([msg])):
            pass

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes["tool_calls"] == ["bash"]

    @pytest.mark.asyncio
    async def test_wrap_stream_registers_task_subagent(self):
        adapter, hooks = self._create_adapter()

        msg = AssistantMessage(
            model="claude-sonnet-4",
            content=[
                ToolUseBlock(
                    name="Task",
                    id="task-sub-001",
                    input={"subagent_type": "coder"},
                ),
            ],
        )

        async for _ in adapter.wrap_stream(_async_iter([msg])):
            pass

        assert "task-sub-001" in hooks._subagent_contexts
        assert hooks._subagent_contexts["task-sub-001"].subagent_type == "coder"

    @pytest.mark.asyncio
    async def test_wrap_stream_captures_thinking_blocks(self):
        adapter, hooks = self._create_adapter()

        msg = AssistantMessage(
            model="claude-sonnet-4",
            content=[
                ThinkingBlock(thinking="Let me analyze...", signature="sig-abc"),
                TextBlock("Here is my answer"),
            ],
        )

        async for _ in adapter.wrap_stream(_async_iter([msg])):
            pass

        assert len(hooks._thinking_blocks) == 1
        assert hooks._thinking_blocks[0] == {
            "text": "Let me analyze...",
            "signature": "sig-abc",
        }

    @pytest.mark.asyncio
    async def test_wrap_stream_sets_parent_context(self):
        adapter, hooks = self._create_adapter()

        msg = AssistantMessage(
            model="claude-sonnet-4",
            content=[TextBlock("Sub-agent response")],
            parent_tool_use_id="parent-task-123",
        )

        async for _ in adapter.wrap_stream(_async_iter([msg])):
            pass

        assert hooks._current_parent_tool_use_id == "parent-task-123"

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes.get("parent_tool_use_id") == "parent-task-123"

    @pytest.mark.asyncio
    async def test_wrap_stream_result_message_finalizes_run(self):
        adapter, hooks = self._create_adapter()
        # Set up a root span for finalize_run
        root_span = Span.new(name="root", span_type="session")
        hooks._span_stack.append(SpanState(span=root_span, start_time=time.time()))

        result_msg = ResultMessage(
            duration_ms=3000,
            num_turns=2,
            total_cost_usd=0.03,
        )

        async for _ in adapter.wrap_stream(_async_iter([result_msg])):
            pass

        assert root_span.attributes["result_duration_ms"] == 3000
        assert root_span.attributes["num_turns"] == 2
        assert root_span.attributes["total_cost_usd"] == 0.03

    @pytest.mark.asyncio
    async def test_wrap_stream_yields_all_messages(self):
        adapter, hooks = self._create_adapter()

        class UnknownMessage:
            pass

        msgs = [
            AssistantMessage(model="claude-sonnet-4", content=[]),
            UnknownMessage(),
            ResultMessage(),
        ]

        collected = []
        async for m in adapter.wrap_stream(_async_iter(msgs)):
            collected.append(m)

        # All messages should pass through
        assert len(collected) == 3

    @pytest.mark.asyncio
    async def test_wrap_stream_empty_content_no_crash(self):
        adapter, hooks = self._create_adapter()
        msg = AssistantMessage(model="claude-sonnet-4", content=[])

        async for _ in adapter.wrap_stream(_async_iter([msg])):
            pass

        spans = _find_spans_by_type("llm_call")
        assert len(spans) == 1
        # No output_text or tool_calls
        assert "output_preview" not in spans[0].attributes
        assert "tool_calls" not in spans[0].attributes


class TestClaudeAgentSDKInstrumentor:
    """Tests for ClaudeAgentSDKInstrumentor properties and behavior."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            ClaudeAgentSDKInstrumentor,
        )
        return ClaudeAgentSDKInstrumentor()

    def test_name(self):
        inst = self._create_instrumentor()
        assert inst.name == "claude_agent_sdk"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = self._create_instrumentor()
        assert inst.instrumentor_type == InstrumentorType.HOOK_BASED

    def test_is_available_without_sdk(self):
        inst = self._create_instrumentor()
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_is_available_with_sdk(self):
        inst = self._create_instrumentor()
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec", return_value=mock_spec):
            assert inst.is_available() is True

    def test_is_instrumented_initially_false(self):
        inst = self._create_instrumentor()
        assert inst.is_instrumented() is False

    def test_instrument_fails_when_not_available(self):
        inst = self._create_instrumentor()
        with patch.object(inst, "is_available", return_value=False):
            assert inst.instrument() is False

    def test_is_instrumented_after_instrument(self):
        inst = self._create_instrumentor()
        with patch.object(inst, "is_available", return_value=True):
            with patch.object(inst, "_patch_send_message"):
                result = inst.instrument()
                assert result is True
                assert inst.is_instrumented() is True

    def test_uninstrument(self):
        inst = self._create_instrumentor()
        with patch.object(inst, "is_available", return_value=True):
            with patch.object(inst, "_patch_send_message"):
                inst.instrument()

        with patch.object(inst, "_unpatch_send_message"):
            result = inst.uninstrument()
            assert result is True
            assert inst.is_instrumented() is False

    def test_get_hooks_returns_hooks(self):
        inst = self._create_instrumentor()
        with patch.object(inst, "is_available", return_value=True):
            with patch.object(inst, "_patch_send_message"):
                hooks = inst.get_hooks()
                assert hooks is not None

    def test_get_hooks_calls_instrument_if_needed(self):
        inst = self._create_instrumentor()
        assert inst._hooks is None

        with patch.object(inst, "instrument") as mock_inst:
            from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
                EvalynAgentHooks,
            )

            def set_hooks():
                inst._hooks = EvalynAgentHooks()
                return True

            mock_inst.side_effect = set_hooks
            hooks = inst.get_hooks()
            mock_inst.assert_called_once()
            assert hooks is not None


class TestBackwardsCompatibilityAlias:
    """Test that the AnthropicAgentsInstrumentor alias works."""

    def test_alias_exists(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            AnthropicAgentsInstrumentor,
            ClaudeAgentSDKInstrumentor,
        )
        assert AnthropicAgentsInstrumentor is ClaudeAgentSDKInstrumentor


class TestCreateAgentHooks:
    """Tests for the create_agent_hooks factory function."""

    def test_create_agent_hooks(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            create_agent_hooks,
        )
        hooks = create_agent_hooks()
        assert hooks is not None
        assert hooks._user_hooks is None

    def test_create_agent_hooks_with_user_hooks(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            create_agent_hooks,
        )
        user_hooks = MockUserHooks()
        hooks = create_agent_hooks(user_hooks=user_hooks)
        assert hooks._user_hooks is user_hooks


class TestCreateClaudeStreamAdapter:
    """Tests for the create_stream_adapter factory function."""

    def test_create_stream_adapter(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            create_agent_hooks,
            create_stream_adapter,
        )
        hooks = create_agent_hooks()
        adapter = create_stream_adapter(hooks)
        assert adapter is not None
        assert adapter._hooks is hooks


# ===========================================================================
# Integration / End-to-end flow tests
# ===========================================================================

class TestADKEndToEnd:
    """Test a full ADK agent call flow: agent -> LLM -> tool -> LLM -> finish."""

    def setup_method(self):
        _reset_span_context()

    def test_full_agent_flow(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )

        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext(
            agent_name="research_agent", invocation_id="inv-100"
        )

        # 1. Agent starts
        cb.before_agent_callback(ctx)

        # 2. First LLM call (decides to use a tool)
        req1 = MockLlmRequest(model="gemini-2.5-flash")
        cb.before_model_callback(ctx, req1)

        content1 = MockContent(
            parts=[
                MockTextPart("I'll search for that."),
                MockFunctionCallPart(name="web_search", args={"q": "test"}),
            ]
        )
        resp1 = MockLlmResponse(
            content=content1,
            usage_metadata=MockUsageMetadata(100, 30, 130),
        )
        cb.after_model_callback(ctx, resp1)

        # 3. Tool call
        tool = MockBaseTool(name="web_search")
        tool_ctx = MockToolContext(
            agent_name="research_agent", invocation_id="inv-100"
        )
        cb.before_tool_callback(tool, {"q": "test"}, tool_ctx)
        cb.after_tool_callback(
            tool, {"q": "test"}, tool_ctx, {"results": ["r1", "r2"]}
        )

        # 4. Second LLM call (final answer)
        req2 = MockLlmRequest(model="gemini-2.5-flash")
        cb.before_model_callback(ctx, req2)

        content2 = MockContent(parts=[MockTextPart("Based on my search, the answer is...")])
        resp2 = MockLlmResponse(
            content=content2,
            usage_metadata=MockUsageMetadata(200, 60, 260),
        )
        cb.after_model_callback(ctx, resp2)

        # 5. Agent finishes
        cb.after_agent_callback(ctx)

        # Verify spans
        all_spans = _get_collected_spans()
        assert len(all_spans) == 4  # 2 LLM + 1 tool + 1 agent

        agent_spans = _find_spans_by_type("agent")
        assert len(agent_spans) == 1
        assert agent_spans[0].name == "agent:research_agent"

        llm_spans = _find_spans_by_type("llm_call")
        assert len(llm_spans) == 2
        assert llm_spans[0].attributes["turn"] == 1
        assert llm_spans[1].attributes["turn"] == 2

        tool_spans = _find_spans_by_type("tool_call")
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "web_search"


class TestClaudeAgentSDKEndToEnd:
    """Test a full Claude Agent SDK flow: user input -> LLM turns -> tool -> result."""

    def setup_method(self):
        _reset_span_context()

    @pytest.mark.asyncio
    async def test_full_agent_flow(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
            MessageStreamAdapter,
        )

        hooks = EvalynAgentHooks()
        adapter = MessageStreamAdapter(hooks)

        # Set up a root span
        root_span = Span.new(name="session", span_type="session")
        hooks._span_stack.append(SpanState(span=root_span, start_time=time.time()))

        # 1. User input
        hooks.capture_user_input("What files are in the current directory?")

        # 2. First assistant message (decides to use bash)
        msg1 = AssistantMessage(
            model="claude-sonnet-4",
            content=[
                TextBlock("I'll check the directory."),
                ToolUseBlock(name="bash", id="tu-001", input={"command": "ls"}),
            ],
        )

        # 3. Tool use hooks
        pre_input = {"tool_name": "bash", "tool_input": {"command": "ls"}}
        context = MockClaudeContext()

        # Process first message via stream adapter
        async for _ in adapter.wrap_stream(_async_iter([msg1])):
            pass

        # Tool hooks
        await hooks.pre_tool_use_hook(pre_input, "tu-001", context)
        await hooks.post_tool_use_hook(
            {"tool_response": "file1.py\nfile2.py"}, "tu-001", context
        )

        # 4. Second assistant message (final answer)
        msg2 = AssistantMessage(
            model="claude-sonnet-4",
            content=[TextBlock("The directory contains file1.py and file2.py.")],
        )

        async for _ in adapter.wrap_stream(_async_iter([msg2])):
            pass

        # 5. Result message
        result_msg = ResultMessage(
            duration_ms=2000,
            num_turns=2,
            total_cost_usd=0.01,
            usage=MockClaudeUsage(input_tokens=300, output_tokens=150),
        )

        async for _ in adapter.wrap_stream(_async_iter([result_msg])):
            pass

        # Verify spans
        user_spans = _find_spans_by_type("user_message")
        assert len(user_spans) == 1

        llm_spans = _find_spans_by_type("llm_call")
        # 2 from stream adapter (AssistantMessages) + 1 from pre_tool_use_hook fallback
        # The hook fallback fires because _try_log_turn_from_context sees last_logged_turn==0
        # and creates an initial span before the tool runs. This is expected behavior
        # when using both adapter and hooks together.
        assert len(llm_spans) == 3

        # Adapter-created spans have the model from the message
        adapter_spans = [s for s in llm_spans if s.attributes["model"] == "claude-sonnet-4"]
        assert len(adapter_spans) == 2

        tool_spans = _find_spans_by_type("tool_call")
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "bash"

        # Root span should have finalized metrics
        assert root_span.attributes["result_duration_ms"] == 2000
        assert root_span.attributes["num_turns"] == 2
        assert root_span.attributes["total_input_tokens"] == 300
        assert root_span.attributes["total_output_tokens"] == 150


# ===========================================================================
# Edge case / error handling tests
# ===========================================================================

class TestADKEdgeCases:
    """Edge cases for ADK callbacks."""

    def setup_method(self):
        _reset_span_context()

    def test_after_agent_without_before(self):
        """after_agent_callback should not crash if before was never called."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext()
        result = cb.after_agent_callback(ctx)
        assert result is None
        assert len(_get_collected_spans()) == 0

    def test_after_model_without_before(self):
        """after_model_callback should not crash if before was never called."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext()
        resp = MockLlmResponse()
        result = cb.after_model_callback(ctx, resp)
        assert result is None
        assert len(_get_collected_spans()) == 0

    def test_after_tool_without_before(self):
        """after_tool_callback should not crash if before was never called."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        tool = MockBaseTool(name="test")
        ctx = MockToolContext()
        result = cb.after_tool_callback(tool, {}, ctx, None)
        assert result is None
        assert len(_get_collected_spans()) == 0

    def test_content_with_none_parts(self):
        """Content with None parts should not crash."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        content = MockContent(parts=None)
        assert cb._extract_text_from_content(content) == ""
        assert cb._extract_tool_calls_from_content(content) == []

    def test_usage_metadata_with_none_values(self):
        """Usage metadata with None attribute values should default to 0."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        usage = MagicMock()
        usage.prompt_token_count = None
        usage.candidates_token_count = None
        usage.total_token_count = None
        usage.cached_content_token_count = None
        usage.thoughts_token_count = None

        result = cb._extract_usage(usage)
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0

    def test_llm_request_with_no_model(self):
        """LlmRequest with no model attr should default to 'unknown'."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext()
        req = MagicMock(spec=[])  # No 'model' attribute

        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, MockLlmResponse())

        spans = _find_spans_by_type("llm_call")
        assert spans[0].attributes["model"] == "unknown"

    def test_agent_stack_pop_resilience(self):
        """after_agent should handle mismatched stack correctly."""
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext()

        cb.before_agent_callback(ctx)
        # Manually corrupt the stack
        span_context._span_stack.set([])

        # Should not crash
        cb.after_agent_callback(ctx)

        spans = _find_spans_by_type("agent")
        assert len(spans) == 1


class TestClaudeAgentSDKEdgeCases:
    """Edge cases for Claude Agent SDK hooks."""

    def setup_method(self):
        _reset_span_context()

    @pytest.mark.asyncio
    async def test_pre_tool_use_unknown_tool(self):
        """Hook should handle missing tool_name gracefully."""
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()
        hook_input = {}  # No tool_name
        context = MockClaudeContext()

        result = await hooks.pre_tool_use_hook(hook_input, "tool-unk", context)
        assert result == {"continue_": True}

        # Span should have "unknown" as tool name
        assert hooks._tool_spans["tool-unk"].span.attributes["tool_name"] == "unknown"

    @pytest.mark.asyncio
    async def test_post_tool_use_no_tool_response(self):
        """PostToolUse with no tool_response key should not crash."""
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()
        hook_input = {"tool_name": "test", "tool_input": {}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-no-resp", context)
        await hooks.post_tool_use_hook({}, "tool-no-resp", context)

        spans = _find_spans_by_type("tool_call")
        assert spans[0].status == "ok"

    @pytest.mark.asyncio
    async def test_user_hook_not_called_when_none(self):
        """If no user hooks provided, _call_user_hook should not crash."""
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks(user_hooks=None)
        # Should not raise
        await hooks._call_user_hook("pre_tool_use_hook", {}, "id", None)

    def test_pop_span_from_context_empty_stack(self):
        """_pop_span_from_context with empty stack should not crash."""
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()
        hooks._pop_span_from_context("nonexistent-span-id")
        # Should not crash

    def test_pop_span_from_context_mismatched_id(self):
        """_pop_span_from_context with wrong ID should not pop."""
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()
        span_context._span_stack.set(["span-1", "span-2"])
        hooks._pop_span_from_context("span-1")  # Not the top
        # Stack should be unchanged
        assert span_context._span_stack.get() == ["span-1", "span-2"]

    def test_pop_span_from_context_correct_id(self):
        """_pop_span_from_context with correct ID should pop."""
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()
        span_context._span_stack.set(["span-1", "span-2"])
        hooks._pop_span_from_context("span-2")  # Top of stack
        assert span_context._span_stack.get() == ["span-1"]


# ===========================================================================
# Span hierarchy / parent-child relationship tests
# ===========================================================================

class TestADKSpanHierarchy:
    """Verify parent-child relationships in ADK spans."""

    def setup_method(self):
        _reset_span_context()

    def test_llm_span_parent_is_agent_span(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext(agent_name="agent1", invocation_id="inv-1")

        # Start agent
        cb.before_agent_callback(ctx)
        agent_span_id = span_context._span_stack.get()[-1]

        # LLM call inside agent
        req = MockLlmRequest(model="gemini-2.5-flash")
        cb.before_model_callback(ctx, req)
        cb.after_model_callback(ctx, MockLlmResponse())

        llm_spans = _find_spans_by_type("llm_call")
        assert llm_spans[0].parent_id == agent_span_id

        cb.after_agent_callback(ctx)

    def test_tool_span_parent_is_agent_span(self):
        from evalyn_sdk.trace.instrumentation.providers.google_adk import (
            EvalynADKCallbacks,
        )
        cb = EvalynADKCallbacks()
        ctx = MockCallbackContext(agent_name="agent1", invocation_id="inv-1")

        # Start agent
        cb.before_agent_callback(ctx)
        agent_span_id = span_context._span_stack.get()[-1]

        # Tool call inside agent
        tool = MockBaseTool(name="search")
        tool_ctx = MockToolContext(agent_name="agent1", invocation_id="inv-1")
        cb.before_tool_callback(tool, {"q": "test"}, tool_ctx)
        cb.after_tool_callback(tool, {"q": "test"}, tool_ctx, {"result": "ok"})

        tool_spans = _find_spans_by_type("tool_call")
        assert tool_spans[0].parent_id == agent_span_id

        cb.after_agent_callback(ctx)


class TestClaudeAgentSDKSpanHierarchy:
    """Verify parent-child relationships in Claude Agent SDK spans."""

    def setup_method(self):
        _reset_span_context()

    @pytest.mark.asyncio
    async def test_tool_span_uses_current_parent(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()

        # Push a parent span
        parent_id = "parent-root-span"
        span_context._span_stack.set([parent_id])

        hook_input = {"tool_name": "bash", "tool_input": {"command": "ls"}}
        context = MockClaudeContext()

        await hooks.pre_tool_use_hook(hook_input, "tool-hier-1", context)
        await hooks.post_tool_use_hook(
            {"tool_response": "ok"}, "tool-hier-1", context
        )

        spans = _find_spans_by_type("tool_call")
        assert spans[0].parent_id == parent_id

    def test_llm_turn_span_uses_current_parent(self):
        from evalyn_sdk.trace.instrumentation.providers.claude_agent_sdk import (
            EvalynAgentHooks,
        )
        hooks = EvalynAgentHooks()

        parent_id = "parent-root-span"
        span_context._span_stack.set([parent_id])

        hooks.log_llm_turn(turn=1, model="claude-sonnet-4")

        spans = _find_spans_by_type("llm_call")
        assert spans[0].parent_id == parent_id
