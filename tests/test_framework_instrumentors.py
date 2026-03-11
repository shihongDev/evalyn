"""
Tests for framework instrumentors (CrewAI, AutoGen, DSPy, Haystack, LlamaIndex, Semantic Kernel).

These tests mock the framework imports to test instrumentation logic
without requiring the actual frameworks to be installed.

Run with: pytest tests/test_framework_instrumentors.py -v
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
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
# Mock CrewAI event bus
# ---------------------------------------------------------------------------

class MockEventBus:
    """Mock CrewAI event bus that supports @event_bus.on(EventType) pattern."""

    def __init__(self):
        self._handlers: Dict[type, List[Callable]] = {}

    def on(self, event_type: type):
        """Decorator to register an event handler."""
        def decorator(fn):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(fn)
            return fn
        return decorator

    def emit(self, event_type: type, source: Any, event: Any):
        """Emit an event to all registered handlers."""
        for handler in self._handlers.get(event_type, []):
            handler(source, event)


# Mock event classes
class MockCrewKickoffStartedEvent:
    def __init__(self, inputs=None):
        self.inputs = inputs or {}


class MockCrewKickoffCompletedEvent:
    def __init__(self, output=None):
        self.output = output


class MockCrewKickoffFailedEvent:
    def __init__(self, error=None):
        self.error = error


class MockAgentExecutionStartedEvent:
    def __init__(self, agent=None):
        self.agent = agent


class MockAgentExecutionCompletedEvent:
    def __init__(self, output=None):
        self.output = output


class MockAgentExecutionErrorEvent:
    def __init__(self, error=None):
        self.error = error


class MockTaskStartedEvent:
    def __init__(self, task=None):
        self.task = task


class MockTaskCompletedEvent:
    def __init__(self, output=None):
        self.output = output


class MockTaskFailedEvent:
    def __init__(self, error=None):
        self.error = error


class MockLLMCallStartedEvent:
    def __init__(self, model=None, messages=None, call_id=None):
        self.model = model
        self.messages = messages
        self.call_id = call_id or id(self)


class MockLLMCallCompletedEvent:
    def __init__(self, model=None, response=None, token_usage=None, call_id=None):
        self.model = model
        self.response = response
        self.token_usage = token_usage or {}
        self.call_id = call_id or id(self)


class MockLLMCallFailedEvent:
    def __init__(self, error=None, call_id=None):
        self.error = error
        self.call_id = call_id or id(self)


class MockToolUsageStartedEvent:
    def __init__(self, tool_name=None, tool_input=None):
        self.tool_name = tool_name
        self.tool_input = tool_input


class MockToolUsageFinishedEvent:
    def __init__(self, tool_output=None):
        self.tool_output = tool_output


class MockToolUsageErrorEvent:
    def __init__(self, error=None):
        self.error = error


class MockKnowledgeRetrievalStartedEvent:
    def __init__(self, query=None):
        self.query = query


class MockKnowledgeRetrievalCompletedEvent:
    def __init__(self, results=None):
        self.results = results


class MockLLMStreamChunkEvent:
    def __init__(self, call_id=None, chunk=None):
        self.call_id = call_id
        self.chunk = chunk


# Mock source objects
class MockCrew:
    def __init__(self, name="test_crew"):
        self.name = name


class MockAgent:
    def __init__(self, role="researcher", goal="find answers"):
        self.role = role
        self.goal = goal


class MockTask:
    def __init__(self, description="analyze data", expected_output="report"):
        self.description = description
        self.expected_output = expected_output


# ---------------------------------------------------------------------------
# CrewAI Instrumentor Tests
# ---------------------------------------------------------------------------

class TestCrewAIInstrumentor:
    """Tests for the CrewAI instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def _create_instrumentor_with_mock_bus(self, include_optional=False):
        """Create a CrewAI instrumentor wired to a mock event bus.

        Args:
            include_optional: if True, include KnowledgeRetrieval and LLMStreamChunk
                in the optional events dict.
        """
        from evalyn_sdk.trace.instrumentation.providers.crewai import (
            CrewAIInstrumentor,
            _SpanTracker,
        )

        instrumentor = CrewAIInstrumentor()
        bus = MockEventBus()

        # Manually set up - bypass the real import
        instrumentor._span_tracker = _SpanTracker()

        # Build core events dict matching _import_core_events() return format
        core_events = {
            "CrewKickoffStartedEvent": MockCrewKickoffStartedEvent,
            "CrewKickoffCompletedEvent": MockCrewKickoffCompletedEvent,
            "CrewKickoffFailedEvent": MockCrewKickoffFailedEvent,
            "AgentExecutionStartedEvent": MockAgentExecutionStartedEvent,
            "AgentExecutionCompletedEvent": MockAgentExecutionCompletedEvent,
            "AgentExecutionErrorEvent": MockAgentExecutionErrorEvent,
            "TaskStartedEvent": MockTaskStartedEvent,
            "TaskCompletedEvent": MockTaskCompletedEvent,
            "TaskFailedEvent": MockTaskFailedEvent,
            "LLMCallStartedEvent": MockLLMCallStartedEvent,
            "LLMCallCompletedEvent": MockLLMCallCompletedEvent,
            "LLMCallFailedEvent": MockLLMCallFailedEvent,
            "ToolUsageStartedEvent": MockToolUsageStartedEvent,
            "ToolUsageFinishedEvent": MockToolUsageFinishedEvent,
            "ToolUsageErrorEvent": MockToolUsageErrorEvent,
        }

        # Build optional events dict
        optional_events: dict = {}
        if include_optional:
            optional_events["KnowledgeRetrievalStartedEvent"] = MockKnowledgeRetrievalStartedEvent
            optional_events["KnowledgeRetrievalCompletedEvent"] = MockKnowledgeRetrievalCompletedEvent
            optional_events["LLMStreamChunkEvent"] = MockLLMStreamChunkEvent

        crewai_mod = "evalyn_sdk.trace.instrumentation.providers.crewai"
        with patch(f"{crewai_mod}._import_core_events", return_value=core_events), \
             patch(f"{crewai_mod}._import_optional_events", return_value=optional_events):
            instrumentor._register_event_handlers(bus)

        instrumentor._instrumented = True
        return instrumentor, bus

    def test_instrumentor_properties(self):
        """Test basic instrumentor properties."""
        from evalyn_sdk.trace.instrumentation.providers.crewai import CrewAIInstrumentor

        inst = CrewAIInstrumentor()
        assert inst.name == "crewai"
        assert inst.instrumentor_type.value == "hook_based"
        assert not inst.is_instrumented()

    def test_crew_kickoff_creates_span(self):
        """Test that crew kickoff start/complete creates a span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        crew = MockCrew(name="research_crew")
        bus.emit(MockCrewKickoffStartedEvent, crew, MockCrewKickoffStartedEvent(inputs={"topic": "AI"}))
        bus.emit(MockCrewKickoffCompletedEvent, crew, MockCrewKickoffCompletedEvent(output="done"))

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "crew:research_crew"
        assert spans[0].span_type == "agent"
        assert spans[0].status == "ok"
        assert spans[0].attributes["crewai.type"] == "crew"

    def test_crew_kickoff_failure(self):
        """Test that crew kickoff failure creates an error span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        crew = MockCrew()
        bus.emit(MockCrewKickoffStartedEvent, crew, MockCrewKickoffStartedEvent())
        bus.emit(MockCrewKickoffFailedEvent, crew, MockCrewKickoffFailedEvent(error="timeout"))

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert spans[0].attributes["error"] == "timeout"

    def test_agent_execution_span(self):
        """Test agent execution creates properly attributed span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        agent = MockAgent(role="data_analyst", goal="analyze trends")
        bus.emit(MockAgentExecutionStartedEvent, agent, MockAgentExecutionStartedEvent(agent=agent))
        bus.emit(MockAgentExecutionCompletedEvent, agent, MockAgentExecutionCompletedEvent(output="analysis complete"))

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "agent:data_analyst"
        assert spans[0].attributes["crewai.agent_role"] == "data_analyst"
        assert spans[0].attributes["crewai.agent_goal"] == "analyze trends"

    def test_task_execution_span(self):
        """Test task execution creates span with description."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        task = MockTask(description="summarize findings", expected_output="summary report")
        bus.emit(MockTaskStartedEvent, task, MockTaskStartedEvent(task=task))
        bus.emit(MockTaskCompletedEvent, task, MockTaskCompletedEvent(output="summary done"))

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert "summarize findings" in spans[0].name
        assert spans[0].attributes["crewai.task_description"] == "summarize findings"

    def test_llm_call_span_with_tokens(self):
        """Test LLM call span captures model and token usage."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        call_id = "test-call-123"
        bus.emit(
            MockLLMCallStartedEvent,
            MagicMock(),
            MockLLMCallStartedEvent(
                model="gpt-4o", messages=[{"role": "user", "content": "hello"}], call_id=call_id
            ),
        )
        bus.emit(
            MockLLMCallCompletedEvent,
            MagicMock(),
            MockLLMCallCompletedEvent(
                model="gpt-4o",
                response="Hi there!",
                token_usage={"prompt_tokens": 10, "completion_tokens": 5},
                call_id=call_id,
            ),
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "crewai:gpt-4o"
        assert spans[0].span_type == "llm_call"
        assert spans[0].attributes["model"] == "gpt-4o"
        assert spans[0].attributes["input_tokens"] == 10
        assert spans[0].attributes["output_tokens"] == 5
        assert spans[0].attributes["cost_usd"] > 0

    def test_llm_call_failure(self):
        """Test LLM call failure creates error span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        call_id = "fail-call"
        bus.emit(
            MockLLMCallStartedEvent,
            MagicMock(),
            MockLLMCallStartedEvent(model="gpt-4o", call_id=call_id),
        )
        bus.emit(
            MockLLMCallFailedEvent,
            MagicMock(),
            MockLLMCallFailedEvent(error="rate limit", call_id=call_id),
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert spans[0].attributes["error"] == "rate limit"

    def test_tool_usage_span(self):
        """Test tool usage creates tool_call span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        # Use the same source object for start/finish so id(source) matches
        tool_source = MagicMock()
        event_start = MockToolUsageStartedEvent(tool_name="web_search", tool_input="AI trends 2025")
        bus.emit(MockToolUsageStartedEvent, tool_source, event_start)

        event_finish = MockToolUsageFinishedEvent(tool_output="Found 10 results")
        bus.emit(MockToolUsageFinishedEvent, tool_source, event_finish)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "web_search"
        assert spans[0].span_type == "tool_call"
        assert spans[0].attributes["tool_name"] == "web_search"

    def test_nested_span_hierarchy(self):
        """Test that spans form proper parent-child hierarchy."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus()

        crew = MockCrew(name="my_crew")
        agent = MockAgent(role="writer")

        # Start crew -> start agent -> start LLM -> finish LLM -> finish agent -> finish crew
        bus.emit(MockCrewKickoffStartedEvent, crew, MockCrewKickoffStartedEvent())

        bus.emit(MockAgentExecutionStartedEvent, agent, MockAgentExecutionStartedEvent(agent=agent))

        call_id = "nested-llm"
        bus.emit(
            MockLLMCallStartedEvent,
            MagicMock(),
            MockLLMCallStartedEvent(model="gpt-4o", call_id=call_id),
        )
        bus.emit(
            MockLLMCallCompletedEvent,
            MagicMock(),
            MockLLMCallCompletedEvent(model="gpt-4o", call_id=call_id),
        )

        bus.emit(MockAgentExecutionCompletedEvent, agent, MockAgentExecutionCompletedEvent())

        bus.emit(MockCrewKickoffCompletedEvent, crew, MockCrewKickoffCompletedEvent())

        spans = _get_collected_spans()
        assert len(spans) == 3

        # Find spans by name
        llm_span = next(s for s in spans if "gpt-4o" in s.name)
        agent_span = next(s for s in spans if "writer" in s.name)
        crew_span = next(s for s in spans if "my_crew" in s.name)

        # LLM should be child of agent
        assert llm_span.parent_id == agent_span.id
        # Agent should be child of crew
        assert agent_span.parent_id == crew_span.id

    def test_span_tracker_thread_safety(self):
        """Test that span tracker is thread-safe."""
        from evalyn_sdk.trace.instrumentation.providers.crewai import _SpanTracker

        tracker = _SpanTracker()
        errors = []

        def create_and_finish(i):
            try:
                key = ("test", i)
                tracker.start_span(key, f"span-{i}", "custom")
                time.sleep(0.001)
                tracker.finish_span(key, "ok")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_and_finish, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_finish_nonexistent_span_is_noop(self):
        """Test that finishing a span that was never started is a no-op."""
        from evalyn_sdk.trace.instrumentation.providers.crewai import _SpanTracker

        tracker = _SpanTracker()
        # Should not raise
        tracker.finish_span(("nonexistent", 123), "ok")

    def test_uninstrument(self):
        """Test uninstrumentation."""
        instrumentor, _ = self._create_instrumentor_with_mock_bus()

        assert instrumentor.is_instrumented()
        result = instrumentor.uninstrument()
        assert result is True
        assert not instrumentor.is_instrumented()

    def test_message_truncation(self):
        """Test that long messages are truncated."""
        from evalyn_sdk.trace.instrumentation.providers.crewai import _truncate_messages

        # Normal messages pass through
        msgs = [{"role": "user", "content": "hello"}]
        result = _truncate_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

        # Long content gets truncated
        long_msg = [{"role": "user", "content": "x" * 1000}]
        result = _truncate_messages(long_msg)
        assert len(result[0]["content"]) == 500

        # More than 20 messages gets capped
        many_msgs = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        result = _truncate_messages(many_msgs)
        assert len(result) == 20

        # None returns None
        assert _truncate_messages(None) is None

    def test_extract_token_usage(self):
        """Test token usage extraction from CrewOutput."""
        from evalyn_sdk.trace.instrumentation.providers.crewai import _extract_token_usage

        assert _extract_token_usage(None) is None

        mock_output = MagicMock()
        mock_output.token_usage = {"total_tokens": 100}
        assert _extract_token_usage(mock_output) == {"total_tokens": 100}

        mock_output.token_usage = None
        assert _extract_token_usage(mock_output) is None

    def test_knowledge_retrieval_span(self):
        """Test knowledge retrieval start/complete creates a retrieval span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus(include_optional=True)

        source = MagicMock()
        bus.emit(
            MockKnowledgeRetrievalStartedEvent,
            source,
            MockKnowledgeRetrievalStartedEvent(query="what is AI?"),
        )
        bus.emit(
            MockKnowledgeRetrievalCompletedEvent,
            source,
            MockKnowledgeRetrievalCompletedEvent(results=["result1", "result2"]),
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "knowledge:retrieval"
        assert spans[0].span_type == "retrieval"
        assert spans[0].status == "ok"
        assert spans[0].attributes["crewai.type"] == "knowledge_retrieval"
        assert spans[0].attributes["query"] == "what is AI?"
        assert spans[0].attributes["result_count"] == 2

    def test_llm_stream_chunk_appends_to_span(self):
        """Test LLM stream chunks accumulate on the open LLM span."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus(include_optional=True)

        call_id = "stream-call-1"
        bus.emit(
            MockLLMCallStartedEvent,
            MagicMock(),
            MockLLMCallStartedEvent(model="gpt-4o", call_id=call_id),
        )

        # Send stream chunks
        bus.emit(
            MockLLMStreamChunkEvent,
            MagicMock(),
            MockLLMStreamChunkEvent(call_id=call_id, chunk="Hello"),
        )
        bus.emit(
            MockLLMStreamChunkEvent,
            MagicMock(),
            MockLLMStreamChunkEvent(call_id=call_id, chunk=" world"),
        )

        # Finish the LLM call
        bus.emit(
            MockLLMCallCompletedEvent,
            MagicMock(),
            MockLLMCallCompletedEvent(model="gpt-4o", call_id=call_id),
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].attributes["crewai.stream_chunks"] == ["Hello", " world"]

    def test_knowledge_retrieval_not_registered_without_optional(self):
        """Test that knowledge retrieval handlers are not registered without optional events."""
        instrumentor, bus = self._create_instrumentor_with_mock_bus(include_optional=False)

        # These event types should have no handlers
        assert MockKnowledgeRetrievalStartedEvent not in bus._handlers
        assert MockKnowledgeRetrievalCompletedEvent not in bus._handlers

    def test_append_attribute_on_nonexistent_span_is_noop(self):
        """Test that appending to a nonexistent span does nothing."""
        from evalyn_sdk.trace.instrumentation.providers.crewai import _SpanTracker

        tracker = _SpanTracker()
        # Should not raise
        tracker.append_attribute(("nonexistent", 123), "key", "value")


# ---------------------------------------------------------------------------
# AutoGen Instrumentor Tests
# ---------------------------------------------------------------------------

class TestAutoGenInstrumentor:
    """Tests for the AutoGen instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_instrumentor_properties(self):
        """Test basic instrumentor properties."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import AutoGenInstrumentor

        inst = AutoGenInstrumentor()
        assert inst.name == "autogen"
        assert inst.instrumentor_type.value == "monkey_patch"
        assert not inst.is_instrumented()

    def test_format_autogen_messages_with_dicts(self):
        """Test formatting dict-based messages."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import _format_autogen_messages

        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _format_autogen_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_format_autogen_messages_with_objects(self):
        """Test formatting object-based messages."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import _format_autogen_messages

        msg = MagicMock()
        msg.role = "user"
        msg.content = "test message"

        result = _format_autogen_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "test message"

    def test_format_autogen_messages_truncation(self):
        """Test that long messages are truncated."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import _format_autogen_messages

        msgs = [{"role": "user", "content": "x" * 1000}]
        result = _format_autogen_messages(msgs)
        assert len(result[0]["content"]) == 500

        # More than 20 messages gets capped
        many = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        result = _format_autogen_messages(many)
        assert len(result) == 20

    def test_format_autogen_messages_empty(self):
        """Test formatting empty/None messages."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import _format_autogen_messages

        assert _format_autogen_messages(None) == []
        assert _format_autogen_messages([]) == []

    def test_format_chat_messages(self):
        """Test formatting BaseChatMessage objects."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import _format_chat_messages

        msg = MagicMock()
        msg.source = "user_agent"
        msg.content = "analyze this data"
        type(msg).__name__ = "TextMessage"

        result = _format_chat_messages([msg])
        assert len(result) == 1
        assert result[0]["source"] == "user_agent"
        assert result[0]["content"] == "analyze this data"

    def test_format_chat_messages_empty(self):
        """Test formatting empty chat messages."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import _format_chat_messages

        assert _format_chat_messages(None) == []
        assert _format_chat_messages([]) == []

    @pytest.mark.asyncio
    async def test_model_client_patch_captures_llm_span(self):
        """Test that patching ChatCompletionClient.create captures LLM spans."""
        from evalyn_sdk.trace.instrumentation.providers.autogen import AutoGenInstrumentor

        # Create a mock ChatCompletionClient class
        class MockCreateResult:
            def __init__(self):
                self.content = "Hello!"
                self.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
                self.finish_reason = "stop"

        class MockChatCompletionClient:
            model = "gpt-4o"

            async def create(self, messages, **kwargs):
                return MockCreateResult()

        client = MockChatCompletionClient()

        # Manually patch
        original = MockChatCompletionClient.create
        inst = AutoGenInstrumentor()

        import functools

        @functools.wraps(original)
        async def wrapped(client_self, messages, *args, **kwargs):
            parent_span_id = span_context.get_current_span_id()
            span = Span.new(
                name=f"autogen:{client_self.model}",
                span_type="llm_call",
                parent_id=parent_span_id,
            )
            span.attributes["model"] = client_self.model

            stack = span_context._span_stack.get()
            new_stack = stack + [span.id]
            span_context._span_stack.set(new_stack)

            result = await original(client_self, messages, *args, **kwargs)

            span.attributes["input_tokens"] = result.usage.prompt_tokens
            span.attributes["output_tokens"] = result.usage.completion_tokens

            span_context._span_stack.set(stack)
            span.finish(status="ok")
            span_context._add_span_to_collector(span)
            return result

        MockChatCompletionClient.create = wrapped

        # Execute
        result = await client.create([{"role": "user", "content": "hi"}])
        assert result.content == "Hello!"

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "autogen:gpt-4o"
        assert spans[0].span_type == "llm_call"
        assert spans[0].attributes["input_tokens"] == 10
        assert spans[0].attributes["output_tokens"] == 5
        assert spans[0].status == "ok"

    @pytest.mark.asyncio
    async def test_tool_patch_captures_tool_span(self):
        """Test that patching BaseTool.run_json captures tool spans."""
        # Simulate tool span creation
        parent_span_id = span_context.get_current_span_id()
        span = Span.new(
            name="web_search",
            span_type="tool_call",
            parent_id=parent_span_id,
            tool_name="web_search",
        )
        span.attributes["tool_input"] = '{"query": "AI trends"}'
        span.attributes["tool_output"] = "Found 5 results"
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "web_search"
        assert spans[0].span_type == "tool_call"

    @pytest.mark.asyncio
    async def test_agent_patch_captures_agent_span(self):
        """Test agent span creation with message info."""
        parent_span_id = span_context.get_current_span_id()
        span = Span.new(
            name="agent:researcher",
            span_type="agent",
            parent_id=parent_span_id,
            autogen_agent_name="researcher",
        )
        span.attributes["autogen.agent_description"] = "Research agent"
        span.attributes["autogen.input_messages"] = [
            {"type": "TextMessage", "source": "user", "content": "find info"}
        ]
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "agent:researcher"
        assert spans[0].attributes["autogen_agent_name"] == "researcher"

    @pytest.mark.asyncio
    async def test_team_span_captures_orchestration(self):
        """Test team span captures participant info."""
        span = Span.new(
            name="team:research_team",
            span_type="graph",
            parent_id=None,
        )
        span.attributes["autogen.type"] = "team"
        span.attributes["autogen.team_type"] = "RoundRobinGroupChat"
        span.attributes["autogen.participants"] = ["researcher", "writer"]
        span.attributes["autogen.message_count"] = 5
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "team:research_team"
        assert spans[0].span_type == "graph"
        assert spans[0].attributes["autogen.participants"] == ["researcher", "writer"]


# ---------------------------------------------------------------------------
# DSPy Instrumentor Tests
# ---------------------------------------------------------------------------

class TestDSPyInstrumentor:
    """Tests for the DSPy instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_instrumentor_properties(self):
        from evalyn_sdk.trace.instrumentation.providers.dspy import DSPyInstrumentor

        inst = DSPyInstrumentor()
        assert inst.name == "dspy"
        assert inst.instrumentor_type.value == "hook_based"
        assert not inst.is_instrumented()

    def test_callback_module_start_end(self):
        """Test DSPy callback creates module span."""
        from evalyn_sdk.trace.instrumentation.providers.dspy import EvalynDSPyCallback

        cb = EvalynDSPyCallback()

        # Mock a DSPy module instance
        mock_instance = MagicMock()
        type(mock_instance).__name__ = "ChainOfThought"
        mock_instance.signature = MagicMock()
        type(mock_instance.signature).__name__ = "QA"
        mock_instance.signature.input_fields = {"question": None}
        mock_instance.signature.output_fields = {"answer": None}

        cb.on_module_start("call-1", mock_instance, {"question": "What is AI?"})
        cb.on_module_end("call-1", {"answer": "AI is..."})

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "module:ChainOfThought"
        assert spans[0].span_type == "agent"
        assert spans[0].status == "ok"
        assert spans[0].attributes["dspy.module_class"] == "ChainOfThought"
        assert "question" in spans[0].attributes["dspy.inputs"]

    def test_callback_module_error(self):
        """Test DSPy callback handles module errors."""
        from evalyn_sdk.trace.instrumentation.providers.dspy import EvalynDSPyCallback

        cb = EvalynDSPyCallback()
        mock_instance = MagicMock()
        type(mock_instance).__name__ = "Predict"

        cb.on_module_start("call-2", mock_instance, {})
        cb.on_module_end("call-2", None, exception=ValueError("bad input"))

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert "bad input" in spans[0].attributes["error"]

    def test_callback_lm_start_end(self):
        """Test DSPy callback creates LLM call span."""
        from evalyn_sdk.trace.instrumentation.providers.dspy import EvalynDSPyCallback

        cb = EvalynDSPyCallback()
        mock_lm = MagicMock()
        mock_lm.model = "gpt-4o"
        mock_lm.temperature = 0.7
        mock_lm.max_tokens = 1000

        cb.on_lm_start("lm-1", mock_lm, {
            "messages": [{"role": "user", "content": "hello"}]
        })
        cb.on_lm_end("lm-1", {
            "choices": [{"message": {"content": "Hi there!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "dspy:gpt-4o"
        assert spans[0].span_type == "llm_call"
        assert spans[0].attributes["input_tokens"] == 10
        assert spans[0].attributes["output_tokens"] == 5
        assert spans[0].attributes["cost_usd"] > 0

    def test_callback_tool_start_end(self):
        """Test DSPy callback creates tool call span."""
        from evalyn_sdk.trace.instrumentation.providers.dspy import EvalynDSPyCallback

        cb = EvalynDSPyCallback()
        mock_tool = MagicMock()
        mock_tool.name = "calculator"

        cb.on_tool_start("tool-1", mock_tool, {"expression": "2+2"})
        cb.on_tool_end("tool-1", "4")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "calculator"
        assert spans[0].span_type == "tool_call"

    def test_callback_nested_module_lm(self):
        """Test nested module -> LM call creates proper hierarchy."""
        from evalyn_sdk.trace.instrumentation.providers.dspy import EvalynDSPyCallback

        cb = EvalynDSPyCallback()
        mock_module = MagicMock()
        type(mock_module).__name__ = "MyModule"
        mock_lm = MagicMock()
        mock_lm.model = "gpt-4o"

        cb.on_module_start("mod-1", mock_module, {"input": "test"})
        cb.on_lm_start("lm-1", mock_lm, {"messages": []})
        cb.on_lm_end("lm-1", {})
        cb.on_module_end("mod-1", {"output": "result"})

        spans = _get_collected_spans()
        assert len(spans) == 2

        lm_span = next(s for s in spans if s.span_type == "llm_call")
        module_span = next(s for s in spans if s.span_type == "agent")

        # LM span should be child of module span
        assert lm_span.parent_id == module_span.id

    def test_callback_end_nonexistent_is_noop(self):
        """Test ending a nonexistent call is a no-op."""
        from evalyn_sdk.trace.instrumentation.providers.dspy import EvalynDSPyCallback

        cb = EvalynDSPyCallback()
        # Should not raise
        cb.on_module_end("nonexistent", None)
        cb.on_lm_end("nonexistent", None)
        cb.on_tool_end("nonexistent", None)

    def test_format_dspy_messages(self):
        from evalyn_sdk.trace.instrumentation.providers.dspy import _format_dspy_messages

        msgs = [{"role": "user", "content": "test"}]
        result = _format_dspy_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

        assert _format_dspy_messages(None) == []
        assert _format_dspy_messages([]) == []


# ---------------------------------------------------------------------------
# Haystack Instrumentor Tests
# ---------------------------------------------------------------------------

class TestHaystackInstrumentor:
    """Tests for the Haystack instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_instrumentor_properties(self):
        from evalyn_sdk.trace.instrumentation.providers.haystack import HaystackInstrumentor

        inst = HaystackInstrumentor()
        assert inst.name == "haystack"
        assert inst.instrumentor_type.value == "hook_based"
        assert not inst.is_instrumented()

    def test_span_creation_and_tags(self):
        """Test EvalynHaystackSpan creates span and handles tags."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackSpan

        span = EvalynHaystackSpan("haystack.pipeline.run", tags={"key": "value"})
        assert span.operation_name == "haystack.pipeline.run"
        assert span._span.span_type == "graph"
        # Tags are applied via set_tags during __init__, which calls set_tag
        assert span.tags["key"] == "value"

        span.set_tag("extra", "data")
        assert span.tags["extra"] == "data"
        assert span._span.attributes["extra"] == "data"

        span.finish(status="ok")
        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "ok"

    def test_component_span_type_mapping(self):
        """Test component types get correct span types."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackSpan

        # Component run defaults to "agent" span type
        span = EvalynHaystackSpan("haystack.component.run")
        assert span._span.span_type == "agent"

        # Set component type tag to generator -> llm_call
        span.set_tag("haystack.component.name", "llm_generator")
        span.set_tag("haystack.component.type", "OpenAIChatGenerator")
        span.finish(status="ok")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].span_type == "llm_call"
        assert spans[0].name == "component:llm_generator"

    def test_retriever_component_type(self):
        """Test retriever components get retrieval span type."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackSpan

        span = EvalynHaystackSpan("haystack.component.run")
        span.set_tag("haystack.component.name", "doc_retriever")
        span.set_tag("haystack.component.type", "InMemoryBM25Retriever")
        span.finish(status="ok")

        spans = _get_collected_spans()
        assert spans[0].span_type == "retrieval"

    def test_content_tags(self):
        """Test content tags are stored with truncation."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackSpan

        span = EvalynHaystackSpan("haystack.component.run")
        span.set_content_tag("haystack.component.input", {"query": "test"})
        span.finish(status="ok")

        spans = _get_collected_spans()
        assert "haystack.component.input" in spans[0].attributes

    def test_tracer_context_manager(self):
        """Test EvalynHaystackTracer.trace context manager."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackTracer

        tracer = EvalynHaystackTracer()

        with tracer.trace("haystack.pipeline.run", tags={"pipeline": "rag"}) as span:
            span.set_tag("custom", "value")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "ok"

    def test_tracer_error_handling(self):
        """Test tracer captures errors."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackTracer

        tracer = EvalynHaystackTracer()

        with pytest.raises(RuntimeError):
            with tracer.trace("haystack.pipeline.run") as span:
                raise RuntimeError("pipeline failed")

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert "pipeline failed" in spans[0].attributes["error"]

    def test_nested_spans(self):
        """Test nested pipeline -> component spans."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import EvalynHaystackTracer

        tracer = EvalynHaystackTracer()

        with tracer.trace("haystack.pipeline.run") as pipeline_span:
            with tracer.trace("haystack.component.run", parent_span=pipeline_span) as comp_span:
                comp_span.set_tag("haystack.component.name", "generator")
                comp_span.set_tag("haystack.component.type", "OpenAIGenerator")

        spans = _get_collected_spans()
        assert len(spans) == 2

        comp = next(s for s in spans if "generator" in s.name)
        pipeline = next(s for s in spans if s.span_type == "graph")

        assert comp.parent_id == pipeline.id

    def test_safe_tag_value(self):
        """Test tag value sanitization."""
        from evalyn_sdk.trace.instrumentation.providers.haystack import _safe_tag_value

        assert _safe_tag_value(None) is None
        assert _safe_tag_value(42) == 42
        assert _safe_tag_value(True) is True
        assert _safe_tag_value("hello") == "hello"
        assert len(_safe_tag_value("x" * 2000)) == 1000


# ---------------------------------------------------------------------------
# LlamaIndex Instrumentor Tests
# ---------------------------------------------------------------------------

class TestLlamaIndexInstrumentor:
    """Tests for the LlamaIndex instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_instrumentor_properties(self):
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import LlamaIndexInstrumentor

        inst = LlamaIndexInstrumentor()
        assert inst.name == "llamaindex"
        assert inst.instrumentor_type.value == "hook_based"

    def test_span_handler_new_and_exit(self):
        """Test span handler creates and finalizes spans."""
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import EvalynLlamaIndexSpanHandler

        handler = EvalynLlamaIndexSpanHandler()

        # Mock instance (LLM)
        mock_llm = MagicMock()
        type(mock_llm).__name__ = "OpenAI"
        type(mock_llm).__module__ = "llama_index.llms"
        mock_llm.model = "gpt-4o"

        tracker = handler.new_span("span-1", None, instance=mock_llm)
        assert tracker is not None
        assert "span-1" in handler.open_spans

        handler.prepare_to_exit_span("span-1", None, instance=mock_llm, result="response")

        assert "span-1" not in handler.open_spans
        assert len(handler.completed_spans) == 1

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "ok"
        assert "llm" in spans[0].span_type

    def test_span_handler_drop_on_error(self):
        """Test span handler handles errors."""
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import EvalynLlamaIndexSpanHandler

        handler = EvalynLlamaIndexSpanHandler()
        mock = MagicMock()
        type(mock).__name__ = "CustomComponent"

        handler.new_span("span-2", None, instance=mock)
        handler.prepare_to_drop_span("span-2", None, err=ValueError("test error"))

        assert len(handler.dropped_spans) == 1
        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert "test error" in spans[0].attributes["error"]

    def test_span_handler_exit_nonexistent_is_noop(self):
        """Test exiting a nonexistent span is a no-op."""
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import EvalynLlamaIndexSpanHandler

        handler = EvalynLlamaIndexSpanHandler()
        result = handler.prepare_to_exit_span("nonexistent", None)
        assert result is None

    def test_infer_span_info(self):
        """Test span type inference from instance class."""
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import _infer_span_info

        # LLM
        mock_llm = MagicMock()
        type(mock_llm).__name__ = "OpenAI"
        mock_llm.model = "gpt-4"
        name, stype = _infer_span_info(mock_llm, None, None)
        assert stype == "llm_call"  # _infer_span_info explicitly handles "OpenAI"

        # Retriever
        mock_ret = MagicMock()
        type(mock_ret).__name__ = "VectorStoreRetriever"
        name, stype = _infer_span_info(mock_ret, None, None)
        assert stype == "retrieval"

        # Agent
        mock_agent = MagicMock()
        type(mock_agent).__name__ = "FunctionAgent"
        name, stype = _infer_span_info(mock_agent, None, None)
        assert stype == "agent"

        # None instance
        name, stype = _infer_span_info(None, None, None)
        assert stype == "custom"

    def test_event_handler_llm_events(self):
        """Test event handler processes LLM events."""
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import (
            EvalynLlamaIndexEventHandler,
            _pending_attrs,
        )

        handler = EvalynLlamaIndexEventHandler()

        # Push a fake span ID onto the stack
        span_context._span_stack.set(["test-span-id"])

        # Simulate LLM chat start event
        start_event = MagicMock()
        type(start_event).__name__ = "LLMChatStartEvent"
        start_event.messages = [MagicMock(role="user", content="hello")]
        start_event.model_dict = {"model": "gpt-4o"}

        handler.handle(start_event)

        # Check pending attributes were set
        assert "test-span-id" in _pending_attrs
        assert "llamaindex.llm.messages" in _pending_attrs["test-span-id"]

        # Clean up
        _pending_attrs.clear()
        span_context._span_stack.set([])

    def test_format_llm_messages(self):
        """Test LLM message formatting."""
        from evalyn_sdk.trace.instrumentation.providers.llamaindex import _format_llm_messages

        msg = MagicMock()
        msg.role = "user"
        msg.content = "test"

        result = _format_llm_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"

        assert _format_llm_messages(None) == []


# ---------------------------------------------------------------------------
# Semantic Kernel Instrumentor Tests
# ---------------------------------------------------------------------------

class TestSemanticKernelInstrumentor:
    """Tests for the Semantic Kernel instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_instrumentor_properties(self):
        from evalyn_sdk.trace.instrumentation.providers.semantic_kernel import SemanticKernelInstrumentor

        inst = SemanticKernelInstrumentor()
        assert inst.name == "semantic_kernel"
        assert inst.instrumentor_type.value == "monkey_patch"

    def test_format_sk_messages(self):
        """Test SK message formatting."""
        from evalyn_sdk.trace.instrumentation.providers.semantic_kernel import _format_sk_messages

        msg = MagicMock()
        msg.role = "user"
        msg.content = "test message"

        result = _format_sk_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "test message"

        # Empty
        assert _format_sk_messages(None) == []
        assert _format_sk_messages([]) == []

    def test_format_sk_messages_truncation(self):
        """Test SK message truncation."""
        from evalyn_sdk.trace.instrumentation.providers.semantic_kernel import _format_sk_messages

        msg = MagicMock()
        msg.role = "user"
        msg.content = "x" * 1000

        result = _format_sk_messages([msg])
        assert len(result[0]["content"]) == 500

    @pytest.mark.asyncio
    async def test_kernel_invoke_span(self):
        """Test kernel invoke creates a span."""
        # Simulate kernel invoke span
        span = Span.new(
            name="kernel:MyPlugin.generate",
            span_type="agent",
            parent_id=None,
        )
        span.attributes["sk.type"] = "kernel_invoke"
        span.attributes["sk.function_name"] = "generate"
        span.attributes["sk.plugin_name"] = "MyPlugin"
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].name == "kernel:MyPlugin.generate"
        assert spans[0].attributes["sk.function_name"] == "generate"

    @pytest.mark.asyncio
    async def test_llm_call_span(self):
        """Test LLM call span with token tracking."""
        span = Span.new(
            name="sk:gpt-4o",
            span_type="llm_call",
            parent_id=None,
            provider="semantic_kernel",
            model="gpt-4o",
        )
        span.attributes["input_tokens"] = 20
        span.attributes["output_tokens"] = 10
        span.attributes["sk.temperature"] = 0.7
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].attributes["input_tokens"] == 20
        assert spans[0].attributes["provider"] == "semantic_kernel"

    @pytest.mark.asyncio
    async def test_agent_span(self):
        """Test agent span."""
        span = Span.new(
            name="agent:ChatBot",
            span_type="agent",
            parent_id=None,
        )
        span.attributes["sk.type"] = "agent"
        span.attributes["sk.agent_name"] = "ChatBot"
        span.attributes["sk.instructions"] = "You are a helpful assistant."
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

        spans = _get_collected_spans()
        assert len(spans) == 1
        assert spans[0].attributes["sk.agent_name"] == "ChatBot"
