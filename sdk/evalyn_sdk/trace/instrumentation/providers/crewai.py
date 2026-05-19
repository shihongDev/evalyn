"""
CrewAI framework instrumentor.

Uses CrewAI's BaseEventListener pattern (official public API) to capture:
- Crew kickoff lifecycle (start/complete/fail)
- Agent execution (start/complete/error)
- Task execution (start/complete/fail)
- LLM calls (start/complete/fail) with token usage
- LLM streaming chunks
- Tool usage (start/finish/error)
- Memory operations (query/save)
- Knowledge retrieval (start/complete)
- Flow execution (start/finish/fail)

Span hierarchy:
    crew:kickoff
      -> agent:{role}
        -> task:{description}
          -> llm_call:{model}
          -> tool_call:{tool_name}
          -> memory:query / memory:save
          -> knowledge:retrieval
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

from ....models import Span
from ... import context as span_context
from ..base import Instrumentor, InstrumentorType
from ._shared import calculate_cost

# ---------------------------------------------------------------------------
# Resilient imports: prefer crewai.events (public API), fall back to
# crewai.utilities.events.* (older versions).
# ---------------------------------------------------------------------------

def _import_event_bus() -> Any:
    """Import crewai_event_bus from the best available path."""
    try:
        from crewai.events import crewai_event_bus
        return crewai_event_bus
    except ImportError:
        pass
    try:
        from crewai.utilities.events.event_listener import crewai_event_bus
        return crewai_event_bus
    except ImportError:
        return None


def _import_base_event_listener() -> Any:
    """Import BaseEventListener from the best available path."""
    try:
        from crewai.events import BaseEventListener
        return BaseEventListener
    except ImportError:
        pass
    return None


def _import_core_events() -> dict[str, type] | None:
    """Import core event classes. Returns dict of name->class or None."""
    try:
        from crewai.events import (
            AgentExecutionCompletedEvent,
            AgentExecutionStartedEvent,
            CrewKickoffCompletedEvent,
            CrewKickoffStartedEvent,
            LLMCallCompletedEvent,
            LLMCallStartedEvent,
            TaskCompletedEvent,
            TaskStartedEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )
        return {
            "CrewKickoffStartedEvent": CrewKickoffStartedEvent,
            "CrewKickoffCompletedEvent": CrewKickoffCompletedEvent,
            "AgentExecutionStartedEvent": AgentExecutionStartedEvent,
            "AgentExecutionCompletedEvent": AgentExecutionCompletedEvent,
            "TaskStartedEvent": TaskStartedEvent,
            "TaskCompletedEvent": TaskCompletedEvent,
            "ToolUsageStartedEvent": ToolUsageStartedEvent,
            "ToolUsageFinishedEvent": ToolUsageFinishedEvent,
            "LLMCallStartedEvent": LLMCallStartedEvent,
            "LLMCallCompletedEvent": LLMCallCompletedEvent,
        }
    except ImportError:
        pass

    # Fallback to old internal import paths
    try:
        from crewai.utilities.events.agent_events import (
            AgentExecutionCompletedEvent,
            AgentExecutionErrorEvent,
            AgentExecutionStartedEvent,
        )
        from crewai.utilities.events.crew_events import (
            CrewKickoffCompletedEvent,
            CrewKickoffFailedEvent,
            CrewKickoffStartedEvent,
        )
        from crewai.utilities.events.llm_events import (
            LLMCallCompletedEvent,
            LLMCallFailedEvent,
            LLMCallStartedEvent,
        )
        from crewai.utilities.events.task_events import (
            TaskCompletedEvent,
            TaskFailedEvent,
            TaskStartedEvent,
        )
        from crewai.utilities.events.tool_events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )
        return {
            "CrewKickoffStartedEvent": CrewKickoffStartedEvent,
            "CrewKickoffCompletedEvent": CrewKickoffCompletedEvent,
            "CrewKickoffFailedEvent": CrewKickoffFailedEvent,
            "AgentExecutionStartedEvent": AgentExecutionStartedEvent,
            "AgentExecutionCompletedEvent": AgentExecutionCompletedEvent,
            "AgentExecutionErrorEvent": AgentExecutionErrorEvent,
            "TaskStartedEvent": TaskStartedEvent,
            "TaskCompletedEvent": TaskCompletedEvent,
            "TaskFailedEvent": TaskFailedEvent,
            "LLMCallStartedEvent": LLMCallStartedEvent,
            "LLMCallCompletedEvent": LLMCallCompletedEvent,
            "LLMCallFailedEvent": LLMCallFailedEvent,
            "ToolUsageStartedEvent": ToolUsageStartedEvent,
            "ToolUsageFinishedEvent": ToolUsageFinishedEvent,
            "ToolUsageErrorEvent": ToolUsageErrorEvent,
        }
    except ImportError:
        return None


def _import_optional_events() -> dict[str, type]:
    """Import optional event classes that may not exist in all versions."""
    events: dict[str, type] = {}

    # Memory events
    try:
        from crewai.events import MemoryQueryCompletedEvent
        events["MemoryQueryCompletedEvent"] = MemoryQueryCompletedEvent
    except ImportError:
        try:
            from crewai.utilities.events.memory_events import (
                MemoryQueryCompleted,
                MemoryQueryStarted,
                MemorySaveCompleted,
                MemorySaveStarted,
            )
            events["MemoryQueryStarted"] = MemoryQueryStarted
            events["MemoryQueryCompleted"] = MemoryQueryCompleted
            events["MemorySaveStarted"] = MemorySaveStarted
            events["MemorySaveCompleted"] = MemorySaveCompleted
        except ImportError:
            pass

    # Knowledge retrieval events
    try:
        from crewai.events import (
            KnowledgeRetrievalCompletedEvent,
            KnowledgeRetrievalStartedEvent,
        )
        events["KnowledgeRetrievalStartedEvent"] = KnowledgeRetrievalStartedEvent
        events["KnowledgeRetrievalCompletedEvent"] = KnowledgeRetrievalCompletedEvent
    except ImportError:
        pass

    # LLM stream chunk event
    try:
        from crewai.events import LLMStreamChunkEvent
        events["LLMStreamChunkEvent"] = LLMStreamChunkEvent
    except ImportError:
        pass

    # Flow events
    try:
        from crewai.utilities.events.flow_events import (
            FlowFinishedEvent,
            FlowStartedEvent,
        )
        events["FlowStartedEvent"] = FlowStartedEvent
        events["FlowFinishedEvent"] = FlowFinishedEvent
    except ImportError:
        pass

    return events


class CrewAIInstrumentor(Instrumentor):
    """Instrumentor for CrewAI framework via its event bus system."""

    _instrumented = False

    @property
    def name(self) -> str:
        return "crewai"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.HOOK_BASED

    def is_available(self) -> bool:
        return importlib.util.find_spec("crewai") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        event_bus = _import_event_bus()
        if event_bus is None:
            return False

        self._span_tracker = _SpanTracker()
        self._register_event_handlers(event_bus)
        self._instrumented = True
        return True

    def _register_event_handlers(self, event_bus: Any) -> None:
        """Register handlers for all CrewAI event types."""
        tracker = self._span_tracker

        core = _import_core_events()
        if core is None:
            return

        optional = _import_optional_events()

        # Crew lifecycle
        CrewKickoffStartedEvent = core.get("CrewKickoffStartedEvent")
        CrewKickoffCompletedEvent = core.get("CrewKickoffCompletedEvent")
        CrewKickoffFailedEvent = core.get("CrewKickoffFailedEvent")

        if CrewKickoffStartedEvent:
            @event_bus.on(CrewKickoffStartedEvent)
            def on_crew_start(source: Any, event: Any) -> None:
                crew_name = getattr(source, "name", None) or "crew"
                inputs = getattr(event, "inputs", None) or {}
                tracker.start_span(
                    key=("crew", id(source)),
                    name=f"crew:{crew_name}",
                    span_type="agent",
                    attributes={
                        "crewai.type": "crew",
                        "crewai.crew_name": crew_name,
                        "crewai.inputs": str(inputs)[:500],
                    },
                )

        if CrewKickoffCompletedEvent:
            @event_bus.on(CrewKickoffCompletedEvent)
            def on_crew_complete(source: Any, event: Any) -> None:
                output = getattr(event, "output", None)
                tracker.finish_span(
                    key=("crew", id(source)),
                    status="ok",
                    attributes={
                        "crewai.output": str(output)[:1000] if output else None,
                        "crewai.token_usage": _extract_token_usage(output),
                    },
                )

        if CrewKickoffFailedEvent:
            @event_bus.on(CrewKickoffFailedEvent)
            def on_crew_fail(source: Any, event: Any) -> None:
                error = getattr(event, "error", None)
                tracker.finish_span(
                    key=("crew", id(source)),
                    status="error",
                    attributes={"error": str(error) if error else "unknown"},
                )

        # Agent lifecycle
        AgentExecutionStartedEvent = core.get("AgentExecutionStartedEvent")
        AgentExecutionCompletedEvent = core.get("AgentExecutionCompletedEvent")
        AgentExecutionErrorEvent = core.get("AgentExecutionErrorEvent")

        if AgentExecutionStartedEvent:
            @event_bus.on(AgentExecutionStartedEvent)
            def on_agent_start(source: Any, event: Any) -> None:
                agent = getattr(event, "agent", source)
                role = getattr(agent, "role", "agent")
                goal = getattr(agent, "goal", "")
                tracker.start_span(
                    key=("agent", id(source)),
                    name=f"agent:{role}",
                    span_type="agent",
                    attributes={
                        "crewai.type": "agent",
                        "crewai.agent_role": str(role),
                        "crewai.agent_goal": str(goal)[:500],
                    },
                )

        if AgentExecutionCompletedEvent:
            @event_bus.on(AgentExecutionCompletedEvent)
            def on_agent_complete(source: Any, event: Any) -> None:
                output = getattr(event, "output", None)
                tracker.finish_span(
                    key=("agent", id(source)),
                    status="ok",
                    attributes={
                        "crewai.agent_output": str(output)[:1000] if output else None,
                    },
                )

        if AgentExecutionErrorEvent:
            @event_bus.on(AgentExecutionErrorEvent)
            def on_agent_error(source: Any, event: Any) -> None:
                error = getattr(event, "error", None)
                tracker.finish_span(
                    key=("agent", id(source)),
                    status="error",
                    attributes={"error": str(error) if error else "unknown"},
                )

        # Task lifecycle
        TaskStartedEvent = core.get("TaskStartedEvent")
        TaskCompletedEvent = core.get("TaskCompletedEvent")
        TaskFailedEvent = core.get("TaskFailedEvent")

        if TaskStartedEvent:
            @event_bus.on(TaskStartedEvent)
            def on_task_start(source: Any, event: Any) -> None:
                task = getattr(event, "task", source)
                description = getattr(task, "description", "task")
                tracker.start_span(
                    key=("task", id(source)),
                    name=f"task:{str(description)[:60]}",
                    span_type="agent",
                    attributes={
                        "crewai.type": "task",
                        "crewai.task_description": str(description)[:500],
                        "crewai.expected_output": str(
                            getattr(task, "expected_output", "")
                        )[:500],
                    },
                )

        if TaskCompletedEvent:
            @event_bus.on(TaskCompletedEvent)
            def on_task_complete(source: Any, event: Any) -> None:
                output = getattr(event, "output", None)
                tracker.finish_span(
                    key=("task", id(source)),
                    status="ok",
                    attributes={
                        "crewai.task_output": str(output)[:1000] if output else None,
                    },
                )

        if TaskFailedEvent:
            @event_bus.on(TaskFailedEvent)
            def on_task_fail(source: Any, event: Any) -> None:
                error = getattr(event, "error", None)
                tracker.finish_span(
                    key=("task", id(source)),
                    status="error",
                    attributes={"error": str(error) if error else "unknown"},
                )

        # LLM calls
        LLMCallStartedEvent = core.get("LLMCallStartedEvent")
        LLMCallCompletedEvent = core.get("LLMCallCompletedEvent")
        LLMCallFailedEvent = core.get("LLMCallFailedEvent")

        if LLMCallStartedEvent:
            @event_bus.on(LLMCallStartedEvent)
            def on_llm_start(source: Any, event: Any) -> None:
                model = getattr(event, "model", None) or getattr(
                    source, "model", "unknown"
                )
                messages = getattr(event, "messages", None)
                call_id = getattr(event, "call_id", None) or id(event)
                tracker.start_span(
                    key=("llm", call_id),
                    name=f"crewai:{model}",
                    span_type="llm_call",
                    attributes={
                        "crewai.type": "llm_call",
                        "provider": "crewai",
                        "model": str(model),
                        "request": {
                            "messages": _truncate_messages(messages),
                        }
                        if messages
                        else {},
                    },
                )

        if LLMCallCompletedEvent:
            @event_bus.on(LLMCallCompletedEvent)
            def on_llm_complete(source: Any, event: Any) -> None:
                call_id = getattr(event, "call_id", None) or id(event)
                model = getattr(event, "model", None) or getattr(
                    source, "model", "unknown"
                )
                response = getattr(event, "response", None)
                token_usage = getattr(event, "token_usage", None) or {}

                input_tokens = 0
                output_tokens = 0
                if isinstance(token_usage, dict):
                    input_tokens = token_usage.get(
                        "prompt_tokens", token_usage.get("input_tokens", 0)
                    )
                    output_tokens = token_usage.get(
                        "completion_tokens", token_usage.get("output_tokens", 0)
                    )

                cost = calculate_cost(str(model), input_tokens, output_tokens)

                tracker.finish_span(
                    key=("llm", call_id),
                    status="ok",
                    attributes={
                        "model": str(model),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost,
                        "response": {
                            "content": str(response)[:1000],
                        }
                        if response
                        else {},
                    },
                )

        if LLMCallFailedEvent:
            @event_bus.on(LLMCallFailedEvent)
            def on_llm_fail(source: Any, event: Any) -> None:
                call_id = getattr(event, "call_id", None) or id(event)
                error = getattr(event, "error", None)
                tracker.finish_span(
                    key=("llm", call_id),
                    status="error",
                    attributes={"error": str(error) if error else "unknown"},
                )

        # Tool usage - keyed by source id for consistent start/finish matching
        ToolUsageStartedEvent = core.get("ToolUsageStartedEvent")
        ToolUsageFinishedEvent = core.get("ToolUsageFinishedEvent")
        ToolUsageErrorEvent = core.get("ToolUsageErrorEvent")

        if ToolUsageStartedEvent:
            @event_bus.on(ToolUsageStartedEvent)
            def on_tool_start(source: Any, event: Any) -> None:
                tool_name = getattr(event, "tool_name", None) or getattr(
                    event, "name", "unknown"
                )
                tool_input = getattr(event, "tool_input", None) or getattr(
                    event, "input", None
                )
                tracker.start_span(
                    key=("tool", id(source)),
                    name=str(tool_name),
                    span_type="tool_call",
                    attributes={
                        "crewai.type": "tool_call",
                        "tool_name": str(tool_name),
                        "tool_input": str(tool_input)[:1000] if tool_input else None,
                    },
                )

        if ToolUsageFinishedEvent:
            @event_bus.on(ToolUsageFinishedEvent)
            def on_tool_finish(source: Any, event: Any) -> None:
                tool_output = getattr(event, "tool_output", None) or getattr(
                    event, "output", None
                )
                tracker.finish_span(
                    key=("tool", id(source)),
                    status="ok",
                    attributes={
                        "tool_output": str(tool_output)[:1000] if tool_output else None,
                    },
                )

        if ToolUsageErrorEvent:
            @event_bus.on(ToolUsageErrorEvent)
            def on_tool_error(source: Any, event: Any) -> None:
                error = getattr(event, "error", None)
                tracker.finish_span(
                    key=("tool", id(source)),
                    status="error",
                    attributes={"error": str(error) if error else "unknown"},
                )

        # --- Optional events (may not exist in all CrewAI versions) ---

        # LLM stream chunks
        LLMStreamChunkEvent = optional.get("LLMStreamChunkEvent")
        if LLMStreamChunkEvent:
            @event_bus.on(LLMStreamChunkEvent)
            def on_llm_stream_chunk(source: Any, event: Any) -> None:
                call_id = getattr(event, "call_id", None) or id(source)
                chunk = getattr(event, "chunk", None) or ""
                # Append chunk to the open LLM span's streamed content
                tracker.append_attribute(
                    key=("llm", call_id),
                    attr_key="crewai.stream_chunks",
                    value=str(chunk),
                )

        # Knowledge retrieval events
        KnowledgeRetrievalStartedEvent = optional.get("KnowledgeRetrievalStartedEvent")
        KnowledgeRetrievalCompletedEvent = optional.get("KnowledgeRetrievalCompletedEvent")

        if KnowledgeRetrievalStartedEvent:
            @event_bus.on(KnowledgeRetrievalStartedEvent)
            def on_knowledge_retrieval_start(source: Any, event: Any) -> None:
                query = getattr(event, "query", None) or ""
                tracker.start_span(
                    key=("knowledge", id(source)),
                    name="knowledge:retrieval",
                    span_type="retrieval",
                    attributes={
                        "crewai.type": "knowledge_retrieval",
                        "query": str(query)[:500],
                    },
                )

        if KnowledgeRetrievalCompletedEvent:
            @event_bus.on(KnowledgeRetrievalCompletedEvent)
            def on_knowledge_retrieval_complete(source: Any, event: Any) -> None:
                results = getattr(event, "results", None)
                tracker.finish_span(
                    key=("knowledge", id(source)),
                    status="ok",
                    attributes={
                        "result_count": len(results) if results else 0,
                    },
                )

        # Memory events (old-style from crewai.utilities.events.memory_events)
        MemoryQueryStarted = optional.get("MemoryQueryStarted")
        MemoryQueryCompleted = optional.get("MemoryQueryCompleted")
        MemorySaveStarted = optional.get("MemorySaveStarted")
        MemorySaveCompleted = optional.get("MemorySaveCompleted")
        MemoryQueryCompletedEvent = optional.get("MemoryQueryCompletedEvent")

        if MemoryQueryStarted:
            @event_bus.on(MemoryQueryStarted)
            def on_memory_query_start(source: Any, event: Any) -> None:
                tracker.start_span(
                    key=("memory_query", id(source)),
                    name="memory:query",
                    span_type="retrieval",
                    attributes={
                        "crewai.type": "memory_query",
                        "query": str(getattr(event, "query", ""))[:500],
                    },
                )

        if MemoryQueryCompleted:
            @event_bus.on(MemoryQueryCompleted)
            def on_memory_query_complete(source: Any, event: Any) -> None:
                results = getattr(event, "results", None)
                tracker.finish_span(
                    key=("memory_query", id(source)),
                    status="ok",
                    attributes={
                        "result_count": len(results) if results else 0,
                    },
                )

        if MemoryQueryCompletedEvent and not MemoryQueryCompleted:
            # New-style public API memory event
            @event_bus.on(MemoryQueryCompletedEvent)
            def on_memory_query_complete_new(source: Any, event: Any) -> None:
                results = getattr(event, "results", None)
                tracker.finish_span(
                    key=("memory_query", id(source)),
                    status="ok",
                    attributes={
                        "result_count": len(results) if results else 0,
                    },
                )

        if MemorySaveStarted:
            @event_bus.on(MemorySaveStarted)
            def on_memory_save_start(source: Any, event: Any) -> None:
                tracker.start_span(
                    key=("memory_save", id(source)),
                    name="memory:save",
                    span_type="custom",
                    attributes={"crewai.type": "memory_save"},
                )

        if MemorySaveCompleted:
            @event_bus.on(MemorySaveCompleted)
            def on_memory_save_complete(source: Any, event: Any) -> None:
                tracker.finish_span(
                    key=("memory_save", id(source)),
                    status="ok",
                )

        # Flow events
        FlowStartedEvent = optional.get("FlowStartedEvent")
        FlowFinishedEvent = optional.get("FlowFinishedEvent")

        if FlowStartedEvent:
            @event_bus.on(FlowStartedEvent)
            def on_flow_start(source: Any, event: Any) -> None:
                flow_name = getattr(source, "name", None) or getattr(
                    source, "__class__", type(source)
                ).__name__
                tracker.start_span(
                    key=("flow", id(source)),
                    name=f"flow:{flow_name}",
                    span_type="graph",
                    attributes={
                        "crewai.type": "flow",
                        "crewai.flow_name": str(flow_name),
                    },
                )

        if FlowFinishedEvent:
            @event_bus.on(FlowFinishedEvent)
            def on_flow_finish(source: Any, event: Any) -> None:
                result = getattr(event, "result", None)
                tracker.finish_span(
                    key=("flow", id(source)),
                    status="ok",
                    attributes={
                        "crewai.flow_result": str(result)[:1000]
                        if result
                        else None,
                    },
                )

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        # CrewAI event bus does not provide a clean unsubscribe API
        # for decorator-registered handlers, but we mark as uninstrumented
        self._instrumented = False
        self._span_tracker = None
        return True


class _SpanTracker:
    """Thread-safe tracker for mapping event keys to open spans."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_spans: dict[tuple, _OpenSpan] = {}

    def has_span(self, key: tuple) -> bool:
        with self._lock:
            return key in self._open_spans

    def start_span(
        self,
        key: tuple,
        name: str,
        span_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        parent_span_id = span_context.get_current_span_id()
        span = Span.new(
            name=name,
            span_type=span_type,
            parent_id=parent_span_id,
        )
        if attributes:
            span.attributes.update(
                {k: v for k, v in attributes.items() if v is not None}
            )

        # Push onto span stack so child events become children
        stack = span_context._span_stack.get()
        new_stack = stack + [span.id]
        span_context._span_stack.set(new_stack)

        with self._lock:
            self._open_spans[key] = _OpenSpan(span=span, prev_stack=stack)

    def finish_span(
        self,
        key: tuple,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            open_span = self._open_spans.pop(key, None)

        if open_span is None:
            return

        if attributes:
            open_span.span.attributes.update(
                {k: v for k, v in attributes.items() if v is not None}
            )

        # Restore span stack
        span_context._span_stack.set(open_span.prev_stack)

        open_span.span.finish(status=status)
        span_context._add_span_to_collector(open_span.span)

    def append_attribute(
        self,
        key: tuple,
        attr_key: str,
        value: str,
    ) -> None:
        """Append a value to a list attribute on an open span (thread-safe)."""
        with self._lock:
            open_span = self._open_spans.get(key)
            if open_span is None:
                return
            existing = open_span.span.attributes.get(attr_key, [])
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(value)
            open_span.span.attributes[attr_key] = existing


class _OpenSpan:
    """Internal state for a span that has been started but not finished."""

    __slots__ = ("span", "prev_stack")

    def __init__(self, span: Span, prev_stack: list) -> None:
        self.span = span
        self.prev_stack = prev_stack


def _truncate_messages(messages: Any) -> Any:
    """Truncate message content for storage."""
    if not messages:
        return messages
    if not isinstance(messages, (list, tuple)):
        return str(messages)[:1000]

    truncated = []
    for msg in messages[:20]:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            truncated.append(
                {
                    "role": msg.get("role", "unknown"),
                    "content": str(content)[:500] if content else "",
                }
            )
        else:
            truncated.append(str(msg)[:500])
    return truncated


def _extract_token_usage(output: Any) -> dict[str, Any] | None:
    """Extract token usage from CrewOutput."""
    if output is None:
        return None
    token_usage = getattr(output, "token_usage", None)
    if token_usage is None:
        return None
    if isinstance(token_usage, dict):
        return token_usage
    # Convert non-dict token usage to a dict with a raw representation
    return {"raw": str(token_usage)}
