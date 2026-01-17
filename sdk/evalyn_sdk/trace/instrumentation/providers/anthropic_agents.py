"""
Anthropic Agent SDK instrumentor.

Uses hook-based instrumentation with EvalynAgentHooks.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

from ..base import Instrumentor, InstrumentorType
from ....models import Span
from ... import context as span_context


@dataclass
class SpanState:
    """Track span state across hook callbacks."""

    span: Span
    start_time: float


class EvalynAgentHooks:
    """
    Hook adapter for Anthropic Agent SDK.

    Captures agent execution, tool calls, and model calls as spans.

    Usage:
        from evalyn_sdk.trace.instrumentation import create_agent_hooks

        hooks = create_agent_hooks()
        agent = Agent(hooks=hooks)
    """

    def __init__(self, user_hooks: Optional[Any] = None):
        """
        Initialize hooks.

        Args:
            user_hooks: Optional user-provided hooks to compose with.
                        Our hooks will run before user hooks.
        """
        self._user_hooks = user_hooks
        self._span_stack: List[SpanState] = []
        self._tool_spans: Dict[str, SpanState] = {}

    # Agent lifecycle hooks
    async def on_agent_start(self, context: Any, agent: Any) -> None:
        """Called when agent starts execution."""
        agent_name = getattr(agent, "name", "agent")

        parent_id = span_context.get_current_span_id()
        span = Span.new(
            name=f"agent:{agent_name}",
            span_type="agent",
            parent_id=parent_id,
            agent_name=agent_name,
        )

        # Push to span stack so child spans have correct parent
        stack = span_context._span_stack.get()
        span_context._span_stack.set(stack + [span.id])

        self._span_stack.append(SpanState(span=span, start_time=time.time()))

        # Call user hooks
        if self._user_hooks and hasattr(self._user_hooks, "on_agent_start"):
            await self._user_hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Called when agent finishes execution."""
        if self._span_stack:
            state = self._span_stack.pop()
            state.span.finish(status="ok", output=str(output)[:500])
            span_context._add_span_to_collector(state.span)

            # Pop from span stack
            stack = span_context._span_stack.get()
            if stack and stack[-1] == state.span.id:
                span_context._span_stack.set(stack[:-1])

        if self._user_hooks and hasattr(self._user_hooks, "on_agent_end"):
            await self._user_hooks.on_agent_end(context, agent, output)

    async def on_agent_error(self, context: Any, agent: Any, error: Exception) -> None:
        """Called when agent encounters an error."""
        if self._span_stack:
            state = self._span_stack.pop()
            state.span.finish(status="error", error=str(error))
            span_context._add_span_to_collector(state.span)

            stack = span_context._span_stack.get()
            if stack and stack[-1] == state.span.id:
                span_context._span_stack.set(stack[:-1])

        if self._user_hooks and hasattr(self._user_hooks, "on_agent_error"):
            await self._user_hooks.on_agent_error(context, agent, error)

    # Tool hooks
    async def on_tool_start(self, context: Any, tool: Any, input_data: Any) -> None:
        """Called before tool execution."""
        tool_name = getattr(tool, "name", str(tool))
        tool_id = f"{id(tool)}_{time.time()}"

        parent_id = span_context.get_current_span_id()
        span = Span.new(
            name=tool_name,
            span_type="tool_call",
            parent_id=parent_id,
            tool_name=tool_name,
            input=str(input_data)[:500],
        )

        self._tool_spans[tool_id] = SpanState(span=span, start_time=time.time())

        # Store tool_id on context for retrieval in on_tool_end
        if hasattr(context, "_evalyn_tool_id"):
            context._evalyn_tool_id = tool_id

        if self._user_hooks and hasattr(self._user_hooks, "on_tool_start"):
            await self._user_hooks.on_tool_start(context, tool, input_data)

    async def on_tool_end(self, context: Any, tool: Any, output: Any) -> None:
        """Called after tool execution."""
        tool_id = getattr(context, "_evalyn_tool_id", None)

        # Try to find matching span
        state = None
        if tool_id and tool_id in self._tool_spans:
            state = self._tool_spans.pop(tool_id)
        elif self._tool_spans:
            # Fall back to most recent
            tool_id = next(iter(self._tool_spans))
            state = self._tool_spans.pop(tool_id)

        if state:
            duration_ms = (time.time() - state.start_time) * 1000
            state.span.start_time = state.span.start_time - timedelta(
                milliseconds=duration_ms
            )
            state.span.finish(status="ok", output=str(output)[:500])
            span_context._add_span_to_collector(state.span)

        if self._user_hooks and hasattr(self._user_hooks, "on_tool_end"):
            await self._user_hooks.on_tool_end(context, tool, output)

    async def on_tool_error(self, context: Any, tool: Any, error: Exception) -> None:
        """Called when tool execution fails."""
        tool_id = getattr(context, "_evalyn_tool_id", None)

        state = None
        if tool_id and tool_id in self._tool_spans:
            state = self._tool_spans.pop(tool_id)
        elif self._tool_spans:
            tool_id = next(iter(self._tool_spans))
            state = self._tool_spans.pop(tool_id)

        if state:
            state.span.finish(status="error", error=str(error))
            span_context._add_span_to_collector(state.span)

        if self._user_hooks and hasattr(self._user_hooks, "on_tool_error"):
            await self._user_hooks.on_tool_error(context, tool, error)

    # Model hooks
    async def on_model_start(self, context: Any, model: str, messages: Any) -> None:
        """Called before model invocation."""
        parent_id = span_context.get_current_span_id()
        span = Span.new(
            name=f"anthropic:{model}",
            span_type="llm_call",
            parent_id=parent_id,
            provider="anthropic",
            model=model,
        )

        self._span_stack.append(SpanState(span=span, start_time=time.time()))

        if self._user_hooks and hasattr(self._user_hooks, "on_model_start"):
            await self._user_hooks.on_model_start(context, model, messages)

    async def on_model_end(self, context: Any, response: Any) -> None:
        """Called after model invocation."""
        if self._span_stack:
            state = self._span_stack.pop()
            duration_ms = (time.time() - state.start_time) * 1000

            # Extract token usage if available
            usage = getattr(response, "usage", None)
            if usage:
                state.span.attributes["input_tokens"] = getattr(
                    usage, "input_tokens", 0
                )
                state.span.attributes["output_tokens"] = getattr(
                    usage, "output_tokens", 0
                )

            state.span.start_time = state.span.start_time - timedelta(
                milliseconds=duration_ms
            )
            state.span.finish(status="ok")
            span_context._add_span_to_collector(state.span)

        if self._user_hooks and hasattr(self._user_hooks, "on_model_end"):
            await self._user_hooks.on_model_end(context, response)


class AnthropicAgentsInstrumentor(Instrumentor):
    """
    Instrumentor for Anthropic Agent SDK.

    Uses hook-based approach: creates EvalynAgentHooks that users
    must explicitly pass to their Agent.
    """

    _hooks: Optional[EvalynAgentHooks] = None

    @property
    def name(self) -> str:
        return "anthropic_agents"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.HOOK_BASED

    def is_available(self) -> bool:
        # Check for anthropic agent SDK
        # First check if anthropic is installed
        if importlib.util.find_spec("anthropic") is None:
            return False
        # Then check for the agents submodule
        try:
            return importlib.util.find_spec("anthropic.agents") is not None
        except (ModuleNotFoundError, ImportError):
            return False

    def is_instrumented(self) -> bool:
        # Hook-based instrumentors are "instrumented" when hooks are created
        return self._hooks is not None

    def instrument(self) -> bool:
        if not self.is_available():
            return False

        # Create hooks adapter
        self._hooks = EvalynAgentHooks()
        return True

    def uninstrument(self) -> bool:
        self._hooks = None
        return True

    def get_hooks(self) -> Optional[EvalynAgentHooks]:
        """Get the hooks adapter to pass to an Agent."""
        if self._hooks is None:
            self.instrument()
        return self._hooks


def create_agent_hooks(user_hooks: Optional[Any] = None) -> EvalynAgentHooks:
    """
    Create Evalyn agent hooks for Anthropic Agent SDK.

    Usage:
        from evalyn_sdk.trace.instrumentation import create_agent_hooks
        from anthropic.agents import Agent

        hooks = create_agent_hooks()
        agent = Agent(hooks=hooks)

    Args:
        user_hooks: Optional user-provided hooks to compose with.

    Returns:
        EvalynAgentHooks instance
    """
    return EvalynAgentHooks(user_hooks=user_hooks)
