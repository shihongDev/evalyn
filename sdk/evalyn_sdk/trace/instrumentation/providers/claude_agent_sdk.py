"""
Claude Agent SDK instrumentor.

Uses hook-based instrumentation with EvalynAgentHooks for claude_agent_sdk.

Captures:
- Tool calls via PreToolUse/PostToolUse hooks
- Subagent hierarchy tracking via message stream
- Token usage with cache metrics
- ResultMessage metrics
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict

from ..base import Instrumentor, InstrumentorType
from ._shared import calculate_cost_with_cache
from ....models import Span
from ... import context as span_context


@dataclass
class SpanState:
    """Track span state across hook callbacks."""

    span: Span
    start_time: float


@dataclass
class SubagentContext:
    """Track subagent context for hierarchy reconstruction."""

    span_id: str
    tool_use_id: str  # Task tool_use_id that spawned this subagent
    subagent_type: str
    start_time: float


class TokenUsage(TypedDict, total=False):
    """Token usage with cache information."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class EvalynAgentHooks:
    """
    Hook adapter for Claude Agent SDK (claude_agent_sdk).

    Captures tool calls and subagent hierarchy as spans.
    Designed to work with claude_agent_sdk's HookMatcher system.

    Usage:
        from evalyn_sdk.trace.instrumentation import create_agent_hooks
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        evalyn_hooks = create_agent_hooks()

        hooks = {
            'PreToolUse': [HookMatcher(matcher=None, hooks=[evalyn_hooks.pre_tool_use_hook])],
            'PostToolUse': [HookMatcher(matcher=None, hooks=[evalyn_hooks.post_tool_use_hook])],
        }

        options = ClaudeAgentOptions(hooks=hooks, ...)
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
        self._subagent_contexts: Dict[str, SubagentContext] = {}
        self._current_parent_tool_use_id: Optional[str] = None
        self._thinking_blocks: List[str] = []
        self._run_start_time: Optional[float] = None

    def _extract_usage(self, usage: Any) -> TokenUsage:
        """Extract token usage including cache tokens."""
        return {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        }

    async def _call_user_hook(self, hook_name: str, *args: Any) -> None:
        """Call user hook if it exists."""
        if self._user_hooks and hasattr(self._user_hooks, hook_name):
            await getattr(self._user_hooks, hook_name)(*args)

    def _pop_span_from_context(self, span_id: str) -> None:
        """Pop a span ID from the span context stack."""
        stack = span_context._span_stack.get()
        if stack and stack[-1] == span_id:
            span_context._span_stack.set(stack[:-1])

    def set_current_context(self, parent_tool_use_id: Optional[str]) -> None:
        """
        Set the current parent tool use ID for context tracking.

        Call this from message stream processing when you receive
        an AssistantMessage with a parent_tool_use_id.
        """
        self._current_parent_tool_use_id = parent_tool_use_id

    def register_subagent(self, tool_use_id: str, tool_input: Dict[str, Any]) -> None:
        """
        Register a subagent spawn from a Task tool call.

        Call this when you detect a Task tool use in the message stream.
        """
        subagent_type = tool_input.get("subagent_type", "unknown")
        self._subagent_contexts[tool_use_id] = SubagentContext(
            span_id="",
            tool_use_id=tool_use_id,
            subagent_type=subagent_type,
            start_time=time.time(),
        )

    def capture_thinking(self, thinking_text: str) -> None:
        """Capture a thinking block from extended thinking."""
        self._thinking_blocks.append(thinking_text)

    # SDK PreToolUse/PostToolUse hooks - compatible with claude_agent_sdk
    async def pre_tool_use_hook(
        self, hook_input: Dict[str, Any], tool_use_id: str, context: Any
    ) -> Dict[str, Any]:
        """
        PreToolUse hook for claude_agent_sdk.

        Args:
            hook_input: {'tool_name': str, 'tool_input': dict}
            tool_use_id: Unique ID for this tool use
            context: SDK context object

        Returns:
            {'continue_': True} to proceed with tool execution
        """
        tool_name = hook_input.get("tool_name", "unknown")
        tool_input = hook_input.get("tool_input", {})

        parent_id = span_context.get_current_span_id()

        # Check if this is a Task tool (subagent spawn)
        extra_attrs = {}
        if tool_name == "Task":
            subagent_type = tool_input.get("subagent_type", "unknown")
            extra_attrs["subagent_type"] = subagent_type
            extra_attrs["description"] = tool_input.get("description", "")
            self.register_subagent(tool_use_id, tool_input)

        # Track current subagent context
        if self._current_parent_tool_use_id:
            extra_attrs["parent_tool_use_id"] = self._current_parent_tool_use_id
            subagent_ctx = self._subagent_contexts.get(self._current_parent_tool_use_id)
            if subagent_ctx:
                extra_attrs["executing_subagent"] = subagent_ctx.subagent_type

        span = Span.new(
            name=tool_name,
            span_type="tool_call",
            parent_id=parent_id,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            input=str(tool_input)[:1000],
            **extra_attrs,
        )

        self._tool_spans[tool_use_id] = SpanState(span=span, start_time=time.time())

        await self._call_user_hook("pre_tool_use_hook", hook_input, tool_use_id, context)

        return {"continue_": True}

    async def post_tool_use_hook(
        self, hook_input: Dict[str, Any], tool_use_id: str, context: Any
    ) -> Dict[str, Any]:
        """
        PostToolUse hook for claude_agent_sdk.

        Args:
            hook_input: {'tool_response': ...} or {'error': ...}
            tool_use_id: Unique ID for this tool use
            context: SDK context object

        Returns:
            {'continue_': True} to proceed
        """
        state = self._tool_spans.pop(tool_use_id, None)

        if state:
            duration_ms = (time.time() - state.start_time) * 1000
            state.span.start_time = state.span.start_time - timedelta(
                milliseconds=duration_ms
            )

            # claude_agent_sdk uses 'tool_response' key
            tool_response = hook_input.get("tool_response")
            tool_error = None

            # Check for error in response
            if isinstance(tool_response, dict):
                tool_error = tool_response.get("error")

            if tool_error:
                state.span.finish(status="error", error=str(tool_error)[:500])
            else:
                output_str = str(tool_response)[:500] if tool_response else ""
                state.span.finish(status="ok", output=output_str)

            span_context._add_span_to_collector(state.span)

        await self._call_user_hook("post_tool_use_hook", hook_input, tool_use_id, context)

        return {"continue_": True}

    def finalize_run(self, result: Any) -> None:
        """
        Finalize the run with ResultMessage data.

        Call this when you receive a ResultMessage in the stream.
        """
        if result is None:
            return

        # Extract metrics from ResultMessage
        duration_ms = getattr(result, "duration_ms", None)
        total_cost_usd = getattr(result, "total_cost_usd", None)
        usage = getattr(result, "usage", None)
        num_turns = getattr(result, "num_turns", None)

        # Store on root span if we have one
        if self._span_stack:
            root_state = self._span_stack[0]
            if duration_ms is not None:
                root_state.span.attributes["result_duration_ms"] = duration_ms
            if total_cost_usd is not None:
                root_state.span.attributes["total_cost_usd"] = total_cost_usd
            if num_turns is not None:
                root_state.span.attributes["num_turns"] = num_turns
            if usage:
                token_usage = self._extract_usage(usage)
                root_state.span.attributes["total_input_tokens"] = token_usage.get(
                    "input_tokens", 0
                )
                root_state.span.attributes["total_output_tokens"] = token_usage.get(
                    "output_tokens", 0
                )
                root_state.span.attributes["total_cache_creation_tokens"] = (
                    token_usage.get("cache_creation_input_tokens", 0)
                )
                root_state.span.attributes["total_cache_read_tokens"] = (
                    token_usage.get("cache_read_input_tokens", 0)
                )

        # Store thinking blocks
        if self._thinking_blocks and self._span_stack:
            root_state = self._span_stack[0]
            root_state.span.attributes["thinking_blocks"] = self._thinking_blocks

    def get_hook_matchers(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get hook configuration for ClaudeAgentOptions.

        Returns format compatible with claude_agent_sdk:
            {
                'PreToolUse': [{'matcher': None, 'hooks': [...]}],
                'PostToolUse': [{'matcher': None, 'hooks': [...]}],
            }

        Note: You'll need to wrap with HookMatcher yourself since
        we don't import claude_agent_sdk here.
        """
        return {
            "PreToolUse": [{"matcher": None, "hooks": [self.pre_tool_use_hook]}],
            "PostToolUse": [{"matcher": None, "hooks": [self.post_tool_use_hook]}],
        }


class MessageStreamAdapter:
    """
    Wraps message stream to intercept messages for instrumentation.

    Use this to process the receive_response() stream and automatically
    track subagent context and capture metrics.

    Captures:
    - AssistantMessage: parent_tool_use_id for hierarchy, Task tool uses
    - ThinkingBlock: Extended thinking content
    - ResultMessage: Final metrics
    """

    def __init__(self, hooks: EvalynAgentHooks):
        self._hooks = hooks

    async def wrap_stream(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """
        Wrap a message stream to intercept and instrument messages.

        Usage:
            adapter = MessageStreamAdapter(hooks)
            async for msg in adapter.wrap_stream(client.receive_response()):
                # Process messages - instrumentation happens automatically
                ...

        Args:
            stream: The original message stream from receive_response()

        Yields:
            Messages from the stream after instrumentation
        """
        async for msg in stream:
            msg_type = type(msg).__name__

            if msg_type == "AssistantMessage":
                # Update context from parent_tool_use_id
                parent_id = getattr(msg, "parent_tool_use_id", None)
                self._hooks.set_current_context(parent_id)

                # Process content blocks
                content = getattr(msg, "content", [])
                for block in content:
                    block_type = type(block).__name__

                    if block_type == "ToolUseBlock":
                        # Check for Task tool (subagent spawn)
                        tool_name = getattr(block, "name", "")
                        if tool_name == "Task":
                            block_id = getattr(block, "id", "")
                            block_input = getattr(block, "input", {})
                            self._hooks.register_subagent(block_id, block_input)

                    elif block_type == "ThinkingBlock":
                        # Capture thinking content
                        thinking = getattr(block, "thinking", "")
                        if thinking:
                            self._hooks.capture_thinking(thinking)

            elif msg_type == "ResultMessage":
                # Finalize run with result metrics
                self._hooks.finalize_run(msg)

            yield msg


class ClaudeAgentSDKInstrumentor(Instrumentor):
    """
    Instrumentor for Claude Agent SDK (claude_agent_sdk).

    Uses hook-based approach: creates EvalynAgentHooks that users
    must explicitly pass to their ClaudeAgentOptions.
    """

    _hooks: Optional[EvalynAgentHooks] = None

    @property
    def name(self) -> str:
        return "claude_agent_sdk"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.HOOK_BASED

    def is_available(self) -> bool:
        return importlib.util.find_spec("claude_agent_sdk") is not None

    def is_instrumented(self) -> bool:
        return self._hooks is not None

    def instrument(self) -> bool:
        if not self.is_available():
            return False
        self._hooks = EvalynAgentHooks()
        return True

    def uninstrument(self) -> bool:
        self._hooks = None
        return True

    def get_hooks(self) -> Optional[EvalynAgentHooks]:
        """Get the hooks adapter to pass to ClaudeAgentOptions."""
        if self._hooks is None:
            self.instrument()
        return self._hooks


# Backwards compatibility aliases
AnthropicAgentsInstrumentor = ClaudeAgentSDKInstrumentor


def create_agent_hooks(user_hooks: Optional[Any] = None) -> EvalynAgentHooks:
    """
    Create Evalyn agent hooks for Claude Agent SDK.

    Usage:
        from evalyn_sdk.trace.instrumentation import create_agent_hooks
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        evalyn_hooks = create_agent_hooks()

        hooks = {
            'PreToolUse': [HookMatcher(matcher=None, hooks=[evalyn_hooks.pre_tool_use_hook])],
            'PostToolUse': [HookMatcher(matcher=None, hooks=[evalyn_hooks.post_tool_use_hook])],
        }

        options = ClaudeAgentOptions(hooks=hooks, ...)

    Args:
        user_hooks: Optional user-provided hooks to compose with.

    Returns:
        EvalynAgentHooks instance
    """
    return EvalynAgentHooks(user_hooks=user_hooks)


def create_stream_adapter(hooks: EvalynAgentHooks) -> MessageStreamAdapter:
    """
    Create a message stream adapter for additional instrumentation.

    Usage:
        hooks = create_agent_hooks()
        adapter = create_stream_adapter(hooks)

        async for msg in adapter.wrap_stream(client.receive_response()):
            # Process messages with automatic instrumentation

    Args:
        hooks: The EvalynAgentHooks instance to use

    Returns:
        MessageStreamAdapter instance
    """
    return MessageStreamAdapter(hooks)
