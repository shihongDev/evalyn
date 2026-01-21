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
        # Track turns for automatic LLM span creation from hooks
        self._last_logged_turn: int = 0
        self._seen_tool_use_ids: set = set()

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

    def _try_log_turn_from_context(self, context: Any, tool_use_id: str) -> None:
        """Try to detect and log a new LLM turn from hook context.

        This is a fallback mechanism for when MessageStreamAdapter isn't used.
        It tries to extract turn info from the SDK context and creates LLM spans.
        """
        # Skip if we've already seen this tool_use_id
        if tool_use_id in self._seen_tool_use_ids:
            return
        self._seen_tool_use_ids.add(tool_use_id)

        # Try to get turn number from context
        turn = None
        model = None

        # Try various attribute names the SDK might use
        for turn_attr in ("turn", "turn_number", "num_turns", "current_turn"):
            turn = getattr(context, turn_attr, None)
            if turn is not None:
                break

        # Try to get model from context
        for model_attr in ("model", "model_id", "model_name"):
            model = getattr(context, model_attr, None)
            if model is not None:
                break

        # If we found turn info and it's a new turn, log it
        if turn is not None and turn > self._last_logged_turn:
            self.log_llm_turn(
                turn=turn,
                model=model or "unknown",
                output_text=None,
                tool_calls=None,
                parent_tool_use_id=self._current_parent_tool_use_id,
            )
            self._last_logged_turn = turn
        elif self._last_logged_turn == 0:
            # Fallback: if no turn info available and we haven't logged any turn yet,
            # create an initial LLM span to indicate agent activity
            self._last_logged_turn = 1
            self.log_llm_turn(
                turn=1,
                model=model or "claude-agent-sdk",
                output_text=None,
                tool_calls=None,
                parent_tool_use_id=self._current_parent_tool_use_id,
            )

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

    def capture_thinking(
        self, thinking_text: str, signature: Optional[str] = None
    ) -> None:
        """Capture a thinking block from extended thinking."""
        if signature:
            self._thinking_blocks.append(
                {"text": thinking_text, "signature": signature}
            )
        else:
            self._thinking_blocks.append(thinking_text)

    def log_llm_turn(
        self,
        turn: int,
        model: str,
        output_text: Optional[str] = None,
        tool_calls: Optional[List[str]] = None,
        parent_tool_use_id: Optional[str] = None,
    ) -> None:
        """
        Log an LLM turn as a span.

        Called when an AssistantMessage is received in the stream.
        Creates an llm_call span to track each model response.
        """
        parent_id = span_context.get_current_span_id()

        # Build attributes
        attrs = {
            "turn": turn,
            "model": model,
            "provider": "anthropic",
        }
        if output_text:
            attrs["output_preview"] = output_text[:500]
        if tool_calls:
            attrs["tool_calls"] = tool_calls
        if parent_tool_use_id:
            attrs["parent_tool_use_id"] = parent_tool_use_id

        # Create and immediately finish the span (we don't know duration)
        span = Span.new(
            name=f"llm_turn_{turn}",
            span_type="llm_call",
            parent_id=parent_id,
            model=model,
            **attrs,
        )
        span.finish(status="ok", output=output_text[:200] if output_text else None)
        span_context._add_span_to_collector(span)

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

        # Try to detect and log LLM turn from context (fallback for no adapter)
        self._try_log_turn_from_context(context, tool_use_id)

        parent_id = span_context.get_current_span_id()

        # Extract session_id from hook input (available in all hooks)
        session_id = hook_input.get("session_id")

        # Track input size and truncation
        input_str = str(tool_input)
        input_size = len(input_str)
        input_truncated = input_size > 1000

        # Check if this is a Task tool (subagent spawn)
        extra_attrs = {}
        if session_id:
            extra_attrs["session_id"] = session_id
        extra_attrs["input_size"] = input_size
        extra_attrs["input_truncated"] = input_truncated

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

        await self._call_user_hook(
            "pre_tool_use_hook", hook_input, tool_use_id, context
        )

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

            # Track output size and truncation
            output_full = str(tool_response) if tool_response else ""
            output_size = len(output_full)
            output_truncated = output_size > 500
            state.span.attributes["output_size"] = output_size
            state.span.attributes["output_truncated"] = output_truncated

            # Check for error in response
            if isinstance(tool_response, dict):
                tool_error = tool_response.get("error")

            if tool_error:
                state.span.finish(status="error", error=str(tool_error)[:500])
            else:
                output_str = output_full[:500]
                state.span.finish(status="ok", output=output_str)

            span_context._add_span_to_collector(state.span)

        await self._call_user_hook(
            "post_tool_use_hook", hook_input, tool_use_id, context
        )

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
        duration_api_ms = getattr(result, "duration_api_ms", None)
        total_cost_usd = getattr(result, "total_cost_usd", None)
        usage = getattr(result, "usage", None)
        num_turns = getattr(result, "num_turns", None)

        # Extract additional fields from ResultMessage
        is_error = getattr(result, "is_error", False)
        result_text = getattr(result, "result", None)
        structured_output = getattr(result, "structured_output", None)
        session_id = getattr(result, "session_id", None)

        # Store on root span if we have one
        if self._span_stack:
            root_state = self._span_stack[0]
            if duration_ms is not None:
                root_state.span.attributes["result_duration_ms"] = duration_ms
            if duration_api_ms is not None:
                root_state.span.attributes["duration_api_ms"] = duration_api_ms
            if total_cost_usd is not None:
                root_state.span.attributes["total_cost_usd"] = total_cost_usd
            if num_turns is not None:
                root_state.span.attributes["num_turns"] = num_turns

            # Store error and result info
            root_state.span.attributes["is_error"] = is_error
            if session_id:
                root_state.span.attributes["session_id"] = session_id
            if result_text:
                root_state.span.attributes["result_preview"] = result_text[:500]
                root_state.span.attributes["result_size"] = len(result_text)
            if structured_output is not None:
                root_state.span.attributes["has_structured_output"] = True

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
                root_state.span.attributes["total_cache_read_tokens"] = token_usage.get(
                    "cache_read_input_tokens", 0
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
    - AssistantMessage: LLM turns with model and content
    - ThinkingBlock: Extended thinking content
    - ResultMessage: Final metrics
    """

    def __init__(self, hooks: EvalynAgentHooks):
        self._hooks = hooks
        self._turn_count = 0

    async def wrap_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
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
                self._turn_count += 1

                # Update context from parent_tool_use_id
                parent_id = getattr(msg, "parent_tool_use_id", None)
                self._hooks.set_current_context(parent_id)

                # Extract model and content for LLM span
                model = getattr(msg, "model", "unknown")

                # Collect text content and tool uses
                text_content = []
                tool_uses = []
                content = getattr(msg, "content", [])
                for block in content:
                    block_type = type(block).__name__

                    if block_type == "TextBlock":
                        text = getattr(block, "text", "")
                        if text:
                            text_content.append(text)

                    elif block_type == "ToolUseBlock":
                        tool_name = getattr(block, "name", "")
                        tool_uses.append(tool_name)
                        # Check for Task tool (subagent spawn)
                        if tool_name == "Task":
                            block_id = getattr(block, "id", "")
                            block_input = getattr(block, "input", {})
                            self._hooks.register_subagent(block_id, block_input)

                    elif block_type == "ThinkingBlock":
                        # Capture thinking content with signature
                        thinking = getattr(block, "thinking", "")
                        signature = getattr(block, "signature", None)
                        if thinking:
                            self._hooks.capture_thinking(thinking, signature)

                # Create LLM span for this turn
                self._hooks.log_llm_turn(
                    turn=self._turn_count,
                    model=model,
                    output_text="\n".join(text_content) if text_content else None,
                    tool_calls=tool_uses if tool_uses else None,
                    parent_tool_use_id=parent_id,
                )

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
