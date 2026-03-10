"""
AutoGen framework instrumentor (v0.4+ / autogen-agentchat).

Patches key methods to capture:
- Agent invocations (run, on_messages)
- LLM calls (ChatCompletionClient.create)
- Tool execution (BaseTool.run_json)
- Team orchestration (BaseGroupChat.run)

Span hierarchy:
    team:run (if team)
      -> agent:{name}
        -> llm_call:{model}
        -> tool_call:{tool_name}
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Any

from ..base import Instrumentor, InstrumentorType
from ....models import Span
from ... import context as span_context
from ._shared import calculate_cost


class AutoGenInstrumentor(Instrumentor):
    """Instrumentor for Microsoft AutoGen (autogen-agentchat v0.4+)."""

    _instrumented = False
    _originals: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "autogen"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("autogen_agentchat") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        patched_any = False

        # Patch ChatCompletionClient.create for LLM calls
        patched_any = self._patch_model_client() or patched_any

        # Patch BaseChatAgent for agent invocations
        patched_any = self._patch_agent() or patched_any

        # Patch BaseTool for tool execution
        patched_any = self._patch_tool() or patched_any

        # Patch BaseGroupChat for team orchestration
        patched_any = self._patch_team() or patched_any

        if patched_any:
            self._instrumented = True
        return patched_any

    def _patch_model_client(self) -> bool:
        """Patch ChatCompletionClient.create to capture LLM calls."""
        try:
            from autogen_core.models import ChatCompletionClient
        except ImportError:
            return False

        if not hasattr(ChatCompletionClient, "create"):
            return False

        original = ChatCompletionClient.create
        self._originals["model_client_create"] = original

        @functools.wraps(original)
        async def wrapped_create(client_self, messages, *args, **kwargs):
            model = (
                getattr(client_self, "model", None)
                or getattr(client_self, "_model_name", None)
                or getattr(client_self, "_model", "unknown")
            )

            parent_span_id = span_context.get_current_span_id()
            span = Span.new(
                name=f"autogen:{model}",
                span_type="llm_call",
                parent_id=parent_span_id,
                provider="autogen",
                model=str(model),
            )

            # Capture request messages
            if messages:
                span.attributes["request"] = {
                    "messages": _format_autogen_messages(messages),
                }

            # Capture tools if provided
            tools = kwargs.get("tools") or (args[0] if args else None)
            if tools:
                span.attributes["tool_schemas"] = [
                    getattr(t, "name", str(t)) for t in tools
                ][:20]

            stack = span_context._span_stack.get()
            new_stack = stack + [span.id]
            span_context._span_stack.set(new_stack)

            try:
                result = await original(client_self, messages, *args, **kwargs)

                # Extract token usage from CreateResult
                usage = getattr(result, "usage", None)
                input_tokens = 0
                output_tokens = 0
                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0

                span.attributes["input_tokens"] = input_tokens
                span.attributes["output_tokens"] = output_tokens
                span.attributes["cost_usd"] = calculate_cost(
                    str(model), input_tokens, output_tokens
                )

                # Capture response content
                content = getattr(result, "content", None)
                if content:
                    span.attributes["response"] = {
                        "content": str(content)[:1000],
                    }

                finish_reason = getattr(result, "finish_reason", None)
                if finish_reason:
                    span.attributes["finish_reason"] = str(finish_reason)

                span_context._span_stack.set(stack)
                span.finish(status="ok")
                span_context._add_span_to_collector(span)
                return result

            except Exception as e:
                span_context._span_stack.set(stack)
                span.finish(status="error", error=str(e))
                span_context._add_span_to_collector(span)
                raise

        ChatCompletionClient.create = wrapped_create
        return True

    def _patch_agent(self) -> bool:
        """Patch BaseChatAgent.on_messages to capture agent invocations."""
        try:
            from autogen_agentchat.agents import BaseChatAgent
        except ImportError:
            return False

        if not hasattr(BaseChatAgent, "on_messages"):
            return False

        original = BaseChatAgent.on_messages
        self._originals["agent_on_messages"] = original

        @functools.wraps(original)
        async def wrapped_on_messages(agent_self, messages, *args, **kwargs):
            agent_name = getattr(agent_self, "name", "agent")
            description = getattr(agent_self, "description", "")

            parent_span_id = span_context.get_current_span_id()
            span = Span.new(
                name=f"agent:{agent_name}",
                span_type="agent",
                parent_id=parent_span_id,
                autogen_agent_name=str(agent_name),
            )

            if description:
                span.attributes["autogen.agent_description"] = str(description)[:500]

            # Capture input messages
            if messages:
                span.attributes["autogen.input_messages"] = _format_chat_messages(
                    messages
                )

            stack = span_context._span_stack.get()
            new_stack = stack + [span.id]
            span_context._span_stack.set(new_stack)

            try:
                result = await original(agent_self, messages, *args, **kwargs)

                # Extract response info
                if result:
                    chat_message = getattr(result, "chat_message", None)
                    if chat_message:
                        span.attributes["autogen.response_type"] = type(
                            chat_message
                        ).__name__
                        content = getattr(chat_message, "content", None)
                        if content:
                            span.attributes["autogen.response"] = str(content)[:1000]

                    # Capture inner messages (tool calls, thoughts, etc.)
                    inner = getattr(result, "inner_messages", None)
                    if inner:
                        span.attributes["autogen.inner_message_count"] = len(inner)
                        span.attributes["autogen.inner_message_types"] = [
                            type(m).__name__ for m in inner
                        ][:20]

                span_context._span_stack.set(stack)
                span.finish(status="ok")
                span_context._add_span_to_collector(span)
                return result

            except Exception as e:
                span_context._span_stack.set(stack)
                span.finish(status="error", error=str(e))
                span_context._add_span_to_collector(span)
                raise

        BaseChatAgent.on_messages = wrapped_on_messages
        return True

    def _patch_tool(self) -> bool:
        """Patch BaseTool.run_json to capture tool executions."""
        try:
            from autogen_core.tools import BaseTool
        except ImportError:
            return False

        if not hasattr(BaseTool, "run_json"):
            return False

        original = BaseTool.run_json
        self._originals["tool_run_json"] = original

        @functools.wraps(original)
        async def wrapped_run_json(tool_self, args_dict, *args, **kwargs):
            tool_name = (
                getattr(tool_self, "name", None)
                or getattr(tool_self, "_name", "unknown")
            )

            parent_span_id = span_context.get_current_span_id()
            span = Span.new(
                name=str(tool_name),
                span_type="tool_call",
                parent_id=parent_span_id,
                tool_name=str(tool_name),
            )

            span.attributes["tool_input"] = str(args_dict)[:1000]

            # Capture call_id if provided
            call_id = kwargs.get("call_id")
            if call_id:
                span.attributes["autogen.call_id"] = str(call_id)

            stack = span_context._span_stack.get()
            new_stack = stack + [span.id]
            span_context._span_stack.set(new_stack)

            try:
                result = await original(tool_self, args_dict, *args, **kwargs)

                span.attributes["tool_output"] = str(result)[:1000]

                # Check if result indicates error
                is_error = False
                if hasattr(result, "is_error"):
                    is_error = result.is_error
                elif isinstance(result, dict):
                    is_error = result.get("is_error", False)

                span_context._span_stack.set(stack)
                span.finish(status="error" if is_error else "ok")
                span_context._add_span_to_collector(span)
                return result

            except Exception as e:
                span_context._span_stack.set(stack)
                span.finish(status="error", error=str(e))
                span_context._add_span_to_collector(span)
                raise

        BaseTool.run_json = wrapped_run_json
        return True

    def _patch_team(self) -> bool:
        """Patch BaseGroupChat.run to capture team orchestration."""
        try:
            from autogen_agentchat.teams import BaseGroupChat
        except ImportError:
            return False

        if not hasattr(BaseGroupChat, "run"):
            return False

        original = BaseGroupChat.run
        self._originals["team_run"] = original

        @functools.wraps(original)
        async def wrapped_run(team_self, *args, **kwargs):
            team_name = (
                getattr(team_self, "name", None)
                or getattr(team_self, "_name", "team")
            )
            team_type = type(team_self).__name__

            parent_span_id = span_context.get_current_span_id()
            span = Span.new(
                name=f"team:{team_name}",
                span_type="graph",
                parent_id=parent_span_id,
            )

            span.attributes["autogen.type"] = "team"
            span.attributes["autogen.team_name"] = str(team_name)
            span.attributes["autogen.team_type"] = team_type

            # Capture participant names
            participants = getattr(team_self, "_participants", None)
            if participants:
                span.attributes["autogen.participants"] = [
                    getattr(p, "name", str(p)) for p in participants
                ][:20]

            # Capture task input
            task = kwargs.get("task") or (args[0] if args else None)
            if task:
                span.attributes["autogen.task"] = str(task)[:500]

            stack = span_context._span_stack.get()
            new_stack = stack + [span.id]
            span_context._span_stack.set(new_stack)

            try:
                result = await original(team_self, *args, **kwargs)

                # Extract TaskResult info
                if result:
                    messages = getattr(result, "messages", None)
                    if messages:
                        span.attributes["autogen.message_count"] = len(messages)

                    stop_reason = getattr(result, "stop_reason", None)
                    if stop_reason:
                        span.attributes["autogen.stop_reason"] = str(stop_reason)

                span_context._span_stack.set(stack)
                span.finish(status="ok")
                span_context._add_span_to_collector(span)
                return result

            except Exception as e:
                span_context._span_stack.set(stack)
                span.finish(status="error", error=str(e))
                span_context._add_span_to_collector(span)
                raise

        BaseGroupChat.run = wrapped_run
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            if "model_client_create" in self._originals:
                from autogen_core.models import ChatCompletionClient

                ChatCompletionClient.create = self._originals["model_client_create"]
        except ImportError:
            pass

        try:
            if "agent_on_messages" in self._originals:
                from autogen_agentchat.agents import BaseChatAgent

                BaseChatAgent.on_messages = self._originals["agent_on_messages"]
        except ImportError:
            pass

        try:
            if "tool_run_json" in self._originals:
                from autogen_core.tools import BaseTool

                BaseTool.run_json = self._originals["tool_run_json"]
        except ImportError:
            pass

        try:
            if "team_run" in self._originals:
                from autogen_agentchat.teams import BaseGroupChat

                BaseGroupChat.run = self._originals["team_run"]
        except ImportError:
            pass

        self._originals.clear()
        self._instrumented = False
        return True


def _format_autogen_messages(messages: Any) -> list:
    """Format AutoGen LLMMessage sequence for storage."""
    if not messages:
        return []

    formatted = []
    for msg in list(messages)[:20]:
        if isinstance(msg, dict):
            formatted.append(
                {
                    "role": msg.get("role", "unknown"),
                    "content": str(msg.get("content", ""))[:500],
                }
            )
        elif hasattr(msg, "content") and hasattr(msg, "role"):
            formatted.append(
                {
                    "role": str(msg.role),
                    "content": str(msg.content)[:500],
                }
            )
        elif hasattr(msg, "content"):
            formatted.append(
                {
                    "type": type(msg).__name__,
                    "content": str(msg.content)[:500],
                }
            )
        else:
            formatted.append({"raw": str(msg)[:500]})
    return formatted


def _format_chat_messages(messages: Any) -> list:
    """Format AutoGen BaseChatMessage sequence for storage."""
    if not messages:
        return []

    formatted = []
    for msg in list(messages)[:20]:
        entry = {"type": type(msg).__name__}
        source = getattr(msg, "source", None)
        if source:
            entry["source"] = str(source)
        content = getattr(msg, "content", None)
        if content:
            entry["content"] = str(content)[:500]
        formatted.append(entry)
    return formatted
