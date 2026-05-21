"""
Semantic Kernel framework instrumentor.

Uses a hybrid approach:
- Filter-based: Patches Kernel.__init__ to auto-register a function invocation
  filter on every new Kernel instance, using SK's official filter API.
- Monkey-patching: Patches ChatCompletionClientBase.get_chat_message_contents
  and Agent.invoke directly, since no filter system exists for those.

Span hierarchy:
    kernel:{plugin}.{function}
      -> llm_call:{model}
      -> tool_call:{function_name}
    agent:{name}
      -> llm_call:{model}
"""

from __future__ import annotations

import functools
import importlib.util
from typing import Any

from ....models import Span
from ... import context as span_context
from ..base import Instrumentor, InstrumentorType
from ._shared import calculate_cost


async def _function_invocation_filter(context: Any, next: Any) -> None:
    """SK function invocation filter that creates tracing spans.

    Registered on each Kernel instance via the patched __init__.
    Uses the official Semantic Kernel filter API:
        kernel.add_filter("function_invocation", filter_fn)
    """
    func_name = getattr(context.function, "name", "unknown")
    plugin_name = getattr(context.function, "plugin_name", "")

    display_name = f"{plugin_name}.{func_name}" if plugin_name else func_name

    span = Span.new(
        name=f"kernel:{display_name}",
        span_type="agent",
        parent_id=span_context.get_current_span_id(),
    )

    span.attributes["sk.type"] = "kernel_invoke"
    span.attributes["sk.function_name"] = func_name
    if plugin_name:
        span.attributes["sk.plugin_name"] = plugin_name

    arguments = getattr(context, "arguments", None)
    if arguments:
        span.attributes["sk.arguments"] = str(arguments)[:500]

    stack = span_context._span_stack.get()
    span_context._span_stack.set(stack + [span.id])

    try:
        await next(context)

        result = getattr(context, "result", None)
        if result:
            value = getattr(result, "value", None)
            if value is not None:
                span.attributes["sk.result"] = str(value)[:1000]
            metadata = getattr(result, "metadata", None)
            if metadata:
                span.attributes["sk.result_metadata"] = str(metadata)[:500]

        span_context._span_stack.set(stack)
        span.finish(status="ok")
        span_context._add_span_to_collector(span)

    except Exception as e:
        span_context._span_stack.set(stack)
        span.finish(status="error", error=str(e))
        span_context._add_span_to_collector(span)
        raise


class SemanticKernelInstrumentor(Instrumentor):
    """Instrumentor for Microsoft Semantic Kernel."""

    _instrumented = False

    def __init__(self) -> None:
        # See RUF012 fix in autogen.py - mutable class-attribute default
        # caused instance-level state to leak across instrumentors.
        self._originals: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "semantic_kernel"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("semantic_kernel") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        patched_any = False
        patched_any = self._patch_kernel_init() or patched_any
        patched_any = self._patch_chat_completion() or patched_any
        patched_any = self._patch_agent() or patched_any

        if patched_any:
            self._instrumented = True
        return patched_any

    def _patch_kernel_init(self) -> bool:
        """Patch Kernel.__init__ to auto-register our function invocation filter.

        Instead of monkey-patching Kernel.invoke, we hook into Kernel.__init__
        so that every new Kernel instance automatically gets our tracing filter
        registered via the official SK filter API (kernel.add_filter).
        """
        try:
            from semantic_kernel.kernel import Kernel
        except ImportError:
            return False

        if not hasattr(Kernel, "__init__"):
            return False

        original_init = Kernel.__init__
        self._originals["kernel_init"] = original_init

        @functools.wraps(original_init)
        def patched_init(kernel_self, *args, **kwargs):
            original_init(kernel_self, *args, **kwargs)
            # Register our filter via the official SK API
            if hasattr(kernel_self, "add_filter"):
                kernel_self.add_filter("function_invocation", _function_invocation_filter)

        Kernel.__init__ = patched_init
        return True

    def _patch_chat_completion(self) -> bool:
        """Patch ChatCompletionClientBase for LLM call tracing."""
        try:
            from semantic_kernel.connectors.ai.chat_completion_client_base import (
                ChatCompletionClientBase,
            )
        except ImportError:
            return False

        if not hasattr(ChatCompletionClientBase, "get_chat_message_contents"):
            return False

        original = ChatCompletionClientBase.get_chat_message_contents
        self._originals["chat_completion"] = original

        @functools.wraps(original)
        async def wrapped_get_chat(service_self, chat_history, settings, **kwargs):
            model = (
                getattr(service_self, "ai_model_id", None)
                or getattr(settings, "ai_model_id", None)
                or "unknown"
            )

            span = Span.new(
                name=f"sk:{model}",
                span_type="llm_call",
                parent_id=span_context.get_current_span_id(),
                provider="semantic_kernel",
                model=str(model),
            )

            if chat_history:
                messages = getattr(chat_history, "messages", None)
                if messages:
                    span.attributes["request"] = {"messages": _format_sk_messages(messages)}
                    span.attributes["sk.message_count"] = len(messages)

            if settings:
                temp = getattr(settings, "temperature", None)
                if temp is not None:
                    span.attributes["sk.temperature"] = temp
                max_tokens = getattr(settings, "max_tokens", None)
                if max_tokens:
                    span.attributes["sk.max_tokens"] = max_tokens

            stack = span_context._span_stack.get()
            span_context._span_stack.set(stack + [span.id])

            try:
                result = await original(service_self, chat_history, settings, **kwargs)

                if result:
                    for msg in result[:3]:
                        content = getattr(msg, "content", None)
                        if content:
                            span.attributes["response"] = {
                                "content": str(content)[:1000],
                            }
                            break

                    for msg in result:
                        metadata = getattr(msg, "metadata", {}) or {}
                        usage = metadata.get("usage")
                        if usage:
                            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                            output_tokens = getattr(usage, "completion_tokens", 0) or 0
                            span.attributes["input_tokens"] = input_tokens
                            span.attributes["output_tokens"] = output_tokens
                            span.attributes["cost_usd"] = calculate_cost(
                                str(model), input_tokens, output_tokens
                            )
                            break

                span_context._span_stack.set(stack)
                span.finish(status="ok")
                span_context._add_span_to_collector(span)
                return result

            except Exception as e:
                span_context._span_stack.set(stack)
                span.finish(status="error", error=str(e))
                span_context._add_span_to_collector(span)
                raise

        ChatCompletionClientBase.get_chat_message_contents = wrapped_get_chat
        return True

    def _patch_agent(self) -> bool:
        """Patch Agent.invoke for agent orchestration tracing."""
        try:
            from semantic_kernel.agents.agent import Agent
        except ImportError:
            return False

        if not hasattr(Agent, "invoke"):
            return False

        original = Agent.invoke
        self._originals["agent_invoke"] = original

        @functools.wraps(original)
        async def wrapped_invoke(agent_self, *args, **kwargs):
            agent_name = getattr(agent_self, "name", "agent")
            agent_id = getattr(agent_self, "id", "")

            span = Span.new(
                name=f"agent:{agent_name}",
                span_type="agent",
                parent_id=span_context.get_current_span_id(),
            )

            span.attributes["sk.type"] = "agent"
            span.attributes["sk.agent_name"] = str(agent_name)
            if agent_id:
                span.attributes["sk.agent_id"] = str(agent_id)

            instructions = getattr(agent_self, "instructions", None)
            if instructions:
                span.attributes["sk.instructions"] = str(instructions)[:500]

            stack = span_context._span_stack.get()
            span_context._span_stack.set(stack + [span.id])

            try:
                results = []
                async for item in original(agent_self, *args, **kwargs):
                    results.append(item)
                    yield item

                span.attributes["sk.response_count"] = len(results)

                span_context._span_stack.set(stack)
                span.finish(status="ok")
                span_context._add_span_to_collector(span)

            except Exception as e:
                span_context._span_stack.set(stack)
                span.finish(status="error", error=str(e))
                span_context._add_span_to_collector(span)
                raise

        Agent.invoke = wrapped_invoke
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            if "kernel_init" in self._originals:
                from semantic_kernel.kernel import Kernel

                Kernel.__init__ = self._originals["kernel_init"]
        except ImportError:
            pass

        try:
            if "chat_completion" in self._originals:
                from semantic_kernel.connectors.ai.chat_completion_client_base import (
                    ChatCompletionClientBase,
                )

                ChatCompletionClientBase.get_chat_message_contents = self._originals[
                    "chat_completion"
                ]
        except ImportError:
            pass

        try:
            if "agent_invoke" in self._originals:
                from semantic_kernel.agents.agent import Agent

                Agent.invoke = self._originals["agent_invoke"]
        except ImportError:
            pass

        self._originals.clear()
        self._instrumented = False
        return True


def _format_sk_messages(messages: Any) -> list:
    """Format Semantic Kernel ChatMessage objects for storage."""
    if not messages:
        return []

    formatted = []
    for msg in list(messages)[:20]:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        entry = {}
        if role:
            entry["role"] = str(role)
        if content:
            entry["content"] = str(content)[:500]
        if entry:
            formatted.append(entry)
    return formatted
