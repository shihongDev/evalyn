"""
DSPy framework instrumentor (v2.5+).

Uses DSPy's callback system (BaseCallback) to capture:
- Module forward passes (Predict, ChainOfThought, etc.)
- LM calls with prompts and completions
- Tool/function calls
- Optimization steps

DSPy uses a global settings object: dspy.settings.configure(callbacks=[...])
Callbacks receive structured call info via on_lm_start/end, on_module_start/end.

Span hierarchy:
    module:{module_name}
      -> lm_call:{model}
      -> module:{nested_module}
        -> lm_call:{model}
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ....models import Span
from ... import context as span_context
from ..base import Instrumentor, InstrumentorType
from ._shared import calculate_cost


class DSPyInstrumentor(Instrumentor):
    """Instrumentor for DSPy via its callback system."""

    _instrumented = False
    _callback: Any | None = None

    @property
    def name(self) -> str:
        return "dspy"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.HOOK_BASED

    def is_available(self) -> bool:
        return importlib.util.find_spec("dspy") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        try:
            import dspy

            # Probe that BaseCallback exists in the installed dspy version;
            # we register a subclass below and need this symbol to be present.
            from dspy.utils.callback import BaseCallback  # noqa: F401
        except ImportError:
            return False

        self._callback = EvalynDSPyCallback()

        # Add our callback to DSPy's global callbacks
        existing = dspy.settings.get("callbacks", []) or []
        if self._callback not in existing:
            dspy.settings.configure(callbacks=existing + [self._callback])

        self._instrumented = True
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            import dspy

            existing = dspy.settings.get("callbacks", []) or []
            filtered = [cb for cb in existing if cb is not self._callback]
            dspy.settings.configure(callbacks=filtered)
        except (ImportError, AttributeError):
            pass

        self._callback = None
        self._instrumented = False
        return True


class EvalynDSPyCallback:
    """
    DSPy callback that creates evalyn spans.

    Implements the BaseCallback protocol methods:
    - on_module_start / on_module_end
    - on_lm_start / on_lm_end
    - on_tool_start / on_tool_end (if available)
    """

    def __init__(self) -> None:
        self._open_spans: dict[str, _DSPyOpenSpan] = {}

    def _push_span(self, call_id: str, span: Span) -> None:
        """Push span onto context stack and track it as open."""
        prev_stack = span_context._span_stack.get()
        span_context._span_stack.set(prev_stack + [span.id])
        self._open_spans[call_id] = _DSPyOpenSpan(span=span, prev_stack=prev_stack)

    def _finish_span(
        self,
        call_id: str,
        exception: Exception | None = None,
    ) -> _DSPyOpenSpan | None:
        """Pop open span, restore stack, finish and collect it.

        Returns the open span if found, None otherwise (caller can use this
        to add attributes before the span was finished - but note the span
        is already finished/collected when this returns).
        """
        open_span = self._open_spans.pop(call_id, None)
        if open_span is None:
            return None

        span_context._span_stack.set(open_span.prev_stack)

        if exception:
            open_span.span.attributes["error"] = str(exception)
            open_span.span.finish(status="error")
        else:
            open_span.span.finish(status="ok")

        span_context._add_span_to_collector(open_span.span)
        return open_span

    # Module lifecycle callbacks

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        module_name = type(instance).__name__

        span = Span.new(
            name=f"module:{module_name}",
            span_type="agent",
            parent_id=span_context.get_current_span_id(),
        )

        span.attributes["dspy.type"] = "module"
        span.attributes["dspy.module_class"] = module_name

        signature = getattr(instance, "signature", None)
        if signature:
            sig_name = getattr(signature, "__name__", None) or type(
                signature
            ).__name__
            span.attributes["dspy.signature"] = sig_name

            input_fields = getattr(signature, "input_fields", None)
            output_fields = getattr(signature, "output_fields", None)
            if input_fields:
                span.attributes["dspy.input_fields"] = list(input_fields.keys())[:10]
            if output_fields:
                span.attributes["dspy.output_fields"] = list(
                    output_fields.keys()
                )[:10]

        if inputs:
            span.attributes["dspy.inputs"] = {
                k: str(v)[:300] for k, v in list(inputs.items())[:10]
            }

        self._push_span(call_id, span)

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        open_span = self._open_spans.get(call_id)
        if open_span is None:
            return

        if outputs:
            open_span.span.attributes["dspy.outputs"] = {
                k: str(v)[:300]
                for k, v in (
                    outputs.items() if hasattr(outputs, "items") else [("result", outputs)]
                )
            }

        self._finish_span(call_id, exception)

    # LM call lifecycle callbacks

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        model = getattr(instance, "model", None) or "unknown"

        span = Span.new(
            name=f"dspy:{model}",
            span_type="llm_call",
            parent_id=span_context.get_current_span_id(),
            provider="dspy",
            model=str(model),
        )

        span.attributes["dspy.type"] = "lm_call"

        messages = inputs.get("messages") or inputs.get("prompt")
        if messages:
            if isinstance(messages, list):
                span.attributes["request"] = {
                    "messages": _format_dspy_messages(messages)
                }
            else:
                span.attributes["request"] = {"prompt": str(messages)[:1000]}

        temperature = getattr(instance, "temperature", None)
        if temperature is not None:
            span.attributes["dspy.temperature"] = temperature
        max_tokens = getattr(instance, "max_tokens", None)
        if max_tokens:
            span.attributes["dspy.max_tokens"] = max_tokens

        self._push_span(call_id, span)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        open_span = self._open_spans.get(call_id)
        if open_span is None:
            return

        if outputs:
            usage = _extract_lm_outputs(open_span.span, outputs)
            if usage:
                input_tokens = _get_usage_field(usage, "prompt_tokens")
                output_tokens = _get_usage_field(usage, "completion_tokens")
                model = open_span.span.attributes.get("model", "unknown")
                open_span.span.attributes["input_tokens"] = input_tokens
                open_span.span.attributes["output_tokens"] = output_tokens
                open_span.span.attributes["cost_usd"] = calculate_cost(
                    str(model), input_tokens, output_tokens
                )

        self._finish_span(call_id, exception)

    # Tool call lifecycle callbacks (DSPy 2.6+)

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        tool_name = getattr(instance, "name", None) or getattr(
            instance, "func_name", type(instance).__name__
        )

        span = Span.new(
            name=str(tool_name),
            span_type="tool_call",
            parent_id=span_context.get_current_span_id(),
            tool_name=str(tool_name),
        )

        span.attributes["dspy.type"] = "tool_call"
        if inputs:
            span.attributes["tool_input"] = str(inputs)[:1000]

        self._push_span(call_id, span)

    def on_tool_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        open_span = self._open_spans.get(call_id)
        if open_span is None:
            return

        if outputs:
            open_span.span.attributes["tool_output"] = str(outputs)[:1000]

        self._finish_span(call_id, exception)


class _DSPyOpenSpan:
    """Internal state for a span that has been started but not finished."""

    __slots__ = ("span", "prev_stack")

    def __init__(self, span: Span, prev_stack: list) -> None:
        self.span = span
        self.prev_stack = prev_stack


def _get_usage_field(usage: Any, field: str) -> int:
    """Extract a token count field from a usage dict or object."""
    if isinstance(usage, dict):
        return usage.get(field, 0) or 0
    return getattr(usage, field, 0) or 0


def _extract_lm_outputs(span: Span, outputs: Any) -> Any:
    """Extract response content and return the usage object (if any).

    Mutates span.attributes to store the response content when found.
    Returns the usage dict/object, or None.
    """
    if isinstance(outputs, dict):
        choices = outputs.get("choices", [])
        if choices:
            first = choices[0] if isinstance(choices, list) else choices
            if isinstance(first, dict):
                message = first.get("message", {})
            else:
                message = getattr(first, "message", None)

            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "") if message else ""

            if content:
                span.attributes["response"] = {"content": str(content)[:1000]}

        return outputs.get("usage")

    if hasattr(outputs, "usage"):
        return outputs.usage

    return None


def _format_dspy_messages(messages: Any) -> list:
    """Format DSPy messages for storage."""
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
        else:
            formatted.append({"raw": str(msg)[:500]})
    return formatted
