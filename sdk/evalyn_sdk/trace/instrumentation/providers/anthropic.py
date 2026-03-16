"""
Anthropic SDK instrumentor.

Patches Anthropic client to auto-capture message completions.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import time
from typing import Any, Optional

from ..base import Instrumentor, InstrumentorType
from ._shared import log_llm_call

logger = logging.getLogger(__name__)


def _extract_response(response: Any) -> dict:
    """Extract response content from Anthropic Message response."""
    resp: dict[str, Any] = {}
    content_blocks = getattr(response, "content", None)
    if content_blocks:
        texts = []
        tool_calls = []
        for block in content_blocks:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                texts.append(str(getattr(block, "text", ""))[:500])
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "name": getattr(block, "name", ""),
                        "input": str(getattr(block, "input", ""))[:300],
                    }
                )
        if texts:
            resp["content"] = " ".join(texts)[:1000]
        if tool_calls:
            resp["tool_calls"] = tool_calls

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason:
        resp["stop_reason"] = stop_reason

    return resp


def _extract_usage(response: Any) -> dict:
    """Extract token usage from Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.debug("Anthropic response missing usage data (may be streaming)")
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }


class AnthropicInstrumentor(Instrumentor):
    """Instrumentor for Anthropic SDK (claude client, not Agent SDK)."""

    _instrumented = False
    _original_create: Optional[Any] = None
    _original_acreate: Optional[Any] = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("anthropic") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        try:
            from anthropic.resources import messages as messages_module
        except ImportError:
            return False

        # Patch sync messages.create
        if hasattr(messages_module, "Messages"):
            self._original_create = messages_module.Messages.create

            @functools.wraps(self._original_create)
            def patched_create(inst, *args, **kwargs):
                start = time.time()
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                try:
                    response = self._original_create(inst, *args, **kwargs)

                    if kwargs.get("stream"):
                        from ._streaming import (
                            LoggingWrapper,
                            StreamingSpanWrapper,
                        )

                        class AnthropicStreamWrapper(StreamingSpanWrapper):
                            def __next__(self):
                                chunk = super().__next__()
                                event_type = getattr(chunk, "type", "")
                                if event_type == "message_start":
                                    msg = getattr(chunk, "message", None)
                                    if msg:
                                        usage = getattr(msg, "usage", None)
                                        if usage:
                                            self.accumulated_input_tokens = getattr(
                                                usage, "input_tokens", 0
                                            )
                                elif event_type == "message_delta":
                                    usage = getattr(chunk, "usage", None)
                                    if usage:
                                        self.accumulated_output_tokens = getattr(
                                            usage, "output_tokens", 0
                                        )
                                return chunk

                        wrapper = AnthropicStreamWrapper(
                            response, request_start_time=start
                        )

                        def _on_exhaust(w):
                            duration_ms = (time.time() - start) * 1000
                            log_llm_call(
                                provider="anthropic",
                                model=model,
                                input_tokens=w.accumulated_input_tokens,
                                output_tokens=w.accumulated_output_tokens,
                                duration_ms=duration_ms,
                                success=True,
                                request={"messages": str(messages)[:500]},
                                streaming_attributes=w.as_span_attributes(),
                            )

                        return LoggingWrapper(wrapper, _on_exhaust)

                    duration_ms = (time.time() - start) * 1000
                    usage = _extract_usage(response)

                    log_llm_call(
                        provider="anthropic",
                        model=model,
                        duration_ms=duration_ms,
                        success=True,
                        request={"messages": str(messages)[:500]},
                        response=_extract_response(response),
                        **usage,
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider="anthropic",
                        model=model,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                    )
                    raise

            messages_module.Messages.create = patched_create

        # Patch async messages.create
        if hasattr(messages_module, "AsyncMessages"):
            self._original_acreate = messages_module.AsyncMessages.create

            @functools.wraps(self._original_acreate)
            async def patched_acreate(inst, *args, **kwargs):
                start = time.time()
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                try:
                    response = await self._original_acreate(inst, *args, **kwargs)

                    if kwargs.get("stream"):
                        from ._streaming import (
                            AsyncLoggingWrapper,
                            StreamingSpanWrapper,
                        )

                        def _extract_anthropic_tokens(wrapper, chunk):
                            event_type = getattr(chunk, "type", "")
                            if event_type == "message_start":
                                msg = getattr(chunk, "message", None)
                                if msg:
                                    usage = getattr(msg, "usage", None)
                                    if usage:
                                        wrapper.accumulated_input_tokens = (
                                            getattr(usage, "input_tokens", 0)
                                        )
                            elif event_type == "message_delta":
                                usage = getattr(chunk, "usage", None)
                                if usage:
                                    wrapper.accumulated_output_tokens = (
                                        getattr(usage, "output_tokens", 0)
                                    )

                        class AnthropicAsyncStreamWrapper(AsyncLoggingWrapper):
                            def _extract_tokens(self, chunk):
                                _extract_anthropic_tokens(self._wrapper, chunk)

                        tracker = StreamingSpanWrapper(iter([]), request_start_time=start)

                        def _on_exhaust(w):
                            duration_ms = (time.time() - start) * 1000
                            log_llm_call(
                                provider="anthropic",
                                model=model,
                                input_tokens=w.accumulated_input_tokens,
                                output_tokens=w.accumulated_output_tokens,
                                duration_ms=duration_ms,
                                success=True,
                                request={"messages": str(messages)[:500]},
                                streaming_attributes=w.as_span_attributes(),
                            )

                        return AnthropicAsyncStreamWrapper(response, tracker, _on_exhaust)

                    duration_ms = (time.time() - start) * 1000
                    usage = _extract_usage(response)

                    log_llm_call(
                        provider="anthropic",
                        model=model,
                        duration_ms=duration_ms,
                        success=True,
                        request={"messages": str(messages)[:500]},
                        response=_extract_response(response),
                        **usage,
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider="anthropic",
                        model=model,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                    )
                    raise

            messages_module.AsyncMessages.create = patched_acreate

        self._instrumented = True
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            from anthropic.resources import messages as messages_module

            if self._original_create and hasattr(messages_module, "Messages"):
                messages_module.Messages.create = self._original_create
            if self._original_acreate and hasattr(messages_module, "AsyncMessages"):
                messages_module.AsyncMessages.create = self._original_acreate

            self._instrumented = False
            self._original_create = None
            self._original_acreate = None
            return True
        except ImportError:
            return False
