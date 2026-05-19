"""
OpenAI SDK instrumentor.

Patches OpenAI client to auto-capture chat completions.
"""

from __future__ import annotations

import functools
import importlib.util
import time
from typing import Any

from ..base import Instrumentor, InstrumentorType
from ._shared import log_llm_call


def _detect_provider(completions_instance) -> str:
    """
    Detect provider from client base_url.

    xAI uses the OpenAI-compatible API with base_url="https://api.x.ai/v1".
    """
    try:
        # Navigate from Completions -> Chat -> Client
        client = getattr(completions_instance, "_client", None)
        if client is None:
            return "openai"
        base_url = str(getattr(client, "base_url", ""))
        if "x.ai" in base_url:
            return "xai"
    except Exception:
        pass
    return "openai"


def _build_request_dict(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a request dict capturing messages and optional params."""
    req: dict[str, Any] = {"messages": str(kwargs.get("messages", []))[:500]}
    if "temperature" in kwargs:
        req["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        req["max_tokens"] = kwargs["max_tokens"]
    return req


def _build_response_dict(response: Any) -> dict[str, Any]:
    """Build a response dict from an OpenAI ChatCompletion response."""
    resp: dict[str, Any] = {}
    if not getattr(response, "choices", None):
        return resp

    choice = response.choices[0]
    message = getattr(choice, "message", None)

    if message is not None:
        resp["content"] = str(getattr(message, "content", ""))[:500]

        # Capture tool calls if present
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            resp["tool_calls"] = [
                {
                    "name": getattr(tc.function, "name", ""),
                    "arguments": str(getattr(tc.function, "arguments", ""))[:300],
                }
                for tc in tool_calls
            ]

    resp["finish_reason"] = getattr(choice, "finish_reason", None)
    return resp


class OpenAIInstrumentor(Instrumentor):
    """Instrumentor for OpenAI SDK."""

    _instrumented = False
    _original_create: Any | None = None
    _original_acreate: Any | None = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("openai") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        if not self.is_available():
            return False

        try:
            from openai.resources.chat import completions as chat_completions
        except ImportError:
            return False

        # Patch sync completions.create
        if hasattr(chat_completions, "Completions"):
            self._original_create = chat_completions.Completions.create

            @functools.wraps(self._original_create)
            def patched_create(inst, *args, **kwargs):
                start = time.time()
                model = kwargs.get("model", "unknown")
                provider = _detect_provider(inst)

                try:
                    response = self._original_create(inst, *args, **kwargs)

                    if kwargs.get("stream"):
                        from ._streaming import (
                            LoggingWrapper,
                            StreamingSpanWrapper,
                        )

                        class OpenAIStreamWrapper(StreamingSpanWrapper):
                            def __next__(self):
                                chunk = super().__next__()
                                usage = getattr(chunk, "usage", None)
                                if usage:
                                    self.accumulated_input_tokens = (
                                        getattr(usage, "prompt_tokens", 0) or 0
                                    )
                                    self.accumulated_output_tokens = (
                                        getattr(usage, "completion_tokens", 0) or 0
                                    )
                                return chunk

                        wrapper = OpenAIStreamWrapper(response, request_start_time=start)

                        def _on_exhaust(w):
                            duration_ms = (time.time() - start) * 1000
                            log_llm_call(
                                provider=provider,
                                model=model,
                                input_tokens=w.accumulated_input_tokens,
                                output_tokens=w.accumulated_output_tokens,
                                duration_ms=duration_ms,
                                success=True,
                                request=_build_request_dict(kwargs),
                                streaming_attributes=w.as_span_attributes(),
                            )

                        return LoggingWrapper(wrapper, _on_exhaust)

                    duration_ms = (time.time() - start) * 1000

                    usage = getattr(response, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = (
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    )

                    log_llm_call(
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=True,
                        request=_build_request_dict(kwargs),
                        response=_build_response_dict(response),
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider=provider,
                        model=model,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                    )
                    raise

            chat_completions.Completions.create = patched_create

        # Patch async completions.create
        if hasattr(chat_completions, "AsyncCompletions"):
            self._original_acreate = chat_completions.AsyncCompletions.create

            @functools.wraps(self._original_acreate)
            async def patched_acreate(inst, *args, **kwargs):
                start = time.time()
                model = kwargs.get("model", "unknown")
                provider = _detect_provider(inst)

                try:
                    response = await self._original_acreate(inst, *args, **kwargs)

                    if kwargs.get("stream"):
                        from ._streaming import (
                            AsyncLoggingWrapper,
                            StreamingSpanWrapper,
                        )

                        class OpenAIAsyncStreamWrapper(AsyncLoggingWrapper):
                            def _extract_tokens(self, chunk):
                                usage = getattr(chunk, "usage", None)
                                if usage:
                                    self._wrapper.accumulated_input_tokens = (
                                        getattr(usage, "prompt_tokens", 0) or 0
                                    )
                                    self._wrapper.accumulated_output_tokens = (
                                        getattr(usage, "completion_tokens", 0) or 0
                                    )

                        tracker = StreamingSpanWrapper(iter([]), request_start_time=start)

                        def _on_exhaust(w):
                            duration_ms = (time.time() - start) * 1000
                            log_llm_call(
                                provider=provider,
                                model=model,
                                input_tokens=w.accumulated_input_tokens,
                                output_tokens=w.accumulated_output_tokens,
                                duration_ms=duration_ms,
                                success=True,
                                request=_build_request_dict(kwargs),
                                streaming_attributes=w.as_span_attributes(),
                            )

                        return OpenAIAsyncStreamWrapper(response, tracker, _on_exhaust)

                    duration_ms = (time.time() - start) * 1000

                    usage = getattr(response, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = (
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    )

                    log_llm_call(
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=True,
                        request=_build_request_dict(kwargs),
                        response=_build_response_dict(response),
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider=provider,
                        model=model,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                    )
                    raise

            chat_completions.AsyncCompletions.create = patched_acreate

        self._instrumented = True
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            from openai.resources.chat import completions as chat_completions

            if self._original_create and hasattr(chat_completions, "Completions"):
                chat_completions.Completions.create = self._original_create
            if self._original_acreate and hasattr(chat_completions, "AsyncCompletions"):
                chat_completions.AsyncCompletions.create = self._original_acreate

            self._instrumented = False
            self._original_create = None
            self._original_acreate = None
            return True
        except ImportError:
            return False
