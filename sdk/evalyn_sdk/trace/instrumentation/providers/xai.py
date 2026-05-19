"""
xAI SDK instrumentor.

Patches xai-sdk client to auto-capture chat completions.

Note: This instrumentor handles the official xai-sdk gRPC-based package.
For xAI calls via OpenAI-compatible API (base_url="https://api.x.ai/v1"),
the OpenAI instrumentor handles detection via _detect_provider().
"""

from __future__ import annotations

import functools
import importlib.util
import time
from typing import Any

from ..base import Instrumentor, InstrumentorType
from ._shared import log_llm_call


def _extract_usage(response: Any) -> tuple[int, int]:
    """Extract input/output token counts from response."""
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0

    input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
        usage, "input_tokens", 0
    )
    output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
        usage, "output_tokens", 0
    )
    return input_tokens, output_tokens


def _extract_content(response: Any) -> str:
    """Extract response content, truncated to 500 chars."""
    if hasattr(response, "content"):
        return str(response.content)[:500]
    if hasattr(response, "text"):
        return str(response.text)[:500]
    if hasattr(response, "message"):
        return str(getattr(response.message, "content", ""))[:500]
    return ""


class XAIInstrumentor(Instrumentor):
    """Instrumentor for official xai-sdk package."""

    _instrumented = False
    _original_sample: Any | None = None
    _original_asample: Any | None = None

    @property
    def name(self) -> str:
        return "xai"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("xai_sdk") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        if not self.is_available():
            return False

        try:
            from xai_sdk.chat import Chat
        except ImportError:
            return False

        # Store original methods
        self._original_sample = Chat.sample

        def _extract_request(args, kwargs):
            """Build a request dict from sample() arguments."""
            parts = []
            for a in args:
                parts.append(str(a)[:500])
            prompt = kwargs.get("prompt") or kwargs.get("message")
            if prompt:
                parts.append(str(prompt)[:500])
            return {"messages": parts} if parts else None

        @functools.wraps(self._original_sample)
        def patched_sample(chat_self, *args, **kwargs):
            start = time.time()
            # Extract model from Chat instance
            model = getattr(chat_self, "_model", None) or getattr(
                chat_self, "model", "unknown"
            )
            request = _extract_request(args, kwargs)

            try:
                response = self._original_sample(chat_self, *args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                input_tokens, output_tokens = _extract_usage(response)
                content = _extract_content(response)

                log_llm_call(
                    provider="xai",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=duration_ms,
                    success=True,
                    request=request,
                    response={"content": content} if content else None,
                )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                log_llm_call(
                    provider="xai",
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    request=request,
                )
                raise

        Chat.sample = patched_sample

        # Patch async sample if it exists
        if hasattr(Chat, "asample"):
            self._original_asample = Chat.asample

            @functools.wraps(self._original_asample)
            async def patched_asample(chat_self, *args, **kwargs):
                start = time.time()
                model = getattr(chat_self, "_model", None) or getattr(
                    chat_self, "model", "unknown"
                )
                request = _extract_request(args, kwargs)

                try:
                    response = await self._original_asample(chat_self, *args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    input_tokens, output_tokens = _extract_usage(response)
                    content = _extract_content(response)

                    log_llm_call(
                        provider="xai",
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=True,
                        request=request,
                        response={"content": content} if content else None,
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider="xai",
                        model=model,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        request=request,
                    )
                    raise

            Chat.asample = patched_asample

        self._instrumented = True
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            from xai_sdk.chat import Chat

            if self._original_sample:
                Chat.sample = self._original_sample
            if self._original_asample and hasattr(Chat, "asample"):
                Chat.asample = self._original_asample

            self._instrumented = False
            self._original_sample = None
            self._original_asample = None
            return True
        except ImportError:
            return False
