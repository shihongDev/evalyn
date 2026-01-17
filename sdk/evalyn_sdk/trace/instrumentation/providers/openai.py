"""
OpenAI SDK instrumentor.

Patches OpenAI client to auto-capture chat completions.
"""

from __future__ import annotations

import functools
import importlib.util
import time
from typing import Any, Optional

from ..base import Instrumentor, InstrumentorType
from ._shared import log_llm_call


class OpenAIInstrumentor(Instrumentor):
    """Instrumentor for OpenAI SDK."""

    _instrumented = False
    _original_create: Optional[Any] = None
    _original_acreate: Optional[Any] = None

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
                messages = kwargs.get("messages", [])

                try:
                    response = self._original_create(inst, *args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    # Extract token usage
                    usage = getattr(response, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = (
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    )

                    log_llm_call(
                        provider="openai",
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=True,
                        request={"messages": str(messages)[:500]},
                        response={
                            "content": str(
                                getattr(response.choices[0].message, "content", "")
                            )[:500]
                            if response.choices
                            else None
                        },
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider="openai",
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

                try:
                    response = await self._original_acreate(inst, *args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    usage = getattr(response, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = (
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    )

                    log_llm_call(
                        provider="openai",
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=True,
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider="openai",
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
