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
                    duration_ms = (time.time() - start) * 1000
                    usage = _extract_usage(response)

                    log_llm_call(
                        provider="anthropic",
                        model=model,
                        duration_ms=duration_ms,
                        success=True,
                        request={"messages": str(messages)[:500]},
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
                    duration_ms = (time.time() - start) * 1000
                    usage = _extract_usage(response)

                    log_llm_call(
                        provider="anthropic",
                        model=model,
                        duration_ms=duration_ms,
                        success=True,
                        request={"messages": str(messages)[:500]},
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
