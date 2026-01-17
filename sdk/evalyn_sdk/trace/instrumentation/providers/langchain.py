"""
LangChain SDK instrumentor.

Uses LangChain's callback system to capture LLM and tool calls.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Dict, Optional

from ..base import Instrumentor, InstrumentorType
from ._shared import log_llm_call, log_tool_call


class LangChainInstrumentor(Instrumentor):
    """Instrumentor for LangChain SDK."""

    _instrumented = False
    _handler: Optional[Any] = None

    @property
    def name(self) -> str:
        return "langchain"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        return importlib.util.find_spec("langchain_core") is not None

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            return False

        # Create a callback handler that logs to Evalyn
        class EvalynCallbackHandler(BaseCallbackHandler):
            def __init__(self):
                self._start_times: Dict[str, float] = {}

            def on_llm_start(self, serialized, prompts, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                self._start_times[run_id] = time.time()

            def on_llm_end(self, response, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                start = self._start_times.pop(run_id, time.time())
                duration_ms = (time.time() - start) * 1000

                # Extract info from response
                llm_output = getattr(response, "llm_output", {}) or {}
                token_usage = llm_output.get("token_usage", {})
                model = llm_output.get("model_name", "unknown")

                log_llm_call(
                    provider="langchain",
                    model=model,
                    input_tokens=token_usage.get("prompt_tokens", 0),
                    output_tokens=token_usage.get("completion_tokens", 0),
                    duration_ms=duration_ms,
                    success=True,
                )

            def on_llm_error(self, error, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                start = self._start_times.pop(run_id, time.time())
                duration_ms = (time.time() - start) * 1000

                log_llm_call(
                    provider="langchain",
                    model="unknown",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(error),
                )

            def on_tool_start(self, serialized, input_str, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                self._start_times[f"tool_{run_id}"] = time.time()

            def on_tool_end(self, output, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                start = self._start_times.pop(f"tool_{run_id}", time.time())
                duration_ms = (time.time() - start) * 1000

                tool_name = kwargs.get("name", "unknown")
                log_tool_call(
                    tool_name=tool_name,
                    tool_input=kwargs.get("input", ""),
                    tool_output=output,
                    duration_ms=duration_ms,
                    success=True,
                )

            def on_tool_error(self, error, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                start = self._start_times.pop(f"tool_{run_id}", time.time())
                duration_ms = (time.time() - start) * 1000

                tool_name = kwargs.get("name", "unknown")
                log_tool_call(
                    tool_name=tool_name,
                    tool_input=kwargs.get("input", ""),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(error),
                )

        # Store handler for later use
        self._handler = EvalynCallbackHandler()

        # Make handler available on the module for backwards compat
        import evalyn_sdk.trace.auto_instrument as auto_instrument_module

        auto_instrument_module.langchain_handler = self._handler

        self._instrumented = True
        return True

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        self._handler = None
        self._instrumented = False
        return True

    def get_handler(self) -> Optional[Any]:
        """Get the LangChain callback handler for use with chains."""
        return self._handler
