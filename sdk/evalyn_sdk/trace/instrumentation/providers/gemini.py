"""
Google Gemini SDK instrumentor.

Patches Gemini client to auto-capture content generation.
"""

from __future__ import annotations

import functools
import importlib.util
import time
from typing import Any, Optional

from ..base import Instrumentor, InstrumentorType
from ._shared import log_llm_call


class GeminiInstrumentor(Instrumentor):
    """Instrumentor for Google Gemini SDK."""

    _instrumented = False
    _original_generate: Optional[Any] = None
    _original_legacy_generate: Optional[Any] = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def instrumentor_type(self) -> InstrumentorType:
        return InstrumentorType.MONKEY_PATCH

    def is_available(self) -> bool:
        # Check for new google-genai or legacy google-generativeai
        try:
            if importlib.util.find_spec("google.genai") is not None:
                return True
        except (ModuleNotFoundError, ImportError):
            pass
        try:
            if importlib.util.find_spec("google.generativeai") is not None:
                return True
        except (ModuleNotFoundError, ImportError):
            pass
        return False

    def is_instrumented(self) -> bool:
        return self._instrumented

    def instrument(self) -> bool:
        if self._instrumented:
            return True

        # Try new google-genai package first
        try:
            from google.genai import models as models_module

            self._patch_new_genai(models_module)
            self._instrumented = True
            return True
        except ImportError:
            pass

        # Fall back to legacy google-generativeai
        try:
            import google.generativeai as genai

            self._patch_legacy_genai(genai)
            self._instrumented = True
            return True
        except ImportError:
            return False

    def _patch_new_genai(self, models_module: Any) -> None:
        """Patch the new google-genai package."""
        if hasattr(models_module, "Models"):
            self._original_generate = models_module.Models.generate_content

            @functools.wraps(self._original_generate)
            def patched_generate(inst, *args, **kwargs):
                start = time.time()
                model = kwargs.get("model", args[0] if args else "unknown")
                if hasattr(model, "name"):
                    model = model.name

                try:
                    response = self._original_generate(inst, *args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    # Extract token usage
                    usage = getattr(response, "usage_metadata", None)
                    input_tokens = (
                        getattr(usage, "prompt_token_count", 0) if usage else 0
                    )
                    output_tokens = (
                        getattr(usage, "candidates_token_count", 0) if usage else 0
                    )
                    tool_tokens = (
                        getattr(usage, "tool_use_prompt_token_count", 0) if usage else 0
                    )

                    # Extract grounding metadata
                    search_queries = None
                    sources = None
                    if response.candidates:
                        gm = getattr(response.candidates[0], "grounding_metadata", None)
                        if gm:
                            search_queries = getattr(gm, "web_search_queries", None)
                            chunks = getattr(gm, "grounding_chunks", None)
                            if chunks:
                                sources = []
                                for chunk in chunks[:5]:
                                    web = getattr(chunk, "web", None)
                                    if web:
                                        sources.append(
                                            {
                                                "title": getattr(web, "title", "")[
                                                    :100
                                                ],
                                                "uri": getattr(web, "uri", "")[:200],
                                            }
                                        )

                    log_llm_call(
                        provider="gemini",
                        model=str(model),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=duration_ms,
                        success=True,
                        response={"text": str(getattr(response, "text", ""))[:500]},
                        tool_tokens=tool_tokens,
                        search_queries=search_queries,
                        sources=sources,
                    )

                    return response
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    log_llm_call(
                        provider="gemini",
                        model=str(model),
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                    )
                    raise

            models_module.Models.generate_content = patched_generate

    def _patch_legacy_genai(self, genai: Any) -> None:
        """Patch older google-generativeai package."""
        if hasattr(genai, "GenerativeModel"):
            self._original_legacy_generate = genai.GenerativeModel.generate_content

            @functools.wraps(self._original_legacy_generate)
            def patched_generate(inst, *args, **kwargs):
                start = time.time()
                model = getattr(inst, "model_name", "unknown")

                try:
                    response = self._original_legacy_generate(inst, *args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    # Extract token usage
                    usage = getattr(response, "usage_metadata", None)
                    input_tokens = (
                        getattr(usage, "prompt_token_count", 0) if usage else 0
                    )
                    output_tokens = (
                        getattr(usage, "candidates_token_count", 0) if usage else 0
                    )

                    log_llm_call(
                        provider="gemini",
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
                        provider="gemini",
                        model=model,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                    )
                    raise

            genai.GenerativeModel.generate_content = patched_generate

    def uninstrument(self) -> bool:
        if not self._instrumented:
            return True

        try:
            if self._original_generate:
                from google.genai import models as models_module

                if hasattr(models_module, "Models"):
                    models_module.Models.generate_content = self._original_generate
        except ImportError:
            pass

        try:
            if self._original_legacy_generate:
                import google.generativeai as genai

                if hasattr(genai, "GenerativeModel"):
                    genai.GenerativeModel.generate_content = (
                        self._original_legacy_generate
                    )
        except ImportError:
            pass

        self._instrumented = False
        self._original_generate = None
        self._original_legacy_generate = None
        return True
