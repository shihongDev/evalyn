"""
Tests for LLM provider instrumentors (OpenAI, Anthropic, Gemini, xAI).

These tests mock the SDK imports to test instrumentation logic
without requiring the actual SDKs to be installed.

Run with: pytest tests/test_llm_provider_instrumentors.py -v
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace import context as span_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_span_context():
    """Reset span context for test isolation."""
    span_context._span_stack.set([])
    span_context._span_collector.set([])
    span_context._active_call.set(None)


def _get_collected_spans() -> List[Span]:
    """Get spans from the collector."""
    return span_context._span_collector.get()


def _make_mock_tracer():
    """Create a mock tracer for log_llm_call."""
    tracer = MagicMock()
    tracer.log_event = MagicMock()
    return tracer


def _make_module(name: str, is_package: bool = False) -> types.ModuleType:
    """Create a module with proper __spec__ so importlib.util.find_spec works."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    if is_package:
        mod.__path__ = []
    return mod


# ---------------------------------------------------------------------------
# Shared utility tests (_shared.py)
# ---------------------------------------------------------------------------

class TestCalculateCost:
    """Tests for calculate_cost and calculate_cost_with_cache."""

    def test_known_model_cost(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.50 + 10.00)

    def test_known_model_cost_case_insensitive(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("GPT-4O", 1_000_000, 0)
        assert cost == pytest.approx(2.50)

    def test_unknown_model_fallback(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.0)

    def test_substring_matching_specificity(self):
        """gpt-4o-mini should match before gpt-4o."""
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost_mini = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        cost_4o = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost_mini < cost_4o

    def test_cache_cost_anthropic(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import (
            calculate_cost_with_cache,
        )
        cost = calculate_cost_with_cache(
            "claude-sonnet-4",
            input_tokens=500_000,
            output_tokens=500_000,
            cache_creation_tokens=100_000,
            cache_read_tokens=200_000,
        )
        assert cost == pytest.approx(1.5 + 7.5 + 0.375 + 0.06)

    def test_is_model_pricing_known(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import (
            is_model_pricing_known,
        )
        assert is_model_pricing_known("gpt-4o") is True
        assert is_model_pricing_known("claude-sonnet-4") is True
        assert is_model_pricing_known("totally-unknown-model") is False

    def test_zero_tokens(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_xai_model_cost(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("grok-4", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_gemini_model_cost(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.10 + 0.40)


class TestLogLlmCall:
    """Tests for log_llm_call span creation."""

    def setup_method(self):
        _reset_span_context()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_creates_span_with_correct_attributes(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        log_llm_call(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            duration_ms=500.0,
            success=True,
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s.name == "openai:gpt-4o"
        assert s.span_type == "llm_call"
        assert s.attributes["provider"] == "openai"
        assert s.attributes["model"] == "gpt-4o"
        assert s.attributes["input_tokens"] == 100
        assert s.attributes["output_tokens"] == 50
        assert s.status == "ok"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_error_creates_error_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        log_llm_call(
            provider="openai",
            model="gpt-4o",
            duration_ms=100.0,
            success=False,
            error="Rate limit exceeded",
        )

        spans = _get_collected_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s.status == "error"
        assert s.attributes["error"] == "Rate limit exceeded"

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_cache_tokens_in_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        log_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=200,
            cache_read_tokens=300,
        )

        spans = _get_collected_spans()
        s = spans[0]
        assert s.attributes["cache_creation_input_tokens"] == 200
        assert s.attributes["cache_read_input_tokens"] == 300

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_request_response_in_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        log_llm_call(
            provider="openai",
            model="gpt-4o",
            request={"messages": "hello"},
            response={"content": "world"},
        )

        spans = _get_collected_spans()
        s = spans[0]
        assert s.attributes["request"] == {"messages": "hello"}
        assert s.attributes["response"] == {"content": "world"}

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_tool_tokens_and_search_in_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        log_llm_call(
            provider="gemini",
            model="gemini-2.5-flash",
            tool_tokens=42,
            search_queries=["test query"],
            sources=[{"title": "Example", "uri": "https://example.com"}],
        )

        spans = _get_collected_spans()
        s = spans[0]
        assert s.attributes["tool_tokens"] == 42
        assert s.attributes["search_queries"] == ["test query"]
        assert len(s.attributes["sources"]) == 1

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_logs_trace_event(self, mock_get_tracer):
        tracer = _make_mock_tracer()
        mock_get_tracer.return_value = tracer
        from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call

        log_llm_call(provider="openai", model="gpt-4o")

        tracer.log_event.assert_called_once()
        call_args = tracer.log_event.call_args
        assert call_args[0][0] == "openai.completion"
        detail = call_args[0][1]
        assert "span_id" in detail


# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------

def _build_mock_openai_response(
    content="Hello!",
    prompt_tokens=10,
    completion_tokens=20,
    model="gpt-4o",
    finish_reason="stop",
):
    """Build a mock OpenAI ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.usage = usage
    response.choices = [choice]
    response.model = model
    return response


def _build_mock_anthropic_response(
    content="Hello from Claude!",
    input_tokens=15,
    output_tokens=25,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
    model="claude-sonnet-4-20250514",
    stop_reason="end_turn",
):
    """Build a mock Anthropic Message response."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens

    content_block = MagicMock()
    content_block.text = content
    content_block.type = "text"

    response = MagicMock()
    response.usage = usage
    response.content = [content_block]
    response.model = model
    response.stop_reason = stop_reason
    response.id = "msg_test123"
    return response


def _build_mock_gemini_response(
    text="Hello from Gemini!",
    prompt_token_count=12,
    candidates_token_count=18,
    tool_use_prompt_token_count=0,
):
    """Build a mock Gemini GenerateContentResponse."""
    usage = MagicMock()
    usage.prompt_token_count = prompt_token_count
    usage.candidates_token_count = candidates_token_count
    usage.tool_use_prompt_token_count = tool_use_prompt_token_count

    candidate = MagicMock()
    candidate.grounding_metadata = None

    response = MagicMock()
    response.usage_metadata = usage
    response.text = text
    response.candidates = [candidate]
    return response


def _build_mock_xai_response(
    content="Hello from Grok!",
    prompt_tokens=8,
    completion_tokens=15,
):
    """Build a mock xAI response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.input_tokens = 0
    usage.output_tokens = 0

    response = MagicMock()
    response.usage = usage
    response.content = content
    response.text = content
    return response


# ---------------------------------------------------------------------------
# Mock SDK installers
#
# Each installs fake modules into sys.modules with proper __spec__ so that
# importlib.util.find_spec() works and `from x import y` succeeds.
#
# The key pattern for span-capture tests:
#   1. Install mock modules
#   2. Call instrumentor.instrument() (patches the class methods)
#   3. Replace instrumentor._original_create with a callable mock
#      (The patched wrapper references self._original_create at call time,
#       so swapping it after patching works.)
#   4. Call the patched method
#   5. Assert spans are collected
# ---------------------------------------------------------------------------

def _install_mock_openai():
    """Install mock openai module hierarchy into sys.modules."""
    openai_mod = _make_module("openai", is_package=True)
    resources_mod = _make_module("openai.resources", is_package=True)
    chat_mod = _make_module("openai.resources.chat", is_package=True)
    completions_mod = _make_module("openai.resources.chat.completions")

    class Completions:
        def create(self, *args, **kwargs):
            pass

    class AsyncCompletions:
        async def create(self, *args, **kwargs):
            pass

    completions_mod.Completions = Completions
    completions_mod.AsyncCompletions = AsyncCompletions

    sys.modules["openai"] = openai_mod
    sys.modules["openai.resources"] = resources_mod
    sys.modules["openai.resources.chat"] = chat_mod
    sys.modules["openai.resources.chat.completions"] = completions_mod

    return completions_mod


def _cleanup_openai():
    for key in list(sys.modules.keys()):
        if key.startswith("openai"):
            del sys.modules[key]


def _install_mock_anthropic():
    """Install mock anthropic module hierarchy into sys.modules."""
    anthropic_mod = _make_module("anthropic", is_package=True)
    resources_mod = _make_module("anthropic.resources", is_package=True)
    messages_mod = _make_module("anthropic.resources.messages")

    class Messages:
        def create(self, *args, **kwargs):
            pass

    class AsyncMessages:
        async def create(self, *args, **kwargs):
            pass

    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages

    sys.modules["anthropic"] = anthropic_mod
    sys.modules["anthropic.resources"] = resources_mod
    sys.modules["anthropic.resources.messages"] = messages_mod

    return messages_mod


def _cleanup_anthropic():
    for key in list(sys.modules.keys()):
        if key.startswith("anthropic"):
            del sys.modules[key]


def _install_mock_genai():
    """Install mock google.genai module hierarchy into sys.modules."""
    google_mod = _make_module("google", is_package=True)
    genai_mod = _make_module("google.genai", is_package=True)
    models_mod = _make_module("google.genai.models")

    class Models:
        def generate_content(self, *args, **kwargs):
            pass

    models_mod.Models = Models

    google_mod.genai = genai_mod
    genai_mod.models = models_mod

    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.models"] = models_mod

    return models_mod


def _cleanup_genai():
    for key in list(sys.modules.keys()):
        if key.startswith("google"):
            del sys.modules[key]


def _install_mock_xai():
    """Install mock xai_sdk module hierarchy into sys.modules."""
    xai_mod = _make_module("xai_sdk", is_package=True)
    chat_mod = _make_module("xai_sdk.chat")

    class Chat:
        def __init__(self):
            self._model = "grok-4"

        def sample(self, *args, **kwargs):
            pass

        async def asample(self, *args, **kwargs):
            pass

    chat_mod.Chat = Chat

    xai_mod.chat = chat_mod

    sys.modules["xai_sdk"] = xai_mod
    sys.modules["xai_sdk.chat"] = chat_mod

    return chat_mod


def _cleanup_xai():
    for key in list(sys.modules.keys()):
        if key.startswith("xai_sdk"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# OpenAI Instrumentor Tests
# ---------------------------------------------------------------------------

class TestOpenAIInstrumentor:
    """Tests for the OpenAI instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_name(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import (
            OpenAIInstrumentor,
        )
        inst = OpenAIInstrumentor()
        assert inst.name == "openai"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import (
            OpenAIInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = OpenAIInstrumentor()
        assert inst.instrumentor_type == InstrumentorType.MONKEY_PATCH

    def test_is_available_when_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import (
            OpenAIInstrumentor,
        )
        inst = OpenAIInstrumentor()
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert inst.is_available() is True

    def test_is_available_when_not_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import (
            OpenAIInstrumentor,
        )
        inst = OpenAIInstrumentor()
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_instrument_patches_create(self):
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            original_create = completions_mod.Completions.create

            result = inst.instrument()

            assert result is True
            assert inst.is_instrumented() is True
            assert completions_mod.Completions.create is not original_create
            assert inst._original_create is original_create

            inst.uninstrument()
        finally:
            _cleanup_openai()

    def test_instrument_patches_async_create(self):
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            original_acreate = completions_mod.AsyncCompletions.create

            inst.instrument()

            assert completions_mod.AsyncCompletions.create is not original_acreate
            assert inst._original_acreate is original_acreate

            inst.uninstrument()
        finally:
            _cleanup_openai()

    def test_uninstrument_restores_originals(self):
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            original_create = completions_mod.Completions.create
            original_acreate = completions_mod.AsyncCompletions.create

            inst.instrument()
            assert completions_mod.Completions.create is not original_create

            result = inst.uninstrument()

            assert result is True
            assert inst.is_instrumented() is False
            assert completions_mod.Completions.create is original_create
            assert completions_mod.AsyncCompletions.create is original_acreate
        finally:
            _cleanup_openai()

    def test_idempotent_instrument(self):
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()
            patched = completions_mod.Completions.create
            inst.instrument()
            assert completions_mod.Completions.create is patched

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_openai_response(
                content="Hi there!",
                prompt_tokens=42,
                completion_tokens=17,
            )
            # Swap the original with our mock - the patched closure reads
            # self._original_create at call time
            inst._original_create = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            fake_instance._client = None
            result = completions_mod.Completions.create(
                fake_instance,
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.span_type == "llm_call"
            assert s.attributes["provider"] == "openai"
            assert s.attributes["model"] == "gpt-4o"
            assert s.attributes["input_tokens"] == 42
            assert s.attributes["output_tokens"] == 17
            assert s.status == "ok"
            assert s.attributes["cost_usd"] > 0

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_captures_request_response(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_openai_response(content="The answer is 42")
            inst._original_create = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            fake_instance._client = None
            messages = [{"role": "user", "content": "What is the answer?"}]
            completions_mod.Completions.create(
                fake_instance, model="gpt-4o", messages=messages,
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert "request" in s.attributes
            assert "response" in s.attributes

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_error_creates_error_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            def failing_create(self, *args, **kwargs):
                raise RuntimeError("API rate limit exceeded")

            inst._original_create = failing_create

            fake_instance = MagicMock()
            fake_instance._client = None

            with pytest.raises(RuntimeError, match="API rate limit exceeded"):
                completions_mod.Completions.create(
                    fake_instance, model="gpt-4o", messages=[],
                )

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.status == "error"
            assert "API rate limit exceeded" in s.attributes.get("error", "")

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_xai_provider_detection(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_openai_response(
                content="Grok says hi", prompt_tokens=5, completion_tokens=10,
            )
            inst._original_create = lambda self, *a, **kw: mock_response

            # Create a fake instance with xAI base_url
            fake_instance = MagicMock()
            fake_instance._client = MagicMock()
            fake_instance._client.base_url = "https://api.x.ai/v1"

            completions_mod.Completions.create(
                fake_instance, model="grok-4", messages=[],
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes["provider"] == "xai"

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_async_create_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_openai_response(
                content="Async response",
                prompt_tokens=20,
                completion_tokens=30,
            )

            async def mock_acreate(self, *args, **kwargs):
                return mock_response

            inst._original_acreate = mock_acreate

            fake_instance = MagicMock()
            fake_instance._client = None

            async def run():
                return await completions_mod.AsyncCompletions.create(
                    fake_instance, model="gpt-4o-mini", messages=[],
                )

            result = asyncio.get_event_loop().run_until_complete(run())
            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.attributes["provider"] == "openai"
            assert s.attributes["model"] == "gpt-4o-mini"
            assert s.attributes["input_tokens"] == 20
            assert s.attributes["output_tokens"] == 30

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_async_create_error(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            async def failing_acreate(self, *args, **kwargs):
                raise ConnectionError("Network timeout")

            inst._original_acreate = failing_acreate

            fake_instance = MagicMock()
            fake_instance._client = None

            async def run():
                return await completions_mod.AsyncCompletions.create(
                    fake_instance, model="gpt-4o", messages=[],
                )

            with pytest.raises(ConnectionError, match="Network timeout"):
                asyncio.get_event_loop().run_until_complete(run())

            spans = _get_collected_spans()
            assert len(spans) == 1
            assert spans[0].status == "error"

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_no_usage_data(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            response = MagicMock()
            response.usage = None
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Hi"

            inst._original_create = lambda self, *a, **kw: response

            fake_instance = MagicMock()
            fake_instance._client = None
            completions_mod.Completions.create(
                fake_instance, model="gpt-4o", messages=[],
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes["input_tokens"] == 0
            assert s.attributes["output_tokens"] == 0

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_message_truncation(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            long_content = "x" * 1000
            mock_response = _build_mock_openai_response(content=long_content)
            inst._original_create = lambda self, *a, **kw: mock_response

            long_messages = [{"role": "user", "content": "y" * 1000}]
            fake_instance = MagicMock()
            fake_instance._client = None
            completions_mod.Completions.create(
                fake_instance, model="gpt-4o", messages=long_messages,
            )

            spans = _get_collected_spans()
            s = spans[0]
            req_msg = s.attributes.get("request", {}).get("messages", "")
            assert len(req_msg) <= 500
            resp_content = s.attributes.get("response", {}).get("content", "")
            assert len(resp_content) <= 500

            inst.uninstrument()
        finally:
            _cleanup_openai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_empty_choices(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        completions_mod = _install_mock_openai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.openai import (
                OpenAIInstrumentor,
            )
            inst = OpenAIInstrumentor()
            inst.instrument()

            response = MagicMock()
            response.usage = MagicMock()
            response.usage.prompt_tokens = 5
            response.usage.completion_tokens = 0
            response.choices = []

            inst._original_create = lambda self, *a, **kw: response

            fake_instance = MagicMock()
            fake_instance._client = None
            completions_mod.Completions.create(
                fake_instance, model="gpt-4o", messages=[],
            )

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            # Response content should be None when choices is empty
            assert s.attributes.get("response", {}).get("content") is None

            inst.uninstrument()
        finally:
            _cleanup_openai()


# ---------------------------------------------------------------------------
# Anthropic Instrumentor Tests
# ---------------------------------------------------------------------------

class TestAnthropicInstrumentor:
    """Tests for the Anthropic instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_name(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            AnthropicInstrumentor,
        )
        inst = AnthropicInstrumentor()
        assert inst.name == "anthropic"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            AnthropicInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = AnthropicInstrumentor()
        assert inst.instrumentor_type == InstrumentorType.MONKEY_PATCH

    def test_is_available(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            AnthropicInstrumentor,
        )
        inst = AnthropicInstrumentor()
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert inst.is_available() is True
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_instrument_patches_create(self):
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            original = messages_mod.Messages.create

            result = inst.instrument()

            assert result is True
            assert inst.is_instrumented() is True
            assert messages_mod.Messages.create is not original
            assert inst._original_create is original

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    def test_instrument_patches_async_create(self):
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            original = messages_mod.AsyncMessages.create

            inst.instrument()

            assert messages_mod.AsyncMessages.create is not original
            assert inst._original_acreate is original

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    def test_uninstrument_restores_originals(self):
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            original = messages_mod.Messages.create
            original_async = messages_mod.AsyncMessages.create

            inst.instrument()
            result = inst.uninstrument()

            assert result is True
            assert inst.is_instrumented() is False
            assert messages_mod.Messages.create is original
            assert messages_mod.AsyncMessages.create is original_async
        finally:
            _cleanup_anthropic()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            mock_response = _build_mock_anthropic_response(
                input_tokens=50,
                output_tokens=30,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=200,
            )
            inst._original_create = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            messages = [{"role": "user", "content": "Tell me a joke"}]
            result = messages_mod.Messages.create(
                fake_instance,
                model="claude-sonnet-4-20250514",
                messages=messages,
            )

            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.span_type == "llm_call"
            assert s.attributes["provider"] == "anthropic"
            assert s.attributes["model"] == "claude-sonnet-4-20250514"
            assert s.attributes["input_tokens"] == 50
            assert s.attributes["output_tokens"] == 30
            assert s.status == "ok"
            assert s.attributes.get("cache_creation_input_tokens") == 100
            assert s.attributes.get("cache_read_input_tokens") == 200

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_captures_request(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            mock_response = _build_mock_anthropic_response()
            inst._original_create = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            messages_mod.Messages.create(
                fake_instance,
                model="claude-sonnet-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert "request" in s.attributes

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_error(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            def failing(self, *args, **kwargs):
                raise ValueError("Overloaded")

            inst._original_create = failing

            fake_instance = MagicMock()
            with pytest.raises(ValueError, match="Overloaded"):
                messages_mod.Messages.create(
                    fake_instance, model="claude-sonnet-4", messages=[],
                )

            spans = _get_collected_spans()
            assert len(spans) == 1
            assert spans[0].status == "error"
            assert "Overloaded" in spans[0].attributes.get("error", "")

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_async_create_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            mock_response = _build_mock_anthropic_response(
                input_tokens=10, output_tokens=20,
            )

            async def mock_acreate(self, *args, **kwargs):
                return mock_response

            inst._original_acreate = mock_acreate

            fake_instance = MagicMock()

            async def run():
                return await messages_mod.AsyncMessages.create(
                    fake_instance,
                    model="claude-sonnet-4",
                    messages=[{"role": "user", "content": "Hi"}],
                )

            result = asyncio.get_event_loop().run_until_complete(run())
            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.attributes["provider"] == "anthropic"
            assert s.attributes["input_tokens"] == 10
            assert s.attributes["output_tokens"] == 20

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_no_usage_data(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            response = MagicMock()
            response.usage = None
            inst._original_create = lambda self, *a, **kw: response

            fake_instance = MagicMock()
            messages_mod.Messages.create(
                fake_instance, model="claude-sonnet-4", messages=[],
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes["input_tokens"] == 0
            assert s.attributes["output_tokens"] == 0

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    def test_extract_usage_helper(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            _extract_usage,
        )

        usage_obj = MagicMock()
        usage_obj.input_tokens = 100
        usage_obj.output_tokens = 200
        usage_obj.cache_creation_input_tokens = 50
        usage_obj.cache_read_input_tokens = 75

        response = MagicMock()
        response.usage = usage_obj

        result = _extract_usage(response)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["cache_creation_tokens"] == 50
        assert result["cache_read_tokens"] == 75

    def test_extract_usage_no_usage(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            _extract_usage,
        )

        response = MagicMock()
        response.usage = None

        result = _extract_usage(response)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["cache_creation_tokens"] == 0
        assert result["cache_read_tokens"] == 0

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_async_create_error(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            async def failing(self, *args, **kwargs):
                raise ValueError("Overloaded async")

            inst._original_acreate = failing

            fake_instance = MagicMock()

            async def run():
                return await messages_mod.AsyncMessages.create(
                    fake_instance, model="claude-sonnet-4", messages=[],
                )

            with pytest.raises(ValueError, match="Overloaded async"):
                asyncio.get_event_loop().run_until_complete(run())

            spans = _get_collected_spans()
            assert len(spans) == 1
            assert spans[0].status == "error"

            inst.uninstrument()
        finally:
            _cleanup_anthropic()

    def test_extract_response_text(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            _extract_response,
        )

        block = MagicMock()
        block.type = "text"
        block.text = "Hello from Claude!"

        response = MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"

        result = _extract_response(response)
        assert result["content"] == "Hello from Claude!"
        assert result["stop_reason"] == "end_turn"
        assert "tool_calls" not in result

    def test_extract_response_tool_use(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            _extract_response,
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me search"

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "web_search"
        tool_block.input = {"query": "weather today"}

        response = MagicMock()
        response.content = [text_block, tool_block]
        response.stop_reason = "tool_use"

        result = _extract_response(response)
        assert "Let me search" in result["content"]
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "web_search"
        assert result["stop_reason"] == "tool_use"

    def test_extract_response_empty(self):
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            _extract_response,
        )

        response = MagicMock()
        response.content = []
        response.stop_reason = None

        result = _extract_response(response)
        assert "content" not in result
        assert "tool_calls" not in result

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sync_create_captures_response(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        messages_mod = _install_mock_anthropic()
        try:
            from evalyn_sdk.trace.instrumentation.providers.anthropic import (
                AnthropicInstrumentor,
            )
            inst = AnthropicInstrumentor()
            inst.instrument()

            mock_response = _build_mock_anthropic_response(
                content="Here is a joke",
                stop_reason="end_turn",
            )
            inst._original_create = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            messages_mod.Messages.create(
                fake_instance,
                model="claude-sonnet-4",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert "response" in s.attributes
            resp = s.attributes["response"]
            assert "Here is a joke" in resp.get("content", "")
            assert resp.get("stop_reason") == "end_turn"

            inst.uninstrument()
        finally:
            _cleanup_anthropic()


# ---------------------------------------------------------------------------
# Gemini Instrumentor Tests
# ---------------------------------------------------------------------------

class TestGeminiInstrumentor:
    """Tests for the Google Gemini instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_name(self):
        from evalyn_sdk.trace.instrumentation.providers.gemini import (
            GeminiInstrumentor,
        )
        inst = GeminiInstrumentor()
        assert inst.name == "gemini"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.providers.gemini import (
            GeminiInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = GeminiInstrumentor()
        assert inst.instrumentor_type == InstrumentorType.MONKEY_PATCH

    def test_is_available_with_new_genai(self):
        from evalyn_sdk.trace.instrumentation.providers.gemini import (
            GeminiInstrumentor,
        )
        inst = GeminiInstrumentor()

        def find_spec_side_effect(name):
            if name == "google.genai":
                return MagicMock()
            return None

        with patch("importlib.util.find_spec", side_effect=find_spec_side_effect):
            assert inst.is_available() is True

    def test_is_available_not_installed(self):
        from evalyn_sdk.trace.instrumentation.providers.gemini import (
            GeminiInstrumentor,
        )
        inst = GeminiInstrumentor()

        def find_spec_raises(name):
            if name in ("google.genai", "google.generativeai"):
                raise ModuleNotFoundError()
            return None

        with patch("importlib.util.find_spec", side_effect=find_spec_raises):
            assert inst.is_available() is False

    def test_instrument_patches_generate_content(self):
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            original = models_mod.Models.generate_content

            result = inst.instrument()

            assert result is True
            assert inst.is_instrumented() is True
            assert models_mod.Models.generate_content is not original
            assert inst._original_generate is original

            inst.uninstrument()
        finally:
            _cleanup_genai()

    def test_uninstrument_restores_original(self):
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            original = models_mod.Models.generate_content

            inst.instrument()
            result = inst.uninstrument()

            assert result is True
            assert inst.is_instrumented() is False
            assert models_mod.Models.generate_content is original
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_generate_content_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            mock_response = _build_mock_gemini_response(
                text="Gemini response",
                prompt_token_count=30,
                candidates_token_count=40,
                tool_use_prompt_token_count=5,
            )
            inst._original_generate = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            result = models_mod.Models.generate_content(
                fake_instance, model="gemini-2.0-flash", contents="Hello",
            )

            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.span_type == "llm_call"
            assert s.attributes["provider"] == "gemini"
            assert s.attributes["model"] == "gemini-2.0-flash"
            assert s.attributes["input_tokens"] == 30
            assert s.attributes["output_tokens"] == 40
            assert s.status == "ok"

            inst.uninstrument()
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_generate_content_captures_tool_tokens(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            mock_response = _build_mock_gemini_response(
                tool_use_prompt_token_count=42,
            )
            inst._original_generate = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            models_mod.Models.generate_content(
                fake_instance, model="gemini-2.5-flash", contents="Use tools",
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes.get("tool_tokens") == 42

            inst.uninstrument()
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_generate_content_captures_grounding(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            web_chunk = MagicMock()
            web_chunk.web = MagicMock()
            web_chunk.web.title = "Example Article"
            web_chunk.web.uri = "https://example.com/article"

            grounding = MagicMock()
            grounding.web_search_queries = ["test query"]
            grounding.grounding_chunks = [web_chunk]

            candidate = MagicMock()
            candidate.grounding_metadata = grounding

            usage = MagicMock()
            usage.prompt_token_count = 10
            usage.candidates_token_count = 20
            usage.tool_use_prompt_token_count = 0

            response = MagicMock()
            response.usage_metadata = usage
            response.text = "Grounded answer"
            response.candidates = [candidate]

            inst._original_generate = lambda self, *a, **kw: response

            fake_instance = MagicMock()
            models_mod.Models.generate_content(
                fake_instance, model="gemini-2.5-flash", contents="Search something",
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes.get("search_queries") == ["test query"]
            assert len(s.attributes.get("sources", [])) == 1
            assert s.attributes["sources"][0]["title"] == "Example Article"

            inst.uninstrument()
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_generate_content_error(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            def failing(self, *args, **kwargs):
                raise RuntimeError("Quota exceeded")

            inst._original_generate = failing

            fake_instance = MagicMock()
            with pytest.raises(RuntimeError, match="Quota exceeded"):
                models_mod.Models.generate_content(
                    fake_instance, model="gemini-2.0-flash", contents="Hello",
                )

            spans = _get_collected_spans()
            assert len(spans) == 1
            assert spans[0].status == "error"
            assert "Quota exceeded" in spans[0].attributes.get("error", "")

            inst.uninstrument()
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_model_from_positional_arg(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            mock_response = _build_mock_gemini_response()
            inst._original_generate = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            # Pass model as first positional arg (after self)
            models_mod.Models.generate_content(
                fake_instance, "gemini-2.5-flash", contents="Hello",
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert "gemini-2.5-flash" in s.attributes["model"]

            inst.uninstrument()
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_response_text_captured(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            mock_response = _build_mock_gemini_response(text="Short response")
            inst._original_generate = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            models_mod.Models.generate_content(
                fake_instance, model="gemini-2.0-flash", contents="Hi",
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert "response" in s.attributes
            assert s.attributes["response"]["text"] == "Short response"

            inst.uninstrument()
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_no_usage_metadata(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        models_mod = _install_mock_genai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            response = MagicMock()
            response.usage_metadata = None
            response.text = "Response without usage"
            response.candidates = []

            inst._original_generate = lambda self, *a, **kw: response

            fake_instance = MagicMock()
            models_mod.Models.generate_content(
                fake_instance, model="gemini-2.0-flash", contents="Hi",
            )

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes["input_tokens"] == 0
            assert s.attributes["output_tokens"] == 0

            inst.uninstrument()
        finally:
            _cleanup_genai()

    def test_legacy_genai_patching(self):
        genai_mod = _make_module("google.generativeai", is_package=True)

        class GenerativeModel:
            def __init__(self):
                self.model_name = "gemini-1.5-pro"

            def generate_content(self, *args, **kwargs):
                pass

        genai_mod.GenerativeModel = GenerativeModel

        google_mod = _make_module("google", is_package=True)
        google_mod.generativeai = genai_mod

        sys.modules["google"] = google_mod
        sys.modules["google.generativeai"] = genai_mod

        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            original = GenerativeModel.generate_content

            result = inst.instrument()

            assert result is True
            assert inst.is_instrumented() is True
            assert GenerativeModel.generate_content is not original
            assert inst._original_legacy_generate is original

            result = inst.uninstrument()
            assert result is True
            assert GenerativeModel.generate_content is original
        finally:
            _cleanup_genai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_legacy_genai_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()

        genai_mod = _make_module("google.generativeai", is_package=True)

        class GenerativeModel:
            def __init__(self):
                self.model_name = "gemini-1.5-pro"

            def generate_content(self, *args, **kwargs):
                pass

        genai_mod.GenerativeModel = GenerativeModel

        google_mod = _make_module("google", is_package=True)
        google_mod.generativeai = genai_mod

        sys.modules["google"] = google_mod
        sys.modules["google.generativeai"] = genai_mod

        try:
            from evalyn_sdk.trace.instrumentation.providers.gemini import (
                GeminiInstrumentor,
            )
            inst = GeminiInstrumentor()
            inst.instrument()

            mock_response = _build_mock_gemini_response(
                prompt_token_count=25, candidates_token_count=35,
            )
            inst._original_legacy_generate = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            fake_instance.model_name = "gemini-1.5-pro"
            GenerativeModel.generate_content(fake_instance, "Hello")

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.attributes["provider"] == "gemini"
            assert s.attributes["input_tokens"] == 25
            assert s.attributes["output_tokens"] == 35

            inst.uninstrument()
        finally:
            _cleanup_genai()


# ---------------------------------------------------------------------------
# xAI Instrumentor Tests
# ---------------------------------------------------------------------------

class TestXAIInstrumentor:
    """Tests for the xAI SDK instrumentor."""

    def setup_method(self):
        _reset_span_context()

    def test_name(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
        inst = XAIInstrumentor()
        assert inst.name == "xai"

    def test_instrumentor_type(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
        from evalyn_sdk.trace.instrumentation.base import InstrumentorType
        inst = XAIInstrumentor()
        assert inst.instrumentor_type == InstrumentorType.MONKEY_PATCH

    def test_is_available(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
        inst = XAIInstrumentor()
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert inst.is_available() is True
        with patch("importlib.util.find_spec", return_value=None):
            assert inst.is_available() is False

    def test_instrument_patches_sample(self):
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            original = chat_mod.Chat.sample

            result = inst.instrument()

            assert result is True
            assert inst.is_instrumented() is True
            assert chat_mod.Chat.sample is not original
            assert inst._original_sample is original

            inst.uninstrument()
        finally:
            _cleanup_xai()

    def test_instrument_patches_asample(self):
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            original = chat_mod.Chat.asample

            inst.instrument()

            assert chat_mod.Chat.asample is not original
            assert inst._original_asample is original

            inst.uninstrument()
        finally:
            _cleanup_xai()

    def test_uninstrument_restores_originals(self):
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            original_sample = chat_mod.Chat.sample
            original_asample = chat_mod.Chat.asample

            inst.instrument()
            result = inst.uninstrument()

            assert result is True
            assert inst.is_instrumented() is False
            assert chat_mod.Chat.sample is original_sample
            assert chat_mod.Chat.asample is original_asample
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sample_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_xai_response(
                content="Grok response",
                prompt_tokens=20,
                completion_tokens=30,
            )
            inst._original_sample = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            fake_instance._model = "grok-4"
            fake_instance.model = "grok-4"
            result = chat_mod.Chat.sample(fake_instance, "Hello")

            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.span_type == "llm_call"
            assert s.attributes["provider"] == "xai"
            assert s.attributes["model"] == "grok-4"
            assert s.attributes["input_tokens"] == 20
            assert s.attributes["output_tokens"] == 30
            assert s.status == "ok"

            inst.uninstrument()
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sample_captures_response_content(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_xai_response(content="Grok says hello")
            inst._original_sample = lambda self, *a, **kw: mock_response

            fake_instance = MagicMock()
            fake_instance._model = "grok-4"
            chat_mod.Chat.sample(fake_instance, "Hello")

            spans = _get_collected_spans()
            s = spans[0]
            assert "response" in s.attributes
            assert s.attributes["response"]["content"] == "Grok says hello"

            inst.uninstrument()
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_sample_error(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            def failing(self, *args, **kwargs):
                raise ConnectionError("Service unavailable")

            inst._original_sample = failing

            fake_instance = MagicMock()
            fake_instance._model = "grok-4"
            with pytest.raises(ConnectionError, match="Service unavailable"):
                chat_mod.Chat.sample(fake_instance, "Hello")

            spans = _get_collected_spans()
            assert len(spans) == 1
            assert spans[0].status == "error"
            assert "Service unavailable" in spans[0].attributes.get("error", "")

            inst.uninstrument()
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_async_sample_captures_span(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_xai_response(
                content="Async grok", prompt_tokens=5, completion_tokens=10,
            )

            async def mock_asample(self, *args, **kwargs):
                return mock_response

            inst._original_asample = mock_asample

            fake_instance = MagicMock()
            fake_instance._model = "grok-4"

            async def run():
                return await chat_mod.Chat.asample(fake_instance, "Hello")

            result = asyncio.get_event_loop().run_until_complete(run())
            assert result is mock_response

            spans = _get_collected_spans()
            assert len(spans) == 1
            s = spans[0]
            assert s.attributes["provider"] == "xai"
            assert s.attributes["input_tokens"] == 5
            assert s.attributes["output_tokens"] == 10

            inst.uninstrument()
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_async_sample_error(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            async def failing(self, *args, **kwargs):
                raise TimeoutError("Request timed out")

            inst._original_asample = failing

            fake_instance = MagicMock()
            fake_instance._model = "grok-4"

            async def run():
                return await chat_mod.Chat.asample(fake_instance, "Hello")

            with pytest.raises(TimeoutError, match="Request timed out"):
                asyncio.get_event_loop().run_until_complete(run())

            spans = _get_collected_spans()
            assert len(spans) == 1
            assert spans[0].status == "error"

            inst.uninstrument()
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_model_from_model_attr(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            mock_response = _build_mock_xai_response()
            inst._original_sample = lambda self, *a, **kw: mock_response

            # Instance with _model=None, falling back to .model
            fake_instance = MagicMock()
            fake_instance._model = None
            fake_instance.model = "grok-4-0709"
            chat_mod.Chat.sample(fake_instance, "Hello")

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes["model"] == "grok-4-0709"

            inst.uninstrument()
        finally:
            _cleanup_xai()

    @patch("evalyn_sdk.trace.instrumentation.providers._shared._get_tracer")
    def test_no_usage_data(self, mock_get_tracer):
        mock_get_tracer.return_value = _make_mock_tracer()
        chat_mod = _install_mock_xai()
        try:
            from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor
            inst = XAIInstrumentor()
            inst.instrument()

            response = MagicMock()
            response.usage = None
            response.content = "No usage data"

            inst._original_sample = lambda self, *a, **kw: response

            fake_instance = MagicMock()
            fake_instance._model = "grok-4"
            chat_mod.Chat.sample(fake_instance, "Hello")

            spans = _get_collected_spans()
            s = spans[0]
            assert s.attributes["input_tokens"] == 0
            assert s.attributes["output_tokens"] == 0

            inst.uninstrument()
        finally:
            _cleanup_xai()

    def test_extract_usage_helper(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import _extract_usage

        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 200
        usage.input_tokens = 0
        usage.output_tokens = 0
        response = MagicMock()
        response.usage = usage

        inp, out = _extract_usage(response)
        assert inp == 100
        assert out == 200

    def test_extract_usage_no_usage(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import _extract_usage

        response = MagicMock()
        response.usage = None

        inp, out = _extract_usage(response)
        assert inp == 0
        assert out == 0

    def test_extract_content_helper(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import _extract_content

        r1 = MagicMock(spec=["content"])
        r1.content = "Hello content"
        assert _extract_content(r1) == "Hello content"

        r2 = MagicMock(spec=["text"])
        r2.text = "Hello text"
        assert _extract_content(r2) == "Hello text"

        r3 = MagicMock(spec=["message"])
        r3.message = MagicMock()
        r3.message.content = "Hello message"
        assert _extract_content(r3) == "Hello message"

    def test_extract_content_truncation(self):
        from evalyn_sdk.trace.instrumentation.providers.xai import _extract_content

        r = MagicMock(spec=["content"])
        r.content = "x" * 1000
        result = _extract_content(r)
        assert len(result) == 500


# ---------------------------------------------------------------------------
# OpenAI _detect_provider tests
# ---------------------------------------------------------------------------

class TestDetectProvider:
    """Tests for the _detect_provider helper."""

    def test_default_openai(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider
        inst = MagicMock()
        inst._client = None
        assert _detect_provider(inst) == "openai"

    def test_xai_detected(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider
        inst = MagicMock()
        inst._client = MagicMock()
        inst._client.base_url = "https://api.x.ai/v1"
        assert _detect_provider(inst) == "xai"

    def test_other_base_url(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider
        inst = MagicMock()
        inst._client = MagicMock()
        inst._client.base_url = "https://api.openai.com/v1"
        assert _detect_provider(inst) == "openai"

    def test_no_client_attribute(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider
        inst = MagicMock(spec=[])
        assert _detect_provider(inst) == "openai"

    def test_exception_in_detection(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import _detect_provider
        inst = MagicMock()
        inst._client = MagicMock()
        type(inst._client).base_url = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert _detect_provider(inst) == "openai"


# ---------------------------------------------------------------------------
# Cross-provider state management
# ---------------------------------------------------------------------------

class TestInstrumentorStateManagement:
    """Test state management across all instrumentors."""

    def test_uninstrument_before_instrument(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import (
            OpenAIInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.providers.anthropic import (
            AnthropicInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.providers.gemini import (
            GeminiInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor

        for cls in [OpenAIInstrumentor, AnthropicInstrumentor, GeminiInstrumentor, XAIInstrumentor]:
            inst = cls()
            assert inst.uninstrument() is True
            assert inst.is_instrumented() is False

    def test_instrument_returns_false_when_not_available(self):
        from evalyn_sdk.trace.instrumentation.providers.openai import (
            OpenAIInstrumentor,
        )
        from evalyn_sdk.trace.instrumentation.providers.xai import XAIInstrumentor

        with patch("importlib.util.find_spec", return_value=None):
            for cls in [OpenAIInstrumentor, XAIInstrumentor]:
                inst = cls()
                assert inst.instrument() is False
                assert inst.is_instrumented() is False


# ---------------------------------------------------------------------------
# Cost calculation edge cases
# ---------------------------------------------------------------------------

class TestCostEdgeCases:
    """Edge cases for cost calculation."""

    def test_model_with_version_suffix(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost = calculate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        expected = 3.00 + 15.00
        assert cost == pytest.approx(expected)

    def test_cache_cost_unknown_model(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import (
            calculate_cost_with_cache,
        )
        cost = calculate_cost_with_cache(
            "mystery-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        assert cost == pytest.approx(4.0)

    def test_cache_fallback_rate_for_model_without_explicit_cache(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import (
            calculate_cost_with_cache,
        )
        cost = calculate_cost_with_cache(
            "gpt-4o",
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        # cache_write = 2.50 * 1.25 = 3.125, cache_read = 2.50 * 0.1 = 0.25
        assert cost == pytest.approx(3.125 + 0.25)

    def test_grok_model_specificity(self):
        from evalyn_sdk.trace.instrumentation.providers._shared import calculate_cost
        cost_specific = calculate_cost("grok-4-1-fast-reasoning", 1_000_000, 1_000_000)
        cost_generic = calculate_cost("grok-4", 1_000_000, 1_000_000)
        assert cost_specific < cost_generic
