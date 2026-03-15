"""Tests for streaming instrumentation across providers."""
import time
from evalyn_sdk.trace.instrumentation.providers._streaming import StreamingSpanWrapper


def test_openai_stream_wrapper_token_extraction():
    """Test that OpenAI-style usage objects are extracted."""

    class MockChunk:
        def __init__(self, usage=None):
            self.usage = usage

    class MockUsage:
        def __init__(self, prompt_tokens=0, completion_tokens=0):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

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

    chunks = [MockChunk(), MockChunk(), MockChunk(MockUsage(100, 50))]
    wrapper = OpenAIStreamWrapper(iter(chunks), request_start_time=time.time())
    collected = list(wrapper)
    assert len(collected) == 3
    assert wrapper.accumulated_input_tokens == 100
    assert wrapper.accumulated_output_tokens == 50
    assert wrapper.chunk_count == 3


def test_anthropic_stream_wrapper_token_extraction():
    """Test Anthropic-style event token extraction."""

    class MockEvent:
        def __init__(self, event_type, message=None, usage=None):
            self.type = event_type
            self.message = message
            self.usage = usage

    class MockMessage:
        def __init__(self, usage):
            self.usage = usage

    class MockUsage:
        def __init__(self, input_tokens=0, output_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

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

    chunks = [
        MockEvent(
            "message_start", message=MockMessage(MockUsage(input_tokens=200))
        ),
        MockEvent("content_block_delta"),
        MockEvent("message_delta", usage=MockUsage(output_tokens=75)),
    ]
    wrapper = AnthropicStreamWrapper(iter(chunks), request_start_time=time.time())
    collected = list(wrapper)
    assert len(collected) == 3
    assert wrapper.accumulated_input_tokens == 200
    assert wrapper.accumulated_output_tokens == 75


def test_gemini_stream_wrapper_token_extraction():
    """Test Gemini-style usage_metadata extraction."""

    class MockChunk:
        def __init__(self, usage_metadata=None):
            self.usage_metadata = usage_metadata

    class MockUsageMetadata:
        def __init__(self, prompt_token_count=0, candidates_token_count=0):
            self.prompt_token_count = prompt_token_count
            self.candidates_token_count = candidates_token_count

    class GeminiStreamWrapper(StreamingSpanWrapper):
        def __next__(self):
            chunk = super().__next__()
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                self.accumulated_input_tokens = getattr(
                    usage, "prompt_token_count", 0
                )
                self.accumulated_output_tokens = getattr(
                    usage, "candidates_token_count", 0
                )
            return chunk

    chunks = [MockChunk(), MockChunk(MockUsageMetadata(150, 60))]
    wrapper = GeminiStreamWrapper(iter(chunks), request_start_time=time.time())
    collected = list(wrapper)
    assert len(collected) == 2
    assert wrapper.accumulated_input_tokens == 150
    assert wrapper.accumulated_output_tokens == 60


def test_streaming_attributes_param_in_log_llm_call():
    """Verify log_llm_call accepts streaming_attributes parameter."""
    from evalyn_sdk.trace.instrumentation.providers._shared import log_llm_call
    import inspect

    sig = inspect.signature(log_llm_call)
    assert "streaming_attributes" in sig.parameters


def test_openai_stream_wrapper_no_usage_chunks():
    """Test wrapper handles chunks with no usage data."""

    class MockChunk:
        def __init__(self):
            self.usage = None

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

    chunks = [MockChunk(), MockChunk()]
    wrapper = OpenAIStreamWrapper(iter(chunks), request_start_time=time.time())
    list(wrapper)
    assert wrapper.accumulated_input_tokens == 0
    assert wrapper.accumulated_output_tokens == 0
    assert wrapper.chunk_count == 2


def test_stream_wrapper_span_attributes():
    """Test that as_span_attributes includes streaming flag."""

    class SimpleWrapper(StreamingSpanWrapper):
        pass

    wrapper = SimpleWrapper(iter(["a", "b", "c"]), request_start_time=time.time())
    list(wrapper)
    attrs = wrapper.as_span_attributes()
    assert attrs["streaming"] is True
    assert attrs["chunk_count"] == 3
    assert "time_to_first_token_ms" in attrs
    assert "streaming_duration_ms" in attrs
