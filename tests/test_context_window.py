from evalyn_sdk.trace.instrumentation.providers._shared import (
    get_model_context_window,
    MODEL_CONTEXT_WINDOWS,
)

def test_known_model_context_window():
    assert get_model_context_window("gpt-4o") == 128_000
    assert get_model_context_window("claude-sonnet-4-5") == 200_000
    assert get_model_context_window("gemini-2.5-flash") == 1_048_576

def test_unknown_model_returns_none():
    assert get_model_context_window("totally-unknown-model") is None

def test_substring_matching():
    # "gpt-4o-2024-08-06" should match "gpt-4o"
    assert get_model_context_window("gpt-4o-2024-08-06") == 128_000
