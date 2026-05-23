"""Tests for `evalyn replay` - trace replay against a different model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from evalyn_sdk.cli.commands import replay as replay_cli
from evalyn_sdk.evaluation.replay import (
    ReplayableSpan,
    ReplayComparison,
    ReplayPair,
    ReplayResult,
    _messages_to_prompt,
    _parse_messages_field,
    build_replay_comparison,
    count_unreplayable_llm_spans,
    extract_replayable_spans,
    replay_span,
)
from evalyn_sdk.models import FunctionCall, Span


# ---------------------------------------------------------------------------
# Test fixtures / builders
# ---------------------------------------------------------------------------


BASE_TS = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)


def make_llm_span(
    name: str = "openai:gpt-4o",
    *,
    span_id: str = "span-1",
    provider: str = "openai",
    model: str = "gpt-4o",
    messages: object = None,
    response_content: str = "original answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.001,
    duration_ms: float = 500.0,
    parent_id: str | None = None,
) -> Span:
    """Build an llm_call Span like the instrumentors emit."""
    if messages is None:
        messages = [{"role": "user", "content": "What is 2+2?"}]
    request = {"messages": str(messages)[:500]}
    response = {"content": response_content, "finish_reason": "stop"}

    span = Span(
        id=span_id,
        name=name,
        span_type="llm_call",
        parent_id=parent_id,
        start_time=BASE_TS,
        end_time=BASE_TS + timedelta(milliseconds=duration_ms),
        status="ok",
        attributes={
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "request": request,
            "response": response,
        },
    )
    return span


def make_call_with_spans(spans: list[Span], call_id: str = "call-abc") -> FunctionCall:
    return FunctionCall(
        id=call_id,
        function_name="agent_main",
        inputs={"prompt": "hello"},
        output="final answer",
        error=None,
        started_at=BASE_TS,
        ended_at=BASE_TS + timedelta(seconds=1),
        duration_ms=1000.0,
        session_id="sess-1",
        trace=[],
        metadata={},
        spans=spans,
    )


def fake_caller_factory(
    text: str = "replayed answer",
    input_tokens: int = 80,
    output_tokens: int = 40,
):
    """Build a deterministic LLM-call stub for replay_span/build_replay_comparison."""

    def _fake(provider, model, messages):  # noqa: ARG001
        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}

    return _fake


# ---------------------------------------------------------------------------
# _parse_messages_field
# ---------------------------------------------------------------------------


def test_parse_messages_field_already_list():
    msgs = [{"role": "user", "content": "hi"}]
    assert _parse_messages_field(msgs) == msgs


def test_parse_messages_field_json_string():
    raw = '[{"role": "user", "content": "hi"}]'
    out = _parse_messages_field(raw)
    assert out == [{"role": "user", "content": "hi"}]


def test_parse_messages_field_python_repr():
    raw = "[{'role': 'user', 'content': 'hi'}]"
    out = _parse_messages_field(raw)
    assert out == [{"role": "user", "content": "hi"}]


def test_parse_messages_field_truncated_returns_empty():
    raw = "[{'role': 'user', 'content': 'hello there how are y"  # mid-string truncation
    assert _parse_messages_field(raw) == []


def test_parse_messages_field_empty_inputs():
    assert _parse_messages_field(None) == []
    assert _parse_messages_field("") == []
    assert _parse_messages_field(123) == []


def test_parse_messages_field_filters_non_dicts():
    raw = '[{"role": "user", "content": "hi"}, "not-a-dict", 42]'
    out = _parse_messages_field(raw)
    assert out == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# extract_replayable_spans
# ---------------------------------------------------------------------------


def test_extract_no_llm_spans():
    call = make_call_with_spans(spans=[])
    assert extract_replayable_spans(call) == []


def test_extract_single_llm_span():
    span = make_llm_span()
    call = make_call_with_spans([span])
    out = extract_replayable_spans(call)
    assert len(out) == 1
    r = out[0]
    assert r.span_id == "span-1"
    assert r.original_provider == "openai"
    assert r.original_model == "gpt-4o"
    assert r.messages == [{"role": "user", "content": "What is 2+2?"}]
    assert r.original_input_tokens == 100
    assert r.original_output_tokens == 50
    assert r.original_cost_usd == 0.001
    assert r.original_duration_ms == 500.0
    assert r.original_output == "original answer"


def test_extract_multiple_llm_spans():
    spans = [
        make_llm_span(span_id="s1", model="gpt-4o"),
        make_llm_span(span_id="s2", model="gpt-4o-mini"),
        make_llm_span(span_id="s3", model="gpt-3.5-turbo"),
    ]
    call = make_call_with_spans(spans)
    out = extract_replayable_spans(call)
    assert len(out) == 3
    assert [r.span_id for r in out] == ["s1", "s2", "s3"]


def test_extract_skips_non_llm_spans():
    spans = [
        make_llm_span(span_id="s1"),
        Span(
            id="s2",
            name="tool_call",
            span_type="tool_call",
            parent_id=None,
            start_time=BASE_TS,
            end_time=BASE_TS,
            attributes={},
        ),
    ]
    call = make_call_with_spans(spans)
    out = extract_replayable_spans(call)
    assert len(out) == 1
    assert out[0].span_id == "s1"


def test_extract_skips_unparseable_messages():
    bad_span = make_llm_span(span_id="bad")
    # Corrupt the request so parsing fails
    bad_span.attributes["request"] = {"messages": "[{not valid python or json"}
    good_span = make_llm_span(span_id="good")
    call = make_call_with_spans([bad_span, good_span])
    out = extract_replayable_spans(call)
    assert len(out) == 1
    assert out[0].span_id == "good"


def test_extract_handles_missing_request_attr():
    span = make_llm_span()
    del span.attributes["request"]
    call = make_call_with_spans([span])
    assert extract_replayable_spans(call) == []


def test_count_unreplayable_spans():
    good = make_llm_span(span_id="g")
    bad = make_llm_span(span_id="b")
    bad.attributes["request"] = {"messages": "broken"}
    call = make_call_with_spans([good, bad])
    assert count_unreplayable_llm_spans(call) == 1


# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------


def test_messages_to_prompt_simple_user_only():
    msgs = [{"role": "user", "content": "Hello"}]
    prompt, system = _messages_to_prompt(msgs)
    assert prompt == "Hello"
    assert system is None


def test_messages_to_prompt_with_system():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    prompt, system = _messages_to_prompt(msgs)
    assert prompt == "Hi"
    assert system == "You are helpful."


def test_messages_to_prompt_multi_turn():
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    prompt, system = _messages_to_prompt(msgs)
    assert "Q1" in prompt
    assert "assistant: A1" in prompt
    assert "Q2" in prompt
    assert system is None


def test_messages_to_prompt_handles_content_blocks():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]},
    ]
    prompt, _ = _messages_to_prompt(msgs)
    assert "Hello" in prompt
    assert "World" in prompt


# ---------------------------------------------------------------------------
# replay_span
# ---------------------------------------------------------------------------


def test_replay_span_success():
    span = make_llm_span()
    replayable = extract_replayable_spans(make_call_with_spans([span]))[0]

    caller = fake_caller_factory(text="hello from gpt-4o-mini", input_tokens=120, output_tokens=60)
    result = replay_span(replayable, "gpt-4o-mini", "openai", caller=caller)

    assert result.error is None
    assert result.new_provider == "openai"
    assert result.new_model == "gpt-4o-mini"
    assert result.new_output == "hello from gpt-4o-mini"
    assert result.new_input_tokens == 120
    assert result.new_output_tokens == 60
    # Cost should be calculated from gpt-4o-mini pricing (real value > 0)
    assert result.new_cost_usd > 0
    assert result.new_duration_ms >= 0


def test_replay_span_error_path():
    span = make_llm_span()
    replayable = extract_replayable_spans(make_call_with_spans([span]))[0]

    def boom(provider, model, messages):  # noqa: ARG001
        raise RuntimeError("API down")

    result = replay_span(replayable, "gpt-4o-mini", "openai", caller=boom)
    assert result.error == "API down"
    assert result.new_output == ""
    assert result.new_input_tokens == 0
    assert result.new_cost_usd == 0.0


def test_replay_span_uses_target_model_for_cost():
    span = make_llm_span()
    replayable = extract_replayable_spans(make_call_with_spans([span]))[0]

    caller = fake_caller_factory(input_tokens=1_000_000, output_tokens=0)
    # gpt-4o-mini is $0.15 per 1M input
    result = replay_span(replayable, "gpt-4o-mini", "openai", caller=caller)
    assert result.new_cost_usd == pytest.approx(0.15, rel=1e-3)


# ---------------------------------------------------------------------------
# build_replay_comparison (deltas)
# ---------------------------------------------------------------------------


def test_build_comparison_totals_and_deltas():
    s1 = make_llm_span(
        span_id="s1", cost_usd=0.005, duration_ms=400, input_tokens=200, output_tokens=100
    )
    s2 = make_llm_span(
        span_id="s2", cost_usd=0.010, duration_ms=600, input_tokens=400, output_tokens=200
    )
    call = make_call_with_spans([s1, s2])

    caller = fake_caller_factory(text="cheap reply", input_tokens=50, output_tokens=25)
    comp = build_replay_comparison(call, "openai", "gpt-4o-mini", caller=caller)

    assert len(comp.pairs) == 2
    # All per-span deltas wired
    for pair in comp.pairs:
        assert pair.cost_delta_usd == pytest.approx(
            pair.replayed.new_cost_usd - pair.original.original_cost_usd
        )
        assert pair.latency_delta_ms == pytest.approx(
            pair.replayed.new_duration_ms - pair.original.original_duration_ms
        )

    # Totals
    assert comp.total_original_cost_usd == pytest.approx(0.015)
    assert comp.total_replayed_cost_usd == pytest.approx(
        sum(p.replayed.new_cost_usd for p in comp.pairs)
    )
    assert comp.total_cost_delta_usd == pytest.approx(
        comp.total_replayed_cost_usd - comp.total_original_cost_usd
    )
    assert comp.total_original_latency_ms == 1000.0
    assert comp.total_latency_delta_ms == pytest.approx(
        comp.total_replayed_latency_ms - 1000.0
    )


def test_build_comparison_skipped_spans_recorded():
    good = make_llm_span(span_id="good")
    bad = make_llm_span(span_id="bad")
    bad.attributes["request"] = {"messages": "broken-truncated-["}
    call = make_call_with_spans([good, bad])

    comp = build_replay_comparison(
        call, "openai", "gpt-4o-mini", caller=fake_caller_factory()
    )
    assert len(comp.pairs) == 1
    assert len(comp.skipped) == 1
    assert comp.skipped[0]["span_id"] == "bad"


def test_build_comparison_empty_spans():
    call = make_call_with_spans([])
    comp = build_replay_comparison(call, "openai", "gpt-4o-mini", caller=fake_caller_factory())
    assert comp.pairs == []
    assert comp.skipped == []
    assert comp.total_cost_delta_usd == 0.0


def test_comparison_as_dict_roundtrip_structure():
    span = make_llm_span()
    call = make_call_with_spans([span])
    comp = build_replay_comparison(call, "openai", "gpt-4o-mini", caller=fake_caller_factory())

    d = comp.as_dict()
    assert d["call_id"] == call.id
    assert d["target_provider"] == "openai"
    assert d["target_model"] == "gpt-4o-mini"
    assert "pairs" in d and len(d["pairs"]) == 1
    assert "totals" in d
    assert "cost_delta_usd" in d["totals"]
    assert "latency_delta_ms" in d["totals"]
    # Make sure it serializes without exotic types
    json.dumps(d, default=str)


# ---------------------------------------------------------------------------
# CLI command (cmd_replay)
# ---------------------------------------------------------------------------


class FakeStorage:
    """Minimal SQLiteStorage stand-in for cmd_replay tests."""

    def __init__(self, calls: dict[str, FunctionCall]):
        self._calls = calls

    def resolve_call_id(self, prefix: str) -> str | None:
        matches = [cid for cid in self._calls if cid.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        return None

    def count_call_prefix_matches(self, prefix: str) -> int:
        return sum(1 for cid in self._calls if cid.startswith(prefix))

    def get_call(self, call_id: str) -> FunctionCall | None:
        return self._calls.get(call_id)


def _make_cli_args(**overrides):
    ns = argparse.Namespace(
        id="abc",
        model="gpt-4o-mini",
        provider="openai",
        db=None,
        output=None,
        format="table",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_cmd_replay_table_output(capsys):
    span = make_llm_span(span_id="s1")
    call = make_call_with_spans([span], call_id="abcdef12-call")
    storage = FakeStorage({call.id: call})

    fake_caller = fake_caller_factory(text="cheap output", input_tokens=10, output_tokens=5)

    with patch.object(replay_cli, "_get_storage", return_value=storage), patch.object(
        replay_cli,
        "build_replay_comparison",
        side_effect=lambda c, target_provider, target_model: build_replay_comparison(
            c, target_provider, target_model, caller=fake_caller
        ),
    ):
        replay_cli.cmd_replay(_make_cli_args(id="abcdef"))

    out = capsys.readouterr().out
    assert "REPLAY COMPARISON" in out
    assert "openai" in out
    assert "gpt-4o-mini" in out
    assert "cheap output" in out
    assert "TOTALS" in out


def test_cmd_replay_json_output(capsys):
    span = make_llm_span(span_id="s1")
    call = make_call_with_spans([span], call_id="abcdef12-call")
    storage = FakeStorage({call.id: call})

    fake_caller = fake_caller_factory(text="json reply", input_tokens=10, output_tokens=5)

    with patch.object(replay_cli, "_get_storage", return_value=storage), patch.object(
        replay_cli,
        "build_replay_comparison",
        side_effect=lambda c, target_provider, target_model: build_replay_comparison(
            c, target_provider, target_model, caller=fake_caller
        ),
    ):
        replay_cli.cmd_replay(_make_cli_args(id="abcdef", format="json"))

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["target_provider"] == "openai"
    assert payload["target_model"] == "gpt-4o-mini"
    assert payload["call_id"] == call.id
    assert len(payload["pairs"]) == 1


def test_cmd_replay_writes_to_output_file(tmp_path, capsys):
    span = make_llm_span(span_id="s1")
    call = make_call_with_spans([span], call_id="abcdef12-call")
    storage = FakeStorage({call.id: call})
    out_file = tmp_path / "subdir" / "compare.json"

    fake_caller = fake_caller_factory()

    with patch.object(replay_cli, "_get_storage", return_value=storage), patch.object(
        replay_cli,
        "build_replay_comparison",
        side_effect=lambda c, target_provider, target_model: build_replay_comparison(
            c, target_provider, target_model, caller=fake_caller
        ),
    ):
        replay_cli.cmd_replay(
            _make_cli_args(id="abcdef", format="json", output=str(out_file))
        )

    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload["call_id"] == call.id
    msg = capsys.readouterr().out
    assert "Wrote replay comparison" in msg


def test_cmd_replay_missing_call(capsys):
    storage = FakeStorage({})
    with patch.object(replay_cli, "_get_storage", return_value=storage):
        with pytest.raises(SystemExit):
            replay_cli.cmd_replay(_make_cli_args(id="missing"))


def test_cmd_replay_call_without_spans(capsys):
    call = make_call_with_spans([], call_id="abcdef-no-spans")
    storage = FakeStorage({call.id: call})
    with patch.object(replay_cli, "_get_storage", return_value=storage):
        with pytest.raises(SystemExit):
            replay_cli.cmd_replay(_make_cli_args(id="abcdef"))


# ---------------------------------------------------------------------------
# Round-trip on dataclasses
# ---------------------------------------------------------------------------


def test_replayable_span_as_dict_keys():
    span = make_llm_span()
    r = extract_replayable_spans(make_call_with_spans([span]))[0]
    d = r.as_dict()
    for key in (
        "span_id",
        "span_name",
        "original_provider",
        "original_model",
        "messages",
        "original_input_tokens",
        "original_output_tokens",
        "original_cost_usd",
        "original_duration_ms",
        "original_output",
    ):
        assert key in d


def test_replay_result_as_dict_keys():
    result = ReplayResult(
        span_id="s",
        new_provider="openai",
        new_model="gpt-4o-mini",
        new_output="x",
        new_input_tokens=1,
        new_output_tokens=1,
        new_cost_usd=0.001,
        new_duration_ms=10.0,
    )
    d = result.as_dict()
    assert d["span_id"] == "s"
    assert d["error"] is None
