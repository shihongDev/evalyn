from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.trace.trace_replay import (
    ReplayInput,
    ReplayOutput,
    ReplayComparison,
    TraceReplayPlan,
    TraceReplayReport,
    extract_replay_inputs,
    build_replay_plan,
    compute_text_similarity,
    compare_replay,
    build_replay_report,
)


def _span(sid, span_type="llm_call", start_offset_ms=0, **attrs):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(id=sid, name=f"span-{sid}", span_type=span_type, parent_id=None, start_time=t, attributes=attrs)


# --- extract_replay_inputs ---


def test_extract_replay_inputs_from_llm_spans():
    spans = [
        _span("s1", messages=[{"role": "user", "content": "hello"}], model="gpt-4", output="hi"),
        _span("s2", input="what is 2+2?", model="gpt-4", output="4"),
    ]
    inputs = extract_replay_inputs(spans)
    assert len(inputs) == 2
    assert inputs[0].span_id == "s1"
    assert inputs[0].original_model == "gpt-4"
    assert inputs[0].input_messages == [{"role": "user", "content": "hello"}]
    assert inputs[0].original_output == "hi"
    assert inputs[1].input_messages == [{"role": "user", "content": "what is 2+2?"}]


def test_extract_replay_inputs_skips_non_llm():
    spans = [
        _span("s1", span_type="tool_call", input="data", model="gpt-4"),
        _span("s2", span_type="custom", input="data"),
        _span("s3", span_type="llm_call", prompt="test prompt", model="claude-3"),
    ]
    inputs = extract_replay_inputs(spans)
    assert len(inputs) == 1
    assert inputs[0].span_id == "s3"


def test_extract_replay_inputs_prompt_attribute():
    spans = [_span("s1", prompt="summarize this", model="claude-3", output="summary")]
    inputs = extract_replay_inputs(spans)
    assert len(inputs) == 1
    assert inputs[0].input_messages == [{"role": "user", "content": "summarize this"}]
    assert inputs[0].original_output == "summary"


def test_extract_replay_inputs_no_model():
    spans = [_span("s1", input="hi")]
    inputs = extract_replay_inputs(spans)
    assert len(inputs) == 1
    assert inputs[0].original_model == ""


def test_extract_replay_inputs_empty_spans():
    inputs = extract_replay_inputs([])
    assert inputs == []


def test_extract_replay_inputs_no_input_attrs():
    spans = [_span("s1", model="gpt-4")]
    inputs = extract_replay_inputs(spans)
    assert len(inputs) == 1
    assert inputs[0].input_messages == []
    assert inputs[0].original_output == ""


# --- build_replay_plan ---


def test_build_replay_plan_correct_counts():
    inp1 = ReplayInput(span_id="s1", original_model="gpt-4", input_messages=[{"role": "user", "content": "hi"}])
    inp2 = ReplayInput(span_id="s2", original_model="gpt-4", input_messages=[{"role": "user", "content": "bye"}])
    plan = build_replay_plan([inp1, inp2], "claude-3")
    assert plan.target_model == "claude-3"
    assert plan.estimated_calls == 2
    assert len(plan.inputs) == 2


def test_build_replay_plan_empty():
    plan = build_replay_plan([], "claude-3")
    assert plan.estimated_calls == 0
    assert plan.inputs == []
    assert plan.target_model == "claude-3"


# --- compute_text_similarity ---


def test_compute_text_similarity_identical():
    assert compute_text_similarity("hello world", "hello world") == 1.0


def test_compute_text_similarity_no_overlap():
    assert compute_text_similarity("hello world", "foo bar") == 0.0


def test_compute_text_similarity_partial():
    sim = compute_text_similarity("the cat sat", "the dog sat")
    # Overlap: {the, sat}, Union: {the, cat, sat, dog} -> 2/4 = 0.5
    assert abs(sim - 0.5) < 1e-9


def test_compute_text_similarity_empty_both():
    assert compute_text_similarity("", "") == 1.0


def test_compute_text_similarity_one_empty():
    assert compute_text_similarity("hello", "") == 0.0
    assert compute_text_similarity("", "hello") == 0.0


def test_compute_text_similarity_case_insensitive():
    assert compute_text_similarity("Hello World", "hello world") == 1.0


# --- compare_replay ---


def test_compare_replay_builds_diff():
    original = ReplayInput(span_id="s1", original_model="gpt-4", original_output="hello world")
    replayed = ReplayOutput(span_id="s1", replayed_model="claude-3", replayed_output="hello there", cost_usd=0.01)
    comp = compare_replay(original, replayed)
    assert comp.span_id == "s1"
    assert comp.original_output == "hello world"
    assert comp.replayed_output == "hello there"
    assert comp.output_diff != ""
    assert comp.cost_delta == 0.01
    assert 0.0 < comp.similarity < 1.0


def test_compare_replay_identical():
    original = ReplayInput(span_id="s1", original_model="gpt-4", original_output="same text")
    replayed = ReplayOutput(span_id="s1", replayed_model="claude-3", replayed_output="same text")
    comp = compare_replay(original, replayed)
    assert comp.similarity == 1.0
    assert comp.output_diff == ""


def test_compare_replay_empty_outputs():
    original = ReplayInput(span_id="s1", original_model="gpt-4", original_output="")
    replayed = ReplayOutput(span_id="s1", replayed_model="claude-3", replayed_output="")
    comp = compare_replay(original, replayed)
    assert comp.similarity == 1.0
    assert comp.output_diff == ""


# --- build_replay_report ---


def test_build_replay_report_averages():
    c1 = ReplayComparison(span_id="s1", original_output="a", replayed_output="b", similarity=0.8, cost_delta=0.01)
    c2 = ReplayComparison(span_id="s2", original_output="c", replayed_output="d", similarity=0.6, cost_delta=0.02)
    report = build_replay_report([c1, c2])
    assert abs(report.avg_similarity - 0.7) < 1e-9
    assert abs(report.total_cost_delta - 0.03) < 1e-9
    assert len(report.comparisons) == 2


def test_build_replay_report_empty():
    report = build_replay_report([])
    assert report.avg_similarity == 0.0
    assert report.total_cost_delta == 0.0
    assert report.comparisons == []


def test_build_replay_report_single():
    c = ReplayComparison(span_id="s1", original_output="x", replayed_output="y", similarity=0.5, cost_delta=0.1)
    report = build_replay_report([c])
    assert report.avg_similarity == 0.5
    assert report.total_cost_delta == 0.1


# --- format_text ---


def test_trace_replay_plan_format_text():
    inp = ReplayInput(span_id="s1", original_model="gpt-4", input_messages=[{"role": "user", "content": "hi"}])
    plan = TraceReplayPlan(inputs=[inp], target_model="claude-3", estimated_calls=1)
    text = plan.format_text()
    assert "claude-3" in text
    assert "1 calls" in text
    assert "s1" in text
    assert "gpt-4" in text


def test_replay_comparison_format_text():
    comp = ReplayComparison(
        span_id="s1", original_output="hello", replayed_output="world",
        output_diff="--- original\n+++ replayed\n-hello\n+world", similarity=0.0, cost_delta=0.05,
    )
    text = comp.format_text()
    assert "s1" in text
    assert "0.00%" in text
    assert "0.0500" in text
    assert "Diff:" in text


def test_replay_report_format_text():
    c = ReplayComparison(span_id="s1", original_output="a", replayed_output="b", similarity=0.9, cost_delta=0.0)
    report = TraceReplayReport(comparisons=[c], avg_similarity=0.9, total_cost_delta=0.0)
    text = report.format_text()
    assert "1 comparisons" in text
    assert "90.00%" in text


# --- as_dict / from_dict roundtrips ---


def test_replay_input_roundtrip():
    inp = ReplayInput(
        span_id="s1", original_model="gpt-4",
        input_messages=[{"role": "user", "content": "hi"}], original_output="hello",
    )
    d = inp.as_dict()
    restored = ReplayInput.from_dict(d)
    assert restored.span_id == inp.span_id
    assert restored.original_model == inp.original_model
    assert restored.input_messages == inp.input_messages
    assert restored.original_output == inp.original_output


def test_trace_replay_plan_roundtrip():
    inp = ReplayInput(span_id="s1", original_model="gpt-4")
    plan = TraceReplayPlan(inputs=[inp], target_model="claude-3", estimated_calls=1)
    d = plan.as_dict()
    restored = TraceReplayPlan.from_dict(d)
    assert restored.target_model == plan.target_model
    assert restored.estimated_calls == plan.estimated_calls
    assert len(restored.inputs) == 1
    assert restored.inputs[0].span_id == "s1"


def test_trace_replay_report_roundtrip():
    c = ReplayComparison(span_id="s1", original_output="a", replayed_output="b", similarity=0.7, cost_delta=0.01)
    report = TraceReplayReport(comparisons=[c], avg_similarity=0.7, total_cost_delta=0.01)
    d = report.as_dict()
    restored = TraceReplayReport.from_dict(d)
    assert restored.avg_similarity == report.avg_similarity
    assert restored.total_cost_delta == report.total_cost_delta
    assert len(restored.comparisons) == 1


def test_replay_output_as_dict():
    out = ReplayOutput(span_id="s1", replayed_model="claude-3", replayed_output="hi", cost_usd=0.02, duration_ms=150.0)
    d = out.as_dict()
    assert d["span_id"] == "s1"
    assert d["replayed_model"] == "claude-3"
    assert d["cost_usd"] == 0.02
    assert d["duration_ms"] == 150.0


def test_replay_comparison_as_dict():
    comp = ReplayComparison(span_id="s1", original_output="a", replayed_output="b", similarity=0.5)
    d = comp.as_dict()
    assert d["span_id"] == "s1"
    assert d["similarity"] == 0.5
