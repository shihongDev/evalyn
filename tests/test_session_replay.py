from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.trace.session_replay import (
    SessionTurn,
    SessionReplayPlan,
    TurnComparison,
    SessionReplayReport,
    extract_session_turns,
    build_session_replay_plan,
    compare_turn,
    build_session_replay_report,
)


def _span(sid, span_type="llm_call", start_offset_ms=0, **attrs):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=start_offset_ms)
    return Span(id=sid, name=f"span-{sid}", span_type=span_type, parent_id=None, start_time=t, attributes=attrs)


# --- extract_session_turns ---


def test_extract_session_turns_from_spans():
    spans = [
        _span("s1", role="user", content="hello", model="gpt-4"),
        _span("s2", start_offset_ms=100, role="assistant", content="hi there", model="gpt-4"),
    ]
    turns = extract_session_turns(spans)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "hello"
    assert turns[1].role == "assistant"
    assert turns[1].content == "hi there"


def test_extract_session_turns_ordered_by_time():
    spans = [
        _span("s2", start_offset_ms=200, role="assistant", content="second"),
        _span("s1", start_offset_ms=0, role="user", content="first"),
        _span("s3", start_offset_ms=100, role="user", content="middle"),
    ]
    turns = extract_session_turns(spans)
    assert len(turns) == 3
    assert turns[0].content == "first"
    assert turns[1].content == "middle"
    assert turns[2].content == "second"


def test_extract_session_turns_input_output_pattern():
    spans = [
        _span("s1", input="user query", output="assistant response", model="gpt-4"),
    ]
    turns = extract_session_turns(spans)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "user query"
    assert turns[1].role == "assistant"
    assert turns[1].content == "assistant response"


def test_extract_session_turns_input_only():
    spans = [_span("s1", input="just input")]
    turns = extract_session_turns(spans)
    assert len(turns) == 1
    assert turns[0].role == "user"
    assert turns[0].content == "just input"


def test_extract_session_turns_output_only():
    spans = [_span("s1", output="just output")]
    turns = extract_session_turns(spans)
    assert len(turns) == 1
    assert turns[0].role == "assistant"
    assert turns[0].content == "just output"


def test_extract_session_turns_empty_spans():
    turns = extract_session_turns([])
    assert turns == []


def test_extract_session_turns_no_relevant_attrs():
    spans = [_span("s1", model="gpt-4", cost=0.01)]
    turns = extract_session_turns(spans)
    assert turns == []


def test_extract_session_turns_indexes_sequential():
    spans = [
        _span("s1", start_offset_ms=0, role="user", content="a"),
        _span("s2", start_offset_ms=100, role="assistant", content="b"),
        _span("s3", start_offset_ms=200, role="user", content="c"),
    ]
    turns = extract_session_turns(spans)
    assert [t.index for t in turns] == [0, 1, 2]


# --- build_session_replay_plan ---


def test_build_session_replay_plan():
    turns = [
        SessionTurn(index=0, role="user", content="hi", span_id="s1"),
        SessionTurn(index=1, role="assistant", content="hello", span_id="s2"),
    ]
    plan = build_session_replay_plan(turns, "claude-3")
    assert plan.target_model == "claude-3"
    assert plan.total_turns == 2
    assert len(plan.turns) == 2
    assert plan.session_id == "s1"


def test_build_session_replay_plan_empty():
    plan = build_session_replay_plan([], "claude-3")
    assert plan.total_turns == 0
    assert plan.turns == []
    assert plan.session_id == ""


# --- compare_turn ---


def test_compare_turn_similarity_identical():
    turn = SessionTurn(index=0, role="user", content="hello world")
    comp = compare_turn(turn, "hello world")
    assert comp.similarity == 1.0
    assert comp.turn_index == 0
    assert comp.original_content == "hello world"
    assert comp.replayed_content == "hello world"


def test_compare_turn_similarity_no_overlap():
    turn = SessionTurn(index=0, role="user", content="hello world")
    comp = compare_turn(turn, "foo bar")
    assert comp.similarity == 0.0


def test_compare_turn_similarity_partial():
    turn = SessionTurn(index=0, role="user", content="the cat sat")
    comp = compare_turn(turn, "the dog sat")
    assert abs(comp.similarity - 0.5) < 1e-9


def test_compare_turn_empty_both():
    turn = SessionTurn(index=0, role="user", content="")
    comp = compare_turn(turn, "")
    assert comp.similarity == 1.0


# --- build_session_replay_report ---


def test_build_session_replay_report():
    c1 = TurnComparison(turn_index=0, original_content="a", replayed_content="b", similarity=0.8)
    c2 = TurnComparison(turn_index=1, original_content="c", replayed_content="d", similarity=0.6)
    report = build_session_replay_report("sess-1", [c1, c2])
    assert report.session_id == "sess-1"
    assert abs(report.avg_similarity - 0.7) < 1e-9
    assert report.total_turns == 2
    assert len(report.comparisons) == 2


def test_build_session_replay_report_empty():
    report = build_session_replay_report("sess-1", [])
    assert report.avg_similarity == 0.0
    assert report.total_turns == 0
    assert report.comparisons == []


def test_build_session_replay_report_single():
    c = TurnComparison(turn_index=0, original_content="x", replayed_content="y", similarity=0.5)
    report = build_session_replay_report("sess-1", [c])
    assert report.avg_similarity == 0.5
    assert report.total_turns == 1


# --- format_text ---


def test_session_replay_plan_format_text():
    turns = [SessionTurn(index=0, role="user", content="hello", model="gpt-4", span_id="s1")]
    plan = SessionReplayPlan(session_id="sess-1", turns=turns, target_model="claude-3", total_turns=1)
    text = plan.format_text()
    assert "sess-1" in text
    assert "claude-3" in text
    assert "1 turns" in text
    assert "user" in text
    assert "gpt-4" in text


def test_session_replay_report_format_text():
    c = TurnComparison(turn_index=0, original_content="hello", replayed_content="hi", similarity=0.5)
    report = SessionReplayReport(session_id="sess-1", comparisons=[c], avg_similarity=0.5, total_turns=1)
    text = report.format_text()
    assert "sess-1" in text
    assert "50.00%" in text
    assert "Turn 0" in text


# --- as_dict / from_dict roundtrips ---


def test_session_turn_roundtrip():
    turn = SessionTurn(index=0, role="user", content="hello", model="gpt-4", span_id="s1")
    d = turn.as_dict()
    restored = SessionTurn.from_dict(d)
    assert restored.index == turn.index
    assert restored.role == turn.role
    assert restored.content == turn.content
    assert restored.model == turn.model
    assert restored.span_id == turn.span_id


def test_session_replay_plan_roundtrip():
    turn = SessionTurn(index=0, role="user", content="hi", span_id="s1")
    plan = SessionReplayPlan(session_id="sess-1", turns=[turn], target_model="claude-3", total_turns=1)
    d = plan.as_dict()
    restored = SessionReplayPlan.from_dict(d)
    assert restored.session_id == plan.session_id
    assert restored.target_model == plan.target_model
    assert restored.total_turns == plan.total_turns
    assert len(restored.turns) == 1
    assert restored.turns[0].content == "hi"


def test_session_replay_report_roundtrip():
    c = TurnComparison(turn_index=0, original_content="a", replayed_content="b", similarity=0.8)
    report = SessionReplayReport(session_id="sess-1", comparisons=[c], avg_similarity=0.8, total_turns=1)
    d = report.as_dict()
    restored = SessionReplayReport.from_dict(d)
    assert restored.session_id == report.session_id
    assert restored.avg_similarity == report.avg_similarity
    assert restored.total_turns == report.total_turns
    assert len(restored.comparisons) == 1


def test_turn_comparison_as_dict():
    comp = TurnComparison(turn_index=0, original_content="a", replayed_content="b", similarity=0.5)
    d = comp.as_dict()
    assert d["turn_index"] == 0
    assert d["original_content"] == "a"
    assert d["replayed_content"] == "b"
    assert d["similarity"] == 0.5
