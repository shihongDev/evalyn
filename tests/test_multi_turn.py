from __future__ import annotations

from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.evaluation.multi_turn import (
    Turn,
    TurnScore,
    ConversationMetrics,
    MultiTurnReport,
    extract_turns,
    score_turn_quality,
    evaluate_context_consistency,
    evaluate_topic_drift,
    evaluate_conversation,
)


def _span(sid, span_type="llm_call", status="ok", duration_ms=100, **attrs):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        status=status,
        attributes=attrs,
    )


# --- extract_turns ---

def test_extract_turns_role_content():
    spans = [
        _span("1", role="user", content="Hello, can you help me?"),
        _span("2", role="assistant", content="Sure, what do you need?"),
    ]
    turns = extract_turns(spans)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello, can you help me?"
    assert turns[1].role == "assistant"


def test_extract_turns_input_output():
    spans = [
        _span("1", input="What is Python?", output="Python is a programming language"),
    ]
    turns = extract_turns(spans)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "What is Python?"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Python is a programming language"


def test_extract_turns_empty():
    turns = extract_turns([])
    assert turns == []


def test_extract_turns_no_relevant_attrs():
    spans = [_span("1", foo="bar")]
    turns = extract_turns(spans)
    assert turns == []


def test_extract_turns_ordering_by_timestamp():
    t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    span_late = Span(
        id="late", name="late", span_type="llm_call", parent_id=None,
        start_time=t2, end_time=t2 + timedelta(seconds=1), status="ok",
        attributes={"role": "assistant", "content": "second"},
    )
    span_early = Span(
        id="early", name="early", span_type="llm_call", parent_id=None,
        start_time=t1, end_time=t1 + timedelta(seconds=1), status="ok",
        attributes={"role": "user", "content": "first"},
    )
    # Pass in reverse order to test sorting
    turns = extract_turns([span_late, span_early])
    assert turns[0].content == "first"
    assert turns[1].content == "second"


def test_extract_turns_mixed_attrs():
    spans = [
        _span("1", role="user", content="Hello there"),
        _span("2", input="query", output="result"),
    ]
    turns = extract_turns(spans)
    assert len(turns) == 3  # 1 from role/content, 2 from input/output


# --- score_turn_quality ---

def test_score_turn_quality_nontrivial():
    turn = Turn(index=0, role="assistant", content="This is a detailed answer with enough content")
    ts = score_turn_quality(turn, [])
    assert ts.score == 1.0
    assert ts.passed is True


def test_score_turn_quality_trivial_short():
    turn = Turn(index=0, role="assistant", content="Hi")
    ts = score_turn_quality(turn, [])
    assert ts.score == 0.0
    assert ts.passed is False


def test_score_turn_quality_with_context_reference():
    ctx = [Turn(index=0, role="user", content="Tell me about Python programming language")]
    turn = Turn(index=1, role="assistant", content="Python is a great programming language for beginners")
    ts = score_turn_quality(turn, ctx)
    # nontrivial + references context = 1.0
    assert ts.score == 1.0
    assert ts.passed is True


def test_score_turn_quality_with_context_no_reference():
    ctx = [Turn(index=0, role="user", content="Tell me about Python programming language")]
    turn = Turn(index=1, role="assistant", content="The sky is blue today")
    ts = score_turn_quality(turn, ctx)
    # nontrivial but no context reference = 0.5
    assert ts.score == 0.5
    assert ts.passed is True


def test_score_turn_quality_short_no_context_ref():
    ctx = [Turn(index=0, role="user", content="Tell me about quantum computing")]
    turn = Turn(index=1, role="assistant", content="Ok")
    ts = score_turn_quality(turn, ctx)
    # not nontrivial, no context reference = 0.0
    assert ts.score == 0.0
    assert ts.passed is False


# --- evaluate_context_consistency ---

def test_context_consistency_high():
    turns = [
        Turn(index=0, role="user", content="Tell me about machine learning algorithms"),
        Turn(index=1, role="assistant", content="Machine learning algorithms include regression and classification"),
        Turn(index=2, role="user", content="What about deep learning?"),
        Turn(index=3, role="assistant", content="Deep learning is a subset of machine learning using neural networks"),
    ]
    score = evaluate_context_consistency(turns)
    assert score > 0.0  # Should have some overlap between assistant turns


def test_context_consistency_low():
    turns = [
        Turn(index=0, role="assistant", content="Apples bananas cherries dates elderberry"),
        Turn(index=1, role="assistant", content="Xenon yttrium zinc tungsten vanadium"),
    ]
    score = evaluate_context_consistency(turns)
    assert score == 0.0  # No word overlap at all


def test_context_consistency_single_assistant():
    turns = [
        Turn(index=0, role="user", content="Hello"),
        Turn(index=1, role="assistant", content="Hi there"),
    ]
    score = evaluate_context_consistency(turns)
    assert score == 1.0  # Single assistant turn: fully consistent


def test_context_consistency_no_turns():
    score = evaluate_context_consistency([])
    assert score == 1.0


# --- evaluate_topic_drift ---

def test_topic_drift_low():
    turns = [
        Turn(index=0, role="user", content="Python programming language"),
        Turn(index=1, role="assistant", content="Python programming language is great"),
    ]
    score = evaluate_topic_drift(turns)
    assert score < 0.5  # Low drift: high word overlap


def test_topic_drift_high():
    turns = [
        Turn(index=0, role="user", content="Apples bananas cherries"),
        Turn(index=1, role="assistant", content="Xenon yttrium zinc"),
    ]
    score = evaluate_topic_drift(turns)
    assert score == 1.0  # Complete drift: no word overlap


def test_topic_drift_single_turn():
    turns = [Turn(index=0, role="user", content="Hello")]
    score = evaluate_topic_drift(turns)
    assert score == 0.0  # No drift with single turn


def test_topic_drift_empty():
    score = evaluate_topic_drift([])
    assert score == 0.0


# --- evaluate_conversation ---

def test_evaluate_conversation_full():
    turns = [
        Turn(index=0, role="user", content="Tell me about Python programming"),
        Turn(index=1, role="assistant", content="Python is a versatile programming language"),
        Turn(index=2, role="user", content="What are Python's main features?"),
        Turn(index=3, role="assistant", content="Python features include dynamic typing and readability"),
    ]
    report = evaluate_conversation("conv-1", turns)
    assert report.conversation_id == "conv-1"
    assert len(report.turns) == 4
    assert len(report.turn_scores) == 4
    assert report.metrics.turn_count == 4
    assert 0.0 <= report.metrics.avg_turn_score <= 1.0
    assert 0.0 <= report.metrics.context_consistency <= 1.0
    assert 0.0 <= report.metrics.topic_drift_score <= 1.0


def test_evaluate_conversation_empty():
    report = evaluate_conversation("empty", [])
    assert report.metrics.turn_count == 0
    assert report.metrics.avg_turn_score == 0.0


# --- Turn as_dict / from_dict ---

def test_turn_as_dict():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turn = Turn(index=0, role="user", content="Hello", timestamp=t)
    d = turn.as_dict()
    assert d["index"] == 0
    assert d["role"] == "user"
    assert d["content"] == "Hello"
    assert d["timestamp"] is not None


def test_turn_from_dict():
    d = {"index": 1, "role": "assistant", "content": "Hi there", "timestamp": None}
    turn = Turn.from_dict(d)
    assert turn.index == 1
    assert turn.role == "assistant"
    assert turn.content == "Hi there"
    assert turn.timestamp is None


def test_turn_roundtrip():
    t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    original = Turn(index=2, role="user", content="Test content", timestamp=t)
    restored = Turn.from_dict(original.as_dict())
    assert restored.index == original.index
    assert restored.role == original.role
    assert restored.content == original.content


# --- ConversationMetrics as_dict / from_dict ---

def test_conversation_metrics_as_dict():
    m = ConversationMetrics(
        turn_count=4, avg_turn_score=0.8, min_turn_score=0.5,
        max_turn_score=1.0, context_consistency=0.7, topic_drift_score=0.2,
    )
    d = m.as_dict()
    assert d["turn_count"] == 4
    assert d["avg_turn_score"] == 0.8


def test_conversation_metrics_from_dict():
    d = {
        "turn_count": 3, "avg_turn_score": 0.9, "min_turn_score": 0.6,
        "max_turn_score": 1.0, "context_consistency": 0.8, "topic_drift_score": 0.1,
    }
    m = ConversationMetrics.from_dict(d)
    assert m.turn_count == 3
    assert m.context_consistency == 0.8


# --- MultiTurnReport format_text ---

def test_multiturn_report_format_text():
    turns = [Turn(index=0, role="user", content="Hello")]
    scores = [TurnScore(turn_index=0, score=1.0, passed=True)]
    metrics = ConversationMetrics(
        turn_count=1, avg_turn_score=1.0, min_turn_score=1.0,
        max_turn_score=1.0, context_consistency=1.0, topic_drift_score=0.0,
    )
    report = MultiTurnReport(
        conversation_id="test", turns=turns, turn_scores=scores, metrics=metrics,
    )
    text = report.format_text()
    assert "test" in text
    assert "Turn 0" in text
    assert "PASS" in text


def test_multiturn_report_format_text_fail():
    turns = [Turn(index=0, role="user", content="Hi")]
    scores = [TurnScore(turn_index=0, score=0.0, passed=False)]
    metrics = ConversationMetrics(
        turn_count=1, avg_turn_score=0.0, min_turn_score=0.0,
        max_turn_score=0.0, context_consistency=1.0, topic_drift_score=0.0,
    )
    report = MultiTurnReport(
        conversation_id="fail-conv", turns=turns, turn_scores=scores, metrics=metrics,
    )
    text = report.format_text()
    assert "FAIL" in text


# --- MultiTurnReport as_dict / from_dict ---

def test_multiturn_report_roundtrip():
    turns = [
        Turn(index=0, role="user", content="Hello world test content"),
        Turn(index=1, role="assistant", content="Hi there, how can I help?"),
    ]
    report = evaluate_conversation("rt-test", turns)
    d = report.as_dict()
    restored = MultiTurnReport.from_dict(d)
    assert restored.conversation_id == "rt-test"
    assert len(restored.turns) == 2
    assert len(restored.turn_scores) == 2
    assert restored.metrics.turn_count == 2


# --- Edge cases ---

def test_single_turn_conversation():
    turns = [Turn(index=0, role="user", content="Just one turn with enough content")]
    report = evaluate_conversation("single", turns)
    assert report.metrics.turn_count == 1
    assert report.metrics.topic_drift_score == 0.0
    assert report.metrics.context_consistency == 1.0
    assert len(report.turn_scores) == 1
