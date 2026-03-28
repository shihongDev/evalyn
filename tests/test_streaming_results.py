"""Tests for streaming_results module."""

from __future__ import annotations

from evalyn_sdk.evaluation.streaming_results import (
    PartialResult,
    StreamingBuffer,
    StreamingStats,
)


# --- PartialResult ---


def test_partial_result_as_dict():
    r = PartialResult("item1", "accuracy", 0.9, True, 1000.0)
    d = r.as_dict()
    assert d["item_id"] == "item1"
    assert d["metric_id"] == "accuracy"
    assert d["score"] == 0.9
    assert d["passed"] is True
    assert d["timestamp_ms"] == 1000.0


def test_partial_result_defaults():
    r = PartialResult("i", "m", 0.5, False)
    assert r.timestamp_ms == 0.0


# --- StreamingStats ---


def test_streaming_stats_as_dict():
    s = StreamingStats(10, 5, 5, 0.5, 0.8, 0.6)
    d = s.as_dict()
    assert d["total_expected"] == 10
    assert d["completed"] == 5
    assert d["in_progress"] == 5
    assert d["completion_pct"] == 0.5
    assert d["avg_score"] == 0.8
    assert d["pass_rate"] == 0.6


def test_streaming_stats_format_text():
    s = StreamingStats(10, 5, 5, 0.5, 0.8123, 0.6)
    text = s.format_text()
    assert "Streaming Stats" in text
    assert "5/10" in text
    assert "50.0%" in text
    assert "0.8123" in text
    assert "60.0%" in text


def test_streaming_stats_format_text_zero():
    s = StreamingStats(0, 0, 0, 0.0, 0.0, 0.0)
    text = s.format_text()
    assert "0/0" in text


# --- StreamingBuffer: add_result ---


def test_add_result_increments():
    buf = StreamingBuffer(expected_count=5)
    assert len(buf.results) == 0
    buf.add_result(PartialResult("i1", "m1", 0.9, True))
    assert len(buf.results) == 1
    buf.add_result(PartialResult("i2", "m1", 0.8, True))
    assert len(buf.results) == 2


def test_add_result_preserves_order():
    buf = StreamingBuffer(expected_count=3)
    buf.add_result(PartialResult("a", "m", 0.1, False))
    buf.add_result(PartialResult("b", "m", 0.2, True))
    buf.add_result(PartialResult("c", "m", 0.3, True))
    assert [r.item_id for r in buf.results] == ["a", "b", "c"]


# --- StreamingBuffer: get_stats ---


def test_get_stats_empty():
    buf = StreamingBuffer(expected_count=10)
    stats = buf.get_stats()
    assert stats.completed == 0
    assert stats.in_progress == 10
    assert stats.completion_pct == 0.0
    assert stats.avg_score == 0.0
    assert stats.pass_rate == 0.0


def test_get_stats_partial():
    buf = StreamingBuffer(expected_count=4)
    buf.add_result(PartialResult("i1", "m1", 1.0, True))
    buf.add_result(PartialResult("i2", "m1", 0.5, False))
    stats = buf.get_stats()
    assert stats.completed == 2
    assert stats.in_progress == 2
    assert stats.completion_pct == 0.5
    assert stats.avg_score == 0.75
    assert stats.pass_rate == 0.5


def test_get_stats_all_completed():
    buf = StreamingBuffer(expected_count=2)
    buf.add_result(PartialResult("i1", "m1", 0.8, True))
    buf.add_result(PartialResult("i2", "m1", 0.6, True))
    stats = buf.get_stats()
    assert stats.completed == 2
    assert stats.in_progress == 0
    assert stats.completion_pct == 1.0
    assert stats.pass_rate == 1.0


# --- StreamingBuffer: get_results_by_metric ---


def test_get_results_by_metric_filters():
    buf = StreamingBuffer(expected_count=4)
    buf.add_result(PartialResult("i1", "accuracy", 0.9, True))
    buf.add_result(PartialResult("i1", "fluency", 0.8, True))
    buf.add_result(PartialResult("i2", "accuracy", 0.7, False))
    buf.add_result(PartialResult("i2", "fluency", 0.6, True))
    acc = buf.get_results_by_metric("accuracy")
    assert len(acc) == 2
    assert all(r.metric_id == "accuracy" for r in acc)


def test_get_results_by_metric_empty():
    buf = StreamingBuffer(expected_count=2)
    buf.add_result(PartialResult("i1", "accuracy", 0.9, True))
    assert buf.get_results_by_metric("nonexistent") == []


# --- StreamingBuffer: get_results_by_item ---


def test_get_results_by_item_filters():
    buf = StreamingBuffer(expected_count=4)
    buf.add_result(PartialResult("i1", "accuracy", 0.9, True))
    buf.add_result(PartialResult("i1", "fluency", 0.8, True))
    buf.add_result(PartialResult("i2", "accuracy", 0.7, False))
    items = buf.get_results_by_item("i1")
    assert len(items) == 2
    assert all(r.item_id == "i1" for r in items)


def test_get_results_by_item_empty():
    buf = StreamingBuffer(expected_count=2)
    assert buf.get_results_by_item("missing") == []


# --- StreamingBuffer: is_complete ---


def test_is_complete_true():
    buf = StreamingBuffer(expected_count=2)
    buf.add_result(PartialResult("i1", "m1", 1.0, True))
    buf.add_result(PartialResult("i2", "m1", 1.0, True))
    assert buf.is_complete() is True


def test_is_complete_false_partial():
    buf = StreamingBuffer(expected_count=3)
    buf.add_result(PartialResult("i1", "m1", 1.0, True))
    assert buf.is_complete() is False


def test_is_complete_false_zero_expected():
    buf = StreamingBuffer(expected_count=0)
    assert buf.is_complete() is False


def test_is_complete_over_expected():
    buf = StreamingBuffer(expected_count=1)
    buf.add_result(PartialResult("i1", "m1", 1.0, True))
    buf.add_result(PartialResult("i2", "m1", 1.0, True))
    assert buf.is_complete() is True


# --- StreamingBuffer: get_early_insights ---


def test_get_early_insights_empty():
    buf = StreamingBuffer(expected_count=5)
    insights = buf.get_early_insights()
    assert insights["avg_score"] == 0.0
    assert insights["pass_rate"] == 0.0
    assert insights["failing_items"] == []
    assert insights["completed"] == 0


def test_get_early_insights_with_results():
    buf = StreamingBuffer(expected_count=4)
    buf.add_result(PartialResult("i1", "m1", 1.0, True))
    buf.add_result(PartialResult("i2", "m1", 0.0, False))
    buf.add_result(PartialResult("i3", "m1", 0.5, True))
    insights = buf.get_early_insights()
    assert insights["avg_score"] == 0.5
    assert insights["pass_rate"] == 0.6667
    assert insights["failing_items"] == ["i2"]
    assert insights["completed"] == 3


def test_get_early_insights_multiple_failing():
    buf = StreamingBuffer(expected_count=3)
    buf.add_result(PartialResult("i1", "m1", 0.0, False))
    buf.add_result(PartialResult("i2", "m1", 0.0, False))
    buf.add_result(PartialResult("i1", "m2", 0.0, False))
    insights = buf.get_early_insights()
    # Failing items should be unique and sorted
    assert insights["failing_items"] == ["i1", "i2"]


# --- StreamingBuffer: as_dict / from_dict ---


def test_as_dict_empty():
    buf = StreamingBuffer(expected_count=5)
    d = buf.as_dict()
    assert d["results"] == []
    assert d["expected_count"] == 5


def test_as_dict_with_results():
    buf = StreamingBuffer(expected_count=2)
    buf.add_result(PartialResult("i1", "m1", 0.9, True, 100.0))
    d = buf.as_dict()
    assert len(d["results"]) == 1
    assert d["results"][0]["item_id"] == "i1"


def test_from_dict_empty():
    buf = StreamingBuffer.from_dict({"results": [], "expected_count": 3})
    assert len(buf.results) == 0
    assert buf.expected_count == 3


def test_from_dict_with_results():
    data = {
        "results": [
            {
                "item_id": "i1",
                "metric_id": "m1",
                "score": 0.9,
                "passed": True,
                "timestamp_ms": 50.0,
            }
        ],
        "expected_count": 1,
    }
    buf = StreamingBuffer.from_dict(data)
    assert len(buf.results) == 1
    assert buf.results[0].item_id == "i1"
    assert buf.results[0].timestamp_ms == 50.0


def test_roundtrip_as_dict_from_dict():
    buf = StreamingBuffer(expected_count=3)
    buf.add_result(PartialResult("i1", "accuracy", 0.9, True, 10.0))
    buf.add_result(PartialResult("i2", "accuracy", 0.4, False, 20.0))
    restored = StreamingBuffer.from_dict(buf.as_dict())
    assert len(restored.results) == 2
    assert restored.expected_count == 3
    assert restored.results[0].score == 0.9
    assert restored.results[1].passed is False


# --- Edge cases ---


def test_single_result_stats():
    buf = StreamingBuffer(expected_count=1)
    buf.add_result(PartialResult("i1", "m1", 0.75, True))
    stats = buf.get_stats()
    assert stats.completed == 1
    assert stats.avg_score == 0.75
    assert stats.pass_rate == 1.0


def test_all_failing():
    buf = StreamingBuffer(expected_count=2)
    buf.add_result(PartialResult("i1", "m1", 0.0, False))
    buf.add_result(PartialResult("i2", "m1", 0.1, False))
    stats = buf.get_stats()
    assert stats.pass_rate == 0.0
    insights = buf.get_early_insights()
    assert len(insights["failing_items"]) == 2


def test_from_dict_missing_timestamp():
    data = {
        "results": [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "passed": True}
        ],
        "expected_count": 1,
    }
    buf = StreamingBuffer.from_dict(data)
    assert buf.results[0].timestamp_ms == 0.0
