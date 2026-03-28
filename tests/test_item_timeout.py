"""Tests for item_timeout module."""

from __future__ import annotations

from evalyn_sdk.evaluation.item_timeout import (
    TimeoutConfig,
    TimeoutEvent,
    TimeoutReport,
    build_timeout_report,
    check_timeout,
    get_timeout_for_metric,
    identify_slow_items,
    record_timeout_event,
    suggest_timeout_adjustments,
)


# -- TimeoutConfig --


def test_config_defaults():
    cfg = TimeoutConfig()
    assert cfg.default_timeout_seconds == 30.0
    assert cfg.per_metric_timeouts == {}
    assert cfg.abort_on_timeout is False


def test_config_as_dict():
    cfg = TimeoutConfig(default_timeout_seconds=60.0, per_metric_timeouts={"helpfulness": 120.0}, abort_on_timeout=True)
    d = cfg.as_dict()
    assert d["default_timeout_seconds"] == 60.0
    assert d["per_metric_timeouts"] == {"helpfulness": 120.0}
    assert d["abort_on_timeout"] is True


def test_config_from_dict():
    d = {"default_timeout_seconds": 45.0, "per_metric_timeouts": {"speed": 10.0}, "abort_on_timeout": True}
    cfg = TimeoutConfig.from_dict(d)
    assert cfg.default_timeout_seconds == 45.0
    assert cfg.per_metric_timeouts == {"speed": 10.0}
    assert cfg.abort_on_timeout is True


def test_config_from_dict_defaults():
    cfg = TimeoutConfig.from_dict({})
    assert cfg.default_timeout_seconds == 30.0
    assert cfg.per_metric_timeouts == {}
    assert cfg.abort_on_timeout is False


def test_config_roundtrip():
    cfg = TimeoutConfig(default_timeout_seconds=15.0, per_metric_timeouts={"a": 5.0, "b": 10.0}, abort_on_timeout=True)
    restored = TimeoutConfig.from_dict(cfg.as_dict())
    assert restored.default_timeout_seconds == cfg.default_timeout_seconds
    assert restored.per_metric_timeouts == cfg.per_metric_timeouts
    assert restored.abort_on_timeout == cfg.abort_on_timeout


# -- TimeoutEvent --


def test_event_defaults():
    e = TimeoutEvent(item_id="item-1")
    assert e.item_id == "item-1"
    assert e.metric_id == ""
    assert e.elapsed_ms == 0.0
    assert e.limit_ms == 0.0
    assert e.timed_out is False


def test_event_as_dict():
    e = TimeoutEvent(item_id="item-1", metric_id="m1", elapsed_ms=500.0, limit_ms=1000.0, timed_out=False)
    d = e.as_dict()
    assert d["item_id"] == "item-1"
    assert d["metric_id"] == "m1"
    assert d["elapsed_ms"] == 500.0
    assert d["limit_ms"] == 1000.0
    assert d["timed_out"] is False


def test_event_timed_out_as_dict():
    e = TimeoutEvent(item_id="slow", elapsed_ms=2000.0, limit_ms=1000.0, timed_out=True)
    d = e.as_dict()
    assert d["timed_out"] is True


# -- TimeoutReport --


def test_report_defaults():
    r = TimeoutReport()
    assert r.events == []
    assert r.total_items == 0
    assert r.timed_out_count == 0
    assert r.timeout_rate == 0.0
    assert r.avg_elapsed_ms == 0.0


def test_report_as_dict():
    e = TimeoutEvent(item_id="x", elapsed_ms=100.0, limit_ms=200.0)
    r = TimeoutReport(events=[e], total_items=1, timed_out_count=0, timeout_rate=0.0, avg_elapsed_ms=100.0)
    d = r.as_dict()
    assert d["total_items"] == 1
    assert len(d["events"]) == 1
    assert d["events"][0]["item_id"] == "x"


def test_report_from_dict():
    d = {
        "events": [
            {"item_id": "a", "metric_id": "", "elapsed_ms": 500.0, "limit_ms": 1000.0, "timed_out": False},
            {"item_id": "b", "metric_id": "m1", "elapsed_ms": 1500.0, "limit_ms": 1000.0, "timed_out": True},
        ],
        "total_items": 2,
        "timed_out_count": 1,
        "timeout_rate": 0.5,
        "avg_elapsed_ms": 1000.0,
    }
    r = TimeoutReport.from_dict(d)
    assert r.total_items == 2
    assert r.timed_out_count == 1
    assert len(r.events) == 2
    assert r.events[1].timed_out is True


def test_report_from_dict_defaults():
    r = TimeoutReport.from_dict({})
    assert r.total_items == 0
    assert r.events == []


def test_report_roundtrip():
    e1 = TimeoutEvent(item_id="a", elapsed_ms=100.0, limit_ms=200.0, timed_out=False)
    e2 = TimeoutEvent(item_id="b", metric_id="m1", elapsed_ms=300.0, limit_ms=200.0, timed_out=True)
    r = TimeoutReport(events=[e1, e2], total_items=2, timed_out_count=1, timeout_rate=0.5, avg_elapsed_ms=200.0)
    restored = TimeoutReport.from_dict(r.as_dict())
    assert restored.total_items == r.total_items
    assert restored.timed_out_count == r.timed_out_count
    assert len(restored.events) == 2
    assert restored.events[0].item_id == "a"
    assert restored.events[1].timed_out is True


def test_report_format_text_basic():
    r = TimeoutReport(total_items=10, timed_out_count=2, timeout_rate=0.2, avg_elapsed_ms=500.0)
    text = r.format_text()
    assert "Total items: 10" in text
    assert "Timed out: 2" in text
    assert "20.0%" in text
    assert "500.0ms" in text


def test_report_format_text_with_events():
    e = TimeoutEvent(item_id="slow-item", metric_id="helpfulness", elapsed_ms=5000.0, limit_ms=3000.0, timed_out=True)
    r = TimeoutReport(events=[e], total_items=1, timed_out_count=1, timeout_rate=1.0, avg_elapsed_ms=5000.0)
    text = r.format_text()
    assert "slow-item" in text
    assert "helpfulness" in text
    assert "5000.0ms" in text


def test_report_format_text_no_timeouts():
    e = TimeoutEvent(item_id="fast-item", elapsed_ms=100.0, limit_ms=1000.0, timed_out=False)
    r = TimeoutReport(events=[e], total_items=1, timed_out_count=0, timeout_rate=0.0, avg_elapsed_ms=100.0)
    text = r.format_text()
    assert "Timed out: 0" in text
    assert "Timeout events:" not in text


# -- get_timeout_for_metric --


def test_get_timeout_default():
    cfg = TimeoutConfig(default_timeout_seconds=30.0)
    assert get_timeout_for_metric("unknown_metric", cfg) == 30.0


def test_get_timeout_per_metric():
    cfg = TimeoutConfig(default_timeout_seconds=30.0, per_metric_timeouts={"helpfulness": 60.0})
    assert get_timeout_for_metric("helpfulness", cfg) == 60.0


def test_get_timeout_falls_back_to_default():
    cfg = TimeoutConfig(default_timeout_seconds=15.0, per_metric_timeouts={"other": 60.0})
    assert get_timeout_for_metric("helpfulness", cfg) == 15.0


# -- check_timeout --


def test_check_timeout_true():
    assert check_timeout(31.0, 30.0) is True


def test_check_timeout_false():
    assert check_timeout(29.0, 30.0) is False


def test_check_timeout_exact():
    assert check_timeout(30.0, 30.0) is False


def test_check_timeout_zero_limit():
    assert check_timeout(0.1, 0.0) is True


def test_check_timeout_zero_elapsed():
    assert check_timeout(0.0, 30.0) is False


# -- record_timeout_event --


def test_record_timeout_event_timed_out():
    e = record_timeout_event("item-1", "helpfulness", elapsed_ms=5000.0, limit_ms=3000.0)
    assert e.item_id == "item-1"
    assert e.metric_id == "helpfulness"
    assert e.elapsed_ms == 5000.0
    assert e.limit_ms == 3000.0
    assert e.timed_out is True


def test_record_timeout_event_not_timed_out():
    e = record_timeout_event("item-2", "speed", elapsed_ms=1000.0, limit_ms=3000.0)
    assert e.timed_out is False


def test_record_timeout_event_exact_limit():
    e = record_timeout_event("item-3", "m1", elapsed_ms=1000.0, limit_ms=1000.0)
    assert e.timed_out is False


# -- build_timeout_report --


def test_build_timeout_report_basic():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=100.0, limit_ms=500.0, timed_out=False),
        TimeoutEvent(item_id="b", elapsed_ms=600.0, limit_ms=500.0, timed_out=True),
    ]
    report = build_timeout_report(events)
    assert report.total_items == 2
    assert report.timed_out_count == 1
    assert report.timeout_rate == 0.5
    assert report.avg_elapsed_ms == 350.0


def test_build_timeout_report_empty():
    report = build_timeout_report([])
    assert report.total_items == 0
    assert report.timed_out_count == 0
    assert report.timeout_rate == 0.0
    assert report.avg_elapsed_ms == 0.0


def test_build_timeout_report_all_timed_out():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=2000.0, limit_ms=1000.0, timed_out=True),
        TimeoutEvent(item_id="b", elapsed_ms=3000.0, limit_ms=1000.0, timed_out=True),
    ]
    report = build_timeout_report(events)
    assert report.timed_out_count == 2
    assert report.timeout_rate == 1.0


def test_build_timeout_report_none_timed_out():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=100.0, limit_ms=1000.0, timed_out=False),
        TimeoutEvent(item_id="b", elapsed_ms=200.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    assert report.timed_out_count == 0
    assert report.timeout_rate == 0.0


# -- identify_slow_items --


def test_identify_slow_items_basic():
    events = [
        TimeoutEvent(item_id="fast", elapsed_ms=100.0, limit_ms=1000.0, timed_out=False),
        TimeoutEvent(item_id="slow", elapsed_ms=900.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    slow = identify_slow_items(report, threshold_pct=0.8)
    assert "slow" in slow
    assert "fast" not in slow


def test_identify_slow_items_all_fast():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=100.0, limit_ms=1000.0, timed_out=False),
        TimeoutEvent(item_id="b", elapsed_ms=200.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    slow = identify_slow_items(report, threshold_pct=0.8)
    assert slow == []


def test_identify_slow_items_includes_timed_out():
    events = [
        TimeoutEvent(item_id="timed_out_item", elapsed_ms=2000.0, limit_ms=1000.0, timed_out=True),
    ]
    report = build_timeout_report(events)
    slow = identify_slow_items(report, threshold_pct=0.8)
    assert "timed_out_item" in slow


def test_identify_slow_items_zero_limit_skipped():
    events = [
        TimeoutEvent(item_id="zero_limit", elapsed_ms=100.0, limit_ms=0.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    slow = identify_slow_items(report, threshold_pct=0.8)
    assert slow == []


def test_identify_slow_items_custom_threshold():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=600.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    assert identify_slow_items(report, threshold_pct=0.5) == ["a"]
    assert identify_slow_items(report, threshold_pct=0.7) == []


# -- suggest_timeout_adjustments --


def test_suggest_adjustments_empty():
    report = TimeoutReport()
    assert suggest_timeout_adjustments(report) == []


def test_suggest_adjustments_high_timeout_rate():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=2000.0, limit_ms=1000.0, timed_out=True),
        TimeoutEvent(item_id="b", elapsed_ms=3000.0, limit_ms=1000.0, timed_out=True),
        TimeoutEvent(item_id="c", elapsed_ms=500.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    suggestions = suggest_timeout_adjustments(report)
    assert any("increasing" in s.lower() or "increase" in s.lower() for s in suggestions)


def test_suggest_adjustments_all_fast():
    events = [
        TimeoutEvent(item_id="a", elapsed_ms=50.0, limit_ms=1000.0, timed_out=False),
        TimeoutEvent(item_id="b", elapsed_ms=80.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    suggestions = suggest_timeout_adjustments(report)
    assert any("reducing" in s.lower() or "reduce" in s.lower() for s in suggestions)


def test_suggest_adjustments_per_metric():
    events = [
        TimeoutEvent(item_id="a", metric_id="slow_metric", elapsed_ms=2000.0, limit_ms=1000.0, timed_out=True),
        TimeoutEvent(item_id="b", metric_id="slow_metric", elapsed_ms=3000.0, limit_ms=1000.0, timed_out=True),
        TimeoutEvent(item_id="c", metric_id="fast_metric", elapsed_ms=100.0, limit_ms=1000.0, timed_out=False),
    ]
    report = build_timeout_report(events)
    suggestions = suggest_timeout_adjustments(report)
    assert any("slow_metric" in s for s in suggestions)
