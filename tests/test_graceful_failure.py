"""Tests for graceful_failure module."""

from __future__ import annotations

from evalyn_sdk.evaluation.graceful_failure import (
    GracefulConfig,
    GracefulEvaluator,
    GracefulReport,
    ItemFailure,
    classify_error,
    create_graceful_evaluator,
    suggest_recovery,
)


# -- ItemFailure --


def test_item_failure_defaults():
    f = ItemFailure(item_id="item-1")
    assert f.item_id == "item-1"
    assert f.metric_id == ""
    assert f.error == ""
    assert f.error_type == ""
    assert f.recoverable is True


def test_item_failure_as_dict():
    f = ItemFailure(item_id="x", metric_id="m1", error="boom", error_type="api_error", recoverable=False)
    d = f.as_dict()
    assert d["item_id"] == "x"
    assert d["metric_id"] == "m1"
    assert d["error"] == "boom"
    assert d["error_type"] == "api_error"
    assert d["recoverable"] is False


# -- GracefulConfig --


def test_config_defaults():
    cfg = GracefulConfig()
    assert cfg.max_failures == 0
    assert cfg.skip_on_timeout is True
    assert cfg.retry_count == 0
    assert cfg.collect_errors is True


def test_config_as_dict():
    cfg = GracefulConfig(max_failures=5, retry_count=2)
    d = cfg.as_dict()
    assert d["max_failures"] == 5
    assert d["retry_count"] == 2


def test_config_from_dict():
    d = {"max_failures": 10, "skip_on_timeout": False, "retry_count": 3, "collect_errors": False}
    cfg = GracefulConfig.from_dict(d)
    assert cfg.max_failures == 10
    assert cfg.skip_on_timeout is False
    assert cfg.retry_count == 3
    assert cfg.collect_errors is False


def test_config_from_dict_defaults():
    cfg = GracefulConfig.from_dict({})
    assert cfg.max_failures == 0
    assert cfg.skip_on_timeout is True


def test_config_roundtrip():
    cfg = GracefulConfig(max_failures=7, skip_on_timeout=False, retry_count=2, collect_errors=False)
    restored = GracefulConfig.from_dict(cfg.as_dict())
    assert restored.max_failures == cfg.max_failures
    assert restored.skip_on_timeout == cfg.skip_on_timeout
    assert restored.retry_count == cfg.retry_count
    assert restored.collect_errors == cfg.collect_errors


# -- GracefulReport --


def test_report_defaults():
    r = GracefulReport()
    assert r.total_items == 0
    assert r.succeeded == 0
    assert r.failed == 0
    assert r.failure_rate == 0.0
    assert r.should_abort is False
    assert r.failures == []


def test_report_as_dict():
    f = ItemFailure(item_id="x", error="e")
    r = GracefulReport(total_items=10, succeeded=8, failed=2, failures=[f], failure_rate=0.2)
    d = r.as_dict()
    assert d["total_items"] == 10
    assert d["succeeded"] == 8
    assert d["failed"] == 2
    assert len(d["failures"]) == 1
    assert d["failures"][0]["item_id"] == "x"


def test_report_from_dict():
    d = {
        "total_items": 5,
        "succeeded": 3,
        "failed": 2,
        "skipped": 0,
        "failures": [{"item_id": "a", "error": "err", "metric_id": "", "error_type": "timeout", "recoverable": True}],
        "failure_rate": 0.4,
        "should_abort": True,
    }
    r = GracefulReport.from_dict(d)
    assert r.total_items == 5
    assert r.failed == 2
    assert r.should_abort is True
    assert len(r.failures) == 1
    assert r.failures[0].error_type == "timeout"


def test_report_roundtrip():
    f = ItemFailure(item_id="i1", metric_id="m1", error="timeout", error_type="timeout", recoverable=True)
    r = GracefulReport(total_items=10, succeeded=9, failed=1, failures=[f], failure_rate=0.1)
    restored = GracefulReport.from_dict(r.as_dict())
    assert restored.total_items == r.total_items
    assert restored.succeeded == r.succeeded
    assert restored.failed == r.failed
    assert len(restored.failures) == 1
    assert restored.failures[0].item_id == "i1"


def test_report_format_text_basic():
    r = GracefulReport(total_items=10, succeeded=8, failed=2, failure_rate=0.2)
    text = r.format_text()
    assert "Total items: 10" in text
    assert "Succeeded: 8" in text
    assert "Failed: 2" in text
    assert "20.0%" in text


def test_report_format_text_aborted():
    r = GracefulReport(total_items=5, failed=5, failure_rate=1.0, should_abort=True)
    text = r.format_text()
    assert "ABORTED" in text


def test_report_format_text_with_failures():
    f = ItemFailure(item_id="item-1", metric_id="helpfulness", error="timeout", error_type="timeout")
    r = GracefulReport(total_items=1, failed=1, failures=[f], failure_rate=1.0)
    text = r.format_text()
    assert "item-1" in text
    assert "helpfulness" in text
    assert "timeout" in text


# -- GracefulEvaluator --


def test_evaluator_record_success():
    ev = GracefulEvaluator(GracefulConfig())
    ev.record_success("item-1")
    ev.record_success("item-2")
    report = ev.get_report()
    assert report.succeeded == 2
    assert report.failed == 0
    assert report.total_items == 2


def test_evaluator_record_failure():
    ev = GracefulEvaluator(GracefulConfig())
    ev.record_failure("item-1", "connection timeout")
    report = ev.get_report()
    assert report.failed == 1
    assert len(report.failures) == 1
    assert report.failures[0].error_type == "timeout"


def test_evaluator_should_continue_unlimited():
    ev = GracefulEvaluator(GracefulConfig(max_failures=0))
    for i in range(100):
        ev.record_failure(f"item-{i}", "error")
    assert ev.should_continue() is True


def test_evaluator_should_continue_respects_max():
    ev = GracefulEvaluator(GracefulConfig(max_failures=3))
    ev.record_failure("item-1", "error")
    ev.record_failure("item-2", "error")
    assert ev.should_continue() is True
    ev.record_failure("item-3", "error")
    assert ev.should_continue() is False


def test_evaluator_report_failure_rate():
    ev = GracefulEvaluator(GracefulConfig())
    ev.record_success("a")
    ev.record_success("b")
    ev.record_failure("c", "error")
    report = ev.get_report()
    assert abs(report.failure_rate - 1 / 3) < 0.01


def test_evaluator_report_should_abort():
    ev = GracefulEvaluator(GracefulConfig(max_failures=2))
    ev.record_failure("a", "e1")
    ev.record_failure("b", "e2")
    report = ev.get_report()
    assert report.should_abort is True


def test_evaluator_reset():
    ev = GracefulEvaluator(GracefulConfig())
    ev.record_success("a")
    ev.record_failure("b", "error")
    ev.reset()
    report = ev.get_report()
    assert report.total_items == 0
    assert report.succeeded == 0
    assert report.failed == 0
    assert report.failures == []


def test_evaluator_collect_errors_disabled():
    ev = GracefulEvaluator(GracefulConfig(collect_errors=False))
    ev.record_failure("item-1", "error")
    report = ev.get_report()
    assert report.failed == 1
    assert report.failures == []  # not collected


def test_evaluator_failure_with_metric_id():
    ev = GracefulEvaluator(GracefulConfig())
    ev.record_failure("item-1", "API rate limit 429", metric_id="helpfulness")
    report = ev.get_report()
    assert report.failures[0].metric_id == "helpfulness"
    assert report.failures[0].error_type == "api_error"


# -- classify_error --


def test_classify_error_timeout():
    err_type, recoverable = classify_error("Connection timed out after 30s")
    assert err_type == "timeout"
    assert recoverable is True


def test_classify_error_timeout_keyword():
    err_type, _ = classify_error("Request timeout")
    assert err_type == "timeout"


def test_classify_error_parse():
    err_type, recoverable = classify_error("JSON decode error at line 5")
    assert err_type == "parse_error"
    assert recoverable is True


def test_classify_error_parse_keyword():
    err_type, _ = classify_error("Failed to parse response")
    assert err_type == "parse_error"


def test_classify_error_api():
    err_type, recoverable = classify_error("API rate limit exceeded (429)")
    assert err_type == "api_error"
    assert recoverable is True


def test_classify_error_api_500():
    err_type, _ = classify_error("Internal server error 500")
    assert err_type == "api_error"


def test_classify_error_validation():
    err_type, recoverable = classify_error("Schema validation failed: missing required field")
    assert err_type == "validation_error"
    assert recoverable is False


def test_classify_error_unknown():
    err_type, recoverable = classify_error("Something completely unexpected")
    assert err_type == "unknown"
    assert recoverable is False


# -- suggest_recovery --


def test_suggest_recovery_timeout():
    f = ItemFailure(item_id="x", error_type="timeout")
    suggestion = suggest_recovery(f)
    assert "timeout" in suggestion.lower() or "retry" in suggestion.lower()


def test_suggest_recovery_parse_error():
    f = ItemFailure(item_id="x", error_type="parse_error")
    suggestion = suggest_recovery(f)
    assert "prompt" in suggestion.lower() or "format" in suggestion.lower()


def test_suggest_recovery_api_error():
    f = ItemFailure(item_id="x", error_type="api_error")
    suggestion = suggest_recovery(f)
    assert "retry" in suggestion.lower() or "backoff" in suggestion.lower()


def test_suggest_recovery_validation_error():
    f = ItemFailure(item_id="x", error_type="validation_error")
    suggestion = suggest_recovery(f)
    assert "fix" in suggestion.lower() or "schema" in suggestion.lower()


def test_suggest_recovery_unknown():
    f = ItemFailure(item_id="x", error_type="unknown")
    suggestion = suggest_recovery(f)
    assert "inspect" in suggestion.lower() or "manual" in suggestion.lower()


# -- create_graceful_evaluator factory --


def test_create_graceful_evaluator_default():
    ev = create_graceful_evaluator()
    assert ev.config.max_failures == 0
    assert ev.should_continue() is True


def test_create_graceful_evaluator_with_limit():
    ev = create_graceful_evaluator(max_failures=5)
    assert ev.config.max_failures == 5


# -- edge cases --


def test_all_items_fail():
    ev = GracefulEvaluator(GracefulConfig())
    for i in range(10):
        ev.record_failure(f"item-{i}", "error")
    report = ev.get_report()
    assert report.total_items == 10
    assert report.succeeded == 0
    assert report.failed == 10
    assert report.failure_rate == 1.0


def test_all_items_succeed():
    ev = GracefulEvaluator(GracefulConfig())
    for i in range(10):
        ev.record_success(f"item-{i}")
    report = ev.get_report()
    assert report.total_items == 10
    assert report.succeeded == 10
    assert report.failed == 0
    assert report.failure_rate == 0.0
    assert report.failures == []


def test_empty_report_failure_rate():
    ev = GracefulEvaluator(GracefulConfig())
    report = ev.get_report()
    assert report.failure_rate == 0.0
