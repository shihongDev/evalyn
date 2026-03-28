"""Tests for calibration batch processing module."""

from evalyn_sdk.calibration.batch_processing import (
    BatchCalibrationReport,
    BatchCalibrationResult,
    BatchMetricConfig,
    build_batch_report,
    create_batch_plan,
    estimate_batch_time,
    record_batch_result,
)


# ---------------------------------------------------------------------------
# create_batch_plan
# ---------------------------------------------------------------------------


def test_create_batch_plan_sorts_by_priority_descending():
    configs = [
        BatchMetricConfig(metric_id="low", priority=1),
        BatchMetricConfig(metric_id="high", priority=10),
        BatchMetricConfig(metric_id="mid", priority=5),
    ]
    plan = create_batch_plan(configs)
    assert [m.metric_id for m in plan] == ["high", "mid", "low"]


def test_create_batch_plan_stable_order_for_equal_priority():
    configs = [
        BatchMetricConfig(metric_id="a", priority=1),
        BatchMetricConfig(metric_id="b", priority=1),
        BatchMetricConfig(metric_id="c", priority=1),
    ]
    plan = create_batch_plan(configs)
    # stable sort preserves insertion order
    assert [m.metric_id for m in plan] == ["a", "b", "c"]


def test_create_batch_plan_empty():
    plan = create_batch_plan([])
    assert plan == []


def test_create_batch_plan_single():
    configs = [BatchMetricConfig(metric_id="only")]
    plan = create_batch_plan(configs)
    assert len(plan) == 1
    assert plan[0].metric_id == "only"


# ---------------------------------------------------------------------------
# record_batch_result
# ---------------------------------------------------------------------------


def test_record_batch_result_success():
    r = record_batch_result("m1", success=True, before=0.5, after=0.8)
    assert r.metric_id == "m1"
    assert r.success is True
    assert r.alignment_before == 0.5
    assert r.alignment_after == 0.8
    assert abs(r.improvement - 0.3) < 1e-9
    assert r.error_message == ""


def test_record_batch_result_failure():
    r = record_batch_result("m1", success=False, before=0.5, after=0.3, error="timeout")
    assert r.success is False
    assert r.improvement == 0.0
    assert r.error_message == "timeout"


def test_record_batch_result_failure_no_improvement():
    r = record_batch_result("m1", success=False, before=0.2, after=0.9)
    assert r.improvement == 0.0


def test_record_batch_result_defaults():
    r = record_batch_result("m1", success=True)
    assert r.alignment_before == 0.0
    assert r.alignment_after == 0.0
    assert r.improvement == 0.0


def test_record_batch_result_negative_improvement():
    r = record_batch_result("m1", success=True, before=0.8, after=0.6)
    assert r.improvement < 0


# ---------------------------------------------------------------------------
# build_batch_report
# ---------------------------------------------------------------------------


def test_build_batch_report_totals():
    results = [
        record_batch_result("m1", True, 0.3, 0.7),
        record_batch_result("m2", True, 0.5, 0.6),
        record_batch_result("m3", False, 0.4, 0.2, error="err"),
    ]
    report = build_batch_report(results)
    assert report.total_metrics == 3
    assert report.successful == 2
    assert report.failed == 1
    assert report.total_improvement == (0.7 - 0.3) + (0.6 - 0.5) + 0.0


def test_build_batch_report_all_success():
    results = [
        record_batch_result("m1", True, 0.1, 0.9),
        record_batch_result("m2", True, 0.2, 0.8),
    ]
    report = build_batch_report(results)
    assert report.successful == 2
    assert report.failed == 0


def test_build_batch_report_all_failures():
    results = [
        record_batch_result("m1", False, error="e1"),
        record_batch_result("m2", False, error="e2"),
    ]
    report = build_batch_report(results)
    assert report.successful == 0
    assert report.failed == 2
    assert report.total_improvement == 0.0


def test_build_batch_report_empty():
    report = build_batch_report([])
    assert report.total_metrics == 0
    assert report.successful == 0
    assert report.failed == 0
    assert report.total_improvement == 0.0


# ---------------------------------------------------------------------------
# estimate_batch_time
# ---------------------------------------------------------------------------


def test_estimate_batch_time_default():
    configs = [BatchMetricConfig(metric_id=f"m{i}") for i in range(4)]
    assert estimate_batch_time(configs) == 20.0


def test_estimate_batch_time_custom():
    configs = [BatchMetricConfig(metric_id="m1"), BatchMetricConfig(metric_id="m2")]
    assert estimate_batch_time(configs, avg_minutes_per_metric=10.0) == 20.0


def test_estimate_batch_time_empty():
    assert estimate_batch_time([]) == 0.0


# ---------------------------------------------------------------------------
# BatchCalibrationReport format_text
# ---------------------------------------------------------------------------


def test_format_text_contains_header():
    results = [record_batch_result("m1", True, 0.3, 0.7)]
    report = build_batch_report(results)
    text = report.format_text()
    assert "Batch Calibration Report" in text
    assert "1 metrics" in text


def test_format_text_contains_counts():
    results = [
        record_batch_result("m1", True, 0.3, 0.7),
        record_batch_result("m2", False, error="err"),
    ]
    report = build_batch_report(results)
    text = report.format_text()
    assert "Successful: 1" in text
    assert "Failed: 1" in text


def test_format_text_shows_per_metric():
    results = [record_batch_result("quality", True, 0.4, 0.9)]
    report = build_batch_report(results)
    text = report.format_text()
    assert "quality" in text
    assert "ok" in text


def test_format_text_shows_failure():
    results = [record_batch_result("m1", False, error="timeout")]
    report = build_batch_report(results)
    text = report.format_text()
    assert "FAILED" in text
    assert "timeout" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


def test_batch_metric_config_as_dict():
    c = BatchMetricConfig(metric_id="m1", optimizer="opro", priority=5)
    d = c.as_dict()
    assert d["metric_id"] == "m1"
    assert d["optimizer"] == "opro"
    assert d["priority"] == 5


def test_batch_metric_config_roundtrip():
    original = BatchMetricConfig(metric_id="m1", optimizer="textgrad", priority=3)
    d = original.as_dict()
    restored = BatchMetricConfig.from_dict(d)
    assert restored.metric_id == original.metric_id
    assert restored.optimizer == original.optimizer
    assert restored.priority == original.priority


def test_batch_metric_config_defaults():
    c = BatchMetricConfig(metric_id="m1")
    assert c.optimizer == "default"
    assert c.priority == 0


def test_batch_metric_config_roundtrip_defaults():
    c = BatchMetricConfig(metric_id="m1")
    d = c.as_dict()
    restored = BatchMetricConfig.from_dict(d)
    assert restored.optimizer == "default"
    assert restored.priority == 0


def test_batch_calibration_result_as_dict():
    r = record_batch_result("m1", True, 0.3, 0.8)
    d = r.as_dict()
    assert d["metric_id"] == "m1"
    assert d["success"] is True
    assert d["improvement"] == 0.5


def test_batch_calibration_report_roundtrip():
    results = [
        record_batch_result("m1", True, 0.3, 0.7),
        record_batch_result("m2", False, error="err"),
    ]
    original = build_batch_report(results)
    d = original.as_dict()
    restored = BatchCalibrationReport.from_dict(d)
    assert restored.total_metrics == original.total_metrics
    assert restored.successful == original.successful
    assert restored.failed == original.failed
    assert restored.total_improvement == original.total_improvement
    assert len(restored.results) == len(original.results)


def test_batch_calibration_report_roundtrip_empty():
    original = build_batch_report([])
    d = original.as_dict()
    restored = BatchCalibrationReport.from_dict(d)
    assert restored.total_metrics == 0
    assert restored.results == []
