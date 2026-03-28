"""Tests for metrics.snapshot_testing module."""

from __future__ import annotations

from evalyn_sdk.metrics.snapshot_testing import (
    MetricSnapshot,
    SnapshotSuite,
    SnapshotTestResult,
    build_snapshot_report,
    create_snapshot,
    detect_behavior_change,
    update_snapshot,
    verify_snapshot,
)


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------


def test_create_snapshot_from_pairs():
    cases = [
        ({"input": "hello"}, 0.9),
        ({"input": "world"}, 0.7),
    ]
    snap = create_snapshot("helpfulness", cases)
    assert snap.metric_id == "helpfulness"
    assert snap.test_inputs == [{"input": "hello"}, {"input": "world"}]
    assert snap.expected_scores == [0.9, 0.7]
    assert snap.snapshot_hash != ""


def test_create_snapshot_deterministic_hash():
    cases = [({"input": "a"}, 1.0)]
    s1 = create_snapshot("m", cases)
    s2 = create_snapshot("m", cases)
    assert s1.snapshot_hash == s2.snapshot_hash


def test_create_snapshot_empty():
    snap = create_snapshot("m", [])
    assert snap.test_inputs == []
    assert snap.expected_scores == []
    assert snap.snapshot_hash != ""


# ---------------------------------------------------------------------------
# verify_snapshot
# ---------------------------------------------------------------------------


def test_verify_snapshot_all_pass():
    snap = create_snapshot("m", [({"i": "a"}, 0.5), ({"i": "b"}, 0.8)])
    result = verify_snapshot(snap, [0.5, 0.8])
    assert result.passed is True
    assert result.mismatches == []
    assert result.total_tests == 2


def test_verify_snapshot_within_tolerance():
    snap = create_snapshot("m", [({"i": "a"}, 0.5)])
    result = verify_snapshot(snap, [0.505], tolerance=0.01)
    assert result.passed is True


def test_verify_snapshot_some_fail():
    snap = create_snapshot("m", [({"i": "a"}, 0.5), ({"i": "b"}, 0.8)])
    result = verify_snapshot(snap, [0.5, 0.5])
    assert result.passed is False
    assert len(result.mismatches) == 1
    assert result.mismatches[0]["index"] == 1
    assert result.mismatches[0]["expected"] == 0.8
    assert result.mismatches[0]["actual"] == 0.5


def test_verify_snapshot_zero_tolerance():
    snap = create_snapshot("m", [({"i": "a"}, 0.5)])
    result = verify_snapshot(snap, [0.500001], tolerance=0.0)
    assert result.passed is False


def test_verify_snapshot_all_fail():
    snap = create_snapshot("m", [({"i": "a"}, 0.0), ({"i": "b"}, 1.0)])
    result = verify_snapshot(snap, [1.0, 0.0])
    assert result.passed is False
    assert len(result.mismatches) == 2


# ---------------------------------------------------------------------------
# update_snapshot
# ---------------------------------------------------------------------------


def test_update_snapshot_replaces_scores():
    snap = create_snapshot("m", [({"i": "a"}, 0.5), ({"i": "b"}, 0.8)])
    updated = update_snapshot(snap, [0.6, 0.9])
    assert updated.expected_scores == [0.6, 0.9]
    assert updated.test_inputs == snap.test_inputs
    assert updated.metric_id == snap.metric_id


def test_update_snapshot_changes_hash():
    snap = create_snapshot("m", [({"i": "a"}, 0.5)])
    updated = update_snapshot(snap, [0.9])
    assert updated.snapshot_hash != snap.snapshot_hash


def test_update_snapshot_preserves_inputs():
    inputs = [{"i": "x"}, {"i": "y"}]
    snap = create_snapshot("m", [(inputs[0], 0.1), (inputs[1], 0.2)])
    updated = update_snapshot(snap, [0.3, 0.4])
    assert updated.test_inputs == inputs


# ---------------------------------------------------------------------------
# detect_behavior_change
# ---------------------------------------------------------------------------


def test_detect_behavior_change_true():
    assert detect_behavior_change([0.5, 0.5], [0.5, 0.9], threshold=0.05) is True


def test_detect_behavior_change_false():
    assert detect_behavior_change([0.5, 0.5], [0.52, 0.48], threshold=0.05) is False


def test_detect_behavior_change_at_threshold():
    # delta exactly at threshold should not trigger (strictly greater required)
    # Use integer-friendly values to avoid floating-point rounding
    assert detect_behavior_change([0.0], [0.05], threshold=0.05) is False


def test_detect_behavior_change_empty():
    assert detect_behavior_change([], [], threshold=0.05) is False


def test_detect_behavior_change_zero_threshold():
    assert detect_behavior_change([0.5], [0.5000001], threshold=0.0) is True


# ---------------------------------------------------------------------------
# build_snapshot_report
# ---------------------------------------------------------------------------


def test_build_snapshot_report():
    r1 = SnapshotTestResult(metric_id="m1", passed=True, total_tests=2)
    r2 = SnapshotTestResult(metric_id="m2", passed=False, mismatches=[{"index": 0, "expected": 0.5, "actual": 0.9, "delta": 0.4}], total_tests=1)
    report = build_snapshot_report([r1, r2])
    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["all_passed"] is False


def test_build_snapshot_report_all_pass():
    r1 = SnapshotTestResult(metric_id="m1", passed=True, total_tests=3)
    report = build_snapshot_report([r1])
    assert report["all_passed"] is True
    assert report["passed"] == 1
    assert report["failed"] == 0


def test_build_snapshot_report_empty():
    report = build_snapshot_report([])
    assert report["total"] == 0
    assert report["passed"] == 0
    assert report["failed"] == 0
    assert report["all_passed"] is True


# ---------------------------------------------------------------------------
# SnapshotTestResult format_text
# ---------------------------------------------------------------------------


def test_snapshot_test_result_format_text_pass():
    r = SnapshotTestResult(metric_id="m1", passed=True, total_tests=2)
    text = r.format_text()
    assert "m1" in text
    assert "PASS" in text
    assert "2 tests" in text


def test_snapshot_test_result_format_text_fail():
    r = SnapshotTestResult(
        metric_id="m1",
        passed=False,
        mismatches=[{"index": 0, "expected": 0.5, "actual": 0.9, "delta": 0.4}],
        total_tests=1,
    )
    text = r.format_text()
    assert "FAIL" in text
    assert "index 0" in text
    assert "expected=0.5" in text


# ---------------------------------------------------------------------------
# MetricSnapshot as_dict / from_dict
# ---------------------------------------------------------------------------


def test_metric_snapshot_as_dict():
    snap = MetricSnapshot(metric_id="m1", test_inputs=[{"i": "a"}], expected_scores=[0.5], snapshot_hash="h")
    d = snap.as_dict()
    assert d["metric_id"] == "m1"
    assert d["test_inputs"] == [{"i": "a"}]
    assert d["expected_scores"] == [0.5]
    assert d["snapshot_hash"] == "h"


def test_metric_snapshot_from_dict():
    d = {"metric_id": "m1", "test_inputs": [{"i": "b"}], "expected_scores": [0.8]}
    snap = MetricSnapshot.from_dict(d)
    assert snap.metric_id == "m1"
    assert snap.expected_scores == [0.8]


def test_metric_snapshot_roundtrip():
    snap = MetricSnapshot(metric_id="m1", test_inputs=[{"i": "a"}], expected_scores=[0.5], snapshot_hash="h")
    snap2 = MetricSnapshot.from_dict(snap.as_dict())
    assert snap2.metric_id == snap.metric_id
    assert snap2.expected_scores == snap.expected_scores
    assert snap2.snapshot_hash == snap.snapshot_hash


# ---------------------------------------------------------------------------
# SnapshotSuite serialisation
# ---------------------------------------------------------------------------


def test_snapshot_suite_to_json_from_json():
    snap = create_snapshot("m1", [({"i": "a"}, 0.5)])
    suite = SnapshotSuite(snapshots=[snap])
    json_str = suite.to_json()
    suite2 = SnapshotSuite.from_json(json_str)
    assert len(suite2.snapshots) == 1
    assert suite2.snapshots[0].metric_id == "m1"
    assert suite2.snapshots[0].expected_scores == [0.5]


def test_snapshot_suite_as_dict_from_dict():
    snap = create_snapshot("m1", [({"i": "a"}, 0.9)])
    suite = SnapshotSuite(snapshots=[snap])
    d = suite.as_dict()
    suite2 = SnapshotSuite.from_dict(d)
    assert suite2.snapshots[0].snapshot_hash == snap.snapshot_hash


def test_snapshot_suite_empty():
    suite = SnapshotSuite()
    assert suite.as_dict() == {"snapshots": []}
    suite2 = SnapshotSuite.from_dict({"snapshots": []})
    assert suite2.snapshots == []


def test_snapshot_suite_from_dict_missing_key():
    suite = SnapshotSuite.from_dict({})
    assert suite.snapshots == []


# ---------------------------------------------------------------------------
# SnapshotTestResult as_dict roundtrip
# ---------------------------------------------------------------------------


def test_snapshot_test_result_as_dict():
    r = SnapshotTestResult(metric_id="m1", passed=False, mismatches=[{"index": 0, "expected": 0.5, "actual": 0.9, "delta": 0.4}], total_tests=1)
    d = r.as_dict()
    assert d["metric_id"] == "m1"
    assert d["passed"] is False
    assert d["total_tests"] == 1
    assert len(d["mismatches"]) == 1
