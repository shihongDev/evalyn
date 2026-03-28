"""Tests for calibration staleness detection."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.staleness_detection import (
    CalibrationMetadata,
    StalenessAlert,
    StalenessReport,
    check_all_staleness,
    check_staleness,
    compute_dataset_drift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)


def _fresh_meta(metric_id: str = "m1", days_ago: int = 5) -> CalibrationMetadata:
    return CalibrationMetadata(
        metric_id=metric_id,
        calibrated_at=NOW - timedelta(days=days_ago),
        dataset_hash="hash_abc",
        num_items=100,
        alignment_score=0.85,
    )


def _old_meta(metric_id: str = "m1", days_ago: int = 62) -> CalibrationMetadata:
    return CalibrationMetadata(
        metric_id=metric_id,
        calibrated_at=NOW - timedelta(days=days_ago),
        dataset_hash="hash_abc",
        num_items=100,
        alignment_score=0.85,
    )


# ---------------------------------------------------------------------------
# CalibrationMetadata
# ---------------------------------------------------------------------------


class TestCalibrationMetadata:
    def test_as_dict(self):
        meta = _fresh_meta()
        d = meta.as_dict()
        assert d["metric_id"] == "m1"
        assert d["num_items"] == 100
        assert d["alignment_score"] == 0.85
        assert isinstance(d["calibrated_at"], str)

    def test_from_dict(self):
        d = {
            "metric_id": "m2",
            "calibrated_at": "2025-01-01T00:00:00+00:00",
            "dataset_hash": "xyz",
            "num_items": 50,
            "alignment_score": 0.9,
        }
        meta = CalibrationMetadata.from_dict(d)
        assert meta.metric_id == "m2"
        assert meta.num_items == 50
        assert meta.alignment_score == 0.9

    def test_roundtrip(self):
        meta = _fresh_meta("roundtrip_metric")
        meta2 = CalibrationMetadata.from_dict(meta.as_dict())
        assert meta2.metric_id == meta.metric_id
        assert meta2.dataset_hash == meta.dataset_hash
        assert meta2.num_items == meta.num_items
        assert meta2.alignment_score == meta.alignment_score

    def test_from_dict_datetime_object(self):
        """from_dict accepts a datetime object directly."""
        dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        d = {
            "metric_id": "m3",
            "calibrated_at": dt,
            "dataset_hash": "h",
            "num_items": 10,
            "alignment_score": 0.7,
        }
        meta = CalibrationMetadata.from_dict(d)
        assert meta.calibrated_at == dt


# ---------------------------------------------------------------------------
# StalenessAlert
# ---------------------------------------------------------------------------


class TestStalenessAlert:
    def test_as_dict(self):
        alert = StalenessAlert(
            metric_id="m1",
            reason="too old",
            severity="warning",
            days_since_calibration=45,
            dataset_drift=0.1,
        )
        d = alert.as_dict()
        assert d["metric_id"] == "m1"
        assert d["severity"] == "warning"
        assert d["days_since_calibration"] == 45

    def test_defaults(self):
        alert = StalenessAlert(metric_id="m1", reason="test", severity="info")
        assert alert.days_since_calibration == 0
        assert alert.dataset_drift == 0.0


# ---------------------------------------------------------------------------
# StalenessReport
# ---------------------------------------------------------------------------


class TestStalenessReport:
    def test_as_dict(self):
        report = StalenessReport(
            alerts=[StalenessAlert(metric_id="m1", reason="old", severity="warning")],
            stale_metrics=["m1"],
            healthy_metrics=["m2"],
        )
        d = report.as_dict()
        assert len(d["alerts"]) == 1
        assert d["stale_metrics"] == ["m1"]
        assert d["healthy_metrics"] == ["m2"]

    def test_from_dict(self):
        d = {
            "alerts": [
                {
                    "metric_id": "m1",
                    "reason": "drift",
                    "severity": "critical",
                    "days_since_calibration": 10,
                    "dataset_drift": 0.5,
                }
            ],
            "stale_metrics": ["m1"],
            "healthy_metrics": [],
        }
        report = StalenessReport.from_dict(d)
        assert len(report.alerts) == 1
        assert report.alerts[0].severity == "critical"
        assert report.stale_metrics == ["m1"]

    def test_roundtrip(self):
        report = StalenessReport(
            alerts=[
                StalenessAlert(
                    metric_id="x", reason="aged", severity="warning",
                    days_since_calibration=40, dataset_drift=0.05,
                )
            ],
            stale_metrics=["x"],
            healthy_metrics=["y"],
        )
        report2 = StalenessReport.from_dict(report.as_dict())
        assert report2.stale_metrics == report.stale_metrics
        assert report2.healthy_metrics == report.healthy_metrics
        assert len(report2.alerts) == len(report.alerts)

    def test_format_text_healthy(self):
        report = StalenessReport(
            alerts=[], stale_metrics=[], healthy_metrics=["m1", "m2"]
        )
        text = report.format_text()
        assert "Healthy" in text
        assert "m1" in text
        assert "No alerts" in text

    def test_format_text_with_alerts(self):
        report = StalenessReport(
            alerts=[StalenessAlert(metric_id="m1", reason="old", severity="warning")],
            stale_metrics=["m1"],
            healthy_metrics=[],
        )
        text = report.format_text()
        assert "WARNING" in text
        assert "m1" in text

    def test_from_dict_empty(self):
        report = StalenessReport.from_dict({})
        assert report.alerts == []
        assert report.stale_metrics == []
        assert report.healthy_metrics == []


# ---------------------------------------------------------------------------
# compute_dataset_drift
# ---------------------------------------------------------------------------


class TestComputeDatasetDrift:
    def test_same_hash_zero_drift(self):
        assert compute_dataset_drift("abc", "abc", 100, 200) == 0.0

    def test_different_hash_item_change(self):
        drift = compute_dataset_drift("abc", "xyz", 100, 150)
        assert abs(drift - 50 / 150) < 1e-9

    def test_different_hash_same_count(self):
        drift = compute_dataset_drift("abc", "xyz", 100, 100)
        assert drift == 0.0

    def test_zero_items(self):
        drift = compute_dataset_drift("abc", "xyz", 0, 0)
        assert drift == 0.0

    def test_old_zero_new_nonzero(self):
        drift = compute_dataset_drift("abc", "xyz", 0, 10)
        assert abs(drift - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    def test_fresh_no_alerts(self):
        meta = _fresh_meta(days_ago=5)
        alerts = check_staleness(meta, "hash_abc", 100)
        assert alerts == []

    def test_old_calibration_warning(self):
        meta = _old_meta(days_ago=62)
        alerts = check_staleness(meta, "hash_abc", 100, max_age_days=30)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].days_since_calibration >= 60

    def test_drift_warning(self):
        meta = _fresh_meta(days_ago=5)
        # Different hash, big item change -> drift > 0.2
        alerts = check_staleness(meta, "hash_NEW", 200, drift_threshold=0.2)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].dataset_drift > 0.2

    def test_old_and_drifted_critical(self):
        meta = _old_meta(days_ago=62)
        alerts = check_staleness(
            meta, "hash_NEW", 200, max_age_days=30, drift_threshold=0.2
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_exact_boundary_no_alert(self):
        meta = _fresh_meta(days_ago=30)
        alerts = check_staleness(meta, "hash_abc", 100, max_age_days=30)
        assert alerts == []

    def test_one_day_over_boundary(self):
        meta = _fresh_meta(days_ago=32)
        alerts = check_staleness(meta, "hash_abc", 100, max_age_days=30)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"


# ---------------------------------------------------------------------------
# check_all_staleness
# ---------------------------------------------------------------------------


class TestCheckAllStaleness:
    def test_all_healthy(self):
        metas = [_fresh_meta("m1", days_ago=5), _fresh_meta("m2", days_ago=10)]
        datasets = {
            "m1": ("hash_abc", 100),
            "m2": ("hash_abc", 100),
        }
        report = check_all_staleness(metas, datasets, max_age_days=30)
        assert report.stale_metrics == []
        assert set(report.healthy_metrics) == {"m1", "m2"}
        assert report.alerts == []

    def test_mixed_stale_and_healthy(self):
        metas = [_fresh_meta("m1", days_ago=5), _old_meta("m2", days_ago=60)]
        datasets = {
            "m1": ("hash_abc", 100),
            "m2": ("hash_abc", 100),
        }
        report = check_all_staleness(metas, datasets, max_age_days=30)
        assert "m1" in report.healthy_metrics
        assert "m2" in report.stale_metrics
        assert len(report.alerts) >= 1

    def test_missing_dataset_defaults(self):
        """When metric not in current_datasets, uses original hash/items (no drift)."""
        metas = [_fresh_meta("m1", days_ago=5)]
        report = check_all_staleness(metas, {}, max_age_days=30)
        assert "m1" in report.healthy_metrics

    def test_empty_calibrations(self):
        report = check_all_staleness([], {})
        assert report.stale_metrics == []
        assert report.healthy_metrics == []
        assert report.alerts == []

    def test_all_stale(self):
        metas = [_old_meta("m1", days_ago=90), _old_meta("m2", days_ago=90)]
        datasets = {
            "m1": ("hash_abc", 100),
            "m2": ("hash_abc", 100),
        }
        report = check_all_staleness(metas, datasets, max_age_days=30)
        assert set(report.stale_metrics) == {"m1", "m2"}
        assert report.healthy_metrics == []
