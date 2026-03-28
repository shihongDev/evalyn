"""Tests for calibration rollback."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.rollback import (
    CalibrationHistory,
    CalibrationSnapshot,
    RollbackDecision,
)


# -------------------------------------------------------------------
# CalibrationSnapshot
# -------------------------------------------------------------------

class TestCalibrationSnapshot:
    def test_as_dict(self):
        ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        snap = CalibrationSnapshot(
            metric_id="acc",
            prompt="Judge accuracy",
            alignment_score=0.85,
            timestamp=ts,
            version=1,
        )
        d = snap.as_dict()
        assert d["metric_id"] == "acc"
        assert d["prompt"] == "Judge accuracy"
        assert d["alignment_score"] == 0.85
        assert d["version"] == 1
        assert "2026" in d["timestamp"]

    def test_from_dict_roundtrip(self):
        ts = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        snap = CalibrationSnapshot(
            metric_id="f1",
            prompt="Evaluate F1",
            alignment_score=0.9,
            timestamp=ts,
            version=3,
        )
        restored = CalibrationSnapshot.from_dict(snap.as_dict())
        assert restored.metric_id == "f1"
        assert restored.prompt == "Evaluate F1"
        assert restored.alignment_score == 0.9
        assert restored.timestamp == ts
        assert restored.version == 3

    def test_from_dict_datetime_object(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        data = {
            "metric_id": "m",
            "prompt": "p",
            "alignment_score": 0.5,
            "timestamp": ts.isoformat(),
            "version": 1,
        }
        snap = CalibrationSnapshot.from_dict(data)
        assert snap.timestamp == ts


# -------------------------------------------------------------------
# RollbackDecision
# -------------------------------------------------------------------

class TestRollbackDecision:
    def test_fields(self):
        rd = RollbackDecision(
            metric_id="acc",
            should_rollback=True,
            current_score=0.70,
            previous_score=0.85,
            score_delta=-0.15,
            reason="Score dropped",
        )
        assert rd.metric_id == "acc"
        assert rd.should_rollback is True
        assert rd.current_score == 0.70
        assert rd.previous_score == 0.85
        assert rd.score_delta == -0.15

    def test_as_dict(self):
        rd = RollbackDecision(
            metric_id="x",
            should_rollback=False,
            current_score=0.9,
            previous_score=0.8,
            score_delta=0.1,
            reason="Improved",
        )
        d = rd.as_dict()
        assert d["should_rollback"] is False
        assert d["score_delta"] == 0.1
        assert d["reason"] == "Improved"


# -------------------------------------------------------------------
# CalibrationHistory - add_snapshot
# -------------------------------------------------------------------

class TestCalibrationHistoryAdd:
    def test_add_snapshot_creates_with_auto_version(self):
        h = CalibrationHistory()
        snap = h.add_snapshot("acc", "prompt v1", 0.80)
        assert snap.version == 1
        assert snap.metric_id == "acc"
        assert snap.prompt == "prompt v1"
        assert snap.alignment_score == 0.80

    def test_add_snapshot_increments_version(self):
        h = CalibrationHistory()
        s1 = h.add_snapshot("acc", "v1", 0.80)
        s2 = h.add_snapshot("acc", "v2", 0.85)
        s3 = h.add_snapshot("acc", "v3", 0.82)
        assert s1.version == 1
        assert s2.version == 2
        assert s3.version == 3

    def test_add_snapshot_sets_timestamp(self):
        h = CalibrationHistory()
        snap = h.add_snapshot("m", "p", 0.5)
        assert snap.timestamp is not None
        assert snap.timestamp.tzinfo is not None

    def test_add_snapshot_multiple_metrics(self):
        h = CalibrationHistory()
        h.add_snapshot("acc", "p1", 0.80)
        h.add_snapshot("f1", "p2", 0.75)
        h.add_snapshot("acc", "p3", 0.85)
        assert len(h.get_history("acc")) == 2
        assert len(h.get_history("f1")) == 1


# -------------------------------------------------------------------
# CalibrationHistory - get_history, get_latest, get_previous
# -------------------------------------------------------------------

class TestCalibrationHistoryGet:
    def test_get_history_returns_ordered(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.80)
        h.add_snapshot("m", "c", 0.75)
        history = h.get_history("m")
        assert len(history) == 3
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[2].version == 3

    def test_get_history_empty_metric(self):
        h = CalibrationHistory()
        assert h.get_history("nonexistent") == []

    def test_get_latest_returns_most_recent(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.80)
        latest = h.get_latest("m")
        assert latest is not None
        assert latest.version == 2
        assert latest.alignment_score == 0.80

    def test_get_latest_empty(self):
        h = CalibrationHistory()
        assert h.get_latest("m") is None

    def test_get_previous_returns_second_to_last(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.80)
        h.add_snapshot("m", "c", 0.75)
        prev = h.get_previous("m")
        assert prev is not None
        assert prev.version == 2
        assert prev.alignment_score == 0.80

    def test_get_previous_single_snapshot(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        assert h.get_previous("m") is None


# -------------------------------------------------------------------
# CalibrationHistory - rollback
# -------------------------------------------------------------------

class TestCalibrationHistoryRollback:
    def test_rollback_removes_latest(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.80)
        h.add_snapshot("m", "c", 0.60)
        result = h.rollback("m")
        assert result is not None
        assert result.version == 2
        assert len(h.get_history("m")) == 2

    def test_rollback_returns_new_latest(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.80)
        new_latest = h.rollback("m")
        assert new_latest is not None
        assert new_latest.prompt == "a"
        assert new_latest.alignment_score == 0.70

    def test_rollback_on_empty_returns_none(self):
        h = CalibrationHistory()
        assert h.rollback("m") is None

    def test_rollback_single_snapshot_returns_none(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        result = h.rollback("m")
        assert result is None
        assert len(h.get_history("m")) == 0

    def test_rollback_multiple_times(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.80)
        h.add_snapshot("m", "c", 0.60)
        h.rollback("m")
        h.rollback("m")
        assert len(h.get_history("m")) == 1
        assert h.get_latest("m").version == 1


# -------------------------------------------------------------------
# CalibrationHistory - check_rollback
# -------------------------------------------------------------------

class TestCalibrationHistoryCheckRollback:
    def test_recommends_when_degraded(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.85)
        h.add_snapshot("m", "b", 0.70)
        decision = h.check_rollback("m")
        assert decision.should_rollback is True
        assert decision.score_delta < 0
        assert "dropped" in decision.reason.lower() or "drop" in decision.reason.lower()

    def test_no_rollback_when_improved(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.70)
        h.add_snapshot("m", "b", 0.85)
        decision = h.check_rollback("m")
        assert decision.should_rollback is False
        assert decision.score_delta > 0

    def test_no_rollback_within_threshold(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.85)
        h.add_snapshot("m", "b", 0.82)  # dropped 0.03, threshold is 0.05
        decision = h.check_rollback("m")
        assert decision.should_rollback is False

    def test_custom_threshold(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.85)
        h.add_snapshot("m", "b", 0.82)  # dropped 0.03
        decision = h.check_rollback("m", degradation_threshold=0.02)
        assert decision.should_rollback is True

    def test_check_rollback_single_snapshot(self):
        h = CalibrationHistory()
        h.add_snapshot("m", "a", 0.80)
        decision = h.check_rollback("m")
        assert decision.should_rollback is False
        assert "not enough" in decision.reason.lower()

    def test_check_rollback_no_history(self):
        h = CalibrationHistory()
        decision = h.check_rollback("m")
        assert decision.should_rollback is False
        assert decision.current_score == 0.0
        assert decision.previous_score == 0.0


# -------------------------------------------------------------------
# CalibrationHistory - as_dict / from_dict
# -------------------------------------------------------------------

class TestCalibrationHistorySerialize:
    def test_as_dict(self):
        h = CalibrationHistory()
        h.add_snapshot("acc", "p1", 0.80)
        h.add_snapshot("acc", "p2", 0.85)
        h.add_snapshot("f1", "p3", 0.70)
        d = h.as_dict()
        assert "acc" in d
        assert "f1" in d
        assert len(d["acc"]) == 2
        assert len(d["f1"]) == 1

    def test_from_dict_roundtrip(self):
        h = CalibrationHistory()
        h.add_snapshot("acc", "p1", 0.80)
        h.add_snapshot("acc", "p2", 0.85)
        d = h.as_dict()
        restored = CalibrationHistory.from_dict(d)
        assert len(restored.get_history("acc")) == 2
        latest = restored.get_latest("acc")
        assert latest.version == 2
        assert latest.alignment_score == 0.85

    def test_from_dict_empty(self):
        restored = CalibrationHistory.from_dict({})
        assert restored.get_history("anything") == []
