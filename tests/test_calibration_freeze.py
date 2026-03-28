"""Tests for calibration.freeze module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalyn_sdk.calibration.freeze import (
    CalibrationFreezeManager,
    FreezeRecord,
    FreezeStatus,
)


# -- FreezeRecord -------------------------------------------------------------


class TestFreezeRecord:
    def test_fields(self):
        now = datetime.now(timezone.utc)
        rec = FreezeRecord(
            metric_id="m1", frozen_at=now, frozen_by="alice",
            reason="stable", prompt_hash="abc123",
        )
        assert rec.metric_id == "m1"
        assert rec.frozen_at == now
        assert rec.frozen_by == "alice"
        assert rec.reason == "stable"
        assert rec.prompt_hash == "abc123"

    def test_defaults(self):
        now = datetime.now(timezone.utc)
        rec = FreezeRecord(metric_id="m2", frozen_at=now)
        assert rec.frozen_by == ""
        assert rec.reason == ""
        assert rec.prompt_hash == ""

    def test_as_dict(self):
        now = datetime.now(timezone.utc)
        rec = FreezeRecord(
            metric_id="m1", frozen_at=now, frozen_by="bob", reason="locked",
        )
        d = rec.as_dict()
        assert d["metric_id"] == "m1"
        assert d["frozen_at"] == now.isoformat()
        assert d["frozen_by"] == "bob"
        assert d["reason"] == "locked"

    def test_from_dict(self):
        now = datetime.now(timezone.utc)
        data = {
            "metric_id": "m1",
            "frozen_at": now.isoformat(),
            "frozen_by": "carol",
            "reason": "production",
            "prompt_hash": "def456",
        }
        rec = FreezeRecord.from_dict(data)
        assert rec.metric_id == "m1"
        assert rec.frozen_by == "carol"
        assert rec.prompt_hash == "def456"

    def test_from_dict_datetime_object(self):
        now = datetime.now(timezone.utc)
        data = {
            "metric_id": "m1",
            "frozen_at": now,
        }
        rec = FreezeRecord.from_dict(data)
        assert rec.frozen_at == now

    def test_roundtrip(self):
        now = datetime.now(timezone.utc)
        rec = FreezeRecord(
            metric_id="m1", frozen_at=now, frozen_by="dave", reason="ship it",
        )
        restored = FreezeRecord.from_dict(rec.as_dict())
        assert restored.metric_id == rec.metric_id
        assert restored.frozen_by == rec.frozen_by
        assert restored.reason == rec.reason


# -- FreezeStatus -------------------------------------------------------------


class TestFreezeStatus:
    def test_not_frozen(self):
        status = FreezeStatus(is_frozen=False)
        d = status.as_dict()
        assert d["is_frozen"] is False
        assert d["record"] is None

    def test_frozen(self):
        now = datetime.now(timezone.utc)
        rec = FreezeRecord(metric_id="m1", frozen_at=now)
        status = FreezeStatus(is_frozen=True, record=rec)
        d = status.as_dict()
        assert d["is_frozen"] is True
        assert d["record"]["metric_id"] == "m1"


# -- CalibrationFreezeManager -------------------------------------------------


class TestCalibrationFreezeManager:
    def test_freeze_locks_metric(self):
        mgr = CalibrationFreezeManager()
        rec = mgr.freeze("m1", reason="stable")
        assert rec.metric_id == "m1"
        assert rec.reason == "stable"
        assert mgr.is_frozen("m1") is True

    def test_freeze_already_frozen_raises(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1")
        with pytest.raises(ValueError, match="already frozen"):
            mgr.freeze("m1")

    def test_unfreeze_unlocks(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1")
        result = mgr.unfreeze("m1")
        assert result is True
        assert mgr.is_frozen("m1") is False

    def test_unfreeze_not_frozen_returns_false(self):
        mgr = CalibrationFreezeManager()
        result = mgr.unfreeze("m1")
        assert result is False

    def test_is_frozen_true(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1")
        assert mgr.is_frozen("m1") is True

    def test_is_frozen_false(self):
        mgr = CalibrationFreezeManager()
        assert mgr.is_frozen("m1") is False

    def test_get_status_frozen(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1", reason="locked", frozen_by="alice")
        status = mgr.get_status("m1")
        assert status.is_frozen is True
        assert status.record is not None
        assert status.record.metric_id == "m1"
        assert status.record.reason == "locked"
        assert status.record.frozen_by == "alice"

    def test_get_status_not_frozen(self):
        mgr = CalibrationFreezeManager()
        status = mgr.get_status("m1")
        assert status.is_frozen is False
        assert status.record is None

    def test_list_frozen_returns_all(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1")
        mgr.freeze("m2")
        mgr.freeze("m3")
        frozen = mgr.list_frozen()
        ids = {r.metric_id for r in frozen}
        assert ids == {"m1", "m2", "m3"}

    def test_list_frozen_empty(self):
        mgr = CalibrationFreezeManager()
        assert mgr.list_frozen() == []

    def test_check_can_calibrate_when_frozen(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1", reason="production lock")
        can, reason = mgr.check_can_calibrate("m1")
        assert can is False
        assert "frozen" in reason
        assert "production lock" in reason

    def test_check_can_calibrate_when_not_frozen(self):
        mgr = CalibrationFreezeManager()
        can, reason = mgr.check_can_calibrate("m1")
        assert can is True
        assert reason == ""

    def test_check_can_calibrate_no_reason(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1")
        can, reason = mgr.check_can_calibrate("m1")
        assert can is False
        assert "frozen" in reason

    def test_freeze_with_frozen_by(self):
        mgr = CalibrationFreezeManager()
        rec = mgr.freeze("m1", frozen_by="system")
        assert rec.frozen_by == "system"

    def test_freeze_records_timestamp(self):
        mgr = CalibrationFreezeManager()
        before = datetime.now(timezone.utc)
        rec = mgr.freeze("m1")
        after = datetime.now(timezone.utc)
        assert before <= rec.frozen_at <= after

    def test_as_dict(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1", reason="locked")
        d = mgr.as_dict()
        assert "frozen" in d
        assert "m1" in d["frozen"]
        assert d["frozen"]["m1"]["reason"] == "locked"

    def test_from_dict(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1", reason="stable")
        mgr.freeze("m2", reason="production")
        d = mgr.as_dict()
        restored = CalibrationFreezeManager.from_dict(d)
        assert restored.is_frozen("m1") is True
        assert restored.is_frozen("m2") is True
        assert not restored.is_frozen("m3")

    def test_roundtrip(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1", reason="v1", frozen_by="alice")
        mgr.freeze("m2", reason="v2", frozen_by="bob")
        d = mgr.as_dict()
        restored = CalibrationFreezeManager.from_dict(d)
        assert len(restored.list_frozen()) == 2
        s1 = restored.get_status("m1")
        assert s1.record.reason == "v1"
        assert s1.record.frozen_by == "alice"

    def test_from_dict_empty(self):
        mgr = CalibrationFreezeManager.from_dict({})
        assert mgr.list_frozen() == []

    def test_unfreeze_then_refreeze(self):
        mgr = CalibrationFreezeManager()
        mgr.freeze("m1", reason="first")
        mgr.unfreeze("m1")
        rec = mgr.freeze("m1", reason="second")
        assert rec.reason == "second"
        assert mgr.is_frozen("m1") is True
