"""Tests for calibration memory."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.memory import CalibrationAttempt, CalibrationMemoryStore


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_attempt(
    metric_id: str = "acc",
    optimizer: str = "opro",
    score: float = 0.8,
    success: bool = True,
    failure_reason: str = "",
) -> CalibrationAttempt:
    return CalibrationAttempt(
        metric_id=metric_id,
        optimizer=optimizer,
        prompt_summary="test prompt",
        alignment_score=score,
        success=success,
        timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
        failure_reason=failure_reason,
    )


# -------------------------------------------------------------------
# CalibrationAttempt
# -------------------------------------------------------------------

class TestCalibrationAttempt:
    def test_as_dict(self):
        a = _make_attempt()
        d = a.as_dict()
        assert d["metric_id"] == "acc"
        assert d["optimizer"] == "opro"
        assert d["alignment_score"] == 0.8
        assert d["success"] is True
        assert "2026" in d["timestamp"]
        assert d["failure_reason"] == ""

    def test_as_dict_with_failure_reason(self):
        a = _make_attempt(success=False, failure_reason="low score")
        d = a.as_dict()
        assert d["failure_reason"] == "low score"
        assert d["success"] is False

    def test_from_dict_roundtrip(self):
        a = _make_attempt(failure_reason="bad")
        restored = CalibrationAttempt.from_dict(a.as_dict())
        assert restored.metric_id == a.metric_id
        assert restored.optimizer == a.optimizer
        assert restored.prompt_summary == a.prompt_summary
        assert restored.alignment_score == a.alignment_score
        assert restored.success == a.success
        assert restored.timestamp == a.timestamp
        assert restored.failure_reason == a.failure_reason

    def test_from_dict_missing_failure_reason(self):
        d = _make_attempt().as_dict()
        del d["failure_reason"]
        restored = CalibrationAttempt.from_dict(d)
        assert restored.failure_reason == ""

    def test_from_dict_datetime_string(self):
        d = _make_attempt().as_dict()
        assert isinstance(d["timestamp"], str)
        restored = CalibrationAttempt.from_dict(d)
        assert isinstance(restored.timestamp, datetime)


# -------------------------------------------------------------------
# CalibrationMemoryStore - record_attempt
# -------------------------------------------------------------------

class TestRecordAttempt:
    def test_record_adds_attempt(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt())
        assert store.count == 1

    def test_record_multiple(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="a"))
        store.record_attempt(_make_attempt(optimizer="b"))
        assert store.count == 2


# -------------------------------------------------------------------
# CalibrationMemoryStore - get_failed_approaches
# -------------------------------------------------------------------

class TestGetFailedApproaches:
    def test_returns_only_failed(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(success=True))
        store.record_attempt(_make_attempt(success=False, failure_reason="bad"))
        failed = store.get_failed_approaches("acc")
        assert len(failed) == 1
        assert failed[0].success is False

    def test_filters_by_metric(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(metric_id="acc", success=False))
        store.record_attempt(_make_attempt(metric_id="f1", success=False))
        assert len(store.get_failed_approaches("acc")) == 1
        assert len(store.get_failed_approaches("f1")) == 1

    def test_empty_store(self):
        store = CalibrationMemoryStore()
        assert store.get_failed_approaches("acc") == []


# -------------------------------------------------------------------
# CalibrationMemoryStore - get_successful_approaches
# -------------------------------------------------------------------

class TestGetSuccessfulApproaches:
    def test_returns_only_successful(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(success=True))
        store.record_attempt(_make_attempt(success=False))
        succ = store.get_successful_approaches("acc")
        assert len(succ) == 1
        assert succ[0].success is True

    def test_filters_by_metric(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(metric_id="acc", success=True))
        store.record_attempt(_make_attempt(metric_id="f1", success=True))
        assert len(store.get_successful_approaches("acc")) == 1

    def test_empty_store(self):
        store = CalibrationMemoryStore()
        assert store.get_successful_approaches("acc") == []


# -------------------------------------------------------------------
# CalibrationMemoryStore - get_best_approach
# -------------------------------------------------------------------

class TestGetBestApproach:
    def test_picks_highest_score(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="a", score=0.7, success=True))
        store.record_attempt(_make_attempt(optimizer="b", score=0.9, success=True))
        store.record_attempt(_make_attempt(optimizer="c", score=0.8, success=True))
        best = store.get_best_approach("acc")
        assert best is not None
        assert best.optimizer == "b"
        assert best.alignment_score == 0.9

    def test_ignores_failed(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(score=0.95, success=False))
        store.record_attempt(_make_attempt(score=0.7, success=True))
        best = store.get_best_approach("acc")
        assert best is not None
        assert best.alignment_score == 0.7

    def test_no_successes_returns_none(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(success=False))
        assert store.get_best_approach("acc") is None

    def test_empty_store_returns_none(self):
        store = CalibrationMemoryStore()
        assert store.get_best_approach("acc") is None


# -------------------------------------------------------------------
# CalibrationMemoryStore - has_been_tried
# -------------------------------------------------------------------

class TestHasBeenTried:
    def test_true_when_tried(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="opro"))
        assert store.has_been_tried("acc", "opro") is True

    def test_false_when_not_tried(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="opro"))
        assert store.has_been_tried("acc", "ape") is False

    def test_false_different_metric(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(metric_id="acc", optimizer="opro"))
        assert store.has_been_tried("f1", "opro") is False


# -------------------------------------------------------------------
# CalibrationMemoryStore - suggest_next_optimizer
# -------------------------------------------------------------------

class TestSuggestNextOptimizer:
    def test_returns_untried(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="opro"))
        result = store.suggest_next_optimizer("acc", ["opro", "ape", "textgrad"])
        assert result == "ape"

    def test_all_tried_returns_none(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="opro"))
        store.record_attempt(_make_attempt(optimizer="ape"))
        result = store.suggest_next_optimizer("acc", ["opro", "ape"])
        assert result is None

    def test_empty_store_returns_first(self):
        store = CalibrationMemoryStore()
        result = store.suggest_next_optimizer("acc", ["opro", "ape"])
        assert result == "opro"

    def test_empty_optimizers_list(self):
        store = CalibrationMemoryStore()
        assert store.suggest_next_optimizer("acc", []) is None


# -------------------------------------------------------------------
# CalibrationMemoryStore - count property
# -------------------------------------------------------------------

class TestCount:
    def test_empty(self):
        assert CalibrationMemoryStore().count == 0

    def test_after_adds(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt())
        store.record_attempt(_make_attempt())
        assert store.count == 2


# -------------------------------------------------------------------
# CalibrationMemoryStore - serialization
# -------------------------------------------------------------------

class TestSerialization:
    def test_as_dict_from_dict_roundtrip(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="a", score=0.7))
        store.record_attempt(_make_attempt(optimizer="b", score=0.9, success=False))
        restored = CalibrationMemoryStore.from_dict(store.as_dict())
        assert restored.count == 2
        assert restored.attempts[0].optimizer == "a"
        assert restored.attempts[1].success is False

    def test_to_json_from_json_roundtrip(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(optimizer="x"))
        json_str = store.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "attempts" in parsed
        restored = CalibrationMemoryStore.from_json(json_str)
        assert restored.count == 1
        assert restored.attempts[0].optimizer == "x"

    def test_empty_store_roundtrip(self):
        store = CalibrationMemoryStore()
        restored = CalibrationMemoryStore.from_dict(store.as_dict())
        assert restored.count == 0

    def test_from_dict_missing_attempts(self):
        restored = CalibrationMemoryStore.from_dict({})
        assert restored.count == 0


# -------------------------------------------------------------------
# Edge cases - multiple metrics
# -------------------------------------------------------------------

class TestMultipleMetrics:
    def test_separate_metrics(self):
        store = CalibrationMemoryStore()
        store.record_attempt(_make_attempt(metric_id="acc", optimizer="opro", success=True, score=0.8))
        store.record_attempt(_make_attempt(metric_id="f1", optimizer="ape", success=True, score=0.9))
        store.record_attempt(_make_attempt(metric_id="acc", optimizer="ape", success=False))
        assert len(store.get_successful_approaches("acc")) == 1
        assert len(store.get_successful_approaches("f1")) == 1
        assert len(store.get_failed_approaches("acc")) == 1
        assert len(store.get_failed_approaches("f1")) == 0
        best_acc = store.get_best_approach("acc")
        best_f1 = store.get_best_approach("f1")
        assert best_acc is not None and best_acc.alignment_score == 0.8
        assert best_f1 is not None and best_f1.alignment_score == 0.9
