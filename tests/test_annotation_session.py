"""Tests for the annotation session persistence module."""

from __future__ import annotations

import json
import os

import pytest

from evalyn_sdk.annotation_session import (
    AnnotationRecord,
    AnnotationSession,
    SessionStats,
    compute_session_stats,
    format_session_progress,
)


# ---------------------------------------------------------------------------
# AnnotationRecord
# ---------------------------------------------------------------------------


class TestAnnotationRecord:
    def test_create_minimal(self):
        r = AnnotationRecord(
            item_id="i1", label=0.8, annotator_id="a1", timestamp="2026-01-01T00:00:00+00:00"
        )
        assert r.confidence == 1.0
        assert r.skipped is False

    def test_create_full(self):
        r = AnnotationRecord("i1", 0.5, "a1", "2026-01-01T00:00:00+00:00", 0.9, True)
        assert r.confidence == 0.9
        assert r.skipped is True

    def test_as_dict(self):
        r = AnnotationRecord("i1", 0.8, "a1", "2026-01-01T00:00:00+00:00", 0.7, False)
        d = r.as_dict()
        assert d["item_id"] == "i1"
        assert d["label"] == 0.8
        assert d["confidence"] == 0.7
        assert d["skipped"] is False

    def test_from_dict_minimal(self):
        r = AnnotationRecord.from_dict(
            {"item_id": "i1", "label": 1.0, "annotator_id": "a1", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        assert r.confidence == 1.0
        assert r.skipped is False

    def test_from_dict_full(self):
        r = AnnotationRecord.from_dict(
            {
                "item_id": "i1",
                "label": 0.3,
                "annotator_id": "a1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "confidence": 0.5,
                "skipped": True,
            }
        )
        assert r.label == 0.3
        assert r.skipped is True

    def test_roundtrip(self):
        orig = AnnotationRecord("x", 0.42, "ann1", "2026-03-28T12:00:00+00:00", 0.99, False)
        restored = AnnotationRecord.from_dict(orig.as_dict())
        assert restored.item_id == orig.item_id
        assert restored.label == orig.label
        assert restored.confidence == orig.confidence


# ---------------------------------------------------------------------------
# SessionStats
# ---------------------------------------------------------------------------


class TestSessionStats:
    def test_create(self):
        s = SessionStats(
            total_items=10, annotated_count=5, skipped_count=2,
            remaining_count=3, items_per_hour=60.0, elapsed_seconds=300.0,
        )
        assert s.agreement_rate == 0.0

    def test_as_dict(self):
        s = SessionStats(10, 5, 2, 3, 60.0, 300.0, 0.8)
        d = s.as_dict()
        assert d["total_items"] == 10
        assert d["agreement_rate"] == 0.8

    def test_from_dict_minimal(self):
        s = SessionStats.from_dict(
            {
                "total_items": 10,
                "annotated_count": 5,
                "skipped_count": 2,
                "remaining_count": 3,
                "items_per_hour": 60.0,
                "elapsed_seconds": 300.0,
            }
        )
        assert s.agreement_rate == 0.0

    def test_roundtrip(self):
        orig = SessionStats(20, 10, 3, 7, 120.0, 600.0, 0.95)
        restored = SessionStats.from_dict(orig.as_dict())
        assert restored.total_items == orig.total_items
        assert restored.agreement_rate == orig.agreement_rate


# ---------------------------------------------------------------------------
# AnnotationSession - basic operations
# ---------------------------------------------------------------------------


class TestAnnotationSessionBasic:
    def test_init(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"])
        assert s.session_id == "s1"
        assert s.annotator_id == "a1"
        assert len(s.item_ids) == 3

    def test_annotate(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.annotate("i1", 0.8)
        records = s.get_annotations()
        assert len(records) == 1
        assert records[0].item_id == "i1"
        assert records[0].label == 0.8
        assert records[0].skipped is False

    def test_annotate_with_confidence(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 0.5, confidence=0.7)
        assert s.get_annotations()[0].confidence == 0.7

    def test_skip(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.skip("i1")
        records = s.get_annotations()
        assert len(records) == 1
        assert records[0].skipped is True
        assert records[0].label == 0.0

    def test_annotator_id_recorded(self):
        s = AnnotationSession("s1", "ann42", ["i1"])
        s.annotate("i1", 1.0)
        assert s.get_annotations()[0].annotator_id == "ann42"


# ---------------------------------------------------------------------------
# AnnotationSession - undo
# ---------------------------------------------------------------------------


class TestAnnotationSessionUndo:
    def test_undo_last(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.annotate("i1", 0.8)
        s.annotate("i2", 0.3)
        undone = s.undo_last()
        assert undone is not None
        assert undone.item_id == "i2"
        assert len(s.get_annotations()) == 1

    def test_undo_empty(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        assert s.undo_last() is None

    def test_undo_then_reannotate(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 0.5)
        s.undo_last()
        s.annotate("i1", 0.9)
        assert s.get_annotations()[0].label == 0.9

    def test_undo_skip(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        s.skip("i1")
        undone = s.undo_last()
        assert undone is not None
        assert undone.skipped is True
        assert len(s.get_annotations()) == 0


# ---------------------------------------------------------------------------
# AnnotationSession - navigation
# ---------------------------------------------------------------------------


class TestAnnotationSessionNavigation:
    def test_get_next_item(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"])
        assert s.get_next_item() == "i1"

    def test_get_next_after_annotate(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"])
        s.annotate("i1", 0.8)
        assert s.get_next_item() == "i2"

    def test_get_next_after_skip(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.skip("i1")
        assert s.get_next_item() == "i2"

    def test_get_next_all_done(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 1.0)
        assert s.get_next_item() is None

    def test_get_next_respects_order(self):
        s = AnnotationSession("s1", "a1", ["i3", "i1", "i2"])
        s.annotate("i3", 0.5)
        assert s.get_next_item() == "i1"


# ---------------------------------------------------------------------------
# AnnotationSession - progress and completion
# ---------------------------------------------------------------------------


class TestAnnotationSessionProgress:
    def test_is_complete_false(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.annotate("i1", 0.8)
        assert s.is_complete() is False

    def test_is_complete_true(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.annotate("i1", 0.8)
        s.skip("i2")
        assert s.is_complete() is True

    def test_is_complete_empty_items(self):
        s = AnnotationSession("s1", "a1", [])
        assert s.is_complete() is True

    def test_get_progress_counts(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"])
        s.annotate("i1", 0.8)
        s.skip("i2")
        stats = s.get_progress()
        assert stats.total_items == 3
        assert stats.annotated_count == 1
        assert stats.skipped_count == 1
        assert stats.remaining_count == 1

    def test_get_progress_elapsed(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        stats = s.get_progress()
        assert stats.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# AnnotationSession - save and load
# ---------------------------------------------------------------------------


class TestAnnotationSessionPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "session.json")
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"], session_path=path)
        s.annotate("i1", 0.8)
        s.skip("i2")
        s.save()

        loaded = AnnotationSession.load(path)
        assert loaded.session_id == "s1"
        assert loaded.annotator_id == "a1"
        assert loaded.item_ids == ["i1", "i2", "i3"]
        records = loaded.get_annotations()
        assert len(records) == 2
        assert records[0].item_id == "i1"
        assert records[1].skipped is True

    def test_save_with_explicit_path(self, tmp_path):
        path = str(tmp_path / "explicit.json")
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 0.5)
        s.save(path)
        assert os.path.exists(path)

    def test_save_no_path_raises(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        with pytest.raises(ValueError):
            s.save()

    def test_load_preserves_next_item(self, tmp_path):
        path = str(tmp_path / "session.json")
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"])
        s.annotate("i1", 0.8)
        s.save(path)

        loaded = AnnotationSession.load(path)
        assert loaded.get_next_item() == "i2"

    def test_load_preserves_completion(self, tmp_path):
        path = str(tmp_path / "session.json")
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 0.8)
        s.save(path)

        loaded = AnnotationSession.load(path)
        assert loaded.is_complete() is True

    def test_save_produces_valid_json(self, tmp_path):
        path = str(tmp_path / "session.json")
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 0.5)
        s.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["session_id"] == "s1"
        assert len(data["records"]) == 1

    def test_roundtrip_empty_session(self, tmp_path):
        path = str(tmp_path / "empty.json")
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        s.save(path)
        loaded = AnnotationSession.load(path)
        assert loaded.get_annotations() == []
        assert loaded.get_next_item() == "i1"


# ---------------------------------------------------------------------------
# compute_session_stats
# ---------------------------------------------------------------------------


class TestComputeSessionStats:
    def test_basic(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2", "i3"])
        s.annotate("i1", 0.8)
        s.skip("i2")
        stats = compute_session_stats(s)
        assert stats.total_items == 3
        assert stats.annotated_count == 1
        assert stats.skipped_count == 1
        assert stats.remaining_count == 1

    def test_empty_session(self):
        s = AnnotationSession("s1", "a1", ["i1", "i2"])
        stats = compute_session_stats(s)
        assert stats.annotated_count == 0
        assert stats.skipped_count == 0
        assert stats.remaining_count == 2

    def test_all_complete(self):
        s = AnnotationSession("s1", "a1", ["i1"])
        s.annotate("i1", 1.0)
        stats = compute_session_stats(s)
        assert stats.remaining_count == 0

    def test_elapsed_nonnegative(self):
        s = AnnotationSession("s1", "a1", [])
        stats = compute_session_stats(s)
        assert stats.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# format_session_progress
# ---------------------------------------------------------------------------


class TestFormatSessionProgress:
    def test_basic_output(self):
        stats = SessionStats(10, 5, 2, 3, 60.0, 300.0)
        text = format_session_progress(stats)
        assert "70.0%" in text
        assert "Annotated: 5" in text
        assert "Skipped:   2" in text
        assert "Remaining: 3" in text

    def test_zero_total(self):
        stats = SessionStats(0, 0, 0, 0, 0.0, 0.0)
        text = format_session_progress(stats)
        assert "0.0%" in text

    def test_complete(self):
        stats = SessionStats(5, 5, 0, 0, 100.0, 180.0)
        text = format_session_progress(stats)
        assert "100.0%" in text

    def test_agreement_rate_shown(self):
        stats = SessionStats(10, 5, 0, 5, 50.0, 360.0, agreement_rate=0.85)
        text = format_session_progress(stats)
        assert "Agreement: 85" in text

    def test_agreement_rate_hidden_when_zero(self):
        stats = SessionStats(10, 5, 0, 5, 50.0, 360.0, agreement_rate=0.0)
        text = format_session_progress(stats)
        assert "Agreement" not in text

    def test_progress_bar_present(self):
        stats = SessionStats(10, 5, 0, 5, 50.0, 360.0)
        text = format_session_progress(stats)
        assert "[" in text
        assert "]" in text
        assert "#" in text
