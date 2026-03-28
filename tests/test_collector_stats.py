"""Tests for trace.collector_stats module."""

from __future__ import annotations

import pytest

from evalyn_sdk.trace.collector_stats import CollectorStats, SpanCollector


# ---------------------------------------------------------------------------
# CollectorStats
# ---------------------------------------------------------------------------


class TestCollectorStats:
    def test_initial_zero(self):
        s = CollectorStats()
        assert s.collected == 0
        assert s.expected == 0
        assert s.orphaned == 0
        assert s.lost == 0
        assert s.duplicates == 0

    def test_loss_rate_zero_expected(self):
        s = CollectorStats()
        assert s.loss_rate == 0.0

    def test_loss_rate_calculation(self):
        s = CollectorStats(collected=8, expected=10, lost=2)
        assert s.loss_rate == pytest.approx(0.2)

    def test_has_warning_false(self):
        s = CollectorStats(collected=10, expected=10, lost=0)
        assert s.has_warning is False

    def test_has_warning_true(self):
        s = CollectorStats(collected=8, expected=10, lost=2, warning_threshold=0.05)
        assert s.has_warning is True

    def test_has_warning_at_threshold(self):
        # Exactly at threshold should not warn (strictly greater than)
        s = CollectorStats(collected=95, expected=100, lost=5, warning_threshold=0.05)
        assert s.has_warning is False

    def test_as_dict(self):
        s = CollectorStats(collected=5, expected=10, lost=3, duplicates=1)
        d = s.as_dict()
        assert d["collected"] == 5
        assert d["expected"] == 10
        assert d["lost"] == 3
        assert d["duplicates"] == 1
        assert "loss_rate" in d
        assert "has_warning" in d

    def test_from_dict_roundtrip(self):
        original = CollectorStats(
            collected=7, expected=10, orphaned=1, lost=2,
            duplicates=3, warning_threshold=0.1,
        )
        d = original.as_dict()
        restored = CollectorStats.from_dict(d)
        assert restored.collected == original.collected
        assert restored.expected == original.expected
        assert restored.orphaned == original.orphaned
        assert restored.lost == original.lost
        assert restored.duplicates == original.duplicates
        assert restored.warning_threshold == original.warning_threshold

    def test_from_dict_defaults(self):
        s = CollectorStats.from_dict({})
        assert s.collected == 0
        assert s.expected == 0

    def test_format_text(self):
        s = CollectorStats(collected=8, expected=10, orphaned=0, lost=2)
        text = s.format_text()
        assert "Collector Stats" in text
        assert "Collected:" in text
        assert "Lost:" in text
        assert "Loss rate:" in text

    def test_format_text_warning(self):
        s = CollectorStats(collected=5, expected=10, lost=5)
        text = s.format_text()
        assert "WARNING" in text


# ---------------------------------------------------------------------------
# SpanCollector
# ---------------------------------------------------------------------------


class TestSpanCollector:
    def test_record_collected(self):
        c = SpanCollector()
        c.record_collected("s1")
        stats = c.get_stats()
        assert stats.collected == 1

    def test_record_collected_multiple(self):
        c = SpanCollector()
        c.record_collected("s1")
        c.record_collected("s2")
        c.record_collected("s3")
        stats = c.get_stats()
        assert stats.collected == 3

    def test_record_expected(self):
        c = SpanCollector()
        c.record_expected(5)
        stats = c.get_stats()
        assert stats.expected == 5

    def test_record_expected_increments(self):
        c = SpanCollector()
        c.record_expected(3)
        c.record_expected(2)
        stats = c.get_stats()
        assert stats.expected == 5

    def test_duplicate_detection(self):
        c = SpanCollector()
        c.record_collected("s1")
        c.record_collected("s1")
        stats = c.get_stats()
        assert stats.collected == 1
        assert stats.duplicates == 1

    def test_get_stats_computes_lost(self):
        c = SpanCollector()
        c.record_expected(10)
        c.record_collected("s1")
        c.record_collected("s2")
        c.record_orphaned(3)
        stats = c.get_stats()
        assert stats.lost == 5  # 10 - 2 - 3

    def test_get_stats_lost_not_negative(self):
        c = SpanCollector()
        c.record_expected(2)
        c.record_collected("s1")
        c.record_collected("s2")
        c.record_collected("s3")
        stats = c.get_stats()
        assert stats.lost == 0

    def test_record_orphaned(self):
        c = SpanCollector()
        c.record_orphaned(2)
        stats = c.get_stats()
        assert stats.orphaned == 2

    def test_reset_clears_all(self):
        c = SpanCollector()
        c.record_expected(10)
        c.record_collected("s1")
        c.record_collected("s1")  # duplicate
        c.record_orphaned(2)
        c.reset()
        stats = c.get_stats()
        assert stats.collected == 0
        assert stats.expected == 0
        assert stats.orphaned == 0
        assert stats.lost == 0
        assert stats.duplicates == 0

    def test_reset_clears_seen_ids(self):
        c = SpanCollector()
        c.record_collected("s1")
        c.reset()
        c.record_collected("s1")  # should not be duplicate after reset
        stats = c.get_stats()
        assert stats.collected == 1
        assert stats.duplicates == 0

    def test_check_health_no_warnings(self):
        c = SpanCollector()
        c.record_expected(10)
        for i in range(10):
            c.record_collected(f"s{i}")
        warnings = c.check_health()
        assert warnings == []

    def test_check_health_high_loss(self):
        c = SpanCollector()
        c.record_expected(10)
        c.record_collected("s1")
        warnings = c.check_health()
        assert len(warnings) >= 1
        assert any("loss rate" in w.lower() for w in warnings)

    def test_check_health_duplicates(self):
        c = SpanCollector()
        c.record_collected("s1")
        c.record_collected("s1")
        warnings = c.check_health()
        assert any("duplicate" in w.lower() for w in warnings)

    def test_check_health_orphaned(self):
        c = SpanCollector()
        c.record_expected(10)
        for i in range(10):
            c.record_collected(f"s{i}")
        c.record_orphaned(3)
        warnings = c.check_health()
        assert any("orphan" in w.lower() for w in warnings)

    def test_loss_rate_via_stats(self):
        c = SpanCollector()
        c.record_expected(20)
        for i in range(16):
            c.record_collected(f"s{i}")
        stats = c.get_stats()
        assert stats.loss_rate == pytest.approx(0.2)

    def test_has_warning_via_stats(self):
        c = SpanCollector()
        c.record_expected(20)
        for i in range(16):
            c.record_collected(f"s{i}")
        stats = c.get_stats()
        assert stats.has_warning is True
