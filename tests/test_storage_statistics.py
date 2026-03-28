"""
Tests for evalyn_sdk.storage.statistics.

Run with: uv run pytest tests/test_storage_statistics.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.statistics import (
    GrowthRate,
    StorageReport,
    StorageSnapshot,
    TableStats,
    build_storage_report,
    compute_growth_rate,
    create_snapshot,
    format_size,
    render_storage_chart,
    suggest_recommendations,
)


# ---------------------------------------------------------------------------
# TableStats
# ---------------------------------------------------------------------------


class TestTableStats:
    def test_as_dict(self):
        ts = TableStats(table_name="calls", row_count=100, estimated_size_bytes=2048)
        d = ts.as_dict()
        assert d["table_name"] == "calls"
        assert d["row_count"] == 100
        assert d["estimated_size_bytes"] == 2048

    def test_defaults(self):
        ts = TableStats(table_name="spans")
        assert ts.row_count == 0
        assert ts.estimated_size_bytes == 0


# ---------------------------------------------------------------------------
# StorageSnapshot
# ---------------------------------------------------------------------------


class TestStorageSnapshot:
    def test_as_dict_from_dict_roundtrip(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=4096,
            total_rows=200,
            tables=[
                TableStats("calls", 100, 2048),
                TableStats("spans", 100, 2048),
            ],
        )
        d = snap.as_dict()
        restored = StorageSnapshot.from_dict(d)
        assert restored.timestamp == snap.timestamp
        assert restored.total_size_bytes == snap.total_size_bytes
        assert restored.total_rows == snap.total_rows
        assert len(restored.tables) == 2
        assert restored.tables[0].table_name == "calls"

    def test_from_dict_defaults(self):
        snap = StorageSnapshot.from_dict({"timestamp": "2026-01-01T00:00:00+00:00"})
        assert snap.total_size_bytes == 0
        assert snap.total_rows == 0
        assert snap.tables == []


# ---------------------------------------------------------------------------
# GrowthRate
# ---------------------------------------------------------------------------


class TestGrowthRate:
    def test_as_dict(self):
        gr = GrowthRate(
            period="weekly",
            size_growth_bytes=1024,
            row_growth=50,
            growth_pct=12.5,
        )
        d = gr.as_dict()
        assert d["period"] == "weekly"
        assert d["size_growth_bytes"] == 1024
        assert d["row_growth"] == 50
        assert d["growth_pct"] == 12.5

    def test_defaults(self):
        gr = GrowthRate()
        assert gr.period == "daily"
        assert gr.size_growth_bytes == 0
        assert gr.row_growth == 0
        assert gr.growth_pct == 0.0


# ---------------------------------------------------------------------------
# StorageReport
# ---------------------------------------------------------------------------


class TestStorageReport:
    def _make_report(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=1024 * 1024,
            total_rows=500,
            tables=[TableStats("calls", 500, 1024 * 1024)],
        )
        return StorageReport(
            current=snap,
            history=[snap],
            growth=GrowthRate(
                period="daily",
                size_growth_bytes=512,
                row_growth=10,
                growth_pct=5.0,
            ),
            largest_table="calls",
            recommendations=["Consider compaction"],
        )

    def test_as_dict_from_dict_roundtrip(self):
        report = self._make_report()
        d = report.as_dict()
        restored = StorageReport.from_dict(d)
        assert restored.current.total_size_bytes == report.current.total_size_bytes
        assert restored.largest_table == "calls"
        assert restored.growth is not None
        assert restored.growth.growth_pct == 5.0
        assert restored.recommendations == ["Consider compaction"]

    def test_from_dict_no_growth(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=100,
            total_rows=5,
        )
        d = {"current": snap.as_dict()}
        restored = StorageReport.from_dict(d)
        assert restored.growth is None

    def test_format_text_basic(self):
        report = self._make_report()
        text = report.format_text()
        assert "Storage Report" in text
        assert "1.0 MB" in text
        assert "500" in text
        assert "calls" in text
        assert "Recommendations:" in text

    def test_format_text_no_growth(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=100,
            total_rows=5,
        )
        report = StorageReport(current=snap)
        text = report.format_text()
        assert "Storage Report" in text
        assert "Growth" not in text


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    def test_basic(self):
        tables = [
            {"name": "calls", "row_count": 100, "size_bytes": 2048},
            {"name": "spans", "row_count": 50, "size_bytes": 1024},
        ]
        snap = create_snapshot(tables)
        assert snap.total_rows == 150
        assert snap.total_size_bytes == 3072
        assert len(snap.tables) == 2
        assert snap.tables[0].table_name == "calls"

    def test_explicit_total_size(self):
        tables = [{"name": "calls", "row_count": 10, "size_bytes": 100}]
        snap = create_snapshot(tables, total_size_bytes=9999)
        assert snap.total_size_bytes == 9999

    def test_empty_tables(self):
        snap = create_snapshot([])
        assert snap.total_rows == 0
        assert snap.total_size_bytes == 0
        assert snap.tables == []
        assert snap.timestamp  # should have a timestamp

    def test_missing_fields_in_table_dict(self):
        tables = [{"name": "calls"}]
        snap = create_snapshot(tables)
        assert snap.tables[0].row_count == 0
        assert snap.tables[0].estimated_size_bytes == 0


# ---------------------------------------------------------------------------
# compute_growth_rate
# ---------------------------------------------------------------------------


class TestComputeGrowthRate:
    def test_increasing(self):
        s1 = StorageSnapshot(
            timestamp="2026-03-01T00:00:00+00:00",
            total_size_bytes=1000,
            total_rows=100,
        )
        s2 = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=1500,
            total_rows=150,
        )
        gr = compute_growth_rate([s1, s2])
        assert gr is not None
        assert gr.size_growth_bytes == 500
        assert gr.row_growth == 50
        assert gr.growth_pct == 50.0

    def test_no_change(self):
        s = StorageSnapshot(
            timestamp="2026-03-01T00:00:00+00:00",
            total_size_bytes=1000,
            total_rows=100,
        )
        gr = compute_growth_rate([s, s])
        assert gr is not None
        assert gr.size_growth_bytes == 0
        assert gr.row_growth == 0
        assert gr.growth_pct == 0.0

    def test_single_snapshot_returns_none(self):
        s = StorageSnapshot(timestamp="2026-03-01T00:00:00+00:00")
        assert compute_growth_rate([s]) is None

    def test_empty_list_returns_none(self):
        assert compute_growth_rate([]) is None

    def test_zero_base_size(self):
        s1 = StorageSnapshot(
            timestamp="2026-03-01T00:00:00+00:00",
            total_size_bytes=0,
            total_rows=0,
        )
        s2 = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=100,
            total_rows=10,
        )
        gr = compute_growth_rate([s1, s2])
        assert gr is not None
        assert gr.growth_pct == 0.0  # cannot divide by zero
        assert gr.size_growth_bytes == 100


# ---------------------------------------------------------------------------
# build_storage_report
# ---------------------------------------------------------------------------


class TestBuildStorageReport:
    def test_full(self):
        s1 = StorageSnapshot(
            timestamp="2026-03-01T00:00:00+00:00",
            total_size_bytes=1000,
            total_rows=100,
            tables=[TableStats("calls", 100, 1000)],
        )
        s2 = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=2000,
            total_rows=200,
            tables=[
                TableStats("calls", 150, 1500),
                TableStats("spans", 50, 500),
            ],
        )
        report = build_storage_report([s1, s2])
        assert report.current is s2
        assert report.growth is not None
        assert report.growth.size_growth_bytes == 1000
        assert report.largest_table == "calls"

    def test_empty_snapshots(self):
        report = build_storage_report([])
        assert report.current.total_size_bytes == 0
        assert report.growth is None

    def test_single_snapshot(self):
        s = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=500,
            total_rows=50,
            tables=[TableStats("calls", 50, 500)],
        )
        report = build_storage_report([s])
        assert report.current is s
        assert report.growth is None
        assert report.largest_table == "calls"


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_bytes(self):
        assert format_size(0) == "0 B"
        assert format_size(512) == "512 B"
        assert format_size(1023) == "1023 B"

    def test_kilobytes(self):
        assert format_size(1024) == "1.0 KB"
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_size(2 * 1024 * 1024 * 1024) == "2.0 GB"

    def test_negative(self):
        assert format_size(-1024) == "-1.0 KB"


# ---------------------------------------------------------------------------
# suggest_recommendations
# ---------------------------------------------------------------------------


class TestSuggestRecommendations:
    def test_large_db(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=600 * 1024 * 1024,  # 600 MB
            total_rows=500,
        )
        report = StorageReport(current=snap)
        recs = suggest_recommendations(report)
        assert any("compaction" in r.lower() for r in recs)

    def test_very_large_db(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=1100 * 1024 * 1024,  # 1.1 GB
            total_rows=500,
        )
        report = StorageReport(current=snap)
        recs = suggest_recommendations(report)
        assert any("archival" in r.lower() for r in recs)

    def test_high_growth(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=1000,
            total_rows=10,
        )
        growth = GrowthRate(growth_pct=75.0)
        report = StorageReport(current=snap, growth=growth)
        recs = suggest_recommendations(report)
        assert any("growth" in r.lower() for r in recs)

    def test_large_row_count(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=1000,
            total_rows=200_000,
        )
        report = StorageReport(current=snap)
        recs = suggest_recommendations(report)
        assert any("row" in r.lower() for r in recs)

    def test_no_recommendations(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=1000,
            total_rows=10,
        )
        report = StorageReport(current=snap)
        recs = suggest_recommendations(report)
        assert recs == []


# ---------------------------------------------------------------------------
# render_storage_chart
# ---------------------------------------------------------------------------


class TestRenderStorageChart:
    def test_basic_output(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            total_size_bytes=3072,
            total_rows=150,
            tables=[
                TableStats("calls", 100, 2048),
                TableStats("spans", 50, 1024),
            ],
        )
        report = StorageReport(current=snap)
        chart = render_storage_chart(report)
        assert "calls" in chart
        assert "spans" in chart
        assert "#" in chart

    def test_empty_tables(self):
        snap = StorageSnapshot(timestamp="2026-03-28T00:00:00+00:00")
        report = StorageReport(current=snap)
        chart = render_storage_chart(report)
        assert chart == "(no tables)"

    def test_single_table(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            tables=[TableStats("calls", 100, 2048)],
        )
        report = StorageReport(current=snap)
        chart = render_storage_chart(report)
        assert "calls" in chart

    def test_all_zero_sizes(self):
        snap = StorageSnapshot(
            timestamp="2026-03-28T00:00:00+00:00",
            tables=[
                TableStats("calls", 10, 0),
                TableStats("spans", 5, 0),
            ],
        )
        report = StorageReport(current=snap)
        chart = render_storage_chart(report)
        assert "calls" in chart
        assert "0 B" in chart
