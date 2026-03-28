"""
Tests for evalyn_sdk.storage.partitioning.

Run with: uv run pytest tests/test_storage_partitioning.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.partitioning import (
    Partition,
    PartitionConfig,
    PartitionReport,
    assign_to_partition,
    compute_partition_key,
    generate_partition_path,
    merge_partition_reports,
    plan_partitions,
)


# ---------------------------------------------------------------------------
# compute_partition_key
# ---------------------------------------------------------------------------


class TestComputePartitionKey:
    def test_daily(self):
        assert compute_partition_key("2026-03-28T10:30:00", "daily") == "2026-03-28"

    def test_weekly(self):
        key = compute_partition_key("2026-03-28T10:30:00", "weekly")
        assert key == "2026-W13"

    def test_monthly(self):
        assert compute_partition_key("2026-03-28T10:30:00", "monthly") == "2026-03"

    def test_yearly(self):
        assert compute_partition_key("2026-03-28T10:30:00", "yearly") == "2026"

    def test_default_is_monthly(self):
        assert compute_partition_key("2026-03-28T10:30:00") == "2026-03"

    def test_date_only_timestamp(self):
        assert compute_partition_key("2026-01-15", "daily") == "2026-01-15"

    def test_different_months(self):
        assert compute_partition_key("2026-01-01T00:00:00", "monthly") == "2026-01"
        assert compute_partition_key("2026-12-31T23:59:59", "monthly") == "2026-12"

    def test_unknown_period_defaults_monthly(self):
        key = compute_partition_key("2026-03-28T10:30:00", "unknown")
        assert key == "2026-03"

    def test_weekly_different_weeks(self):
        # Jan 1 2026 is a Thursday, week 1
        key1 = compute_partition_key("2026-01-01T00:00:00", "weekly")
        key2 = compute_partition_key("2026-01-12T00:00:00", "weekly")
        assert key1 != key2


# ---------------------------------------------------------------------------
# PartitionConfig
# ---------------------------------------------------------------------------


class TestPartitionConfig:
    def test_defaults(self):
        c = PartitionConfig()
        assert c.period == "monthly"
        assert c.base_dir == "."
        assert c.naming_pattern == "evalyn_{period}"

    def test_as_dict(self):
        c = PartitionConfig(period="daily", base_dir="/data", naming_pattern="part_{period}")
        d = c.as_dict()
        assert d["period"] == "daily"
        assert d["base_dir"] == "/data"
        assert d["naming_pattern"] == "part_{period}"

    def test_from_dict(self):
        d = {"period": "weekly", "base_dir": "/tmp", "naming_pattern": "db_{period}"}
        c = PartitionConfig.from_dict(d)
        assert c.period == "weekly"
        assert c.base_dir == "/tmp"
        assert c.naming_pattern == "db_{period}"

    def test_from_dict_defaults(self):
        c = PartitionConfig.from_dict({})
        assert c.period == "monthly"
        assert c.base_dir == "."
        assert c.naming_pattern == "evalyn_{period}"

    def test_roundtrip(self):
        original = PartitionConfig(period="yearly", base_dir="/archive", naming_pattern="arch_{period}")
        restored = PartitionConfig.from_dict(original.as_dict())
        assert restored.period == original.period
        assert restored.base_dir == original.base_dir
        assert restored.naming_pattern == original.naming_pattern


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


class TestPartition:
    def test_as_dict(self):
        p = Partition(
            partition_id="2026-03",
            period_label="2026-03",
            start_date="2026-03-01",
            end_date="2026-03-31",
            file_path="/data/evalyn_2026-03.db",
            row_count=100,
            size_bytes=4096,
        )
        d = p.as_dict()
        assert d["partition_id"] == "2026-03"
        assert d["period_label"] == "2026-03"
        assert d["start_date"] == "2026-03-01"
        assert d["end_date"] == "2026-03-31"
        assert d["file_path"] == "/data/evalyn_2026-03.db"
        assert d["row_count"] == 100
        assert d["size_bytes"] == 4096

    def test_from_dict(self):
        d = {
            "partition_id": "2026-03",
            "period_label": "2026-03",
            "start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "row_count": 50,
        }
        p = Partition.from_dict(d)
        assert p.partition_id == "2026-03"
        assert p.row_count == 50
        assert p.file_path == ""
        assert p.size_bytes == 0

    def test_from_dict_defaults(self):
        p = Partition.from_dict({})
        assert p.partition_id == ""
        assert p.row_count == 0
        assert p.size_bytes == 0

    def test_roundtrip(self):
        original = Partition(
            partition_id="2026-W13",
            period_label="2026-W13",
            start_date="2026-03-23",
            end_date="2026-03-29",
            file_path="/data/part.db",
            row_count=42,
            size_bytes=2048,
        )
        restored = Partition.from_dict(original.as_dict())
        assert restored.partition_id == original.partition_id
        assert restored.row_count == original.row_count
        assert restored.size_bytes == original.size_bytes


# ---------------------------------------------------------------------------
# PartitionReport
# ---------------------------------------------------------------------------


class TestPartitionReport:
    def test_as_dict(self):
        report = PartitionReport(
            partitions=[
                Partition("2026-03", "2026-03", "2026-03-01", "2026-03-31", row_count=10)
            ],
            total_partitions=1,
            total_rows=10,
            total_size_bytes=1024,
        )
        d = report.as_dict()
        assert len(d["partitions"]) == 1
        assert d["total_partitions"] == 1
        assert d["total_rows"] == 10
        assert d["total_size_bytes"] == 1024

    def test_from_dict_defaults(self):
        report = PartitionReport.from_dict({})
        assert report.partitions == []
        assert report.total_partitions == 0
        assert report.total_rows == 0

    def test_roundtrip(self):
        original = PartitionReport(
            partitions=[
                Partition("2026-01", "2026-01", "2026-01-01", "2026-01-31", row_count=5),
                Partition("2026-02", "2026-02", "2026-02-01", "2026-02-28", row_count=8),
            ],
            total_partitions=2,
            total_rows=13,
            total_size_bytes=2048,
        )
        restored = PartitionReport.from_dict(original.as_dict())
        assert restored.total_partitions == original.total_partitions
        assert restored.total_rows == original.total_rows
        assert len(restored.partitions) == len(original.partitions)

    def test_format_text(self):
        report = PartitionReport(
            partitions=[
                Partition("2026-03", "2026-03", "2026-03-01", "2026-03-31", "/data/p.db", 10, 512)
            ],
            total_partitions=1,
            total_rows=10,
            total_size_bytes=512,
        )
        text = report.format_text()
        assert "Partition Report" in text
        assert "Total partitions: 1" in text
        assert "Total rows: 10" in text
        assert "2026-03" in text
        assert "/data/p.db" in text

    def test_format_text_empty(self):
        report = PartitionReport()
        text = report.format_text()
        assert "Total partitions: 0" in text
        assert "Total rows: 0" in text


# ---------------------------------------------------------------------------
# generate_partition_path
# ---------------------------------------------------------------------------


class TestGeneratePartitionPath:
    def test_default_pattern(self):
        config = PartitionConfig(base_dir="/data")
        path = generate_partition_path(config, "2026-03")
        assert path.endswith("evalyn_2026-03.db")
        assert "/data/" in path or path.startswith("/data")

    def test_custom_pattern(self):
        config = PartitionConfig(base_dir="/archive", naming_pattern="db_{period}")
        path = generate_partition_path(config, "2026")
        assert "db_2026.db" in path

    def test_current_dir(self):
        config = PartitionConfig(base_dir=".")
        path = generate_partition_path(config, "2026-03")
        assert "evalyn_2026-03.db" in path


# ---------------------------------------------------------------------------
# assign_to_partition
# ---------------------------------------------------------------------------


class TestAssignToPartition:
    def test_basic(self):
        config = PartitionConfig(period="monthly")
        record = {"created_at": "2026-03-15T10:00:00", "data": "test"}
        key = assign_to_partition(record, config)
        assert key == "2026-03"

    def test_custom_field(self):
        config = PartitionConfig(period="daily")
        record = {"timestamp": "2026-03-28T12:00:00"}
        key = assign_to_partition(record, config, timestamp_field="timestamp")
        assert key == "2026-03-28"

    def test_missing_field(self):
        config = PartitionConfig(period="monthly")
        record = {"data": "no timestamp"}
        key = assign_to_partition(record, config)
        assert key == ""


# ---------------------------------------------------------------------------
# plan_partitions
# ---------------------------------------------------------------------------


class TestPlanPartitions:
    def test_groups_correctly(self):
        config = PartitionConfig(period="monthly")
        records = [
            {"created_at": "2026-01-10T10:00:00"},
            {"created_at": "2026-01-20T12:00:00"},
            {"created_at": "2026-02-05T08:00:00"},
        ]
        report = plan_partitions(records, config)
        assert report.total_partitions == 2
        assert report.total_rows == 3
        # First partition should be January
        assert report.partitions[0].partition_id == "2026-01"
        assert report.partitions[0].row_count == 2
        assert report.partitions[1].partition_id == "2026-02"
        assert report.partitions[1].row_count == 1

    def test_single_record(self):
        config = PartitionConfig(period="monthly")
        records = [{"created_at": "2026-03-15T10:00:00"}]
        report = plan_partitions(records, config)
        assert report.total_partitions == 1
        assert report.total_rows == 1

    def test_empty_records(self):
        config = PartitionConfig(period="monthly")
        report = plan_partitions([], config)
        assert report.total_partitions == 0
        assert report.total_rows == 0
        assert report.partitions == []

    def test_file_paths_generated(self):
        config = PartitionConfig(period="monthly", base_dir="/data")
        records = [{"created_at": "2026-03-15T10:00:00"}]
        report = plan_partitions(records, config)
        assert report.partitions[0].file_path != ""
        assert "2026-03" in report.partitions[0].file_path

    def test_date_range_within_partition(self):
        config = PartitionConfig(period="monthly")
        records = [
            {"created_at": "2026-03-01T00:00:00"},
            {"created_at": "2026-03-15T12:00:00"},
            {"created_at": "2026-03-31T23:59:59"},
        ]
        report = plan_partitions(records, config)
        p = report.partitions[0]
        assert p.start_date == "2026-03-01T00:00:00"
        assert p.end_date == "2026-03-31T23:59:59"

    def test_custom_timestamp_field(self):
        config = PartitionConfig(period="daily")
        records = [{"ts": "2026-03-28T10:00:00"}]
        report = plan_partitions(records, config, timestamp_field="ts")
        assert report.total_partitions == 1
        assert report.partitions[0].partition_id == "2026-03-28"


# ---------------------------------------------------------------------------
# merge_partition_reports
# ---------------------------------------------------------------------------


class TestMergePartitionReports:
    def test_merge_disjoint(self):
        r1 = PartitionReport(
            partitions=[Partition("2026-01", "2026-01", "2026-01-01", "2026-01-31", row_count=5)],
            total_partitions=1,
            total_rows=5,
        )
        r2 = PartitionReport(
            partitions=[Partition("2026-02", "2026-02", "2026-02-01", "2026-02-28", row_count=8)],
            total_partitions=1,
            total_rows=8,
        )
        merged = merge_partition_reports([r1, r2])
        assert merged.total_partitions == 2
        assert merged.total_rows == 13

    def test_merge_overlapping(self):
        r1 = PartitionReport(
            partitions=[Partition("2026-03", "2026-03", "2026-03-01", "2026-03-15", row_count=5, size_bytes=100)],
            total_partitions=1,
            total_rows=5,
        )
        r2 = PartitionReport(
            partitions=[Partition("2026-03", "2026-03", "2026-03-16", "2026-03-31", row_count=8, size_bytes=200)],
            total_partitions=1,
            total_rows=8,
        )
        merged = merge_partition_reports([r1, r2])
        assert merged.total_partitions == 1
        assert merged.total_rows == 13
        assert merged.total_size_bytes == 300
        # Date range should be extended
        assert merged.partitions[0].start_date == "2026-03-01"
        assert merged.partitions[0].end_date == "2026-03-31"

    def test_merge_empty_list(self):
        merged = merge_partition_reports([])
        assert merged.total_partitions == 0
        assert merged.total_rows == 0
        assert merged.partitions == []

    def test_merge_single_report(self):
        r = PartitionReport(
            partitions=[Partition("2026-01", "2026-01", "2026-01-01", "2026-01-31", row_count=3)],
            total_partitions=1,
            total_rows=3,
        )
        merged = merge_partition_reports([r])
        assert merged.total_partitions == 1
        assert merged.total_rows == 3

    def test_merge_preserves_size(self):
        r1 = PartitionReport(
            partitions=[Partition("2026-01", "2026-01", "2026-01-01", "2026-01-31", row_count=2, size_bytes=512)],
            total_partitions=1,
            total_rows=2,
            total_size_bytes=512,
        )
        r2 = PartitionReport(
            partitions=[Partition("2026-01", "2026-01", "2026-01-01", "2026-01-31", row_count=3, size_bytes=768)],
            total_partitions=1,
            total_rows=3,
            total_size_bytes=768,
        )
        merged = merge_partition_reports([r1, r2])
        assert merged.total_size_bytes == 1280
        assert merged.partitions[0].size_bytes == 1280
