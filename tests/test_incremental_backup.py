"""
Tests for evalyn_sdk.storage.incremental_backup.

Run with: uv run pytest tests/test_incremental_backup.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from evalyn_sdk.storage.incremental_backup import (
    BackupConfig,
    BackupRecord,
    BackupReport,
    build_backup_report,
    cleanup_old_backups,
    create_backup_path,
    generate_backup_id,
    list_backups,
    plan_backup,
    should_backup,
)


# ---------------------------------------------------------------------------
# generate_backup_id
# ---------------------------------------------------------------------------


class TestGenerateBackupId:
    def test_format_starts_with_backup(self):
        bid = generate_backup_id()
        assert bid.startswith("backup_")

    def test_format_has_date_and_time(self):
        bid = generate_backup_id()
        # backup_YYYYMMDD_HHMMSS
        parts = bid.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS

    def test_id_is_string(self):
        bid = generate_backup_id()
        assert isinstance(bid, str)


# ---------------------------------------------------------------------------
# create_backup_path
# ---------------------------------------------------------------------------


class TestCreateBackupPath:
    def test_includes_backup_dir(self):
        config = BackupConfig(backup_dir="/tmp/backups", compress=False)
        path = create_backup_path(config, "backup_20260328_120000")
        assert path.startswith("/tmp/backups")

    def test_compressed_extension(self):
        config = BackupConfig(backup_dir="/tmp/backups", compress=True)
        path = create_backup_path(config, "backup_20260328_120000")
        assert path.endswith(".db.gz")

    def test_uncompressed_extension(self):
        config = BackupConfig(backup_dir="/tmp/backups", compress=False)
        path = create_backup_path(config, "backup_20260328_120000")
        assert path.endswith(".db")

    def test_includes_backup_id(self):
        config = BackupConfig(backup_dir="/tmp/backups", compress=False)
        path = create_backup_path(config, "backup_20260328_120000")
        assert "backup_20260328_120000" in path


# ---------------------------------------------------------------------------
# should_backup
# ---------------------------------------------------------------------------


class TestShouldBackup:
    def test_true_when_no_last_backup(self):
        config = BackupConfig(interval_minutes=60.0)
        assert should_backup(config, None) is True

    def test_true_after_interval(self):
        config = BackupConfig(interval_minutes=60.0)
        old_time = (
            datetime.now(timezone.utc) - timedelta(minutes=120)
        ).isoformat()
        assert should_backup(config, old_time) is True

    def test_false_too_soon(self):
        config = BackupConfig(interval_minutes=60.0)
        recent_time = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
        assert should_backup(config, recent_time) is False

    def test_true_on_invalid_timestamp(self):
        config = BackupConfig(interval_minutes=60.0)
        assert should_backup(config, "not-a-date") is True

    def test_zero_interval_always_true(self):
        config = BackupConfig(interval_minutes=0.0)
        recent_time = datetime.now(timezone.utc).isoformat()
        assert should_backup(config, recent_time) is True


# ---------------------------------------------------------------------------
# plan_backup
# ---------------------------------------------------------------------------


class TestPlanBackup:
    def test_creates_record(self):
        config = BackupConfig(
            source_path="/data/traces.db",
            backup_dir="/backups",
            compress=True,
        )
        record = plan_backup(config)
        assert isinstance(record, BackupRecord)

    def test_record_has_source_path(self):
        config = BackupConfig(source_path="/data/traces.db")
        record = plan_backup(config)
        assert record.source_path == "/data/traces.db"

    def test_record_has_backup_path(self):
        config = BackupConfig(backup_dir="/backups", compress=False)
        record = plan_backup(config)
        assert record.backup_path.startswith("/backups")

    def test_record_compressed_flag(self):
        config = BackupConfig(compress=True)
        record = plan_backup(config)
        assert record.compressed is True

    def test_record_has_timestamp(self):
        config = BackupConfig()
        record = plan_backup(config)
        assert record.timestamp != ""


# ---------------------------------------------------------------------------
# list_backups
# ---------------------------------------------------------------------------


class TestListBackups:
    def test_empty_for_missing_dir(self):
        result = list_backups("/nonexistent/dir/xyz")
        assert result == []

    def test_lists_backup_files(self, tmp_path):
        # Create some fake backup files
        (tmp_path / "backup_20260101_000000.db").write_text("data1")
        (tmp_path / "backup_20260102_000000.db.gz").write_text("data2")
        (tmp_path / "other_file.txt").write_text("ignore")
        records = list_backups(str(tmp_path))
        assert len(records) == 2

    def test_sorted_by_name(self, tmp_path):
        (tmp_path / "backup_20260102_000000.db").write_text("b")
        (tmp_path / "backup_20260101_000000.db").write_text("a")
        records = list_backups(str(tmp_path))
        assert records[0].backup_id == "backup_20260101_000000"
        assert records[1].backup_id == "backup_20260102_000000"

    def test_detects_compressed(self, tmp_path):
        (tmp_path / "backup_20260101_000000.db.gz").write_text("data")
        records = list_backups(str(tmp_path))
        assert records[0].compressed is True

    def test_detects_uncompressed(self, tmp_path):
        (tmp_path / "backup_20260101_000000.db").write_text("data")
        records = list_backups(str(tmp_path))
        assert records[0].compressed is False


# ---------------------------------------------------------------------------
# cleanup_old_backups
# ---------------------------------------------------------------------------


class TestCleanupOldBackups:
    def test_keeps_max(self):
        records = [
            BackupRecord(backup_id=f"backup_{i}") for i in range(10)
        ]
        kept = cleanup_old_backups(records, max_backups=3)
        assert len(kept) == 3

    def test_keeps_newest(self):
        records = [
            BackupRecord(backup_id=f"backup_{i}") for i in range(5)
        ]
        kept = cleanup_old_backups(records, max_backups=2)
        assert kept[0].backup_id == "backup_3"
        assert kept[1].backup_id == "backup_4"

    def test_empty_records(self):
        kept = cleanup_old_backups([], max_backups=5)
        assert kept == []

    def test_fewer_than_max(self):
        records = [BackupRecord(backup_id="b1"), BackupRecord(backup_id="b2")]
        kept = cleanup_old_backups(records, max_backups=10)
        assert len(kept) == 2

    def test_zero_max_returns_empty(self):
        records = [BackupRecord(backup_id="b1")]
        kept = cleanup_old_backups(records, max_backups=0)
        assert kept == []

    def test_single_backup(self):
        records = [BackupRecord(backup_id="only")]
        kept = cleanup_old_backups(records, max_backups=1)
        assert len(kept) == 1
        assert kept[0].backup_id == "only"


# ---------------------------------------------------------------------------
# build_backup_report
# ---------------------------------------------------------------------------


class TestBuildBackupReport:
    def test_empty_records(self):
        report = build_backup_report([])
        assert report.total_backups == 0
        assert report.total_size_bytes == 0

    def test_aggregates_size(self):
        records = [
            BackupRecord(backup_id="b1", size_bytes=100, timestamp="2026-01-01T00:00:00"),
            BackupRecord(backup_id="b2", size_bytes=200, timestamp="2026-01-02T00:00:00"),
        ]
        report = build_backup_report(records)
        assert report.total_size_bytes == 300
        assert report.total_backups == 2

    def test_oldest_newest(self):
        records = [
            BackupRecord(backup_id="b1", timestamp="2026-01-01T00:00:00"),
            BackupRecord(backup_id="b2", timestamp="2026-03-01T00:00:00"),
        ]
        report = build_backup_report(records)
        assert report.oldest_backup == "2026-01-01T00:00:00"
        assert report.newest_backup == "2026-03-01T00:00:00"

    def test_single_record(self):
        records = [
            BackupRecord(backup_id="b1", size_bytes=50, timestamp="2026-02-01T00:00:00"),
        ]
        report = build_backup_report(records)
        assert report.total_backups == 1
        assert report.oldest_backup == "2026-02-01T00:00:00"
        assert report.newest_backup == "2026-02-01T00:00:00"


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_backup_config_roundtrip(self):
        original = BackupConfig(
            source_path="/data/db.sqlite",
            backup_dir="/backups",
            max_backups=10,
            interval_minutes=30.0,
            compress=False,
        )
        restored = BackupConfig.from_dict(original.as_dict())
        assert restored.source_path == original.source_path
        assert restored.backup_dir == original.backup_dir
        assert restored.max_backups == original.max_backups
        assert restored.interval_minutes == original.interval_minutes
        assert restored.compress == original.compress

    def test_backup_record_roundtrip(self):
        original = BackupRecord(
            backup_id="backup_20260328_120000",
            source_path="/data/db.sqlite",
            backup_path="/backups/backup_20260328_120000.db.gz",
            timestamp="2026-03-28T12:00:00",
            size_bytes=1024,
            compressed=True,
        )
        restored = BackupRecord.from_dict(original.as_dict())
        assert restored.backup_id == original.backup_id
        assert restored.size_bytes == original.size_bytes
        assert restored.compressed == original.compressed

    def test_backup_report_roundtrip(self):
        original = BackupReport(
            backups=[
                BackupRecord(backup_id="b1", timestamp="2026-01-01T00:00:00"),
            ],
            total_backups=1,
            total_size_bytes=512,
            oldest_backup="2026-01-01T00:00:00",
            newest_backup="2026-01-01T00:00:00",
        )
        restored = BackupReport.from_dict(original.as_dict())
        assert restored.total_backups == original.total_backups
        assert len(restored.backups) == 1


# ---------------------------------------------------------------------------
# BackupReport.format_text
# ---------------------------------------------------------------------------


class TestBackupReportFormatText:
    def test_contains_header(self):
        report = BackupReport()
        text = report.format_text()
        assert "Backup Report" in text

    def test_contains_counts(self):
        report = BackupReport(total_backups=3, total_size_bytes=1024)
        text = report.format_text()
        assert "3" in text
        assert "1024" in text

    def test_lists_individual_backups(self):
        report = BackupReport(
            backups=[
                BackupRecord(backup_id="b1", size_bytes=100),
                BackupRecord(backup_id="b2", size_bytes=200),
            ]
        )
        text = report.format_text()
        assert "b1" in text
        assert "b2" in text
