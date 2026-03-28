from __future__ import annotations

import os

import pytest

from evalyn_sdk.storage.wal_monitoring import (
    WALConfig,
    WALReport,
    WALStatus,
    build_wal_report,
    check_wal_status,
    estimate_pending_writes,
    needs_checkpoint,
    suggest_wal_actions,
)


# --- WALStatus ---


class TestWALStatus:
    def test_defaults(self):
        s = WALStatus()
        assert s.wal_exists is False
        assert s.wal_size_bytes == 0
        assert s.shm_exists is False
        assert s.shm_size_bytes == 0
        assert s.estimated_pending_writes == 0

    def test_as_dict(self):
        s = WALStatus(wal_exists=True, wal_size_bytes=1024)
        d = s.as_dict()
        assert d["wal_exists"] is True
        assert d["wal_size_bytes"] == 1024

    def test_format_text_contains_fields(self):
        s = WALStatus(wal_exists=True, wal_size_bytes=500)
        text = s.format_text()
        assert "WAL Status" in text
        assert "500" in text
        assert "True" in text


# --- WALConfig ---


class TestWALConfig:
    def test_defaults(self):
        c = WALConfig()
        assert c.max_wal_size_bytes == 10_000_000
        assert c.checkpoint_threshold_bytes == 5_000_000
        assert c.monitor_interval_seconds == 60.0

    def test_as_dict(self):
        c = WALConfig(max_wal_size_bytes=999)
        d = c.as_dict()
        assert d["max_wal_size_bytes"] == 999

    def test_from_dict(self):
        d = {"max_wal_size_bytes": 123, "checkpoint_threshold_bytes": 456}
        c = WALConfig.from_dict(d)
        assert c.max_wal_size_bytes == 123
        assert c.checkpoint_threshold_bytes == 456
        assert c.monitor_interval_seconds == 60.0

    def test_from_dict_empty(self):
        c = WALConfig.from_dict({})
        assert c.max_wal_size_bytes == 10_000_000

    def test_roundtrip(self):
        original = WALConfig(max_wal_size_bytes=42, monitor_interval_seconds=5.0)
        restored = WALConfig.from_dict(original.as_dict())
        assert restored.max_wal_size_bytes == original.max_wal_size_bytes
        assert restored.monitor_interval_seconds == original.monitor_interval_seconds


# --- WALReport ---


class TestWALReport:
    def test_defaults(self):
        r = WALReport()
        assert r.needs_checkpoint is False
        assert r.recommendations == []

    def test_as_dict(self):
        r = WALReport(
            status=WALStatus(wal_exists=True),
            needs_checkpoint=True,
            recommendations=["checkpoint now"],
        )
        d = r.as_dict()
        assert d["needs_checkpoint"] is True
        assert d["status"]["wal_exists"] is True
        assert d["recommendations"] == ["checkpoint now"]

    def test_from_dict(self):
        d = {
            "status": {"wal_exists": True, "wal_size_bytes": 100},
            "needs_checkpoint": True,
            "recommendations": ["do it"],
        }
        r = WALReport.from_dict(d)
        assert r.status.wal_exists is True
        assert r.status.wal_size_bytes == 100
        assert r.needs_checkpoint is True
        assert r.recommendations == ["do it"]

    def test_from_dict_empty(self):
        r = WALReport.from_dict({})
        assert r.status.wal_exists is False
        assert r.needs_checkpoint is False

    def test_roundtrip(self):
        original = WALReport(
            status=WALStatus(wal_exists=True, wal_size_bytes=8192),
            needs_checkpoint=True,
            recommendations=["checkpoint"],
        )
        restored = WALReport.from_dict(original.as_dict())
        assert restored.needs_checkpoint == original.needs_checkpoint
        assert restored.status.wal_size_bytes == original.status.wal_size_bytes
        assert restored.recommendations == original.recommendations

    def test_format_text(self):
        r = WALReport(
            status=WALStatus(wal_exists=True, wal_size_bytes=2048),
            needs_checkpoint=True,
            recommendations=["Run checkpoint"],
        )
        text = r.format_text()
        assert "WAL Report" in text
        assert "Needs checkpoint: True" in text
        assert "Run checkpoint" in text

    def test_format_text_no_recommendations(self):
        r = WALReport()
        text = r.format_text()
        assert "Recommendations" not in text


# --- check_wal_status ---


class TestCheckWalStatus:
    def test_no_wal_file(self, tmp_path):
        db = str(tmp_path / "test.db")
        # Create only the DB file, no WAL
        with open(db, "w") as f:
            f.write("data")
        s = check_wal_status(db)
        assert s.wal_exists is False
        assert s.wal_size_bytes == 0
        assert s.shm_exists is False

    def test_with_wal_file(self, tmp_path):
        db = str(tmp_path / "test.db")
        wal = db + "-wal"
        with open(db, "w") as f:
            f.write("data")
        with open(wal, "wb") as f:
            f.write(b"x" * 8192)
        s = check_wal_status(db)
        assert s.wal_exists is True
        assert s.wal_size_bytes == 8192
        assert s.estimated_pending_writes == 2  # 8192 / 4096

    def test_with_wal_and_shm(self, tmp_path):
        db = str(tmp_path / "test.db")
        wal = db + "-wal"
        shm = db + "-shm"
        with open(db, "w") as f:
            f.write("data")
        with open(wal, "wb") as f:
            f.write(b"w" * 4096)
        with open(shm, "wb") as f:
            f.write(b"s" * 256)
        s = check_wal_status(db)
        assert s.wal_exists is True
        assert s.shm_exists is True
        assert s.shm_size_bytes == 256

    def test_missing_db_file(self, tmp_path):
        db = str(tmp_path / "nonexistent.db")
        s = check_wal_status(db)
        assert s.wal_exists is False
        assert s.shm_exists is False
        assert s.wal_size_bytes == 0


# --- needs_checkpoint ---


class TestNeedsCheckpoint:
    def test_above_threshold(self):
        status = WALStatus(wal_size_bytes=6_000_000)
        config = WALConfig(checkpoint_threshold_bytes=5_000_000)
        assert needs_checkpoint(status, config) is True

    def test_below_threshold(self):
        status = WALStatus(wal_size_bytes=1_000_000)
        config = WALConfig(checkpoint_threshold_bytes=5_000_000)
        assert needs_checkpoint(status, config) is False

    def test_at_threshold(self):
        status = WALStatus(wal_size_bytes=5_000_000)
        config = WALConfig(checkpoint_threshold_bytes=5_000_000)
        assert needs_checkpoint(status, config) is True

    def test_zero_wal(self):
        status = WALStatus(wal_size_bytes=0)
        config = WALConfig(checkpoint_threshold_bytes=5_000_000)
        assert needs_checkpoint(status, config) is False


# --- estimate_pending_writes ---


class TestEstimatePendingWrites:
    def test_basic(self):
        assert estimate_pending_writes(8192) == 2

    def test_custom_avg(self):
        assert estimate_pending_writes(10000, avg_write_bytes=1000) == 10

    def test_zero_wal(self):
        assert estimate_pending_writes(0) == 0

    def test_zero_avg_write(self):
        assert estimate_pending_writes(4096, avg_write_bytes=0) == 0

    def test_negative_wal(self):
        assert estimate_pending_writes(-100) == 0


# --- build_wal_report ---


class TestBuildWalReport:
    def test_no_wal(self, tmp_path):
        db = str(tmp_path / "test.db")
        with open(db, "w") as f:
            f.write("data")
        report = build_wal_report(db)
        assert report.needs_checkpoint is False
        assert report.status.wal_exists is False

    def test_with_large_wal(self, tmp_path):
        db = str(tmp_path / "test.db")
        wal = db + "-wal"
        with open(db, "w") as f:
            f.write("data")
        with open(wal, "wb") as f:
            f.write(b"x" * 6_000_000)
        report = build_wal_report(db)
        assert report.needs_checkpoint is True
        assert report.status.wal_exists is True
        assert len(report.recommendations) > 0

    def test_custom_config(self, tmp_path):
        db = str(tmp_path / "test.db")
        wal = db + "-wal"
        with open(db, "w") as f:
            f.write("data")
        with open(wal, "wb") as f:
            f.write(b"x" * 100)
        config = WALConfig(checkpoint_threshold_bytes=50)
        report = build_wal_report(db, config=config)
        assert report.needs_checkpoint is True


# --- suggest_wal_actions ---


class TestSuggestWalActions:
    def test_no_wal(self):
        report = WALReport(status=WALStatus(wal_exists=False))
        actions = suggest_wal_actions(report)
        assert any("not be in WAL mode" in a for a in actions)

    def test_healthy_wal(self):
        report = WALReport(
            status=WALStatus(wal_exists=True, wal_size_bytes=100),
            needs_checkpoint=False,
        )
        actions = suggest_wal_actions(report)
        assert any("healthy" in a.lower() for a in actions)

    def test_needs_checkpoint(self):
        report = WALReport(
            status=WALStatus(wal_exists=True, wal_size_bytes=6_000_000),
            needs_checkpoint=True,
        )
        actions = suggest_wal_actions(report)
        assert any("checkpoint" in a.lower() for a in actions)

    def test_large_wal(self):
        report = WALReport(
            status=WALStatus(
                wal_exists=True,
                wal_size_bytes=15_000_000,
                estimated_pending_writes=3000,
            ),
            needs_checkpoint=True,
        )
        actions = suggest_wal_actions(report)
        assert any("10 MB" in a for a in actions)
        assert any("pending writes" in a.lower() for a in actions)
