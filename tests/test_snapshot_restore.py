"""Tests for the snapshot restore module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.storage.snapshot_restore import (
    Snapshot,
    SnapshotConfig,
    SnapshotReport,
    build_snapshot_report,
    cleanup_snapshots,
    compute_snapshot_hash,
    create_snapshot,
    find_snapshot,
    generate_snapshot_id,
    list_snapshots,
)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_defaults(self):
        snap = Snapshot(snapshot_id="abc")
        assert snap.source_path == ""
        assert snap.created_at == ""
        assert snap.description == ""
        assert snap.data_hash == ""
        assert snap.size_bytes == 0

    def test_as_dict(self):
        snap = Snapshot(
            snapshot_id="id1",
            source_path="/db.sqlite",
            created_at="2026-01-01T00:00:00",
            description="before cal",
            data_hash="abcd1234",
            size_bytes=1024,
        )
        d = snap.as_dict()
        assert d["snapshot_id"] == "id1"
        assert d["source_path"] == "/db.sqlite"
        assert d["size_bytes"] == 1024

    def test_from_dict(self):
        d = {
            "snapshot_id": "s1",
            "source_path": "/path",
            "created_at": "2026-01-01",
            "description": "test",
            "data_hash": "hash",
            "size_bytes": 512,
        }
        snap = Snapshot.from_dict(d)
        assert snap.snapshot_id == "s1"
        assert snap.size_bytes == 512

    def test_from_dict_defaults(self):
        snap = Snapshot.from_dict({})
        assert snap.snapshot_id == ""
        assert snap.size_bytes == 0

    def test_roundtrip(self):
        snap = Snapshot(
            snapshot_id="rt",
            source_path="/a/b",
            created_at="2026-03-28T12:00:00",
            description="roundtrip",
            data_hash="aabb",
            size_bytes=256,
        )
        restored = Snapshot.from_dict(snap.as_dict())
        assert restored.snapshot_id == snap.snapshot_id
        assert restored.source_path == snap.source_path
        assert restored.created_at == snap.created_at
        assert restored.description == snap.description
        assert restored.data_hash == snap.data_hash
        assert restored.size_bytes == snap.size_bytes


# ---------------------------------------------------------------------------
# SnapshotConfig
# ---------------------------------------------------------------------------


class TestSnapshotConfig:
    def test_defaults(self):
        cfg = SnapshotConfig()
        assert cfg.snapshot_dir == ".evalyn_snapshots"
        assert cfg.max_snapshots == 10
        assert cfg.auto_snapshot_before_calibration is True

    def test_as_dict(self):
        cfg = SnapshotConfig(snapshot_dir="/snaps", max_snapshots=5, auto_snapshot_before_calibration=False)
        d = cfg.as_dict()
        assert d["snapshot_dir"] == "/snaps"
        assert d["max_snapshots"] == 5
        assert d["auto_snapshot_before_calibration"] is False

    def test_from_dict(self):
        d = {"snapshot_dir": "/tmp/snaps", "max_snapshots": 3, "auto_snapshot_before_calibration": False}
        cfg = SnapshotConfig.from_dict(d)
        assert cfg.snapshot_dir == "/tmp/snaps"
        assert cfg.max_snapshots == 3
        assert cfg.auto_snapshot_before_calibration is False

    def test_from_dict_defaults(self):
        cfg = SnapshotConfig.from_dict({})
        assert cfg.snapshot_dir == ".evalyn_snapshots"
        assert cfg.max_snapshots == 10
        assert cfg.auto_snapshot_before_calibration is True

    def test_roundtrip(self):
        cfg = SnapshotConfig(snapshot_dir="/d", max_snapshots=7, auto_snapshot_before_calibration=False)
        restored = SnapshotConfig.from_dict(cfg.as_dict())
        assert restored.snapshot_dir == cfg.snapshot_dir
        assert restored.max_snapshots == cfg.max_snapshots
        assert restored.auto_snapshot_before_calibration == cfg.auto_snapshot_before_calibration


# ---------------------------------------------------------------------------
# SnapshotReport
# ---------------------------------------------------------------------------


class TestSnapshotReport:
    def test_defaults(self):
        report = SnapshotReport()
        assert report.snapshots == []
        assert report.total_snapshots == 0
        assert report.total_size_bytes == 0
        assert report.oldest == ""
        assert report.newest == ""

    def test_as_dict(self):
        snap = Snapshot(snapshot_id="s1")
        report = SnapshotReport(
            snapshots=[snap], total_snapshots=1, total_size_bytes=100,
            oldest="2026-01-01", newest="2026-01-01",
        )
        d = report.as_dict()
        assert d["total_snapshots"] == 1
        assert len(d["snapshots"]) == 1

    def test_from_dict(self):
        d = {
            "snapshots": [{"snapshot_id": "a"}],
            "total_snapshots": 1,
            "total_size_bytes": 200,
            "oldest": "2026-01-01",
            "newest": "2026-02-01",
        }
        report = SnapshotReport.from_dict(d)
        assert report.total_snapshots == 1
        assert report.snapshots[0].snapshot_id == "a"
        assert report.oldest == "2026-01-01"

    def test_from_dict_defaults(self):
        report = SnapshotReport.from_dict({})
        assert report.snapshots == []
        assert report.total_snapshots == 0

    def test_roundtrip(self):
        snap = Snapshot(snapshot_id="rt1", created_at="2026-03-28")
        report = SnapshotReport(
            snapshots=[snap], total_snapshots=1, total_size_bytes=50,
            oldest="2026-03-28", newest="2026-03-28",
        )
        restored = SnapshotReport.from_dict(report.as_dict())
        assert restored.total_snapshots == report.total_snapshots
        assert restored.snapshots[0].snapshot_id == "rt1"

    def test_format_text(self):
        snap = Snapshot(snapshot_id="snap_001", description="my snapshot")
        report = SnapshotReport(
            snapshots=[snap], total_snapshots=1, total_size_bytes=1024,
            oldest="2026-01-01", newest="2026-01-01",
        )
        text = report.format_text()
        assert "Snapshot Report" in text
        assert "Total snapshots: 1" in text
        assert "Total size: 1024 bytes" in text
        assert "snap_001" in text
        assert "my snapshot" in text

    def test_format_text_empty(self):
        report = SnapshotReport()
        text = report.format_text()
        assert "Total snapshots: 0" in text


# ---------------------------------------------------------------------------
# generate_snapshot_id
# ---------------------------------------------------------------------------


class TestGenerateSnapshotId:
    def test_format(self):
        sid = generate_snapshot_id()
        # Should be YYYYMMDD_HHMMSS format
        assert len(sid) == 15
        parts = sid.split("_")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_with_description(self):
        sid = generate_snapshot_id("before calibration")
        assert "before_calibration" in sid
        # Starts with timestamp
        assert sid[:8].isdigit()

    def test_description_sanitization(self):
        sid = generate_snapshot_id("My Test! @Snapshot")
        # Special chars replaced with underscores
        assert "!" not in sid
        assert "@" not in sid


# ---------------------------------------------------------------------------
# compute_snapshot_hash
# ---------------------------------------------------------------------------


class TestComputeSnapshotHash:
    def test_deterministic(self):
        h1 = compute_snapshot_hash("hello world")
        h2 = compute_snapshot_hash("hello world")
        assert h1 == h2

    def test_length(self):
        h = compute_snapshot_hash("test data")
        assert len(h) == 16

    def test_different_data(self):
        h1 = compute_snapshot_hash("alpha")
        h2 = compute_snapshot_hash("beta")
        assert h1 != h2

    def test_empty_data(self):
        h = compute_snapshot_hash("")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    def test_metadata(self):
        cfg = SnapshotConfig()
        snap = create_snapshot("/db.sqlite", cfg, description="test snap")
        assert snap.source_path == "/db.sqlite"
        assert snap.description == "test snap"
        assert snap.created_at != ""
        assert snap.snapshot_id != ""

    def test_without_description(self):
        cfg = SnapshotConfig()
        snap = create_snapshot("/db.sqlite", cfg)
        assert snap.description == ""
        assert snap.snapshot_id != ""


# ---------------------------------------------------------------------------
# list_snapshots
# ---------------------------------------------------------------------------


class TestListSnapshots:
    def test_returns_empty(self):
        cfg = SnapshotConfig()
        result = list_snapshots(cfg)
        assert result == []


# ---------------------------------------------------------------------------
# find_snapshot
# ---------------------------------------------------------------------------


class TestFindSnapshot:
    def test_existing(self):
        snaps = [
            Snapshot(snapshot_id="a"),
            Snapshot(snapshot_id="b"),
            Snapshot(snapshot_id="c"),
        ]
        found = find_snapshot(snaps, "b")
        assert found is not None
        assert found.snapshot_id == "b"

    def test_missing(self):
        snaps = [Snapshot(snapshot_id="a")]
        found = find_snapshot(snaps, "z")
        assert found is None

    def test_empty_list(self):
        found = find_snapshot([], "x")
        assert found is None


# ---------------------------------------------------------------------------
# cleanup_snapshots
# ---------------------------------------------------------------------------


class TestCleanupSnapshots:
    def test_keeps_newest(self):
        snaps = [
            Snapshot(snapshot_id="old", created_at="2026-01-01T00:00:00"),
            Snapshot(snapshot_id="mid", created_at="2026-02-01T00:00:00"),
            Snapshot(snapshot_id="new", created_at="2026-03-01T00:00:00"),
        ]
        keep, remove = cleanup_snapshots(snaps, max_keep=2)
        assert len(keep) == 2
        assert len(remove) == 1
        keep_ids = [s.snapshot_id for s in keep]
        assert "new" in keep_ids
        assert "mid" in keep_ids
        assert remove[0].snapshot_id == "old"

    def test_all_kept(self):
        snaps = [Snapshot(snapshot_id="a", created_at="2026-01-01")]
        keep, remove = cleanup_snapshots(snaps, max_keep=5)
        assert len(keep) == 1
        assert len(remove) == 0

    def test_empty(self):
        keep, remove = cleanup_snapshots([], max_keep=5)
        assert keep == []
        assert remove == []

    def test_single_snapshot(self):
        snaps = [Snapshot(snapshot_id="only", created_at="2026-01-01")]
        keep, remove = cleanup_snapshots(snaps, max_keep=1)
        assert len(keep) == 1
        assert len(remove) == 0


# ---------------------------------------------------------------------------
# build_snapshot_report
# ---------------------------------------------------------------------------


class TestBuildSnapshotReport:
    def test_basic(self):
        snaps = [
            Snapshot(snapshot_id="a", created_at="2026-01-01", size_bytes=100),
            Snapshot(snapshot_id="b", created_at="2026-02-01", size_bytes=200),
        ]
        report = build_snapshot_report(snaps)
        assert report.total_snapshots == 2
        assert report.total_size_bytes == 300
        assert report.oldest == "2026-01-01"
        assert report.newest == "2026-02-01"

    def test_empty(self):
        report = build_snapshot_report([])
        assert report.total_snapshots == 0
        assert report.total_size_bytes == 0
        assert report.oldest == ""
        assert report.newest == ""

    def test_single(self):
        snaps = [Snapshot(snapshot_id="s", created_at="2026-03-28", size_bytes=50)]
        report = build_snapshot_report(snaps)
        assert report.total_snapshots == 1
        assert report.oldest == "2026-03-28"
        assert report.newest == "2026-03-28"
