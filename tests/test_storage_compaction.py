from __future__ import annotations

import os
import tempfile

import pytest

from evalyn_sdk.storage.compaction import (
    CompactionConfig,
    CompactionStats,
    compact_database,
    estimate_compaction_savings,
    format_size,
    get_database_size,
    should_compact,
)


# ---------------------------------------------------------------------------
# CompactionStats
# ---------------------------------------------------------------------------


class TestCompactionStats:
    def test_defaults(self):
        cs = CompactionStats()
        assert cs.original_size_bytes == 0
        assert cs.compacted_size_bytes == 0
        assert cs.savings_bytes == 0
        assert cs.savings_pct == 0.0
        assert cs.duration_ms == 0.0

    def test_as_dict(self):
        cs = CompactionStats(
            original_size_bytes=1000,
            compacted_size_bytes=800,
            savings_bytes=200,
            savings_pct=20.0,
            duration_ms=5.5,
        )
        d = cs.as_dict()
        assert d["original_size_bytes"] == 1000
        assert d["compacted_size_bytes"] == 800
        assert d["savings_bytes"] == 200
        assert d["savings_pct"] == 20.0
        assert d["duration_ms"] == 5.5

    def test_as_dict_roundtrip(self):
        cs = CompactionStats(
            original_size_bytes=5000,
            compacted_size_bytes=4000,
            savings_bytes=1000,
            savings_pct=20.0,
            duration_ms=12.3,
        )
        d = cs.as_dict()
        restored = CompactionStats(**d)
        assert restored == cs

    def test_format_text(self):
        cs = CompactionStats(
            original_size_bytes=1024 * 1024,
            compacted_size_bytes=900 * 1024,
            savings_bytes=124 * 1024,
            savings_pct=12.1,
            duration_ms=45.3,
        )
        text = cs.format_text()
        assert "Compaction Results" in text
        assert "Original size" in text
        assert "Compacted size" in text
        assert "Savings" in text
        assert "Duration" in text
        assert "12.1%" in text

    def test_format_text_zero(self):
        cs = CompactionStats()
        text = cs.format_text()
        assert "0 B" in text
        assert "0.0%" in text


# ---------------------------------------------------------------------------
# CompactionConfig
# ---------------------------------------------------------------------------


class TestCompactionConfig:
    def test_defaults(self):
        cc = CompactionConfig()
        assert cc.vacuum is True
        assert cc.reindex is True
        assert cc.analyze is True
        assert cc.wal_checkpoint is True

    def test_as_dict(self):
        cc = CompactionConfig(vacuum=False, analyze=False)
        d = cc.as_dict()
        assert d["vacuum"] is False
        assert d["reindex"] is True
        assert d["analyze"] is False
        assert d["wal_checkpoint"] is True

    def test_from_dict_roundtrip(self):
        cc = CompactionConfig(vacuum=False, reindex=False, analyze=True, wal_checkpoint=False)
        restored = CompactionConfig.from_dict(cc.as_dict())
        assert restored == cc

    def test_from_dict_missing_keys(self):
        cc = CompactionConfig.from_dict({})
        assert cc.vacuum is True
        assert cc.reindex is True
        assert cc.analyze is True
        assert cc.wal_checkpoint is True

    def test_from_dict_partial(self):
        cc = CompactionConfig.from_dict({"vacuum": False})
        assert cc.vacuum is False
        assert cc.reindex is True


# ---------------------------------------------------------------------------
# get_database_size
# ---------------------------------------------------------------------------


class TestGetDatabaseSize:
    def test_returns_correct_size(self, tmp_path):
        f = tmp_path / "test.db"
        f.write_bytes(b"x" * 512)
        assert get_database_size(str(f)) == 512

    def test_missing_file(self):
        assert get_database_size("/nonexistent/path/db.sqlite") == 0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.db"
        f.write_bytes(b"")
        assert get_database_size(str(f)) == 0


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_zero_bytes(self):
        assert format_size(0) == "0 B"

    def test_kilobytes(self):
        result = format_size(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_megabytes(self):
        result = format_size(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.0" in result

    def test_gigabytes(self):
        result = format_size(2 * 1024 * 1024 * 1024)
        assert "GB" in result
        assert "2.0" in result

    def test_boundary_kb(self):
        # Exactly 1024 bytes should be 1.0 KB
        assert format_size(1024) == "1.0 KB"

    def test_boundary_mb(self):
        assert format_size(1024 * 1024) == "1.0 MB"

    def test_boundary_gb(self):
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"


# ---------------------------------------------------------------------------
# estimate_compaction_savings
# ---------------------------------------------------------------------------


class TestEstimateCompactionSavings:
    def test_with_tmp_file(self, tmp_path):
        f = tmp_path / "test.db"
        f.write_bytes(b"x" * 10000)
        stats = estimate_compaction_savings(str(f))
        assert stats.original_size_bytes == 10000
        assert stats.savings_bytes > 0
        assert stats.compacted_size_bytes < 10000
        assert 10.0 < stats.savings_pct < 20.0

    def test_missing_file(self):
        stats = estimate_compaction_savings("/nonexistent/db.sqlite")
        assert stats.original_size_bytes == 0
        assert stats.savings_bytes == 0

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.db"
        f.write_bytes(b"")
        stats = estimate_compaction_savings(str(f))
        assert stats.original_size_bytes == 0


# ---------------------------------------------------------------------------
# compact_database
# ---------------------------------------------------------------------------


class TestCompactDatabase:
    def test_with_tmp_file(self, tmp_path):
        f = tmp_path / "test.db"
        f.write_bytes(b"x" * 5000)
        stats = compact_database(str(f))
        assert stats.original_size_bytes == 5000
        assert stats.duration_ms >= 0.0

    def test_missing_file(self):
        stats = compact_database("/nonexistent/db.sqlite")
        assert stats.original_size_bytes == 0

    def test_with_custom_config(self, tmp_path):
        f = tmp_path / "test.db"
        f.write_bytes(b"x" * 1000)
        config = CompactionConfig(vacuum=True, reindex=False)
        stats = compact_database(str(f), config=config)
        assert stats.original_size_bytes == 1000

    def test_default_config_used(self, tmp_path):
        f = tmp_path / "test.db"
        f.write_bytes(b"x" * 1000)
        stats = compact_database(str(f))
        assert stats.compacted_size_bytes == 1000  # no actual SQL operations


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------


class TestShouldCompact:
    def test_above_threshold(self, tmp_path):
        f = tmp_path / "big.db"
        # Write 200 MB worth (as a proxy, use threshold_mb=0.001 for a small file)
        f.write_bytes(b"x" * 2000)
        assert should_compact(str(f), threshold_mb=0.001) is True

    def test_below_threshold(self, tmp_path):
        f = tmp_path / "small.db"
        f.write_bytes(b"x" * 100)
        assert should_compact(str(f), threshold_mb=100.0) is False

    def test_missing_file(self):
        assert should_compact("/nonexistent/db.sqlite") is False

    def test_default_threshold(self, tmp_path):
        f = tmp_path / "small.db"
        f.write_bytes(b"x" * 100)
        # Default is 100 MB, 100 bytes is way below
        assert should_compact(str(f)) is False
