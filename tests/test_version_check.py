"""Tests for version_check module."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from evalyn_sdk.version_check import (
    VersionInfo,
    compare_versions,
    format_update_message,
    get_current_version,
    parse_version,
    save_check_timestamp,
    should_check_updates,
)


# ---------------------------------------------------------------------------
# VersionInfo dataclass
# ---------------------------------------------------------------------------


class TestVersionInfo:
    def test_as_dict_roundtrip(self):
        info = VersionInfo(
            current_version="1.0.0",
            latest_version="1.1.0",
            update_available=True,
            checked_at="2026-03-28T00:00:00+00:00",
        )
        d = info.as_dict()
        restored = VersionInfo.from_dict(d)
        assert restored.current_version == "1.0.0"
        assert restored.latest_version == "1.1.0"
        assert restored.update_available is True
        assert restored.checked_at == "2026-03-28T00:00:00+00:00"

    def test_as_dict_keys(self):
        info = VersionInfo("0.1.0", None, False, "2026-01-01")
        d = info.as_dict()
        assert set(d.keys()) == {
            "current_version",
            "latest_version",
            "update_available",
            "checked_at",
        }

    def test_from_dict_defaults(self):
        d = {"current_version": "0.1.0"}
        info = VersionInfo.from_dict(d)
        assert info.latest_version is None
        assert info.update_available is False
        assert info.checked_at == ""

    def test_from_dict_with_none_latest(self):
        d = {
            "current_version": "0.1.0",
            "latest_version": None,
            "update_available": False,
            "checked_at": "2026-01-01",
        }
        info = VersionInfo.from_dict(d)
        assert info.latest_version is None

    def test_as_dict_preserves_none_latest(self):
        info = VersionInfo("1.0.0", None, False, "2026-01-01")
        assert info.as_dict()["latest_version"] is None


# ---------------------------------------------------------------------------
# parse_version
# ---------------------------------------------------------------------------


class TestParseVersion:
    def test_simple(self):
        assert parse_version("1.2.3") == ((0, 1, ""), (0, 2, ""), (0, 3, ""))

    def test_two_parts(self):
        assert parse_version("1.0") == ((0, 1, ""), (0, 0, ""))

    def test_single_part(self):
        assert parse_version("5") == ((0, 5, ""),)

    def test_zero_version(self):
        assert parse_version("0.0.0") == ((0, 0, ""), (0, 0, ""), (0, 0, ""))

    def test_large_numbers(self):
        assert parse_version("100.200.300") == ((0, 100, ""), (0, 200, ""), (0, 300, ""))

    def test_non_numeric_segment(self):
        result = parse_version("1.0.0a1")
        assert result == ((0, 1, ""), (0, 0, ""), (1, 0, "0a1"))

    def test_four_parts(self):
        assert parse_version("1.2.3.4") == ((0, 1, ""), (0, 2, ""), (0, 3, ""), (0, 4, ""))

    def test_prerelease_sorts_after_numeric(self):
        # "1.0.0a1" should compare correctly against "1.0.0"
        assert parse_version("1.0.0") < parse_version("1.0.0a1")

    def test_mixed_type_comparison_no_crash(self):
        # This was a bug: comparing int vs str segments crashed
        v1 = parse_version("1.0.0")
        v2 = parse_version("1.0.0a1")
        # Should not raise TypeError
        assert v1 < v2 or v1 > v2 or v1 == v2


# ---------------------------------------------------------------------------
# compare_versions
# ---------------------------------------------------------------------------


class TestCompareVersions:
    def test_equal(self):
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_less_than_major(self):
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_greater_than_major(self):
        assert compare_versions("2.0.0", "1.0.0") == 1

    def test_less_than_minor(self):
        assert compare_versions("1.0.0", "1.1.0") == -1

    def test_greater_than_minor(self):
        assert compare_versions("1.1.0", "1.0.0") == 1

    def test_less_than_patch(self):
        assert compare_versions("1.0.0", "1.0.1") == -1

    def test_greater_than_patch(self):
        assert compare_versions("1.0.1", "1.0.0") == 1

    def test_different_lengths(self):
        # (1, 0) < (1, 0, 1) because tuples compare element-wise
        assert compare_versions("1.0", "1.0.1") == -1

    def test_zero_versions(self):
        assert compare_versions("0.0.0", "0.0.0") == 0

    def test_zero_less_than_nonzero(self):
        assert compare_versions("0.0.0", "0.0.1") == -1

    def test_large_version_numbers(self):
        assert compare_versions("10.20.30", "10.20.31") == -1


# ---------------------------------------------------------------------------
# get_current_version
# ---------------------------------------------------------------------------


class TestGetCurrentVersion:
    def test_returns_string(self):
        v = get_current_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_matches_init(self):
        from evalyn_sdk import __version__
        assert get_current_version() == __version__

    def test_dot_separated(self):
        v = get_current_version()
        parts = v.split(".")
        assert len(parts) >= 2


# ---------------------------------------------------------------------------
# format_update_message
# ---------------------------------------------------------------------------


class TestFormatUpdateMessage:
    def test_update_available(self):
        info = VersionInfo("0.1.0", "0.2.0", True, "2026-01-01")
        msg = format_update_message(info)
        assert "0.1.0" in msg
        assert "0.2.0" in msg
        assert "pip install" in msg

    def test_up_to_date(self):
        info = VersionInfo("1.0.0", "1.0.0", False, "2026-01-01")
        msg = format_update_message(info)
        assert msg == ""

    def test_no_latest_version(self):
        info = VersionInfo("1.0.0", None, False, "2026-01-01")
        msg = format_update_message(info)
        assert msg == ""

    def test_update_not_available_but_latest_set(self):
        info = VersionInfo("1.0.0", "1.0.0", False, "2026-01-01")
        msg = format_update_message(info)
        assert msg == ""


# ---------------------------------------------------------------------------
# should_check_updates
# ---------------------------------------------------------------------------


class TestShouldCheckUpdates:
    def test_no_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert should_check_updates(path) is True

    def test_recent_check(self, tmp_path):
        path = str(tmp_path / "cache.json")
        with open(path, "w") as f:
            json.dump({"last_check_timestamp": time.time()}, f)
        assert should_check_updates(path, interval_hours=24.0) is False

    def test_stale_check(self, tmp_path):
        path = str(tmp_path / "cache.json")
        old_time = time.time() - 25 * 3600  # 25 hours ago
        with open(path, "w") as f:
            json.dump({"last_check_timestamp": old_time}, f)
        assert should_check_updates(path, interval_hours=24.0) is True

    def test_zero_interval(self, tmp_path):
        path = str(tmp_path / "cache.json")
        with open(path, "w") as f:
            json.dump({"last_check_timestamp": time.time()}, f)
        # 0 hours interval means always check
        assert should_check_updates(path, interval_hours=0.0) is True

    def test_corrupted_json(self, tmp_path):
        path = str(tmp_path / "cache.json")
        with open(path, "w") as f:
            f.write("not json")
        assert should_check_updates(path) is True

    def test_missing_timestamp_key(self, tmp_path):
        path = str(tmp_path / "cache.json")
        with open(path, "w") as f:
            json.dump({"other_key": "value"}, f)
        assert should_check_updates(path) is True

    def test_custom_interval(self, tmp_path):
        path = str(tmp_path / "cache.json")
        recent = time.time() - 0.5 * 3600  # 30 min ago
        with open(path, "w") as f:
            json.dump({"last_check_timestamp": recent}, f)
        assert should_check_updates(path, interval_hours=1.0) is False
        assert should_check_updates(path, interval_hours=0.25) is True


# ---------------------------------------------------------------------------
# save_check_timestamp
# ---------------------------------------------------------------------------


class TestSaveCheckTimestamp:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "cache.json")
        save_check_timestamp(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "last_check_timestamp" in data
        assert isinstance(data["last_check_timestamp"], float)

    def test_preserves_existing_keys(self, tmp_path):
        path = str(tmp_path / "cache.json")
        with open(path, "w") as f:
            json.dump({"existing_key": "value"}, f)
        save_check_timestamp(path)
        with open(path) as f:
            data = json.load(f)
        assert data["existing_key"] == "value"
        assert "last_check_timestamp" in data

    def test_updates_timestamp(self, tmp_path):
        path = str(tmp_path / "cache.json")
        with open(path, "w") as f:
            json.dump({"last_check_timestamp": 1000.0}, f)
        save_check_timestamp(path)
        with open(path) as f:
            data = json.load(f)
        assert data["last_check_timestamp"] > 1000.0

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "cache.json")
        save_check_timestamp(path)
        assert os.path.exists(path)

    def test_roundtrip_with_should_check(self, tmp_path):
        path = str(tmp_path / "cache.json")
        save_check_timestamp(path)
        assert should_check_updates(path, interval_hours=24.0) is False
