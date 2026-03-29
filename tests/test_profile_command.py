"""Tests for the profile command module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.profile_command import (
    StorageProfile,
    EnvironmentProfile,
    ProfileReport,
    scan_data_directory,
    get_environment_profile,
    build_profile,
    format_size_human,
    format_profile_report,
)


# ---------------------------------------------------------------------------
# StorageProfile dataclass
# ---------------------------------------------------------------------------


class TestStorageProfile:
    def test_as_dict(self):
        sp = StorageProfile(
            data_dir="/tmp/data",
            total_size_bytes=1024,
            file_count=10,
            run_count=2,
            trace_count=3,
            annotation_count=1,
        )
        d = sp.as_dict()
        assert d["data_dir"] == "/tmp/data"
        assert d["total_size_bytes"] == 1024
        assert d["file_count"] == 10
        assert d["run_count"] == 2
        assert d["trace_count"] == 3
        assert d["annotation_count"] == 1

    def test_from_dict(self):
        d = {
            "data_dir": "/data",
            "total_size_bytes": 2048,
            "file_count": 5,
            "run_count": 1,
            "trace_count": 2,
            "annotation_count": 0,
        }
        sp = StorageProfile.from_dict(d)
        assert sp.data_dir == "/data"
        assert sp.total_size_bytes == 2048
        assert sp.file_count == 5
        assert sp.run_count == 1
        assert sp.trace_count == 2
        assert sp.annotation_count == 0

    def test_roundtrip(self):
        sp = StorageProfile(
            data_dir="/x", total_size_bytes=0, file_count=0,
            run_count=0, trace_count=0, annotation_count=0,
        )
        assert StorageProfile.from_dict(sp.as_dict()).as_dict() == sp.as_dict()

    def test_all_fields_present(self):
        sp = StorageProfile(
            data_dir=".", total_size_bytes=100, file_count=1,
            run_count=0, trace_count=0, annotation_count=0,
        )
        keys = set(sp.as_dict().keys())
        assert keys == {
            "data_dir", "total_size_bytes", "file_count",
            "run_count", "trace_count", "annotation_count",
        }


# ---------------------------------------------------------------------------
# EnvironmentProfile dataclass
# ---------------------------------------------------------------------------


class TestEnvironmentProfile:
    def test_as_dict(self):
        ep = EnvironmentProfile(
            python_version="3.12.0",
            evalyn_version="0.1.0",
            platform="Linux",
            installed_providers=["openai"],
        )
        d = ep.as_dict()
        assert d["python_version"] == "3.12.0"
        assert d["installed_providers"] == ["openai"]

    def test_from_dict(self):
        d = {
            "python_version": "3.11.0",
            "evalyn_version": "0.2.0",
            "platform": "Darwin",
            "installed_providers": ["anthropic", "google"],
        }
        ep = EnvironmentProfile.from_dict(d)
        assert ep.python_version == "3.11.0"
        assert ep.installed_providers == ["anthropic", "google"]

    def test_from_dict_default_providers(self):
        d = {
            "python_version": "3.10.0",
            "evalyn_version": "0.1.0",
            "platform": "win32",
        }
        ep = EnvironmentProfile.from_dict(d)
        assert ep.installed_providers == []

    def test_roundtrip(self):
        ep = EnvironmentProfile(
            python_version="3.12.0", evalyn_version="0.1.0",
            platform="Linux", installed_providers=["openai", "anthropic"],
        )
        assert EnvironmentProfile.from_dict(ep.as_dict()).as_dict() == ep.as_dict()

    def test_providers_list_is_copy(self):
        ep = EnvironmentProfile(
            python_version="3.12.0", evalyn_version="0.1.0",
            platform="Linux", installed_providers=["openai"],
        )
        d = ep.as_dict()
        d["installed_providers"].append("extra")
        assert ep.installed_providers == ["openai"]


# ---------------------------------------------------------------------------
# ProfileReport dataclass
# ---------------------------------------------------------------------------


class TestProfileReport:
    def test_as_dict(self):
        sp = StorageProfile(
            data_dir="/d", total_size_bytes=0, file_count=0,
            run_count=0, trace_count=0, annotation_count=0,
        )
        ep = EnvironmentProfile(
            python_version="3.12.0", evalyn_version="0.1.0",
            platform="Linux", installed_providers=[],
        )
        pr = ProfileReport(storage=sp, environment=ep, generated_at="2026-01-01T00:00:00Z")
        d = pr.as_dict()
        assert "storage" in d
        assert "environment" in d
        assert d["generated_at"] == "2026-01-01T00:00:00Z"

    def test_from_dict(self):
        d = {
            "storage": {
                "data_dir": "/d", "total_size_bytes": 10, "file_count": 1,
                "run_count": 0, "trace_count": 0, "annotation_count": 0,
            },
            "environment": {
                "python_version": "3.12.0", "evalyn_version": "0.1.0",
                "platform": "Linux", "installed_providers": [],
            },
            "generated_at": "2026-03-01T00:00:00Z",
        }
        pr = ProfileReport.from_dict(d)
        assert pr.storage.data_dir == "/d"
        assert pr.environment.python_version == "3.12.0"
        assert pr.generated_at == "2026-03-01T00:00:00Z"

    def test_roundtrip(self):
        sp = StorageProfile(
            data_dir="/d", total_size_bytes=512, file_count=3,
            run_count=1, trace_count=2, annotation_count=0,
        )
        ep = EnvironmentProfile(
            python_version="3.12.0", evalyn_version="0.1.0",
            platform="Linux", installed_providers=["openai"],
        )
        pr = ProfileReport(storage=sp, environment=ep, generated_at="2026-01-01")
        assert ProfileReport.from_dict(pr.as_dict()).as_dict() == pr.as_dict()


# ---------------------------------------------------------------------------
# scan_data_directory
# ---------------------------------------------------------------------------


class TestScanDataDirectory:
    def test_nonexistent_directory(self):
        sp = scan_data_directory("/nonexistent/path/abc123")
        assert sp.total_size_bytes == 0
        assert sp.file_count == 0
        assert sp.run_count == 0
        assert sp.trace_count == 0
        assert sp.annotation_count == 0

    def test_empty_directory(self, tmp_path):
        sp = scan_data_directory(str(tmp_path))
        assert sp.total_size_bytes == 0
        assert sp.file_count == 0
        assert sp.run_count == 0

    def test_counts_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        sp = scan_data_directory(str(tmp_path))
        assert sp.file_count == 2
        assert sp.total_size_bytes == 10

    def test_counts_run_dirs(self, tmp_path):
        (tmp_path / "run_001").mkdir()
        (tmp_path / "run_002").mkdir()
        (tmp_path / "not_a_run").mkdir()
        sp = scan_data_directory(str(tmp_path))
        assert sp.run_count == 2

    def test_counts_traces(self, tmp_path):
        (tmp_path / "trace1.jsonl").write_text("{}")
        (tmp_path / "trace2.jsonl").write_text("{}")
        (tmp_path / "data.json").write_text("{}")
        sp = scan_data_directory(str(tmp_path))
        assert sp.trace_count == 2

    def test_counts_annotations(self, tmp_path):
        (tmp_path / "annotation_v1.json").write_text("{}")
        (tmp_path / "annotations.jsonl").write_text("{}")
        (tmp_path / "other.json").write_text("{}")
        sp = scan_data_directory(str(tmp_path))
        assert sp.annotation_count == 2

    def test_nested_files(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "trace.jsonl").write_text("line\n")
        (sub / "annotation_data.json").write_text("{}")
        sp = scan_data_directory(str(tmp_path))
        assert sp.file_count == 2
        assert sp.trace_count == 1
        assert sp.annotation_count == 1

    def test_nested_run_dirs(self, tmp_path):
        inner = tmp_path / "eval_runs"
        inner.mkdir()
        (inner / "run_abc").mkdir()
        sp = scan_data_directory(str(tmp_path))
        assert sp.run_count == 1

    def test_data_dir_stored(self, tmp_path):
        sp = scan_data_directory(str(tmp_path))
        assert sp.data_dir == str(tmp_path)

    def test_size_accumulates(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "b.bin").write_bytes(b"\x00" * 200)
        sp = scan_data_directory(str(tmp_path))
        assert sp.total_size_bytes == 300


# ---------------------------------------------------------------------------
# get_environment_profile
# ---------------------------------------------------------------------------


class TestGetEnvironmentProfile:
    def test_returns_environment_profile(self):
        ep = get_environment_profile()
        assert isinstance(ep, EnvironmentProfile)

    def test_python_version_format(self):
        ep = get_environment_profile()
        parts = ep.python_version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_evalyn_version_present(self):
        ep = get_environment_profile()
        assert ep.evalyn_version  # not empty

    def test_platform_not_empty(self):
        ep = get_environment_profile()
        assert len(ep.platform) > 0

    def test_providers_is_list(self):
        ep = get_environment_profile()
        assert isinstance(ep.installed_providers, list)

    def test_provider_detection(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
        ep = get_environment_profile()
        assert "openai" in ep.installed_providers
        assert "anthropic" in ep.installed_providers

    def test_no_providers_when_unset(self, monkeypatch):
        for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                     "COHERE_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY"]:
            monkeypatch.delenv(var, raising=False)
        ep = get_environment_profile()
        assert ep.installed_providers == []


# ---------------------------------------------------------------------------
# build_profile
# ---------------------------------------------------------------------------


class TestBuildProfile:
    def test_returns_profile_report(self, tmp_path):
        pr = build_profile(str(tmp_path))
        assert isinstance(pr, ProfileReport)

    def test_generated_at_present(self, tmp_path):
        pr = build_profile(str(tmp_path))
        assert pr.generated_at  # not empty

    def test_storage_populated(self, tmp_path):
        (tmp_path / "file.txt").write_text("data")
        pr = build_profile(str(tmp_path))
        assert pr.storage.file_count == 1

    def test_environment_populated(self, tmp_path):
        pr = build_profile(str(tmp_path))
        assert pr.environment.python_version


# ---------------------------------------------------------------------------
# format_size_human
# ---------------------------------------------------------------------------


class TestFormatSizeHuman:
    def test_bytes(self):
        assert format_size_human(0) == "0 B"
        assert format_size_human(512) == "512 B"
        assert format_size_human(1023) == "1023 B"

    def test_kilobytes(self):
        assert format_size_human(1024) == "1.0 KB"
        assert format_size_human(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_size_human(1024 * 1024) == "1.0 MB"
        assert format_size_human(1024 * 1024 * 5) == "5.0 MB"

    def test_gigabytes(self):
        assert format_size_human(1024 * 1024 * 1024) == "1.0 GB"
        assert format_size_human(1024 * 1024 * 1024 * 2) == "2.0 GB"

    def test_negative(self):
        assert format_size_human(-1) == "0 B"


# ---------------------------------------------------------------------------
# format_profile_report
# ---------------------------------------------------------------------------


class TestFormatProfileReport:
    def _make_report(self):
        sp = StorageProfile(
            data_dir="/data", total_size_bytes=2048, file_count=5,
            run_count=2, trace_count=3, annotation_count=1,
        )
        ep = EnvironmentProfile(
            python_version="3.12.0", evalyn_version="0.1.0",
            platform="Linux", installed_providers=["openai"],
        )
        return ProfileReport(storage=sp, environment=ep, generated_at="2026-01-01")

    def test_contains_header(self):
        txt = format_profile_report(self._make_report())
        assert "PROFILE REPORT" in txt

    def test_contains_storage_info(self):
        txt = format_profile_report(self._make_report())
        assert "2.0 KB" in txt
        assert "Files:" in txt
        assert "5" in txt

    def test_contains_environment_info(self):
        txt = format_profile_report(self._make_report())
        assert "3.12.0" in txt
        assert "0.1.0" in txt
        assert "openai" in txt

    def test_no_providers_shows_none(self):
        sp = StorageProfile(
            data_dir="/d", total_size_bytes=0, file_count=0,
            run_count=0, trace_count=0, annotation_count=0,
        )
        ep = EnvironmentProfile(
            python_version="3.12.0", evalyn_version="0.1.0",
            platform="Linux", installed_providers=[],
        )
        pr = ProfileReport(storage=sp, environment=ep, generated_at="2026-01-01")
        txt = format_profile_report(pr)
        assert "none" in txt

    def test_contains_generated_at(self):
        txt = format_profile_report(self._make_report())
        assert "2026-01-01" in txt
