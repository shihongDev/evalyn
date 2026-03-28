"""Tests for the connection diagnostics module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.storage.connection_diagnostics import (
    ConnectionHealth,
    DiagnosticItem,
    check_file_exists,
    check_file_permissions,
    check_file_size,
    check_journal_file,
    check_wal_mode,
    run_connection_diagnostics,
    suggest_connection_fixes,
)


# ---------------------------------------------------------------------------
# DiagnosticItem
# ---------------------------------------------------------------------------


class TestDiagnosticItem:
    def test_defaults(self):
        item = DiagnosticItem(name="test", value="val")
        assert item.status == "ok"
        assert item.recommendation == ""

    def test_as_dict(self):
        item = DiagnosticItem(name="n", value="v", status="warning", recommendation="fix it")
        d = item.as_dict()
        assert d["name"] == "n"
        assert d["value"] == "v"
        assert d["status"] == "warning"
        assert d["recommendation"] == "fix it"

    def test_as_dict_roundtrip(self):
        item = DiagnosticItem(name="a", value="b", status="error", recommendation="r")
        d = item.as_dict()
        restored = DiagnosticItem(**d)
        assert restored.name == item.name
        assert restored.value == item.value
        assert restored.status == item.status
        assert restored.recommendation == item.recommendation


# ---------------------------------------------------------------------------
# ConnectionHealth
# ---------------------------------------------------------------------------


class TestConnectionHealth:
    def test_defaults(self):
        health = ConnectionHealth()
        assert health.items == []
        assert health.overall_status == "ok"
        assert health.warnings == []
        assert health.errors == []

    def test_as_dict(self):
        item = DiagnosticItem(name="x", value="y")
        health = ConnectionHealth(items=[item], overall_status="ok", warnings=[], errors=[])
        d = health.as_dict()
        assert len(d["items"]) == 1
        assert d["items"][0]["name"] == "x"
        assert d["overall_status"] == "ok"

    def test_from_dict(self):
        d = {
            "items": [{"name": "a", "value": "b", "status": "ok", "recommendation": ""}],
            "overall_status": "warning",
            "warnings": ["w1"],
            "errors": [],
        }
        health = ConnectionHealth.from_dict(d)
        assert len(health.items) == 1
        assert health.items[0].name == "a"
        assert health.overall_status == "warning"
        assert health.warnings == ["w1"]

    def test_from_dict_defaults(self):
        health = ConnectionHealth.from_dict({})
        assert health.items == []
        assert health.overall_status == "ok"

    def test_roundtrip(self):
        item = DiagnosticItem(name="t", value="v", status="error", recommendation="rec")
        health = ConnectionHealth(
            items=[item], overall_status="error", warnings=["w"], errors=["e"]
        )
        restored = ConnectionHealth.from_dict(health.as_dict())
        assert restored.overall_status == health.overall_status
        assert len(restored.items) == 1
        assert restored.items[0].name == "t"
        assert restored.warnings == ["w"]
        assert restored.errors == ["e"]

    def test_format_text(self):
        item = DiagnosticItem(name="check1", value="good", status="ok")
        health = ConnectionHealth(items=[item], overall_status="ok")
        text = health.format_text()
        assert "Connection Health Report" in text
        assert "Overall status: ok" in text
        assert "[OK] check1: good" in text

    def test_format_text_with_recommendation(self):
        item = DiagnosticItem(name="perm", value="read-only", status="warning", recommendation="fix perms")
        health = ConnectionHealth(items=[item], overall_status="warning", warnings=["perm: fix perms"])
        text = health.format_text()
        assert "Warnings: 1" in text
        assert "fix perms" in text

    def test_format_text_with_errors(self):
        item = DiagnosticItem(name="missing", value="false", status="error", recommendation="check path")
        health = ConnectionHealth(items=[item], overall_status="error", errors=["missing: check path"])
        text = health.format_text()
        assert "Errors: 1" in text


# ---------------------------------------------------------------------------
# check_file_exists
# ---------------------------------------------------------------------------


class TestCheckFileExists:
    def test_existing_file(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        result = check_file_exists(str(db))
        assert result.status == "ok"
        assert result.value == "true"

    def test_missing_file(self):
        result = check_file_exists("/nonexistent/path/db.sqlite")
        assert result.status == "error"
        assert result.value == "false"

    def test_empty_path(self):
        result = check_file_exists("")
        assert result.status == "error"


# ---------------------------------------------------------------------------
# check_file_permissions
# ---------------------------------------------------------------------------


class TestCheckFilePermissions:
    def test_readable_writable(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        result = check_file_permissions(str(db))
        assert result.status == "ok"
        assert result.value == "read/write"


# ---------------------------------------------------------------------------
# check_file_size
# ---------------------------------------------------------------------------


class TestCheckFileSize:
    def test_within_limit(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_bytes(b"x" * 100)
        result = check_file_size(str(db), max_size_mb=1.0)
        assert result.status == "ok"

    def test_over_limit(self, tmp_path):
        db = tmp_path / "test.db"
        # Write just over the limit (1 MB + 1 byte)
        db.write_bytes(b"x" * (1024 * 1024 + 1))
        result = check_file_size(str(db), max_size_mb=1.0)
        assert result.status == "error"

    def test_approaching_limit(self, tmp_path):
        db = tmp_path / "test.db"
        # 85% of 1 MB (above 80% threshold)
        db.write_bytes(b"x" * int(1024 * 1024 * 0.85))
        result = check_file_size(str(db), max_size_mb=1.0)
        assert result.status == "warning"

    def test_missing_file(self):
        result = check_file_size("/nonexistent/file.db")
        assert result.status == "error"
        assert result.value == "unknown"


# ---------------------------------------------------------------------------
# check_wal_mode
# ---------------------------------------------------------------------------


class TestCheckWalMode:
    def test_no_wal(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        result = check_wal_mode(str(db))
        assert result.value == "not detected"

    def test_with_wal(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        wal = tmp_path / "test.db-wal"
        wal.write_text("wal data")
        result = check_wal_mode(str(db))
        assert result.value == "enabled"
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# check_journal_file
# ---------------------------------------------------------------------------


class TestCheckJournalFile:
    def test_no_journal(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        result = check_journal_file(str(db))
        assert result.value == "none"

    def test_with_journal(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        journal = tmp_path / "test.db-journal"
        journal.write_text("journal data")
        result = check_journal_file(str(db))
        assert "journal" in result.value

    def test_with_wal_file(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        wal = tmp_path / "test.db-wal"
        wal.write_text("wal data")
        result = check_journal_file(str(db))
        assert "wal" in result.value

    def test_with_both(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        (tmp_path / "test.db-journal").write_text("j")
        (tmp_path / "test.db-wal").write_text("w")
        result = check_journal_file(str(db))
        assert "journal" in result.value
        assert "wal" in result.value


# ---------------------------------------------------------------------------
# run_connection_diagnostics
# ---------------------------------------------------------------------------


class TestRunConnectionDiagnostics:
    def test_existing_db(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        health = run_connection_diagnostics(str(db))
        assert health.overall_status == "ok"
        assert len(health.items) >= 4

    def test_missing_db(self):
        health = run_connection_diagnostics("/nonexistent/db.sqlite")
        assert health.overall_status == "error"
        assert len(health.errors) > 0

    def test_items_contain_file_exists(self, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("data")
        health = run_connection_diagnostics(str(db))
        names = [item.name for item in health.items]
        assert "file_exists" in names


# ---------------------------------------------------------------------------
# suggest_connection_fixes
# ---------------------------------------------------------------------------


class TestSuggestConnectionFixes:
    def test_no_issues(self):
        health = ConnectionHealth(
            items=[DiagnosticItem(name="ok_check", value="fine", status="ok")]
        )
        fixes = suggest_connection_fixes(health)
        assert fixes == []

    def test_with_warnings(self):
        health = ConnectionHealth(
            items=[
                DiagnosticItem(name="size", value="big", status="warning", recommendation="clean up"),
            ]
        )
        fixes = suggest_connection_fixes(health)
        assert len(fixes) == 1
        assert "clean up" in fixes[0]

    def test_with_errors(self):
        health = ConnectionHealth(
            items=[
                DiagnosticItem(name="missing", value="false", status="error", recommendation="check path"),
            ]
        )
        fixes = suggest_connection_fixes(health)
        assert len(fixes) == 1
        assert "check path" in fixes[0]

    def test_mixed(self):
        health = ConnectionHealth(
            items=[
                DiagnosticItem(name="a", value="v", status="ok"),
                DiagnosticItem(name="b", value="v", status="warning", recommendation="w"),
                DiagnosticItem(name="c", value="v", status="error", recommendation="e"),
            ]
        )
        fixes = suggest_connection_fixes(health)
        assert len(fixes) == 2
