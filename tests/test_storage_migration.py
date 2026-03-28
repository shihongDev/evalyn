"""Tests for evalyn_sdk.storage.migration.

Run with: uv run pytest tests/test_storage_migration.py -v
"""
from __future__ import annotations

import json
import os

import pytest

from evalyn_sdk.storage.migration import (
    MigrationPlan,
    MigrationRecord,
    MigrationResult,
    compute_migration_size,
    export_to_json,
    import_from_json,
    plan_migration,
    validate_migration_plan,
)


# ---------------------------------------------------------------------------
# MigrationRecord
# ---------------------------------------------------------------------------


class TestMigrationRecord:
    def test_as_dict(self):
        rec = MigrationRecord(
            source="/a", destination="/b", item_type="runs", item_count=5, timestamp="2026-01-01"
        )
        d = rec.as_dict()
        assert d["source"] == "/a"
        assert d["destination"] == "/b"
        assert d["item_type"] == "runs"
        assert d["item_count"] == 5

    def test_from_dict(self):
        rec = MigrationRecord.from_dict(
            {"source": "/x", "destination": "/y", "item_type": "metrics", "item_count": 10}
        )
        assert rec.source == "/x"
        assert rec.item_count == 10

    def test_roundtrip(self):
        rec = MigrationRecord(
            source="s", destination="d", item_type="t", item_count=3, timestamp="ts"
        )
        restored = MigrationRecord.from_dict(rec.as_dict())
        assert restored.source == rec.source
        assert restored.destination == rec.destination
        assert restored.item_type == rec.item_type
        assert restored.item_count == rec.item_count
        assert restored.timestamp == rec.timestamp

    def test_defaults(self):
        rec = MigrationRecord(source="a", destination="b", item_type="t")
        assert rec.item_count == 0
        assert rec.timestamp == ""


# ---------------------------------------------------------------------------
# MigrationPlan
# ---------------------------------------------------------------------------


class TestMigrationPlan:
    def test_as_dict(self):
        plan = MigrationPlan(source_path="/src", dest_path="/dst")
        d = plan.as_dict()
        assert d["source_path"] == "/src"
        assert d["format"] == "json"
        assert d["tables"] == []

    def test_from_dict(self):
        plan = MigrationPlan.from_dict(
            {"source_path": "/a", "dest_path": "/b", "format": "sqlite", "tables": ["runs"]}
        )
        assert plan.format == "sqlite"
        assert plan.tables == ["runs"]

    def test_roundtrip(self):
        plan = MigrationPlan(
            source_path="/src",
            dest_path="/dst",
            format="csv",
            tables=["metrics", "runs"],
            estimated_items=42,
        )
        restored = MigrationPlan.from_dict(plan.as_dict())
        assert restored.source_path == plan.source_path
        assert restored.format == plan.format
        assert restored.tables == plan.tables
        assert restored.estimated_items == plan.estimated_items

    def test_format_text(self):
        plan = MigrationPlan(
            source_path="/src",
            dest_path="/dst",
            format="json",
            tables=["runs"],
            estimated_items=10,
        )
        text = plan.format_text()
        assert "/src" in text
        assert "/dst" in text
        assert "json" in text
        assert "runs" in text

    def test_format_text_no_tables(self):
        plan = MigrationPlan(source_path="/s", dest_path="/d")
        text = plan.format_text()
        assert "Tables" not in text


# ---------------------------------------------------------------------------
# MigrationResult
# ---------------------------------------------------------------------------


class TestMigrationResult:
    def test_as_dict(self):
        plan = MigrationPlan(source_path="/a", dest_path="/b")
        result = MigrationResult(plan=plan, exported_count=5, imported_count=5)
        d = result.as_dict()
        assert d["exported_count"] == 5
        assert d["plan"]["source_path"] == "/a"

    def test_from_dict(self):
        data = {
            "plan": {"source_path": "/a", "dest_path": "/b"},
            "exported_count": 3,
            "imported_count": 3,
            "errors": ["oops"],
            "duration_ms": 123.4,
        }
        result = MigrationResult.from_dict(data)
        assert result.exported_count == 3
        assert result.errors == ["oops"]
        assert result.duration_ms == 123.4

    def test_roundtrip(self):
        plan = MigrationPlan(source_path="/x", dest_path="/y", format="csv")
        result = MigrationResult(
            plan=plan,
            exported_count=10,
            imported_count=9,
            errors=["one skipped"],
            duration_ms=50.5,
        )
        restored = MigrationResult.from_dict(result.as_dict())
        assert restored.exported_count == result.exported_count
        assert restored.imported_count == result.imported_count
        assert restored.errors == result.errors
        assert restored.duration_ms == result.duration_ms
        assert restored.plan.format == "csv"

    def test_format_text_ok(self):
        plan = MigrationPlan(source_path="/a", dest_path="/b")
        result = MigrationResult(plan=plan, exported_count=5, imported_count=5, duration_ms=10.0)
        text = result.format_text()
        assert "OK" in text
        assert "Exported: 5" in text

    def test_format_text_errors(self):
        plan = MigrationPlan(source_path="/a", dest_path="/b")
        result = MigrationResult(plan=plan, errors=["bad item", "corrupt"])
        text = result.format_text()
        assert "ERRORS (2)" in text
        assert "bad item" in text


# ---------------------------------------------------------------------------
# plan_migration
# ---------------------------------------------------------------------------


class TestPlanMigration:
    def test_creates_plan(self, tmp_path):
        src = tmp_path / "source.json"
        src.write_text('[{"a": 1}]')
        plan = plan_migration(str(src), str(tmp_path / "dest.json"))
        assert plan.source_path == str(src)
        assert plan.format == "json"
        assert plan.estimated_items >= 0

    def test_nonexistent_source(self, tmp_path):
        plan = plan_migration(str(tmp_path / "nope"), str(tmp_path / "dest"))
        assert plan.estimated_items == 0

    def test_custom_format_and_tables(self, tmp_path):
        plan = plan_migration(
            str(tmp_path), str(tmp_path / "out"), format="sqlite", tables=["runs", "metrics"]
        )
        assert plan.format == "sqlite"
        assert plan.tables == ["runs", "metrics"]


# ---------------------------------------------------------------------------
# export_to_json / import_from_json
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_writes_file(self, tmp_path):
        fpath = str(tmp_path / "out.json")
        data = [{"id": 1}, {"id": 2}]
        count = export_to_json(data, fpath)
        assert count == 2
        assert os.path.isfile(fpath)

    def test_import_reads_back(self, tmp_path):
        fpath = str(tmp_path / "data.json")
        original = [{"key": "val"}]
        export_to_json(original, fpath)
        loaded = import_from_json(fpath)
        assert loaded == original

    def test_roundtrip(self, tmp_path):
        fpath = str(tmp_path / "rt.json")
        data = [{"a": 1, "b": [2, 3]}, {"c": {"nested": True}}]
        export_to_json(data, fpath)
        assert import_from_json(fpath) == data

    def test_export_empty_list(self, tmp_path):
        fpath = str(tmp_path / "empty.json")
        count = export_to_json([], fpath)
        assert count == 0
        assert import_from_json(fpath) == []

    def test_import_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_from_json(str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# validate_migration_plan
# ---------------------------------------------------------------------------


class TestValidateMigrationPlan:
    def test_valid(self, tmp_path):
        src = tmp_path / "src.json"
        src.write_text("[]")
        plan = MigrationPlan(source_path=str(src), dest_path=str(tmp_path / "dst.json"))
        ok, errors = validate_migration_plan(plan)
        assert ok is True
        assert errors == []

    def test_invalid_missing_source(self, tmp_path):
        plan = MigrationPlan(
            source_path=str(tmp_path / "nope"), dest_path=str(tmp_path / "dst")
        )
        ok, errors = validate_migration_plan(plan)
        assert ok is False
        assert any("does not exist" in e for e in errors)

    def test_invalid_format(self, tmp_path):
        src = tmp_path / "s.json"
        src.write_text("[]")
        plan = MigrationPlan(source_path=str(src), dest_path="/dst", format="xml")
        ok, errors = validate_migration_plan(plan)
        assert ok is False
        assert any("unsupported format" in e for e in errors)

    def test_empty_source_path(self):
        plan = MigrationPlan(source_path="", dest_path="/dst")
        ok, errors = validate_migration_plan(plan)
        assert ok is False
        assert any("source_path is empty" in e for e in errors)

    def test_empty_dest_path(self, tmp_path):
        src = tmp_path / "s.json"
        src.write_text("[]")
        plan = MigrationPlan(source_path=str(src), dest_path="")
        ok, errors = validate_migration_plan(plan)
        assert ok is False
        assert any("dest_path is empty" in e for e in errors)


# ---------------------------------------------------------------------------
# compute_migration_size
# ---------------------------------------------------------------------------


class TestComputeMigrationSize:
    def test_nonempty(self):
        items = [{"a": 1}, {"b": 2}]
        size = compute_migration_size(items)
        assert size > 0
        # Should match the byte length of json.dumps
        assert size == len(json.dumps(items).encode("utf-8"))

    def test_empty(self):
        assert compute_migration_size([]) == len(json.dumps([]).encode("utf-8"))
