"""Tests for datasets_schema_evolution module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_schema_evolution import (
    SchemaField,
    SchemaVersion,
    SchemaMigration,
    MigrationResult,
    detect_schema,
    compare_schemas,
    apply_migration,
    validate_against_schema,
    build_migration_chain,
)


# ---------------------------------------------------------------------------
# SchemaField
# ---------------------------------------------------------------------------


class TestSchemaField:
    def test_as_dict_defaults(self):
        f = SchemaField(name="x")
        d = f.as_dict()
        assert d == {"name": "x", "field_type": "str", "required": True, "default": None}

    def test_from_dict_roundtrip(self):
        original = SchemaField(name="score", field_type="float", required=False, default=0.0)
        restored = SchemaField.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.field_type == original.field_type
        assert restored.required == original.required
        assert restored.default == original.default

    def test_from_dict_minimal(self):
        f = SchemaField.from_dict({"name": "id"})
        assert f.field_type == "str"
        assert f.required is True


# ---------------------------------------------------------------------------
# SchemaVersion
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_as_dict_empty_fields(self):
        sv = SchemaVersion(version=1)
        d = sv.as_dict()
        assert d["version"] == 1
        assert d["fields"] == []
        assert d["description"] == ""

    def test_from_dict_roundtrip(self):
        original = SchemaVersion(
            version=2,
            fields=[SchemaField(name="a"), SchemaField(name="b", field_type="int")],
            description="v2 schema",
        )
        restored = SchemaVersion.from_dict(original.as_dict())
        assert restored.version == 2
        assert len(restored.fields) == 2
        assert restored.description == "v2 schema"

    def test_from_dict_missing_optional(self):
        sv = SchemaVersion.from_dict({"version": 3})
        assert sv.fields == []
        assert sv.description == ""


# ---------------------------------------------------------------------------
# SchemaMigration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_as_dict_roundtrip(self):
        mig = SchemaMigration(
            from_version=1,
            to_version=2,
            add_fields=[SchemaField(name="new_col", default="n/a")],
            remove_fields=["old_col"],
            rename_fields={"name": "full_name"},
        )
        restored = SchemaMigration.from_dict(mig.as_dict())
        assert restored.from_version == 1
        assert restored.to_version == 2
        assert len(restored.add_fields) == 1
        assert restored.remove_fields == ["old_col"]
        assert restored.rename_fields == {"name": "full_name"}


# ---------------------------------------------------------------------------
# MigrationResult
# ---------------------------------------------------------------------------


class TestMigrationResult:
    def test_as_dict(self):
        r = MigrationResult(migrated_count=5, from_version=1, to_version=2)
        d = r.as_dict()
        assert d["migrated_count"] == 5
        assert d["errors"] == []

    def test_format_text_no_errors(self):
        r = MigrationResult(migrated_count=10, from_version=1, to_version=2)
        text = r.format_text()
        assert "v1 -> v2" in text
        assert "10 records migrated" in text

    def test_format_text_with_errors(self):
        r = MigrationResult(
            migrated_count=8,
            errors=["Record 2: bad value", "Record 5: missing key"],
            from_version=1,
            to_version=2,
        )
        text = r.format_text()
        assert "2 error(s)" in text
        assert "Record 2" in text


# ---------------------------------------------------------------------------
# detect_schema
# ---------------------------------------------------------------------------


class TestDetectSchema:
    def test_infers_fields(self):
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        schema = detect_schema(records)
        assert schema.version == 1
        names = {f.name for f in schema.fields}
        assert "name" in names
        assert "age" in names

    def test_infers_types(self):
        records = [{"x": 1, "y": 2.5, "z": True}]
        schema = detect_schema(records)
        type_map = {f.name: f.field_type for f in schema.fields}
        assert type_map["x"] == "int"
        assert type_map["y"] == "float"
        assert type_map["z"] == "bool"

    def test_optional_field_detection(self):
        records = [
            {"a": 1, "b": 2},
            {"a": 3},
        ]
        schema = detect_schema(records)
        field_map = {f.name: f for f in schema.fields}
        assert field_map["a"].required is True
        assert field_map["b"].required is False

    def test_empty_records(self):
        schema = detect_schema([])
        assert schema.version == 1
        assert schema.fields == []

    def test_single_record_single_field(self):
        schema = detect_schema([{"id": "abc"}])
        assert len(schema.fields) == 1
        assert schema.fields[0].name == "id"
        assert schema.fields[0].required is True


# ---------------------------------------------------------------------------
# compare_schemas
# ---------------------------------------------------------------------------


class TestCompareSchemas:
    def test_finds_added_fields(self):
        old = SchemaVersion(version=1, fields=[SchemaField(name="a")])
        new = SchemaVersion(version=2, fields=[SchemaField(name="a"), SchemaField(name="b")])
        mig = compare_schemas(old, new)
        assert mig.from_version == 1
        assert mig.to_version == 2
        added_names = [f.name for f in mig.add_fields]
        assert "b" in added_names

    def test_finds_removed_fields(self):
        old = SchemaVersion(version=1, fields=[SchemaField(name="a"), SchemaField(name="b")])
        new = SchemaVersion(version=2, fields=[SchemaField(name="a")])
        mig = compare_schemas(old, new)
        assert "b" in mig.remove_fields

    def test_finds_renames(self):
        old = SchemaVersion(
            version=1,
            fields=[SchemaField(name="name", field_type="str")],
        )
        new = SchemaVersion(
            version=2,
            fields=[SchemaField(name="full_name", field_type="str")],
        )
        mig = compare_schemas(old, new)
        assert mig.rename_fields == {"name": "full_name"}
        # Renamed fields should not also appear as added or removed
        assert mig.add_fields == []
        assert mig.remove_fields == []

    def test_identical_schemas(self):
        s = SchemaVersion(version=1, fields=[SchemaField(name="a")])
        mig = compare_schemas(s, s)
        assert mig.add_fields == []
        assert mig.remove_fields == []
        assert mig.rename_fields == {}


# ---------------------------------------------------------------------------
# apply_migration
# ---------------------------------------------------------------------------


class TestApplyMigration:
    def test_adds_defaults(self):
        records = [{"a": 1}]
        mig = SchemaMigration(
            from_version=1,
            to_version=2,
            add_fields=[SchemaField(name="b", default=99)],
        )
        migrated, result = apply_migration(records, mig)
        assert migrated[0]["b"] == 99
        assert result.migrated_count == 1

    def test_removes_fields(self):
        records = [{"a": 1, "b": 2}]
        mig = SchemaMigration(from_version=1, to_version=2, remove_fields=["b"])
        migrated, result = apply_migration(records, mig)
        assert "b" not in migrated[0]
        assert migrated[0]["a"] == 1

    def test_renames_fields(self):
        records = [{"name": "Alice"}]
        mig = SchemaMigration(
            from_version=1,
            to_version=2,
            rename_fields={"name": "full_name"},
        )
        migrated, result = apply_migration(records, mig)
        assert migrated[0]["full_name"] == "Alice"
        assert "name" not in migrated[0]

    def test_combined_operations(self):
        records = [{"a": 1, "old": "x"}]
        mig = SchemaMigration(
            from_version=1,
            to_version=2,
            add_fields=[SchemaField(name="new_field", default="default")],
            remove_fields=["old"],
            rename_fields={"a": "alpha"},
        )
        migrated, result = apply_migration(records, mig)
        assert migrated[0] == {"alpha": 1, "new_field": "default"}

    def test_empty_records(self):
        mig = SchemaMigration(from_version=1, to_version=2)
        migrated, result = apply_migration([], mig)
        assert migrated == []
        assert result.migrated_count == 0


# ---------------------------------------------------------------------------
# validate_against_schema
# ---------------------------------------------------------------------------


class TestValidateAgainstSchema:
    def test_valid_records(self):
        schema = SchemaVersion(
            version=1,
            fields=[SchemaField(name="a"), SchemaField(name="b")],
        )
        records = [{"a": 1, "b": 2}]
        valid, errors = validate_against_schema(records, schema)
        assert valid is True
        assert errors == []

    def test_missing_required_field(self):
        schema = SchemaVersion(
            version=1,
            fields=[SchemaField(name="a", required=True), SchemaField(name="b", required=True)],
        )
        records = [{"a": 1}]
        valid, errors = validate_against_schema(records, schema)
        assert valid is False
        assert any("'b'" in e for e in errors)

    def test_optional_field_not_required(self):
        schema = SchemaVersion(
            version=1,
            fields=[
                SchemaField(name="a", required=True),
                SchemaField(name="b", required=False),
            ],
        )
        records = [{"a": 1}]
        valid, errors = validate_against_schema(records, schema)
        assert valid is True

    def test_empty_records_valid(self):
        schema = SchemaVersion(version=1, fields=[SchemaField(name="a")])
        valid, errors = validate_against_schema([], schema)
        assert valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# build_migration_chain
# ---------------------------------------------------------------------------


class TestBuildMigrationChain:
    def test_sequential_chain(self):
        v1 = SchemaVersion(version=1, fields=[SchemaField(name="a")])
        v2 = SchemaVersion(version=2, fields=[SchemaField(name="a"), SchemaField(name="b")])
        v3 = SchemaVersion(version=3, fields=[SchemaField(name="b"), SchemaField(name="c")])
        chain = build_migration_chain([v1, v2, v3])
        assert len(chain) == 2
        assert chain[0].from_version == 1
        assert chain[0].to_version == 2
        assert chain[1].from_version == 2
        assert chain[1].to_version == 3

    def test_single_version_no_chain(self):
        v1 = SchemaVersion(version=1, fields=[SchemaField(name="a")])
        chain = build_migration_chain([v1])
        assert chain == []

    def test_empty_versions(self):
        chain = build_migration_chain([])
        assert chain == []

    def test_unsorted_input(self):
        v3 = SchemaVersion(version=3, fields=[SchemaField(name="c")])
        v1 = SchemaVersion(version=1, fields=[SchemaField(name="a")])
        v2 = SchemaVersion(version=2, fields=[SchemaField(name="b")])
        chain = build_migration_chain([v3, v1, v2])
        assert chain[0].from_version == 1
        assert chain[1].from_version == 2
