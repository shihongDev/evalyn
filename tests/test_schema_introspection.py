"""
Tests for evalyn_sdk.storage.schema_introspection.

Run with: uv run pytest tests/test_schema_introspection.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.schema_introspection import (
    ColumnInfo,
    SchemaReport,
    TableSchema,
    build_schema_report,
    build_table_schema,
    compare_schemas,
    render_schema_ascii,
    suggest_schema_optimizations,
)


# ---------------------------------------------------------------------------
# ColumnInfo
# ---------------------------------------------------------------------------


class TestColumnInfo:
    def test_as_dict(self):
        col = ColumnInfo(name="id", data_type="INTEGER", nullable=False, primary_key=True)
        d = col.as_dict()
        assert d["name"] == "id"
        assert d["data_type"] == "INTEGER"
        assert d["nullable"] is False
        assert d["primary_key"] is True

    def test_defaults(self):
        col = ColumnInfo(name="value", data_type="TEXT")
        assert col.nullable is True
        assert col.primary_key is False

    def test_as_dict_defaults(self):
        col = ColumnInfo(name="x", data_type="REAL")
        d = col.as_dict()
        assert d["nullable"] is True
        assert d["primary_key"] is False


# ---------------------------------------------------------------------------
# TableSchema
# ---------------------------------------------------------------------------


class TestTableSchema:
    def test_as_dict(self):
        ts = TableSchema(
            name="runs",
            columns=[ColumnInfo("id", "INTEGER", False, True)],
            row_count=50,
            index_count=1,
        )
        d = ts.as_dict()
        assert d["name"] == "runs"
        assert len(d["columns"]) == 1
        assert d["row_count"] == 50
        assert d["index_count"] == 1

    def test_from_dict(self):
        data = {
            "name": "metrics",
            "columns": [
                {"name": "id", "data_type": "INTEGER", "nullable": False, "primary_key": True},
                {"name": "value", "data_type": "REAL"},
            ],
            "row_count": 100,
            "index_count": 2,
        }
        ts = TableSchema.from_dict(data)
        assert ts.name == "metrics"
        assert len(ts.columns) == 2
        assert ts.columns[0].primary_key is True
        assert ts.columns[1].nullable is True
        assert ts.row_count == 100

    def test_from_dict_defaults(self):
        ts = TableSchema.from_dict({"name": "empty"})
        assert ts.columns == []
        assert ts.row_count == 0
        assert ts.index_count == 0

    def test_roundtrip(self):
        ts = TableSchema(
            name="spans",
            columns=[
                ColumnInfo("id", "TEXT", False, True),
                ColumnInfo("data", "BLOB", True, False),
            ],
            row_count=200,
            index_count=3,
        )
        restored = TableSchema.from_dict(ts.as_dict())
        assert restored.name == ts.name
        assert len(restored.columns) == len(ts.columns)
        assert restored.row_count == ts.row_count
        assert restored.index_count == ts.index_count
        assert restored.columns[0].name == "id"
        assert restored.columns[1].nullable is True


# ---------------------------------------------------------------------------
# SchemaReport
# ---------------------------------------------------------------------------


class TestSchemaReport:
    def test_as_dict(self):
        report = SchemaReport(
            tables=[TableSchema("t1", [], 10, 1)],
            total_tables=1,
            total_columns=0,
            total_rows=10,
            total_indexes=1,
        )
        d = report.as_dict()
        assert d["total_tables"] == 1
        assert len(d["tables"]) == 1

    def test_from_dict(self):
        data = {
            "tables": [{"name": "t1", "columns": [], "row_count": 5, "index_count": 0}],
            "total_tables": 1,
            "total_columns": 0,
            "total_rows": 5,
            "total_indexes": 0,
        }
        r = SchemaReport.from_dict(data)
        assert r.total_tables == 1
        assert r.tables[0].name == "t1"

    def test_from_dict_defaults(self):
        r = SchemaReport.from_dict({})
        assert r.tables == []
        assert r.total_tables == 0
        assert r.total_columns == 0

    def test_roundtrip(self):
        report = SchemaReport(
            tables=[
                TableSchema("a", [ColumnInfo("id", "INT")], 10, 1),
                TableSchema("b", [ColumnInfo("name", "TEXT")], 20, 0),
            ],
            total_tables=2,
            total_columns=2,
            total_rows=30,
            total_indexes=1,
        )
        restored = SchemaReport.from_dict(report.as_dict())
        assert restored.total_tables == 2
        assert restored.total_rows == 30
        assert len(restored.tables) == 2

    def test_format_text(self):
        report = SchemaReport(
            tables=[
                TableSchema(
                    "users",
                    [ColumnInfo("id", "INTEGER", False, True), ColumnInfo("name", "TEXT")],
                    50,
                    1,
                ),
            ],
            total_tables=1,
            total_columns=2,
            total_rows=50,
            total_indexes=1,
        )
        text = report.format_text()
        assert "Schema Report" in text
        assert "Tables: 1" in text
        assert "Total columns: 2" in text
        assert "users" in text
        assert "[PK]" in text
        assert "NOT NULL" in text

    def test_format_text_empty(self):
        report = SchemaReport()
        text = report.format_text()
        assert "Schema Report" in text
        assert "Tables: 0" in text


# ---------------------------------------------------------------------------
# build_table_schema
# ---------------------------------------------------------------------------


class TestBuildTableSchema:
    def test_basic(self):
        cols = [
            {"name": "id", "type": "INTEGER", "nullable": False, "pk": True},
            {"name": "value", "type": "REAL"},
        ]
        ts = build_table_schema("metrics", cols, row_count=100)
        assert ts.name == "metrics"
        assert len(ts.columns) == 2
        assert ts.columns[0].primary_key is True
        assert ts.columns[0].nullable is False
        assert ts.columns[1].data_type == "REAL"
        assert ts.columns[1].nullable is True
        assert ts.row_count == 100

    def test_empty_columns(self):
        ts = build_table_schema("empty", [])
        assert ts.name == "empty"
        assert ts.columns == []
        assert ts.row_count == 0

    def test_minimal_column_dict(self):
        cols = [{"name": "x", "type": "TEXT"}]
        ts = build_table_schema("t", cols)
        assert ts.columns[0].nullable is True
        assert ts.columns[0].primary_key is False


# ---------------------------------------------------------------------------
# build_schema_report
# ---------------------------------------------------------------------------


class TestBuildSchemaReport:
    def test_aggregates(self):
        tables = [
            TableSchema("a", [ColumnInfo("id", "INT"), ColumnInfo("v", "TEXT")], 10, 1),
            TableSchema("b", [ColumnInfo("id", "INT")], 20, 2),
        ]
        report = build_schema_report(tables)
        assert report.total_tables == 2
        assert report.total_columns == 3
        assert report.total_rows == 30
        assert report.total_indexes == 3

    def test_empty_tables(self):
        report = build_schema_report([])
        assert report.total_tables == 0
        assert report.total_columns == 0
        assert report.total_rows == 0

    def test_single_table(self):
        tables = [TableSchema("only", [ColumnInfo("x", "INT")], 5, 0)]
        report = build_schema_report(tables)
        assert report.total_tables == 1
        assert report.total_columns == 1


# ---------------------------------------------------------------------------
# render_schema_ascii
# ---------------------------------------------------------------------------


class TestRenderSchemaAscii:
    def test_output_contains_table_names(self):
        tables = [
            TableSchema("users", [ColumnInfo("id", "INT")], 100, 1),
            TableSchema("orders", [ColumnInfo("id", "INT"), ColumnInfo("total", "REAL")], 200, 0),
        ]
        report = build_schema_report(tables)
        text = render_schema_ascii(report)
        assert "users" in text
        assert "orders" in text
        assert "TOTAL" in text

    def test_empty_report(self):
        report = build_schema_report([])
        text = render_schema_ascii(report)
        assert "TOTAL" in text
        assert "Table" in text


# ---------------------------------------------------------------------------
# compare_schemas
# ---------------------------------------------------------------------------


class TestCompareSchemas:
    def test_identical(self):
        tables = [TableSchema("t1", [ColumnInfo("id", "INT")], 10, 0)]
        a = build_schema_report(tables)
        b = build_schema_report(tables)
        diff = compare_schemas(a, b)
        assert diff["added_tables"] == []
        assert diff["removed_tables"] == []
        assert diff["changed_tables"] == []

    def test_finds_added_table(self):
        a = build_schema_report([TableSchema("t1", [], 0, 0)])
        b = build_schema_report([
            TableSchema("t1", [], 0, 0),
            TableSchema("t2", [], 0, 0),
        ])
        diff = compare_schemas(a, b)
        assert "t2" in diff["added_tables"]
        assert diff["removed_tables"] == []

    def test_finds_removed_table(self):
        a = build_schema_report([
            TableSchema("t1", [], 0, 0),
            TableSchema("t2", [], 0, 0),
        ])
        b = build_schema_report([TableSchema("t1", [], 0, 0)])
        diff = compare_schemas(a, b)
        assert "t2" in diff["removed_tables"]

    def test_finds_changed_columns(self):
        a = build_schema_report([
            TableSchema("t1", [ColumnInfo("id", "INT")], 10, 0),
        ])
        b = build_schema_report([
            TableSchema("t1", [ColumnInfo("id", "INT"), ColumnInfo("name", "TEXT")], 10, 0),
        ])
        diff = compare_schemas(a, b)
        assert len(diff["changed_tables"]) == 1
        assert "name" in diff["changed_tables"][0]["added_columns"]

    def test_finds_row_count_change(self):
        a = build_schema_report([TableSchema("t1", [ColumnInfo("id", "INT")], 10, 0)])
        b = build_schema_report([TableSchema("t1", [ColumnInfo("id", "INT")], 50, 0)])
        diff = compare_schemas(a, b)
        assert len(diff["changed_tables"]) == 1
        assert diff["changed_tables"][0]["row_count_a"] == 10
        assert diff["changed_tables"][0]["row_count_b"] == 50


# ---------------------------------------------------------------------------
# suggest_schema_optimizations
# ---------------------------------------------------------------------------


class TestSuggestSchemaOptimizations:
    def test_no_indexes(self):
        tables = [TableSchema("big", [ColumnInfo("id", "INT")], 100, 0)]
        report = build_schema_report(tables)
        suggestions = suggest_schema_optimizations(report)
        assert any("no indexes" in s for s in suggestions)

    def test_no_primary_key(self):
        tables = [TableSchema("t", [ColumnInfo("val", "TEXT", True, False)], 10, 1)]
        report = build_schema_report(tables)
        suggestions = suggest_schema_optimizations(report)
        assert any("no primary key" in s for s in suggestions)

    def test_all_nullable(self):
        tables = [
            TableSchema(
                "t",
                [ColumnInfo("a", "TEXT", True, False), ColumnInfo("b", "TEXT", True, False)],
                10,
                1,
            )
        ]
        report = build_schema_report(tables)
        suggestions = suggest_schema_optimizations(report)
        assert any("nullable" in s.lower() for s in suggestions)

    def test_well_optimized(self):
        tables = [
            TableSchema(
                "t",
                [ColumnInfo("id", "INT", False, True), ColumnInfo("val", "TEXT", False, False)],
                10,
                1,
            )
        ]
        report = build_schema_report(tables)
        suggestions = suggest_schema_optimizations(report)
        assert suggestions == []

    def test_empty_report(self):
        report = build_schema_report([])
        suggestions = suggest_schema_optimizations(report)
        assert suggestions == []
