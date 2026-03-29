"""Tests for the parquet export module."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.parquet_export import (
    EVAL_RESULT_SCHEMA,
    ParquetDataset,
    ParquetRow,
    ParquetSchema,
    build_parquet_dataset,
    check_pyarrow_available,
    convert_results_to_rows,
    export_as_csv_fallback,
    export_as_json_rows,
    format_schema_info,
)


# ---------------------------------------------------------------------------
# ParquetSchema
# ---------------------------------------------------------------------------


class TestParquetSchema:
    def test_basic_creation(self):
        s = ParquetSchema(columns=[("a", "str"), ("b", "int")])
        assert len(s.columns) == 2
        assert s.description == ""

    def test_with_description(self):
        s = ParquetSchema(columns=[], description="test desc")
        assert s.description == "test desc"

    def test_as_dict(self):
        s = ParquetSchema(columns=[("x", "float")], description="d")
        d = s.as_dict()
        assert d["columns"] == [["x", "float"]]
        assert d["description"] == "d"

    def test_from_dict(self):
        d = {"columns": [["a", "str"], ["b", "int"]], "description": "desc"}
        s = ParquetSchema.from_dict(d)
        assert len(s.columns) == 2
        assert s.columns[0] == ("a", "str")
        assert s.description == "desc"

    def test_from_dict_no_description(self):
        d = {"columns": [["a", "str"]]}
        s = ParquetSchema.from_dict(d)
        assert s.description == ""

    def test_roundtrip(self):
        s = ParquetSchema(columns=[("a", "str"), ("b", "float")], description="rt")
        restored = ParquetSchema.from_dict(s.as_dict())
        assert restored.columns == s.columns
        assert restored.description == s.description


# ---------------------------------------------------------------------------
# ParquetRow
# ---------------------------------------------------------------------------


class TestParquetRow:
    def test_basic_creation(self):
        r = ParquetRow(values={"a": 1, "b": "two"})
        assert r.values["a"] == 1

    def test_as_dict(self):
        r = ParquetRow(values={"x": 42})
        d = r.as_dict()
        assert d == {"values": {"x": 42}}

    def test_from_dict(self):
        r = ParquetRow.from_dict({"values": {"k": "v"}})
        assert r.values["k"] == "v"

    def test_roundtrip(self):
        r = ParquetRow(values={"a": 1, "b": "two", "c": True})
        restored = ParquetRow.from_dict(r.as_dict())
        assert restored.values == r.values

    def test_as_dict_copies_values(self):
        r = ParquetRow(values={"a": 1})
        d = r.as_dict()
        d["values"]["a"] = 999
        assert r.values["a"] == 1


# ---------------------------------------------------------------------------
# ParquetDataset
# ---------------------------------------------------------------------------


class TestParquetDataset:
    def test_basic_creation(self):
        s = ParquetSchema(columns=[("a", "str")])
        ds = ParquetDataset(schema=s, rows=[], row_count=0)
        assert ds.row_count == 0
        assert ds.rows == []

    def test_as_dict(self):
        s = ParquetSchema(columns=[("a", "str")])
        r = ParquetRow(values={"a": "hello"})
        ds = ParquetDataset(schema=s, rows=[r], row_count=1)
        d = ds.as_dict()
        assert d["row_count"] == 1
        assert len(d["rows"]) == 1
        assert d["schema"]["columns"] == [["a", "str"]]

    def test_from_dict(self):
        d = {
            "schema": {"columns": [["a", "str"]]},
            "rows": [{"values": {"a": "hi"}}],
            "row_count": 1,
        }
        ds = ParquetDataset.from_dict(d)
        assert ds.row_count == 1
        assert ds.rows[0].values["a"] == "hi"

    def test_roundtrip(self):
        s = ParquetSchema(columns=[("x", "int")], description="test")
        rows = [ParquetRow(values={"x": i}) for i in range(3)]
        ds = ParquetDataset(schema=s, rows=rows, row_count=3)
        restored = ParquetDataset.from_dict(ds.as_dict())
        assert restored.row_count == ds.row_count
        assert len(restored.rows) == len(ds.rows)
        for orig, rest in zip(ds.rows, restored.rows):
            assert orig.values == rest.values


# ---------------------------------------------------------------------------
# EVAL_RESULT_SCHEMA
# ---------------------------------------------------------------------------


class TestEvalResultSchema:
    def test_has_eight_columns(self):
        assert len(EVAL_RESULT_SCHEMA.columns) == 8

    def test_column_names(self):
        names = [c[0] for c in EVAL_RESULT_SCHEMA.columns]
        expected = [
            "item_id", "metric_id", "score", "passed",
            "input_text", "output_text", "run_id", "timestamp",
        ]
        assert names == expected

    def test_has_description(self):
        assert EVAL_RESULT_SCHEMA.description != ""

    def test_column_types(self):
        types = {c[0]: c[1] for c in EVAL_RESULT_SCHEMA.columns}
        assert types["score"] == "float"
        assert types["passed"] == "bool"
        assert types["item_id"] == "str"


# ---------------------------------------------------------------------------
# convert_results_to_rows
# ---------------------------------------------------------------------------


class TestConvertResultsToRows:
    def test_empty_results(self):
        rows = convert_results_to_rows([])
        assert rows == []

    def test_single_result(self):
        results = [{"item_id": "i1", "metric_id": "m1", "score": 0.9, "passed": True}]
        rows = convert_results_to_rows(results)
        assert len(rows) == 1
        assert rows[0].values["item_id"] == "i1"
        assert rows[0].values["score"] == 0.9

    def test_missing_fields_get_defaults(self):
        rows = convert_results_to_rows([{}])
        assert rows[0].values["item_id"] == ""
        assert rows[0].values["score"] == 0.0
        assert rows[0].values["passed"] is False

    def test_join_with_items(self):
        results = [{"item_id": "i1", "metric_id": "m1", "score": 1.0, "passed": True}]
        items = [{"item_id": "i1", "input_text": "hello", "output_text": "world"}]
        rows = convert_results_to_rows(results, items)
        assert rows[0].values["input_text"] == "hello"
        assert rows[0].values["output_text"] == "world"

    def test_item_text_overrides_result_text(self):
        results = [
            {"item_id": "i1", "input_text": "from_result", "output_text": "from_result"}
        ]
        items = [
            {"item_id": "i1", "input_text": "from_item", "output_text": "from_item"}
        ]
        rows = convert_results_to_rows(results, items)
        assert rows[0].values["input_text"] == "from_item"

    def test_no_matching_item(self):
        results = [{"item_id": "i1", "input_text": "fallback"}]
        items = [{"item_id": "i2", "input_text": "other"}]
        rows = convert_results_to_rows(results, items)
        assert rows[0].values["input_text"] == "fallback"

    def test_multiple_results(self):
        results = [
            {"item_id": f"i{n}", "metric_id": "m", "score": float(n)}
            for n in range(5)
        ]
        rows = convert_results_to_rows(results)
        assert len(rows) == 5
        assert rows[2].values["score"] == 2.0

    def test_items_none(self):
        results = [{"item_id": "i1"}]
        rows = convert_results_to_rows(results, None)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# build_parquet_dataset
# ---------------------------------------------------------------------------


class TestBuildParquetDataset:
    def test_empty(self):
        ds = build_parquet_dataset([])
        assert ds.row_count == 0
        assert ds.rows == []
        assert ds.schema is EVAL_RESULT_SCHEMA

    def test_uses_default_schema(self):
        ds = build_parquet_dataset([{"item_id": "x"}])
        assert ds.schema is EVAL_RESULT_SCHEMA

    def test_custom_schema(self):
        custom = ParquetSchema(columns=[("a", "str")])
        ds = build_parquet_dataset([{"item_id": "x"}], schema=custom)
        assert ds.schema is custom

    def test_row_count_matches(self):
        results = [{"item_id": f"i{n}"} for n in range(7)]
        ds = build_parquet_dataset(results)
        assert ds.row_count == 7
        assert len(ds.rows) == 7

    def test_with_items(self):
        results = [{"item_id": "i1", "score": 0.5}]
        items = [{"item_id": "i1", "input_text": "inp"}]
        ds = build_parquet_dataset(results, items)
        assert ds.rows[0].values["input_text"] == "inp"


# ---------------------------------------------------------------------------
# export_as_csv_fallback
# ---------------------------------------------------------------------------


class TestExportAsCsvFallback:
    def test_empty_dataset(self):
        ds = build_parquet_dataset([])
        csv_str = export_as_csv_fallback(ds)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1  # header only
        assert rows[0][0] == "item_id"

    def test_single_row(self):
        ds = build_parquet_dataset([{"item_id": "i1", "score": 0.8}])
        csv_str = export_as_csv_fallback(ds)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2
        header = rows[0]
        assert "item_id" in header
        assert "score" in header

    def test_values_in_output(self):
        ds = build_parquet_dataset([{"item_id": "abc", "metric_id": "m1"}])
        csv_str = export_as_csv_fallback(ds)
        assert "abc" in csv_str
        assert "m1" in csv_str

    def test_multiple_rows(self):
        results = [{"item_id": f"i{n}"} for n in range(3)]
        ds = build_parquet_dataset(results)
        csv_str = export_as_csv_fallback(ds)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 4  # header + 3 data rows


# ---------------------------------------------------------------------------
# export_as_json_rows
# ---------------------------------------------------------------------------


class TestExportAsJsonRows:
    def test_empty_dataset(self):
        ds = build_parquet_dataset([])
        result = export_as_json_rows(ds)
        parsed = json.loads(result)
        assert parsed == []

    def test_single_row(self):
        ds = build_parquet_dataset([{"item_id": "j1", "score": 0.5}])
        result = export_as_json_rows(ds)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["item_id"] == "j1"
        assert parsed[0]["score"] == 0.5

    def test_valid_json(self):
        results = [{"item_id": f"i{n}"} for n in range(5)]
        ds = build_parquet_dataset(results)
        result = export_as_json_rows(ds)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 5

    def test_preserves_types(self):
        ds = build_parquet_dataset([{"item_id": "x", "score": 1.5, "passed": True}])
        result = export_as_json_rows(ds)
        parsed = json.loads(result)
        assert isinstance(parsed[0]["score"], float)
        assert isinstance(parsed[0]["passed"], bool)


# ---------------------------------------------------------------------------
# check_pyarrow_available
# ---------------------------------------------------------------------------


class TestCheckPyarrowAvailable:
    def test_returns_bool(self):
        result = check_pyarrow_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# format_schema_info
# ---------------------------------------------------------------------------


class TestFormatSchemaInfo:
    def test_includes_column_names(self):
        s = ParquetSchema(columns=[("a", "str"), ("b", "int")])
        result = format_schema_info(s)
        assert "a (str)" in result
        assert "b (int)" in result

    def test_includes_description(self):
        s = ParquetSchema(columns=[], description="My description")
        result = format_schema_info(s)
        assert "My description" in result

    def test_no_description(self):
        s = ParquetSchema(columns=[("a", "str")])
        result = format_schema_info(s)
        assert "Columns:" in result

    def test_eval_result_schema(self):
        result = format_schema_info(EVAL_RESULT_SCHEMA)
        assert "item_id" in result
        assert "score" in result
        assert "passed" in result
