"""Tests for datasets_import module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_import import (
    ImportConfig,
    ImportResult,
    ImportedItem,
    convert_to_evalyn_format,
    detect_format,
    import_from_csv,
    import_from_dict_list,
    import_from_jsonl,
    validate_import,
)


# ---------------------------------------------------------------------------
# ImportConfig
# ---------------------------------------------------------------------------


class TestImportConfig:
    def test_as_dict(self):
        c = ImportConfig(format="csv", file_path="/tmp/f.csv")
        d = c.as_dict()
        assert d["format"] == "csv"
        assert d["input_column"] == "input"
        assert d["delimiter"] == ","

    def test_from_dict(self):
        d = {
            "format": "jsonl",
            "input_column": "question",
            "output_column": "answer",
            "id_column": "idx",
            "delimiter": "\t",
        }
        c = ImportConfig.from_dict(d)
        assert c.format == "jsonl"
        assert c.input_column == "question"
        assert c.delimiter == "\t"

    def test_roundtrip(self):
        c = ImportConfig(
            format="csv",
            file_path="data.csv",
            input_column="q",
            output_column="a",
            id_column="num",
            delimiter=";",
        )
        assert ImportConfig.from_dict(c.as_dict()).as_dict() == c.as_dict()

    def test_defaults(self):
        c = ImportConfig(format="csv")
        assert c.file_path == ""
        assert c.input_column == "input"
        assert c.output_column == "output"
        assert c.id_column == "id"
        assert c.delimiter == ","


# ---------------------------------------------------------------------------
# ImportedItem
# ---------------------------------------------------------------------------


class TestImportedItem:
    def test_as_dict(self):
        item = ImportedItem(id="1", input_text="hi", output_text="bye")
        d = item.as_dict()
        assert d["id"] == "1"
        assert d["input_text"] == "hi"

    def test_from_dict(self):
        d = {
            "id": "2",
            "input_text": "hello",
            "output_text": "world",
            "metadata": {"src": "test"},
            "source_format": "csv",
        }
        item = ImportedItem.from_dict(d)
        assert item.source_format == "csv"
        assert item.metadata == {"src": "test"}

    def test_roundtrip(self):
        item = ImportedItem(
            id="3",
            input_text="q",
            output_text="a",
            metadata={"x": 1},
            source_format="jsonl",
        )
        assert ImportedItem.from_dict(item.as_dict()).as_dict() == item.as_dict()


# ---------------------------------------------------------------------------
# ImportResult
# ---------------------------------------------------------------------------


class TestImportResult:
    def test_as_dict(self):
        r = ImportResult(
            items=[ImportedItem(id="1", input_text="t")],
            total_imported=1,
            skipped=0,
            errors=[],
            source_format="csv",
        )
        d = r.as_dict()
        assert d["total_imported"] == 1
        assert len(d["items"]) == 1

    def test_from_dict(self):
        d = {
            "items": [{"id": "a", "input_text": "q"}],
            "total_imported": 1,
            "skipped": 2,
            "errors": ["bad line"],
            "source_format": "jsonl",
        }
        r = ImportResult.from_dict(d)
        assert r.skipped == 2
        assert len(r.items) == 1

    def test_roundtrip(self):
        r = ImportResult(
            items=[ImportedItem(id="x", input_text="i", output_text="o")],
            total_imported=1,
            skipped=0,
            errors=[],
            source_format="csv",
        )
        assert ImportResult.from_dict(r.as_dict()).as_dict() == r.as_dict()

    def test_format_text(self):
        r = ImportResult(
            total_imported=10, skipped=2, errors=["e1", "e2"], source_format="csv"
        )
        text = r.format_text()
        assert "Import Result" in text
        assert "Imported" in text
        assert "Skipped" in text
        assert "e1" in text

    def test_format_text_many_errors(self):
        r = ImportResult(errors=[f"err{i}" for i in range(8)])
        text = r.format_text()
        assert "and 3 more" in text


# ---------------------------------------------------------------------------
# import_from_csv
# ---------------------------------------------------------------------------


class TestImportFromCsv:
    def test_basic(self, tmp_path: Path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,input,output\n1,hello,world\n2,foo,bar\n")
        config = ImportConfig(format="csv")
        result = import_from_csv(str(csv_file), config)
        assert result.total_imported == 2
        assert result.items[0].input_text == "hello"
        assert result.items[1].output_text == "bar"

    def test_custom_columns(self, tmp_path: Path):
        csv_file = tmp_path / "custom.csv"
        csv_file.write_text("idx,question,answer\na,What?,42\n")
        config = ImportConfig(
            format="csv",
            input_column="question",
            output_column="answer",
            id_column="idx",
        )
        result = import_from_csv(str(csv_file), config)
        assert result.total_imported == 1
        assert result.items[0].id == "a"
        assert result.items[0].input_text == "What?"

    def test_tab_delimiter(self, tmp_path: Path):
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("id\tinput\toutput\n1\thello\tworld\n")
        config = ImportConfig(format="csv", delimiter="\t")
        result = import_from_csv(str(tsv_file), config)
        assert result.total_imported == 1

    def test_missing_input_skipped(self, tmp_path: Path):
        csv_file = tmp_path / "missing.csv"
        csv_file.write_text("id,input,output\n1,,world\n2,hello,bar\n")
        config = ImportConfig(format="csv")
        result = import_from_csv(str(csv_file), config)
        assert result.total_imported == 1
        assert result.skipped == 1

    def test_file_not_found(self):
        config = ImportConfig(format="csv")
        result = import_from_csv("/nonexistent/path.csv", config)
        assert result.total_imported == 0
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].lower()

    def test_empty_file(self, tmp_path: Path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        config = ImportConfig(format="csv")
        result = import_from_csv(str(csv_file), config)
        assert result.total_imported == 0

    def test_extra_columns_in_metadata(self, tmp_path: Path):
        csv_file = tmp_path / "extra.csv"
        csv_file.write_text("id,input,output,category\n1,hi,bye,greet\n")
        config = ImportConfig(format="csv")
        result = import_from_csv(str(csv_file), config)
        assert result.items[0].metadata.get("category") == "greet"

    def test_auto_id_when_missing(self, tmp_path: Path):
        csv_file = tmp_path / "noid.csv"
        csv_file.write_text("input,output\nhello,world\n")
        config = ImportConfig(format="csv")
        result = import_from_csv(str(csv_file), config)
        assert result.total_imported == 1
        assert result.items[0].id == "1"  # row number used as id


# ---------------------------------------------------------------------------
# import_from_jsonl
# ---------------------------------------------------------------------------


class TestImportFromJsonl:
    def test_basic(self, tmp_path: Path):
        jl = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"id": "1", "input": "hi", "output": "bye"}),
            json.dumps({"id": "2", "input": "foo", "output": "bar"}),
        ]
        jl.write_text("\n".join(lines) + "\n")
        config = ImportConfig(format="jsonl")
        result = import_from_jsonl(str(jl), config)
        assert result.total_imported == 2

    def test_malformed_line(self, tmp_path: Path):
        jl = tmp_path / "bad.jsonl"
        jl.write_text('{"id":"1","input":"ok"}\nnot json\n')
        config = ImportConfig(format="jsonl")
        result = import_from_jsonl(str(jl), config)
        assert result.total_imported == 1
        assert result.skipped == 1
        assert len(result.errors) == 1

    def test_empty_file(self, tmp_path: Path):
        jl = tmp_path / "empty.jsonl"
        jl.write_text("")
        config = ImportConfig(format="jsonl")
        result = import_from_jsonl(str(jl), config)
        assert result.total_imported == 0

    def test_missing_input_skipped(self, tmp_path: Path):
        jl = tmp_path / "skip.jsonl"
        jl.write_text('{"id":"1","output":"no input"}\n')
        config = ImportConfig(format="jsonl")
        result = import_from_jsonl(str(jl), config)
        assert result.total_imported == 0
        assert result.skipped == 1

    def test_file_not_found(self):
        config = ImportConfig(format="jsonl")
        result = import_from_jsonl("/nonexistent.jsonl", config)
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# import_from_dict_list
# ---------------------------------------------------------------------------


class TestImportFromDictList:
    def test_basic(self):
        records = [
            {"id": "a", "input": "q1", "output": "a1"},
            {"id": "b", "input": "q2", "output": "a2"},
        ]
        config = ImportConfig(format="custom")
        result = import_from_dict_list(records, config)
        assert result.total_imported == 2

    def test_missing_input(self):
        records = [{"id": "c", "output": "nothing"}]
        config = ImportConfig(format="custom")
        result = import_from_dict_list(records, config)
        assert result.total_imported == 0
        assert result.skipped == 1

    def test_auto_id(self):
        records = [{"input": "text"}]
        config = ImportConfig(format="custom")
        result = import_from_dict_list(records, config)
        assert result.items[0].id == "1"

    def test_empty_list(self):
        config = ImportConfig(format="custom")
        result = import_from_dict_list([], config)
        assert result.total_imported == 0


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_csv(self):
        assert detect_format("data.csv") == "csv"

    def test_jsonl(self):
        assert detect_format("data.jsonl") == "jsonl"

    def test_json(self):
        assert detect_format("data.json") == "json"

    def test_unknown(self):
        assert detect_format("data.parquet") == "unknown"

    def test_case_insensitive(self):
        assert detect_format("DATA.CSV") == "csv"
        assert detect_format("file.JSONL") == "jsonl"


# ---------------------------------------------------------------------------
# validate_import
# ---------------------------------------------------------------------------


class TestValidateImport:
    def test_valid(self):
        result = ImportResult(
            items=[ImportedItem(id="1", input_text="hi")], total_imported=1
        )
        valid, issues = validate_import(result)
        assert valid is True
        assert issues == []

    def test_missing_input(self):
        result = ImportResult(
            items=[ImportedItem(id="1", input_text="")], total_imported=1
        )
        valid, issues = validate_import(result)
        assert valid is False
        assert len(issues) == 1

    def test_missing_id(self):
        result = ImportResult(
            items=[ImportedItem(id="", input_text="hi")], total_imported=1
        )
        valid, issues = validate_import(result)
        assert valid is False

    def test_empty_result(self):
        result = ImportResult()
        valid, issues = validate_import(result)
        assert valid is True


# ---------------------------------------------------------------------------
# convert_to_evalyn_format
# ---------------------------------------------------------------------------


class TestConvertToEvalynFormat:
    def test_basic(self):
        items = [
            ImportedItem(id="1", input_text="question", output_text="answer"),
        ]
        converted = convert_to_evalyn_format(items)
        assert len(converted) == 1
        assert converted[0]["id"] == "1"
        assert converted[0]["input"] == {"text": "question"}
        assert converted[0]["output"] == "answer"

    def test_no_output(self):
        items = [ImportedItem(id="2", input_text="q")]
        converted = convert_to_evalyn_format(items)
        assert "output" not in converted[0]

    def test_metadata_included(self):
        items = [
            ImportedItem(id="3", input_text="q", metadata={"src": "test"}),
        ]
        converted = convert_to_evalyn_format(items)
        assert converted[0]["metadata"] == {"src": "test"}

    def test_empty_list(self):
        assert convert_to_evalyn_format([]) == []
