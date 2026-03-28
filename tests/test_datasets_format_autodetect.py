from __future__ import annotations

import json
import os

from evalyn_sdk.datasets_format_autodetect import (
    AutoloadResult,
    FormatDetection,
    autoload,
    autoload_from_string,
    detect_format,
    detect_format_from_content,
    detect_format_from_extension,
    supported_formats,
)


# ---------------------------------------------------------------------------
# detect_format_from_extension
# ---------------------------------------------------------------------------


class TestDetectFormatFromExtension:
    def test_json(self):
        result = detect_format_from_extension("data.json")
        assert result.format == "json"
        assert result.confidence > 0.5

    def test_csv(self):
        result = detect_format_from_extension("data.csv")
        assert result.format == "csv"
        assert result.confidence > 0.5

    def test_jsonl(self):
        result = detect_format_from_extension("data.jsonl")
        assert result.format == "jsonl"

    def test_tsv(self):
        result = detect_format_from_extension("data.tsv")
        assert result.format == "tsv"

    def test_yaml(self):
        result = detect_format_from_extension("config.yaml")
        assert result.format == "yaml"

    def test_yml(self):
        result = detect_format_from_extension("config.yml")
        assert result.format == "yaml"

    def test_ndjson(self):
        result = detect_format_from_extension("data.ndjson")
        assert result.format == "jsonl"

    def test_unknown(self):
        result = detect_format_from_extension("data.xyz")
        assert result.format == "unknown"
        assert result.confidence == 0.0

    def test_no_extension(self):
        result = detect_format_from_extension("datafile")
        assert result.format == "unknown"


# ---------------------------------------------------------------------------
# detect_format_from_content
# ---------------------------------------------------------------------------


class TestDetectFormatFromContent:
    def test_json_array(self):
        content = '[{"a": 1}, {"a": 2}]'
        result = detect_format_from_content(content)
        assert result.format == "json"
        assert result.confidence > 0.8

    def test_json_object(self):
        content = '{"key": "value"}'
        result = detect_format_from_content(content)
        assert result.format == "json"

    def test_jsonl(self):
        content = '{"a": 1}\n{"a": 2}\n'
        result = detect_format_from_content(content)
        # The whole string is not valid JSON, so it falls to JSONL check
        assert result.format == "jsonl"

    def test_csv(self):
        content = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        result = detect_format_from_content(content)
        assert result.format == "csv"

    def test_tsv(self):
        content = "name\tage\tcity\nAlice\t30\tNYC\n"
        result = detect_format_from_content(content)
        assert result.format == "tsv"

    def test_empty(self):
        result = detect_format_from_content("")
        assert result.format == "unknown"
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        result = detect_format_from_content("   \n  ")
        assert result.format == "unknown"

    def test_unknown_content(self):
        result = detect_format_from_content("just some plain text without structure")
        assert result.format == "unknown"


# ---------------------------------------------------------------------------
# detect_format (combined)
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_extension_and_content_agree(self):
        content = '[{"a": 1}]'
        result = detect_format("data.json", content=content)
        assert result.format == "json"
        assert result.confidence > 0.9

    def test_extension_wins_over_content(self):
        # Extension says JSON, content looks like CSV
        result = detect_format("data.json", content="a,b,c\n1,2,3\n")
        assert result.format == "json"

    def test_content_fallback(self):
        content = '[{"key": "val"}]'
        result = detect_format("noext", content=content)
        assert result.format == "json"

    def test_unknown_both(self):
        result = detect_format("noext", content="random garbage !@#$")
        assert result.format == "unknown"


# ---------------------------------------------------------------------------
# autoload (file-based)
# ---------------------------------------------------------------------------


class TestAutoload:
    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        items = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]
        f.write_text(json.dumps(items))
        result = autoload(str(f))
        assert result.total_loaded == 2
        assert result.format_detected == "json"
        assert result.items[0]["id"] == "1"

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("id,value\n1,a\n2,b\n")
        result = autoload(str(f))
        assert result.total_loaded == 2
        assert result.format_detected == "csv"

    def test_jsonl_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"id": "1"}\n{"id": "2"}\n')
        result = autoload(str(f))
        assert result.total_loaded == 2
        assert result.format_detected == "jsonl"

    def test_single_json_object(self, tmp_path):
        f = tmp_path / "single.json"
        f.write_text('{"id": "only"}')
        result = autoload(str(f))
        assert result.total_loaded == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        result = autoload(str(f))
        assert result.total_loaded == 0
        assert len(result.errors) > 0

    def test_nonexistent_file(self):
        result = autoload("/tmp/does_not_exist_evalyn_test.json")
        assert result.total_loaded == 0
        assert len(result.errors) > 0

    def test_invalid_content(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("this is not json or csv or anything")
        result = autoload(str(f))
        assert result.total_loaded == 0


# ---------------------------------------------------------------------------
# autoload_from_string
# ---------------------------------------------------------------------------


class TestAutoloadFromString:
    def test_json_string(self):
        content = '[{"a": 1}, {"a": 2}]'
        result = autoload_from_string(content)
        assert result.total_loaded == 2
        assert result.format_detected == "json"

    def test_csv_string(self):
        content = "name,age\nAlice,30\nBob,25\n"
        result = autoload_from_string(content)
        assert result.total_loaded == 2
        assert result.format_detected == "csv"

    def test_jsonl_string(self):
        content = '{"x": 1}\n{"x": 2}\n'
        result = autoload_from_string(content)
        assert result.total_loaded == 2

    def test_with_hint(self):
        content = '[{"a": 1}]'
        result = autoload_from_string(content, hint="json")
        assert result.format_detected == "json"
        assert result.total_loaded == 1

    def test_empty_string(self):
        result = autoload_from_string("")
        assert result.total_loaded == 0
        assert len(result.errors) > 0

    def test_invalid_string(self):
        result = autoload_from_string("not valid data at all!!!")
        assert result.total_loaded == 0


# ---------------------------------------------------------------------------
# supported_formats
# ---------------------------------------------------------------------------


class TestSupportedFormats:
    def test_list(self):
        fmts = supported_formats()
        assert "json" in fmts
        assert "csv" in fmts
        assert "jsonl" in fmts

    def test_returns_list(self):
        fmts = supported_formats()
        assert isinstance(fmts, list)
        assert len(fmts) >= 3


# ---------------------------------------------------------------------------
# AutoloadResult
# ---------------------------------------------------------------------------


class TestAutoloadResult:
    def test_format_text(self):
        r = AutoloadResult(format_detected="json", total_loaded=5)
        text = r.format_text()
        assert "json" in text
        assert "5" in text

    def test_format_text_with_errors(self):
        r = AutoloadResult(format_detected="csv", total_loaded=0, errors=["parse error"])
        text = r.format_text()
        assert "parse error" in text

    def test_as_dict_from_dict(self):
        r = AutoloadResult(
            items=[{"a": 1}],
            format_detected="json",
            total_loaded=1,
            errors=["warn"],
        )
        d = r.as_dict()
        restored = AutoloadResult.from_dict(d)
        assert restored.format_detected == "json"
        assert restored.total_loaded == 1
        assert restored.errors == ["warn"]
        assert restored.items == [{"a": 1}]


# ---------------------------------------------------------------------------
# FormatDetection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_as_dict(self):
        fd = FormatDetection(format="json", confidence=0.9, evidence="test")
        d = fd.as_dict()
        assert d["format"] == "json"
        assert d["confidence"] == 0.9
        assert d["evidence"] == "test"

    def test_defaults(self):
        fd = FormatDetection()
        assert fd.format == "unknown"
        assert fd.confidence == 0.0
