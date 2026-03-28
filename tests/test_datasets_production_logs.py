"""Tests for evalyn_sdk.datasets_production_logs.

Run with: uv run pytest tests/test_datasets_production_logs.py -v
"""

from __future__ import annotations

import json

import pytest

from evalyn_sdk.datasets_production_logs import (
    LogEntry,
    ImportedLogItem,
    LogImportResult,
    parse_log_entry,
    extract_llm_content,
    import_from_logs,
    detect_log_format,
    filter_logs_by_endpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_json(
    url="/v1/chat/completions",
    method="POST",
    status_code=200,
    request_body=None,
    response_body=None,
    timestamp="2026-03-28T10:00:00Z",
    duration_ms=150.0,
) -> str:
    """Build a JSON log line."""
    if request_body is None:
        request_body = json.dumps({
            "messages": [{"role": "user", "content": "Hello"}]
        })
    if response_body is None:
        response_body = json.dumps({
            "choices": [{"message": {"content": "Hi there"}}]
        })
    return json.dumps({
        "timestamp": timestamp,
        "method": method,
        "url": url,
        "request_body": request_body,
        "response_body": response_body,
        "status_code": status_code,
        "duration_ms": duration_ms,
    })


def _openai_request(prompt: str = "Hello") -> str:
    return json.dumps({"messages": [{"role": "user", "content": prompt}]})


def _openai_response(content: str = "Hi there") -> str:
    return json.dumps({"choices": [{"message": {"content": content}}]})


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

class TestLogEntry:
    def test_as_dict(self):
        entry = LogEntry(timestamp="t1", url="/api")
        d = entry.as_dict()
        assert d["timestamp"] == "t1"
        assert d["url"] == "/api"
        assert d["method"] == "POST"

    def test_from_dict(self):
        data = {"timestamp": "t2", "url": "/api", "status_code": 404}
        entry = LogEntry.from_dict(data)
        assert entry.timestamp == "t2"
        assert entry.status_code == 404

    def test_roundtrip(self):
        original = LogEntry(
            timestamp="t1", method="GET", url="/test",
            request_body="req", response_body="resp",
            status_code=201, duration_ms=42.5,
            headers={"Authorization": "Bearer abc"},
        )
        restored = LogEntry.from_dict(original.as_dict())
        assert restored.as_dict() == original.as_dict()

    def test_defaults(self):
        entry = LogEntry()
        assert entry.method == "POST"
        assert entry.status_code == 200
        assert entry.headers == {}


# ---------------------------------------------------------------------------
# ImportedLogItem
# ---------------------------------------------------------------------------

class TestImportedLogItem:
    def test_as_dict(self):
        item = ImportedLogItem(id="abc", input_text="hello")
        d = item.as_dict()
        assert d["id"] == "abc"
        assert d["input_text"] == "hello"

    def test_from_dict(self):
        data = {"id": "x", "input_text": "in", "output_text": "out", "metadata": {"k": 1}}
        item = ImportedLogItem.from_dict(data)
        assert item.id == "x"
        assert item.metadata["k"] == 1

    def test_roundtrip(self):
        original = ImportedLogItem(
            id="item1", input_text="in", output_text="out",
            metadata={"source": "logs"},
        )
        restored = ImportedLogItem.from_dict(original.as_dict())
        assert restored.as_dict() == original.as_dict()


# ---------------------------------------------------------------------------
# LogImportResult
# ---------------------------------------------------------------------------

class TestLogImportResult:
    def test_as_dict(self):
        result = LogImportResult(total_logs=10, imported=8, skipped=1, errors=["e1"])
        d = result.as_dict()
        assert d["total_logs"] == 10
        assert d["imported"] == 8
        assert d["errors"] == ["e1"]

    def test_from_dict(self):
        data = {
            "items": [{"id": "a", "input_text": "x"}],
            "total_logs": 5,
            "imported": 3,
            "skipped": 1,
            "errors": ["bad"],
        }
        result = LogImportResult.from_dict(data)
        assert len(result.items) == 1
        assert result.items[0].id == "a"
        assert result.skipped == 1

    def test_roundtrip(self):
        original = LogImportResult(
            items=[ImportedLogItem(id="i1", input_text="hello")],
            total_logs=1, imported=1, skipped=0, errors=[],
        )
        restored = LogImportResult.from_dict(original.as_dict())
        assert restored.as_dict() == original.as_dict()

    def test_format_text_basic(self):
        result = LogImportResult(total_logs=10, imported=8, skipped=2)
        text = result.format_text()
        assert "8/10 imported" in text
        assert "2 skipped" in text

    def test_format_text_with_errors(self):
        result = LogImportResult(
            total_logs=3, imported=1, skipped=0, errors=["err1", "err2"]
        )
        text = result.format_text()
        assert "Errors (2)" in text
        assert "err1" in text


# ---------------------------------------------------------------------------
# parse_log_entry
# ---------------------------------------------------------------------------

class TestParseLogEntry:
    def test_valid_json(self):
        line = _make_log_json()
        entry = parse_log_entry(line)
        assert entry is not None
        assert entry.status_code == 200
        assert entry.url == "/v1/chat/completions"

    def test_invalid_json_returns_none(self):
        assert parse_log_entry("not json") is None

    def test_empty_line_returns_none(self):
        assert parse_log_entry("") is None
        assert parse_log_entry("   ") is None

    def test_json_array_returns_none(self):
        assert parse_log_entry("[1, 2, 3]") is None

    def test_custom_fields(self):
        line = json.dumps({
            "timestamp": "2026-01-01",
            "method": "GET",
            "url": "/health",
            "status_code": 204,
            "duration_ms": 5.0,
        })
        entry = parse_log_entry(line)
        assert entry is not None
        assert entry.method == "GET"
        assert entry.duration_ms == 5.0

    def test_missing_fields_use_defaults(self):
        line = json.dumps({"url": "/test"})
        entry = parse_log_entry(line)
        assert entry is not None
        assert entry.method == "POST"
        assert entry.status_code == 200

    def test_unsupported_format_returns_none(self):
        assert parse_log_entry("some text", format="csv") is None


# ---------------------------------------------------------------------------
# extract_llm_content
# ---------------------------------------------------------------------------

class TestExtractLlmContent:
    def test_openai_format(self):
        entry = LogEntry(
            request_body=_openai_request("What is 2+2?"),
            response_body=_openai_response("4"),
        )
        input_text, output_text = extract_llm_content(entry)
        assert "What is 2+2?" in input_text
        assert "4" in output_text

    def test_prompt_format(self):
        entry = LogEntry(
            request_body=json.dumps({"prompt": "Tell me a joke"}),
            response_body=json.dumps({"text": "Why did the chicken..."}),
        )
        input_text, output_text = extract_llm_content(entry)
        assert input_text == "Tell me a joke"
        assert "chicken" in output_text

    def test_empty_bodies(self):
        entry = LogEntry(request_body="", response_body="")
        input_text, output_text = extract_llm_content(entry)
        assert input_text == ""
        assert output_text == ""

    def test_non_json_bodies(self):
        entry = LogEntry(request_body="not json", response_body="not json")
        input_text, output_text = extract_llm_content(entry)
        assert input_text == ""
        assert output_text == ""

    def test_choices_with_text_field(self):
        entry = LogEntry(
            request_body=_openai_request("hi"),
            response_body=json.dumps({
                "choices": [{"text": "completion text"}]
            }),
        )
        _, output_text = extract_llm_content(entry)
        assert output_text == "completion text"

    def test_multiple_messages(self):
        req = json.dumps({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        })
        entry = LogEntry(request_body=req, response_body=_openai_response("Hi"))
        input_text, _ = extract_llm_content(entry)
        assert "You are helpful" in input_text
        assert "Hello" in input_text


# ---------------------------------------------------------------------------
# import_from_logs
# ---------------------------------------------------------------------------

class TestImportFromLogs:
    def test_basic_import(self):
        lines = [_make_log_json() for _ in range(3)]
        result = import_from_logs(lines)
        assert result.total_logs == 3
        assert result.imported == 3
        assert result.skipped == 0
        assert len(result.items) == 3

    def test_filters_by_status(self):
        lines = [
            _make_log_json(status_code=200),
            _make_log_json(status_code=500),
            _make_log_json(status_code=201),
        ]
        result = import_from_logs(lines, min_status=200, max_status=299)
        assert result.imported == 2
        assert result.skipped == 1

    def test_skips_parse_errors(self):
        lines = [_make_log_json(), "bad line", _make_log_json()]
        result = import_from_logs(lines)
        assert result.imported == 2
        assert len(result.errors) == 1
        assert "parse error" in result.errors[0]

    def test_empty_logs(self):
        result = import_from_logs([])
        assert result.total_logs == 0
        assert result.imported == 0
        assert len(result.items) == 0

    def test_items_have_ids(self):
        lines = [_make_log_json()]
        result = import_from_logs(lines)
        assert result.items[0].id != ""
        assert len(result.items[0].id) == 16

    def test_items_have_metadata(self):
        lines = [_make_log_json(url="/v1/chat/completions", duration_ms=100.0)]
        result = import_from_logs(lines)
        meta = result.items[0].metadata
        assert meta["url"] == "/v1/chat/completions"
        assert meta["duration_ms"] == 100.0

    def test_custom_status_range(self):
        lines = [
            _make_log_json(status_code=400),
            _make_log_json(status_code=404),
            _make_log_json(status_code=500),
        ]
        result = import_from_logs(lines, min_status=400, max_status=499)
        assert result.imported == 2
        assert result.skipped == 1


# ---------------------------------------------------------------------------
# detect_log_format
# ---------------------------------------------------------------------------

class TestDetectLogFormat:
    def test_json_format(self):
        sample = '{"timestamp": "t1", "url": "/api"}\n{"timestamp": "t2"}'
        assert detect_log_format(sample) == "json"

    def test_csv_format(self):
        sample = "timestamp,method,url,status\n2026-01-01,POST,/api,200"
        assert detect_log_format(sample) == "csv"

    def test_text_format(self):
        sample = "INFO 2026-01-01 request completed"
        assert detect_log_format(sample) == "text"

    def test_empty_returns_text(self):
        assert detect_log_format("") == "text"
        assert detect_log_format("   ") == "text"


# ---------------------------------------------------------------------------
# filter_logs_by_endpoint
# ---------------------------------------------------------------------------

class TestFilterLogsByEndpoint:
    def test_basic_filter(self):
        logs = [
            LogEntry(url="/v1/chat/completions"),
            LogEntry(url="/v1/embeddings"),
            LogEntry(url="/v1/chat/completions"),
            LogEntry(url="/health"),
        ]
        filtered = filter_logs_by_endpoint(logs, "/v1/chat")
        assert len(filtered) == 2

    def test_no_match(self):
        logs = [LogEntry(url="/api/v1")]
        filtered = filter_logs_by_endpoint(logs, "/v2/")
        assert len(filtered) == 0

    def test_empty_list(self):
        assert filter_logs_by_endpoint([], "/api") == []

    def test_empty_pattern_matches_all(self):
        logs = [LogEntry(url="/a"), LogEntry(url="/b")]
        assert len(filter_logs_by_endpoint(logs, "")) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_non_llm_endpoint_produces_empty_content(self):
        line = json.dumps({
            "url": "/health",
            "status_code": 200,
            "request_body": "",
            "response_body": json.dumps({"status": "ok"}),
        })
        result = import_from_logs([line])
        assert result.imported == 1
        assert result.items[0].input_text == ""

    def test_all_lines_invalid(self):
        result = import_from_logs(["bad1", "bad2", "bad3"])
        assert result.imported == 0
        assert len(result.errors) == 3
