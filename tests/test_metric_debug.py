"""Tests for the metric debug mode module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from evalyn_sdk.metric_debug import (
    DebugConfig,
    DebugEntry,
    DebugLog,
    append_debug_entry,
    create_debug_entry,
    filter_debug_entries,
    format_debug_entry,
    format_debug_summary,
    read_debug_log,
    truncate_text,
)


# ---------------------------------------------------------------------------
# DebugEntry
# ---------------------------------------------------------------------------


class TestDebugEntry:
    def test_basic_creation(self):
        e = DebugEntry(item_id="1", metric_id="bleu", input_text="hi", output_text="bye", prompt_sent="p")
        assert e.item_id == "1"
        assert e.score == 0.0
        assert e.judge_response == ""
        assert e.reasoning == ""
        assert e.timestamp == ""

    def test_full_creation(self):
        e = DebugEntry(
            item_id="x", metric_id="m", input_text="in", output_text="out",
            prompt_sent="p", judge_response="r", score=0.9, reasoning="good", timestamp="t"
        )
        assert e.score == 0.9
        assert e.reasoning == "good"

    def test_as_dict(self):
        e = DebugEntry(item_id="1", metric_id="m", input_text="a", output_text="b", prompt_sent="c")
        d = e.as_dict()
        assert d["item_id"] == "1"
        assert d["score"] == 0.0
        assert "prompt_sent" in d

    def test_from_dict(self):
        d = {
            "item_id": "2", "metric_id": "x", "input_text": "i",
            "output_text": "o", "prompt_sent": "p", "score": 0.5,
        }
        e = DebugEntry.from_dict(d)
        assert e.item_id == "2"
        assert e.score == 0.5

    def test_from_dict_defaults(self):
        e = DebugEntry.from_dict({})
        assert e.item_id == ""
        assert e.score == 0.0
        assert e.reasoning == ""

    def test_roundtrip(self):
        original = DebugEntry(
            item_id="r", metric_id="m", input_text="i", output_text="o",
            prompt_sent="p", judge_response="j", score=0.7, reasoning="ok", timestamp="ts"
        )
        restored = DebugEntry.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.score == original.score
        assert restored.timestamp == original.timestamp


# ---------------------------------------------------------------------------
# DebugConfig
# ---------------------------------------------------------------------------


class TestDebugConfig:
    def test_defaults(self):
        c = DebugConfig()
        assert c.enabled is False
        assert c.log_path == ".evalyn/debug.jsonl"
        assert c.include_prompts is True
        assert c.include_responses is True
        assert c.max_text_length == 500

    def test_custom(self):
        c = DebugConfig(enabled=True, log_path="/tmp/x.jsonl", max_text_length=100)
        assert c.enabled is True
        assert c.max_text_length == 100

    def test_as_dict(self):
        c = DebugConfig(enabled=True)
        d = c.as_dict()
        assert d["enabled"] is True
        assert d["max_text_length"] == 500

    def test_from_dict(self):
        d = {"enabled": True, "log_path": "/a/b", "max_text_length": 200}
        c = DebugConfig.from_dict(d)
        assert c.enabled is True
        assert c.log_path == "/a/b"
        assert c.max_text_length == 200

    def test_from_dict_defaults(self):
        c = DebugConfig.from_dict({})
        assert c.enabled is False
        assert c.include_prompts is True

    def test_roundtrip(self):
        original = DebugConfig(enabled=True, log_path="x", include_prompts=False, max_text_length=42)
        restored = DebugConfig.from_dict(original.as_dict())
        assert restored.enabled == original.enabled
        assert restored.log_path == original.log_path
        assert restored.include_prompts == original.include_prompts
        assert restored.max_text_length == original.max_text_length


# ---------------------------------------------------------------------------
# DebugLog
# ---------------------------------------------------------------------------


class TestDebugLog:
    def test_empty(self):
        log = DebugLog()
        assert log.entries == []
        assert isinstance(log.config, DebugConfig)

    def test_with_entries(self):
        e = DebugEntry(item_id="1", metric_id="m", input_text="", output_text="", prompt_sent="")
        log = DebugLog(entries=[e])
        assert len(log.entries) == 1

    def test_as_dict(self):
        log = DebugLog()
        d = log.as_dict()
        assert d["entries"] == []
        assert "config" in d

    def test_from_dict(self):
        d = {
            "entries": [
                {"item_id": "1", "metric_id": "m", "input_text": "", "output_text": "", "prompt_sent": ""}
            ],
            "config": {"enabled": True},
        }
        log = DebugLog.from_dict(d)
        assert len(log.entries) == 1
        assert log.config.enabled is True

    def test_from_dict_defaults(self):
        log = DebugLog.from_dict({})
        assert log.entries == []
        assert log.config.enabled is False

    def test_roundtrip(self):
        e = DebugEntry(item_id="a", metric_id="b", input_text="c", output_text="d", prompt_sent="e", score=0.5)
        original = DebugLog(entries=[e], config=DebugConfig(enabled=True))
        restored = DebugLog.from_dict(original.as_dict())
        assert len(restored.entries) == 1
        assert restored.entries[0].score == 0.5
        assert restored.config.enabled is True


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_short_text(self):
        assert truncate_text("hello", 500) == "hello"

    def test_exact_length(self):
        text = "a" * 500
        assert truncate_text(text, 500) == text

    def test_over_length(self):
        text = "a" * 600
        result = truncate_text(text, 500)
        assert result.endswith("... [truncated]")
        assert result.startswith("a" * 500)

    def test_default_max(self):
        text = "x" * 501
        result = truncate_text(text)
        assert "truncated" in result

    def test_empty_string(self):
        assert truncate_text("", 500) == ""

    def test_small_max(self):
        result = truncate_text("hello world", 5)
        assert result == "hello... [truncated]"


# ---------------------------------------------------------------------------
# create_debug_entry
# ---------------------------------------------------------------------------


class TestCreateDebugEntry:
    def test_basic(self):
        e = create_debug_entry("item1", "metric1", "input", "output", "prompt")
        assert e.item_id == "item1"
        assert e.metric_id == "metric1"
        assert e.timestamp != ""

    def test_truncation(self):
        config = DebugConfig(max_text_length=10)
        long_text = "a" * 100
        e = create_debug_entry("1", "m", long_text, long_text, long_text, config=config)
        assert len(e.input_text) < 100
        assert "truncated" in e.input_text

    def test_prompt_excluded_when_disabled(self):
        config = DebugConfig(include_prompts=False)
        e = create_debug_entry("1", "m", "in", "out", "big prompt", config=config)
        assert e.prompt_sent == ""

    def test_auto_timestamp(self):
        e = create_debug_entry("1", "m", "in", "out", "p")
        assert "T" in e.timestamp  # ISO format

    def test_defaults_score_and_reasoning(self):
        e = create_debug_entry("1", "m", "in", "out", "p")
        assert e.score == 0.0
        assert e.reasoning == ""


# ---------------------------------------------------------------------------
# append_debug_entry / read_debug_log
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_append_and_read(self, tmp_path):
        log_path = str(tmp_path / "debug.jsonl")
        e = DebugEntry(item_id="1", metric_id="m", input_text="i", output_text="o", prompt_sent="p")
        append_debug_entry(log_path, e)
        entries = read_debug_log(log_path)
        assert len(entries) == 1
        assert entries[0].item_id == "1"

    def test_multiple_appends(self, tmp_path):
        log_path = str(tmp_path / "debug.jsonl")
        for i in range(5):
            e = DebugEntry(item_id=str(i), metric_id="m", input_text="", output_text="", prompt_sent="")
            append_debug_entry(log_path, e)
        entries = read_debug_log(log_path)
        assert len(entries) == 5

    def test_read_limit(self, tmp_path):
        log_path = str(tmp_path / "debug.jsonl")
        for i in range(10):
            e = DebugEntry(item_id=str(i), metric_id="m", input_text="", output_text="", prompt_sent="")
            append_debug_entry(log_path, e)
        entries = read_debug_log(log_path, limit=3)
        assert len(entries) == 3
        # Should be the last 3
        assert entries[0].item_id == "7"
        assert entries[2].item_id == "9"

    def test_read_nonexistent(self):
        entries = read_debug_log("/nonexistent/path/file.jsonl")
        assert entries == []

    def test_creates_parent_dirs(self, tmp_path):
        log_path = str(tmp_path / "sub" / "dir" / "debug.jsonl")
        e = DebugEntry(item_id="1", metric_id="m", input_text="", output_text="", prompt_sent="")
        append_debug_entry(log_path, e)
        assert os.path.exists(log_path)

    def test_jsonl_format(self, tmp_path):
        log_path = str(tmp_path / "debug.jsonl")
        e = DebugEntry(item_id="x", metric_id="m", input_text="a", output_text="b", prompt_sent="c", score=0.5)
        append_debug_entry(log_path, e)
        with open(log_path) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["item_id"] == "x"
            assert data["score"] == 0.5


# ---------------------------------------------------------------------------
# filter_debug_entries
# ---------------------------------------------------------------------------


class TestFilterDebugEntries:
    def _make_entries(self):
        return [
            DebugEntry(item_id="1", metric_id="bleu", input_text="", output_text="", prompt_sent=""),
            DebugEntry(item_id="1", metric_id="rouge", input_text="", output_text="", prompt_sent=""),
            DebugEntry(item_id="2", metric_id="bleu", input_text="", output_text="", prompt_sent=""),
            DebugEntry(item_id="2", metric_id="rouge", input_text="", output_text="", prompt_sent=""),
        ]

    def test_filter_by_item(self):
        entries = self._make_entries()
        filtered = filter_debug_entries(entries, item_id="1")
        assert len(filtered) == 2
        assert all(e.item_id == "1" for e in filtered)

    def test_filter_by_metric(self):
        entries = self._make_entries()
        filtered = filter_debug_entries(entries, metric_id="bleu")
        assert len(filtered) == 2
        assert all(e.metric_id == "bleu" for e in filtered)

    def test_filter_by_both(self):
        entries = self._make_entries()
        filtered = filter_debug_entries(entries, item_id="2", metric_id="rouge")
        assert len(filtered) == 1
        assert filtered[0].item_id == "2"
        assert filtered[0].metric_id == "rouge"

    def test_no_filter(self):
        entries = self._make_entries()
        filtered = filter_debug_entries(entries)
        assert len(filtered) == 4

    def test_no_match(self):
        entries = self._make_entries()
        filtered = filter_debug_entries(entries, item_id="999")
        assert filtered == []

    def test_empty_entries(self):
        assert filter_debug_entries([], item_id="1") == []


# ---------------------------------------------------------------------------
# format_debug_entry
# ---------------------------------------------------------------------------


class TestFormatDebugEntry:
    def test_contains_all_fields(self):
        e = DebugEntry(
            item_id="item1", metric_id="bleu", input_text="hello",
            output_text="world", prompt_sent="evaluate", judge_response="ok",
            score=0.8, reasoning="good match", timestamp="2026-01-01T00:00:00"
        )
        text = format_debug_entry(e)
        assert "item1" in text
        assert "bleu" in text
        assert "hello" in text
        assert "world" in text
        assert "evaluate" in text
        assert "ok" in text
        assert "0.8" in text
        assert "good match" in text
        assert "2026-01-01" in text

    def test_empty_entry(self):
        e = DebugEntry(item_id="", metric_id="", input_text="", output_text="", prompt_sent="")
        text = format_debug_entry(e)
        assert "Item:" in text
        assert "Metric:" in text


# ---------------------------------------------------------------------------
# format_debug_summary
# ---------------------------------------------------------------------------


class TestFormatDebugSummary:
    def test_empty(self):
        result = format_debug_summary([])
        assert result == "No debug entries."

    def test_single_entry(self):
        e = DebugEntry(
            item_id="1", metric_id="bleu", input_text="", output_text="",
            prompt_sent="", score=0.9, reasoning="great"
        )
        text = format_debug_summary([e])
        assert "1" in text
        assert "bleu" in text
        assert "0.90" in text

    def test_reasoning_truncation(self):
        long_reasoning = "x" * 100
        e = DebugEntry(
            item_id="1", metric_id="m", input_text="", output_text="",
            prompt_sent="", reasoning=long_reasoning
        )
        text = format_debug_summary([e])
        assert "..." in text

    def test_multiple_entries(self):
        entries = [
            DebugEntry(item_id=str(i), metric_id="m", input_text="", output_text="", prompt_sent="", score=float(i)/10)
            for i in range(3)
        ]
        text = format_debug_summary(entries)
        lines = text.strip().split("\n")
        # header + separator + 3 entries
        assert len(lines) == 5

    def test_header_present(self):
        e = DebugEntry(item_id="1", metric_id="m", input_text="", output_text="", prompt_sent="")
        text = format_debug_summary([e])
        assert "Item" in text
        assert "Metric" in text
        assert "Score" in text
