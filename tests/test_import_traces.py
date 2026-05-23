"""Tests for `evalyn import-traces` and the trace-importer library."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pytest
from evalyn_sdk.cli.commands.import_traces import cmd_import_traces
from evalyn_sdk.integration.trace_importers import (
    FORMAT_IMPORTERS,
    import_anthropic_jsonl,
    import_generic_jsonl,
    import_openai_jsonl,
    import_traces,
)
from evalyn_sdk.storage import SQLiteStorage

FIXTURES = Path(__file__).parent / "fixtures"
OPENAI_SAMPLE = FIXTURES / "openai_log_sample.jsonl"
ANTHROPIC_SAMPLE = FIXTURES / "anthropic_log_sample.jsonl"
GENERIC_SAMPLE = FIXTURES / "generic_log_sample.jsonl"


# ---------------------------------------------------------------------------
# Fixture-file sanity
# ---------------------------------------------------------------------------


def test_fixture_files_exist():
    assert OPENAI_SAMPLE.exists()
    assert ANTHROPIC_SAMPLE.exists()
    assert GENERIC_SAMPLE.exists()


def test_format_importers_registry():
    assert set(FORMAT_IMPORTERS) == {"openai", "anthropic", "generic"}


# ---------------------------------------------------------------------------
# OpenAI importer
# ---------------------------------------------------------------------------


def test_import_openai_basic():
    calls = import_openai_jsonl(OPENAI_SAMPLE, project="demo-app")
    assert len(calls) == 3

    first = calls[0]
    assert first.function_name == "openai.chat.completions"
    assert first.metadata["project_name"] == "demo-app"
    assert first.metadata["imported_from"] == "openai"
    assert first.inputs["model"] == "gpt-4o-mini"
    assert isinstance(first.inputs["input"], list)
    assert "Paris" in str(first.output)
    assert first.duration_ms == 412
    assert first.started_at is not None and first.ended_at is not None

    assert len(first.spans) == 1
    span = first.spans[0]
    assert span.span_type == "llm_call"
    assert span.attributes["provider"] == "openai"
    assert span.attributes["model"] == "gpt-4o-mini"
    assert span.attributes["input_tokens"] == 14
    assert span.attributes["output_tokens"] == 8
    # gpt-4o-mini pricing is known -> non-zero cost
    assert span.attributes["cost_usd"] > 0


def test_import_openai_missing_fields_skips_record(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        # missing model
        json.dumps({"request": {"messages": [{"role": "user", "content": "hi"}]}}) + "\n"
        # missing messages
        + json.dumps({"request": {"model": "gpt-4o"}}) + "\n"
        # malformed JSON
        + "{not json\n"
        # valid record
        + json.dumps(
            {
                "request": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "response": {
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
                "duration_ms": 50,
            }
        )
        + "\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calls = import_openai_jsonl(bad)
    assert len(calls) == 1
    assert calls[0].spans[0].attributes["model"] == "gpt-4o-mini"


def test_import_openai_handles_missing_usage(tmp_path: Path):
    """Records without `usage` should still parse with 0 tokens."""
    log = tmp_path / "no_usage.jsonl"
    log.write_text(
        json.dumps(
            {
                "request": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "response": {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                },
                "duration_ms": 10,
            }
        )
        + "\n"
    )
    calls = import_openai_jsonl(log)
    assert len(calls) == 1
    span = calls[0].spans[0]
    assert span.attributes["input_tokens"] == 0
    assert span.attributes["output_tokens"] == 0


# ---------------------------------------------------------------------------
# Anthropic importer
# ---------------------------------------------------------------------------


def test_import_anthropic_basic():
    calls = import_anthropic_jsonl(ANTHROPIC_SAMPLE)
    assert len(calls) == 3

    first = calls[0]
    assert first.function_name == "anthropic.messages"
    assert first.metadata["imported_from"] == "anthropic"
    assert first.inputs["model"] == "claude-3-5-sonnet-20241022"
    assert "Hello" in str(first.output)

    span = first.spans[0]
    assert span.attributes["provider"] == "anthropic"
    assert span.attributes["input_tokens"] == 10
    assert span.attributes["output_tokens"] == 11
    assert span.attributes["cost_usd"] > 0


def test_import_anthropic_cache_tokens_surface_on_span():
    calls = import_anthropic_jsonl(ANTHROPIC_SAMPLE)
    # Second record has cache_read_input_tokens=4
    span = calls[1].spans[0]
    assert span.attributes.get("cache_read_input_tokens") == 4


def test_import_anthropic_concatenates_text_blocks(tmp_path: Path):
    log = tmp_path / "multi_block.jsonl"
    log.write_text(
        json.dumps(
            {
                "request": {
                    "model": "claude-3-5-haiku",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "response": {
                    "content": [
                        {"type": "text", "text": "Part one."},
                        {"type": "tool_use", "name": "search", "input": {}},
                        {"type": "text", "text": "Part two."},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 6},
                },
                "duration_ms": 100,
            }
        )
        + "\n"
    )
    calls = import_anthropic_jsonl(log)
    assert len(calls) == 1
    assert calls[0].output == "Part one. Part two."


def test_import_anthropic_max_tokens_in_metadata():
    calls = import_anthropic_jsonl(ANTHROPIC_SAMPLE)
    assert calls[0].metadata.get("max_tokens") == 1024


# ---------------------------------------------------------------------------
# Generic importer
# ---------------------------------------------------------------------------


def test_import_generic_basic():
    calls = import_generic_jsonl(GENERIC_SAMPLE, project="custom")
    assert len(calls) == 3

    first = calls[0]
    assert first.function_name == "generic.llm_call"
    assert first.inputs["input"] == "What is 2 + 2?"
    assert first.output == "4"
    assert first.metadata["user_id"] == "u-001"
    assert first.metadata["project_name"] == "custom"

    span = first.spans[0]
    assert span.attributes["provider"] == "generic"
    assert span.attributes["input_tokens"] == 6
    assert span.attributes["output_tokens"] == 1


def test_import_generic_missing_io_skipped(tmp_path: Path):
    log = tmp_path / "missing.jsonl"
    log.write_text(
        json.dumps({"output": "no input"}) + "\n"
        + json.dumps({"input": "no output"}) + "\n"
        + json.dumps({"input": "x", "output": "y"}) + "\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calls = import_generic_jsonl(log)
    assert len(calls) == 1


def test_import_generic_unknown_model_defaults_to_unknown():
    """Last record in the fixture has no `model` field."""
    calls = import_generic_jsonl(GENERIC_SAMPLE)
    # Third record has no model
    third = calls[2]
    assert third.spans[0].attributes["model"] == "unknown"


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def test_import_traces_dispatch_openai():
    calls = import_traces(OPENAI_SAMPLE, fmt="openai")
    assert len(calls) == 3


def test_import_traces_dispatch_anthropic():
    calls = import_traces(ANTHROPIC_SAMPLE, fmt="anthropic")
    assert len(calls) == 3


def test_import_traces_dispatch_generic():
    calls = import_traces(GENERIC_SAMPLE, fmt="generic")
    assert len(calls) == 3


def test_import_traces_unknown_format_raises():
    with pytest.raises(ValueError, match="Unsupported format"):
        import_traces(OPENAI_SAMPLE, fmt="bogus")


def test_import_traces_custom_function_name():
    calls = import_traces(OPENAI_SAMPLE, fmt="openai", function_name="my-agent")
    assert all(c.function_name == "my-agent" for c in calls)


# ---------------------------------------------------------------------------
# CLI command handler -> end-to-end persistence
# ---------------------------------------------------------------------------


def _make_args(file: str, fmt: str, db_path: str, **overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        file=file,
        format=fmt,
        project=None,
        function=None,
        db=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    # Override the storage path resolution by writing direct to a temp DB
    ns._db_path = db_path
    return ns


def test_cmd_import_traces_writes_to_storage(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "import_test.sqlite"
    storage = SQLiteStorage(db_path)
    storage.close()

    # Monkey-patch the resolver to return our temp storage.
    from evalyn_sdk.cli.commands import import_traces as it_mod

    def fake_get_storage(args):
        return SQLiteStorage(db_path)

    monkeypatch.setattr(it_mod, "_get_storage", fake_get_storage)

    args = argparse.Namespace(
        file=str(OPENAI_SAMPLE),
        format="openai",
        project="cli-test",
        function=None,
        db=None,
    )
    cmd_import_traces(args)

    out = capsys.readouterr().out
    assert "Imported 3 call(s)" in out
    assert "cli-test" in out

    # Verify they actually landed in storage and are queryable.
    s2 = SQLiteStorage(db_path)
    listed = s2.list_calls(limit=10, project="cli-test")
    assert len(listed) == 3
    # Make sure spans round-tripped
    sample = s2.get_call(listed[0].id)
    assert sample is not None
    assert len(sample.spans) == 1
    assert sample.spans[0].span_type == "llm_call"
    s2.close()


def test_cmd_import_traces_missing_file_exits(tmp_path: Path, capsys):
    args = argparse.Namespace(
        file=str(tmp_path / "does_not_exist.jsonl"),
        format="openai",
        project=None,
        function=None,
        db=None,
    )
    with pytest.raises(SystemExit):
        cmd_import_traces(args)


def test_cmd_import_traces_unknown_format_exits(tmp_path: Path):
    args = argparse.Namespace(
        file=str(OPENAI_SAMPLE),
        format="bogus",
        project=None,
        function=None,
        db=None,
    )
    with pytest.raises(SystemExit):
        cmd_import_traces(args)


def test_cmd_import_traces_empty_file_prints_no_records(tmp_path: Path, monkeypatch, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    db_path = tmp_path / "empty.sqlite"
    SQLiteStorage(db_path).close()

    from evalyn_sdk.cli.commands import import_traces as it_mod

    monkeypatch.setattr(it_mod, "_get_storage", lambda args: SQLiteStorage(db_path))

    args = argparse.Namespace(
        file=str(empty),
        format="generic",
        project=None,
        function=None,
        db=None,
    )
    cmd_import_traces(args)
    out = capsys.readouterr().out
    assert "No records imported" in out


# ---------------------------------------------------------------------------
# register_commands wiring
# ---------------------------------------------------------------------------


def test_register_commands_wires_subparser():
    from evalyn_sdk.cli.commands.import_traces import register_commands

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_commands(subparsers)

    args = parser.parse_args(
        [
            "import-traces",
            "--file",
            str(OPENAI_SAMPLE),
            "--format",
            "openai",
            "--project",
            "p",
            "--db",
            "test",
        ]
    )
    assert args.command == "import-traces"
    assert args.file == str(OPENAI_SAMPLE)
    assert args.format == "openai"
    assert args.project == "p"
    assert args.db == "test"
    assert callable(args.func)
