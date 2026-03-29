"""Tests for json_output module."""

from __future__ import annotations

import json

from evalyn_sdk.json_output import (
    CommandResult,
    JSONOutputConfig,
    ProgressEvent,
    create_error_result,
    create_failure_result,
    create_success_result,
    determine_exit_code,
    format_json_output,
    format_jsonl_stream,
    format_progress_event,
    wrap_command_output,
)


class TestJSONOutputConfig:
    def test_defaults(self):
        c = JSONOutputConfig()
        assert c.pretty is False
        assert c.streaming is False
        assert c.include_metadata is True

    def test_as_dict_roundtrip(self):
        c = JSONOutputConfig(pretty=True, streaming=True)
        restored = JSONOutputConfig.from_dict(c.as_dict())
        assert restored.pretty is True


class TestCommandResult:
    def test_as_dict(self):
        r = CommandResult(command="run-eval", status="success", exit_code=0, data={"score": 0.9})
        d = r.as_dict()
        assert d["command"] == "run-eval"
        assert d["data"]["score"] == 0.9

    def test_from_dict_roundtrip(self):
        r = CommandResult(command="test", status="failure", exit_code=1, errors=["oops"])
        restored = CommandResult.from_dict(r.as_dict())
        assert restored.errors == ["oops"]

    def test_defaults(self):
        r = CommandResult(command="x", status="success", exit_code=0)
        assert r.data is None
        assert r.errors == []
        assert r.metadata == {}

    def test_as_dict_excludes_empty(self):
        r = CommandResult(command="x", status="success", exit_code=0)
        d = r.as_dict()
        assert "errors" not in d
        assert "metadata" not in d


class TestProgressEvent:
    def test_as_dict(self):
        e = ProgressEvent(event_type="progress", progress=0.5, message="halfway")
        d = e.as_dict()
        assert d["progress"] == 0.5

    def test_from_dict_roundtrip(self):
        e = ProgressEvent(event_type="complete", progress=1.0, message="done", timestamp="now")
        restored = ProgressEvent.from_dict(e.as_dict())
        assert restored.event_type == "complete"


class TestFormatJsonOutput:
    def test_compact(self):
        r = create_success_result("test", {"x": 1})
        output = format_json_output(r)
        assert "\n" not in output
        parsed = json.loads(output)
        assert parsed["status"] == "success"

    def test_pretty(self):
        r = create_success_result("test", {"x": 1})
        config = JSONOutputConfig(pretty=True)
        output = format_json_output(r, config)
        assert "\n" in output

    def test_exclude_metadata(self):
        r = create_success_result("test", "data", {"key": "val"})
        config = JSONOutputConfig(include_metadata=False)
        output = format_json_output(r, config)
        parsed = json.loads(output)
        assert "metadata" not in parsed

    def test_include_metadata(self):
        r = create_success_result("test", "data", {"key": "val"})
        output = format_json_output(r)
        parsed = json.loads(output)
        assert parsed["metadata"]["key"] == "val"

    def test_valid_json(self):
        r = create_failure_result("cmd", ["error1", "error2"])
        output = format_json_output(r)
        json.loads(output)  # should not raise


class TestFormatJsonlStream:
    def test_basic(self):
        events = [
            ProgressEvent("start", 0.0, "starting"),
            ProgressEvent("complete", 1.0, "done"),
        ]
        output = format_jsonl_stream(events)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            json.loads(line)

    def test_empty(self):
        assert format_jsonl_stream([]) == ""


class TestFormatProgressEvent:
    def test_single_line(self):
        e = ProgressEvent("progress", 0.5, "halfway")
        output = format_progress_event(e)
        assert "\n" not in output
        parsed = json.loads(output)
        assert parsed["progress"] == 0.5


class TestCreateSuccessResult:
    def test_exit_code(self):
        r = create_success_result("cmd", "data")
        assert r.exit_code == 0
        assert r.status == "success"

    def test_with_metadata(self):
        r = create_success_result("cmd", "data", {"key": "val"})
        assert r.metadata["key"] == "val"

    def test_without_metadata(self):
        r = create_success_result("cmd", "data")
        assert r.metadata == {}


class TestCreateFailureResult:
    def test_exit_code(self):
        r = create_failure_result("cmd", ["err"])
        assert r.exit_code == 1
        assert r.status == "failure"

    def test_errors(self):
        r = create_failure_result("cmd", ["e1", "e2"])
        assert len(r.errors) == 2

    def test_with_data(self):
        r = create_failure_result("cmd", ["err"], data={"partial": True})
        assert r.data["partial"] is True


class TestCreateErrorResult:
    def test_exit_code(self):
        r = create_error_result("cmd", "boom")
        assert r.exit_code == 2
        assert r.status == "error"
        assert r.errors == ["boom"]


class TestDetermineExitCode:
    def test_success(self):
        assert determine_exit_code(create_success_result("x", None)) == 0

    def test_failure(self):
        assert determine_exit_code(create_failure_result("x", ["e"])) == 1

    def test_error(self):
        assert determine_exit_code(create_error_result("x", "e")) == 2


class TestWrapCommandOutput:
    def test_success_text(self):
        output, code = wrap_command_output("test", lambda: "hello")
        assert output == "hello"
        assert code == 0

    def test_success_json(self):
        output, code = wrap_command_output("test", lambda: {"x": 1}, json_mode=True)
        assert code == 0
        parsed = json.loads(output)
        assert parsed["status"] == "success"

    def test_error_text(self):
        def fail():
            raise ValueError("boom")
        output, code = wrap_command_output("test", fail)
        assert code == 2
        assert "boom" in output

    def test_error_json(self):
        def fail():
            raise RuntimeError("crash")
        output, code = wrap_command_output("test", fail, json_mode=True)
        assert code == 2
        parsed = json.loads(output)
        assert parsed["status"] == "error"

    def test_none_result(self):
        output, code = wrap_command_output("test", lambda: None)
        assert code == 0
        assert output == ""
