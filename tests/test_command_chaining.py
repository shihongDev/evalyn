"""Tests for CLI command chaining."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.command_chaining import (
    ChainConfig,
    ChainResult,
    ChainStep,
    execute_chain,
    format_chain_plan,
    format_chain_result,
    parse_chain,
)


# ---------------------------------------------------------------------------
# Helper executor
# ---------------------------------------------------------------------------


def echo_executor(cmd: str, args: List[str], input_data: str) -> Tuple[int, str]:
    """Simple executor that echoes the command and args."""
    return 0, f"{cmd} {' '.join(args)}".strip()


def fail_executor(cmd: str, args: List[str], input_data: str) -> Tuple[int, str]:
    """Executor that always fails."""
    return 1, "error"


def passthrough_executor(
    cmd: str, args: List[str], input_data: str
) -> Tuple[int, str]:
    """Executor that passes input through with a prefix."""
    return 0, f"[{cmd}] {input_data}"


# ---------------------------------------------------------------------------
# ChainStep
# ---------------------------------------------------------------------------


class TestChainStep:
    def test_defaults(self):
        step = ChainStep(command="ls")
        assert step.command == "ls"
        assert step.args == []
        assert step.input_data == ""
        assert step.output_data == ""
        assert step.exit_code == 0

    def test_custom_values(self):
        step = ChainStep(
            command="grep",
            args=["-i", "pattern"],
            input_data="data",
            output_data="result",
            exit_code=1,
        )
        assert step.command == "grep"
        assert step.args == ["-i", "pattern"]
        assert step.exit_code == 1

    def test_as_dict(self):
        step = ChainStep(command="cat", args=["file.txt"], output_data="contents")
        d = step.as_dict()
        assert d["command"] == "cat"
        assert d["args"] == ["file.txt"]
        assert d["output_data"] == "contents"

    def test_from_dict(self):
        d = {"command": "wc", "args": ["-l"], "exit_code": 0, "output_data": "42"}
        step = ChainStep.from_dict(d)
        assert step.command == "wc"
        assert step.args == ["-l"]
        assert step.output_data == "42"

    def test_from_dict_defaults(self):
        step = ChainStep.from_dict({})
        assert step.command == ""
        assert step.args == []
        assert step.exit_code == 0

    def test_roundtrip(self):
        step = ChainStep(command="sort", args=["-r"], exit_code=0, output_data="z\na")
        restored = ChainStep.from_dict(step.as_dict())
        assert restored.command == step.command
        assert restored.args == step.args
        assert restored.output_data == step.output_data

    def test_args_is_copy(self):
        """as_dict should return a copy of the args list."""
        step = ChainStep(command="x", args=["a"])
        d = step.as_dict()
        d["args"].append("b")
        assert step.args == ["a"]


# ---------------------------------------------------------------------------
# ChainConfig
# ---------------------------------------------------------------------------


class TestChainConfig:
    def test_defaults(self):
        cfg = ChainConfig()
        assert cfg.stop_on_error is True
        assert cfg.pass_output_as == "stdin"

    def test_custom(self):
        cfg = ChainConfig(stop_on_error=False, pass_output_as="arg")
        assert cfg.stop_on_error is False
        assert cfg.pass_output_as == "arg"

    def test_as_dict(self):
        cfg = ChainConfig(stop_on_error=False, pass_output_as="file")
        d = cfg.as_dict()
        assert d == {"stop_on_error": False, "pass_output_as": "file"}

    def test_from_dict(self):
        d = {"stop_on_error": False, "pass_output_as": "arg"}
        cfg = ChainConfig.from_dict(d)
        assert cfg.stop_on_error is False
        assert cfg.pass_output_as == "arg"

    def test_from_dict_defaults(self):
        cfg = ChainConfig.from_dict({})
        assert cfg.stop_on_error is True
        assert cfg.pass_output_as == "stdin"

    def test_roundtrip(self):
        cfg = ChainConfig(stop_on_error=False, pass_output_as="file")
        restored = ChainConfig.from_dict(cfg.as_dict())
        assert restored.stop_on_error == cfg.stop_on_error
        assert restored.pass_output_as == cfg.pass_output_as


# ---------------------------------------------------------------------------
# ChainResult
# ---------------------------------------------------------------------------


class TestChainResult:
    def test_defaults(self):
        r = ChainResult()
        assert r.steps == []
        assert r.final_output == ""
        assert r.total_steps == 0
        assert r.succeeded == 0
        assert r.failed == 0

    def test_as_dict(self):
        step = ChainStep(command="echo", output_data="hi")
        r = ChainResult(
            steps=[step], final_output="hi", total_steps=1, succeeded=1, failed=0
        )
        d = r.as_dict()
        assert d["total_steps"] == 1
        assert len(d["steps"]) == 1
        assert d["steps"][0]["command"] == "echo"

    def test_from_dict(self):
        d = {
            "steps": [{"command": "ls", "args": [], "exit_code": 0}],
            "final_output": "files",
            "total_steps": 1,
            "succeeded": 1,
            "failed": 0,
        }
        r = ChainResult.from_dict(d)
        assert r.total_steps == 1
        assert len(r.steps) == 1
        assert r.steps[0].command == "ls"

    def test_from_dict_defaults(self):
        r = ChainResult.from_dict({})
        assert r.steps == []
        assert r.total_steps == 0

    def test_roundtrip(self):
        step = ChainStep(command="wc", args=["-l"], output_data="10")
        r = ChainResult(
            steps=[step], final_output="10", total_steps=1, succeeded=1, failed=0
        )
        restored = ChainResult.from_dict(r.as_dict())
        assert restored.total_steps == r.total_steps
        assert restored.final_output == r.final_output
        assert len(restored.steps) == len(r.steps)


# ---------------------------------------------------------------------------
# parse_chain
# ---------------------------------------------------------------------------


class TestParseChain:
    def test_single_command(self):
        steps = parse_chain("ls -la")
        assert len(steps) == 1
        assert steps[0].command == "ls"
        assert steps[0].args == ["-la"]

    def test_two_commands(self):
        steps = parse_chain("cat file.txt | grep error")
        assert len(steps) == 2
        assert steps[0].command == "cat"
        assert steps[0].args == ["file.txt"]
        assert steps[1].command == "grep"
        assert steps[1].args == ["error"]

    def test_three_commands(self):
        steps = parse_chain("ls | sort | head -5")
        assert len(steps) == 3
        assert steps[2].command == "head"
        assert steps[2].args == ["-5"]

    def test_custom_separator(self):
        steps = parse_chain("cmd1 >> cmd2 >> cmd3", separator=">>")
        assert len(steps) == 3

    def test_whitespace_handling(self):
        steps = parse_chain("  ls  |  grep foo  ")
        assert len(steps) == 2
        assert steps[0].command == "ls"
        assert steps[1].command == "grep"

    def test_empty_string(self):
        steps = parse_chain("")
        assert steps == []

    def test_only_separator(self):
        steps = parse_chain("|")
        assert steps == []

    def test_command_no_args(self):
        steps = parse_chain("pwd")
        assert len(steps) == 1
        assert steps[0].command == "pwd"
        assert steps[0].args == []

    def test_multiple_args(self):
        steps = parse_chain("find . -name *.py -type f")
        assert len(steps) == 1
        assert steps[0].command == "find"
        assert steps[0].args == [".", "-name", "*.py", "-type", "f"]


# ---------------------------------------------------------------------------
# execute_chain
# ---------------------------------------------------------------------------


class TestExecuteChain:
    def test_single_step_success(self):
        steps = [ChainStep(command="echo", args=["hello"])]
        result = execute_chain(steps, echo_executor)
        assert result.succeeded == 1
        assert result.failed == 0
        assert result.final_output == "echo hello"

    def test_output_piping(self):
        steps = parse_chain("first | second | third")
        result = execute_chain(steps, passthrough_executor)
        assert result.succeeded == 3
        # Each step prefixes input with [cmd]
        assert "[third]" in result.final_output
        assert "[second]" in result.final_output

    def test_stop_on_error(self):
        steps = [
            ChainStep(command="ok"),
            ChainStep(command="fail"),
            ChainStep(command="never"),
        ]

        def mixed_executor(cmd, args, inp):
            if cmd == "fail":
                return 1, "boom"
            return 0, "ok"

        result = execute_chain(steps, mixed_executor, ChainConfig(stop_on_error=True))
        assert result.succeeded == 1
        assert result.failed == 1
        assert len(result.steps) == 2  # third step never ran

    def test_continue_on_error(self):
        steps = [
            ChainStep(command="ok"),
            ChainStep(command="fail"),
            ChainStep(command="also_ok"),
        ]

        def mixed_executor(cmd, args, inp):
            if cmd == "fail":
                return 1, "boom"
            return 0, "fine"

        result = execute_chain(steps, mixed_executor, ChainConfig(stop_on_error=False))
        assert result.succeeded == 2
        assert result.failed == 1
        assert len(result.steps) == 3

    def test_empty_chain(self):
        result = execute_chain([], echo_executor)
        assert result.total_steps == 0
        assert result.final_output == ""

    def test_default_config(self):
        steps = [ChainStep(command="echo")]
        result = execute_chain(steps, echo_executor)
        assert result.succeeded == 1

    def test_input_data_passed(self):
        """Verify that output of step N becomes input of step N+1."""
        received_inputs = []

        def tracking_executor(cmd, args, inp):
            received_inputs.append(inp)
            return 0, f"output_of_{cmd}"

        steps = parse_chain("a | b | c")
        execute_chain(steps, tracking_executor)
        assert received_inputs[0] == ""  # first step gets empty input
        assert received_inputs[1] == "output_of_a"
        assert received_inputs[2] == "output_of_b"

    def test_all_fail(self):
        steps = parse_chain("a | b | c")
        result = execute_chain(steps, fail_executor, ChainConfig(stop_on_error=True))
        assert result.succeeded == 0
        assert result.failed == 1
        assert len(result.steps) == 1

    def test_all_fail_continue(self):
        steps = parse_chain("a | b | c")
        result = execute_chain(steps, fail_executor, ChainConfig(stop_on_error=False))
        assert result.succeeded == 0
        assert result.failed == 3
        assert len(result.steps) == 3

    def test_final_output_is_last_step(self):
        steps = parse_chain("a | b")

        def numbered(cmd, args, inp):
            return 0, f"result_{cmd}"

        result = execute_chain(steps, numbered)
        assert result.final_output == "result_b"


# ---------------------------------------------------------------------------
# format_chain_plan
# ---------------------------------------------------------------------------


class TestFormatChainPlan:
    def test_single_step(self):
        steps = [ChainStep(command="ls")]
        assert format_chain_plan(steps) == "Step 1: ls"

    def test_multiple_steps(self):
        steps = parse_chain("cat file | grep err | wc -l")
        result = format_chain_plan(steps)
        assert "Step 1: cat file" in result
        assert "Step 2: grep err" in result
        assert "Step 3: wc -l" in result
        assert " -> " in result

    def test_empty_steps(self):
        assert format_chain_plan([]) == ""

    def test_step_with_args(self):
        steps = [ChainStep(command="find", args=[".", "-name", "*.py"])]
        result = format_chain_plan(steps)
        assert "find . -name *.py" in result


# ---------------------------------------------------------------------------
# format_chain_result
# ---------------------------------------------------------------------------


class TestFormatChainResult:
    def test_success_result(self):
        step = ChainStep(
            command="echo",
            args=["hello"],
            output_data="hello",
            exit_code=0,
        )
        result = ChainResult(
            steps=[step], final_output="hello", total_steps=1, succeeded=1, failed=0
        )
        text = format_chain_result(result)
        assert "OK" in text
        assert "echo hello" in text
        assert "1/1 succeeded" in text

    def test_failed_result(self):
        step = ChainStep(command="bad", exit_code=1, output_data="error msg")
        result = ChainResult(
            steps=[step], final_output="error msg", total_steps=1, succeeded=0, failed=1
        )
        text = format_chain_result(result)
        assert "FAILED" in text
        assert "exit 1" in text
        assert "0/1 succeeded" in text

    def test_mixed_result(self):
        s1 = ChainStep(command="ok_cmd", exit_code=0, output_data="fine")
        s2 = ChainStep(command="bad_cmd", exit_code=2, output_data="oops")
        result = ChainResult(
            steps=[s1, s2], final_output="oops", total_steps=2, succeeded=1, failed=1
        )
        text = format_chain_result(result)
        assert "1/2 succeeded" in text
        assert "1 failed" in text

    def test_empty_result(self):
        result = ChainResult()
        text = format_chain_result(result)
        assert "No steps executed" in text

    def test_long_output_truncated(self):
        step = ChainStep(command="cat", exit_code=0, output_data="x" * 200)
        result = ChainResult(
            steps=[step], final_output="x" * 200, total_steps=1, succeeded=1, failed=0
        )
        text = format_chain_result(result)
        assert "..." in text

    def test_short_output_not_truncated(self):
        step = ChainStep(command="echo", exit_code=0, output_data="short")
        result = ChainResult(
            steps=[step], final_output="short", total_steps=1, succeeded=1, failed=0
        )
        text = format_chain_result(result)
        assert "..." not in text
