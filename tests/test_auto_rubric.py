"""Tests for `evalyn auto-rubric` and the underlying rubric_generator module.

All LLM calls are mocked. No network access required.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from evalyn_sdk.calibration.rubric_generator import (
    MIN_EXAMPLES,
    SUPPORTED_SCALES,
    build_rubric_prompt,
    generate_rubric,
    load_examples,
    parse_rubric_response,
)
from evalyn_sdk.cli.commands.auto_rubric import cmd_auto_rubric, register_commands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_examples(n: int = 6) -> list[dict[str, Any]]:
    """Build N valid example dicts. Alternates labels so the rubric is non-trivial."""
    out = []
    for i in range(n):
        label = 1.0 if i % 2 == 0 else 0.0
        out.append(
            {
                "input": f"question {i}",
                "output": f"answer {i}" if label == 1.0 else "I don't know",
                "label": label,
                "rationale": (
                    "Directly answers the question."
                    if label == 1.0
                    else "Refuses without trying."
                ),
            }
        )
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _valid_rubric_dict(metric_id: str = "helpfulness", scale: str = "0-1") -> dict[str, Any]:
    """Return a well-formed rubric the parser should accept."""
    return {
        "id": metric_id,
        "type": "subjective",
        "description": "How helpful the response is to the user.",
        "category": "correctness",
        "scope": "overall",
        "prompt": "You are a helpfulness judge. Evaluate the response.",
        "config": {
            "rubric": [
                "Directly addresses the user's question.",
                "Provides actionable, accurate information.",
                "Does not refuse without trying.",
                "Acknowledges uncertainty when appropriate.",
            ],
            "scale": scale,
            "scoring_guidance": (
                f"On the {scale} scale, score 1.0 if the response fully helps, "
                "0.0 if it refuses or fails."
            ),
            "threshold": 0.5,
        },
        "requires_reference": False,
    }


# ---------------------------------------------------------------------------
# load_examples
# ---------------------------------------------------------------------------


class TestLoadExamples:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        rows = _make_examples(MIN_EXAMPLES + 1)
        path = tmp_path / "examples.jsonl"
        _write_jsonl(path, rows)

        loaded = load_examples(path)
        assert len(loaded) == len(rows)
        assert all("input" in r and "output" in r and "label" in r for r in loaded)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        rows = _make_examples(MIN_EXAMPLES)
        path = tmp_path / "examples.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
            for r in rows:
                f.write(json.dumps(r) + "\n\n")

        loaded = load_examples(path)
        assert len(loaded) == MIN_EXAMPLES

    def test_rejects_missing_label(self, tmp_path: Path) -> None:
        rows = _make_examples(MIN_EXAMPLES)
        del rows[0]["label"]
        path = tmp_path / "examples.jsonl"
        _write_jsonl(path, rows)

        with pytest.raises(ValueError, match=r"missing required field 'label'"):
            load_examples(path)

    def test_rejects_missing_input(self, tmp_path: Path) -> None:
        rows = _make_examples(MIN_EXAMPLES)
        del rows[1]["input"]
        path = tmp_path / "examples.jsonl"
        _write_jsonl(path, rows)

        with pytest.raises(ValueError, match=r"missing required field 'input'"):
            load_examples(path)

    def test_rejects_fewer_than_minimum(self, tmp_path: Path) -> None:
        rows = _make_examples(MIN_EXAMPLES - 1)
        path = tmp_path / "examples.jsonl"
        _write_jsonl(path, rows)

        with pytest.raises(ValueError, match=r"At least 5 labeled examples"):
            load_examples(path)

    def test_rejects_non_numeric_label(self, tmp_path: Path) -> None:
        rows = _make_examples(MIN_EXAMPLES)
        rows[2]["label"] = "good"
        path = tmp_path / "examples.jsonl"
        _write_jsonl(path, rows)

        with pytest.raises(ValueError, match=r"'label' must be a number"):
            load_examples(path)

    def test_rejects_invalid_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "examples.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_make_examples(1)[0]) + "\n")
            f.write("{not json\n")

        with pytest.raises(ValueError, match=r"Invalid JSON on line 2"):
            load_examples(path)

    def test_raises_filenotfound_for_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_examples(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# build_rubric_prompt
# ---------------------------------------------------------------------------


class TestBuildRubricPrompt:
    def test_includes_all_examples(self) -> None:
        examples = _make_examples(6)
        prompt = build_rubric_prompt(examples, scale="0-1", metric_id="m1")

        for i, ex in enumerate(examples, start=1):
            assert f'<example index="{i}">' in prompt
            assert f"answer {i - 1}" in prompt or "I don't know" in prompt
            # Labels should also appear
            assert f"LABEL: {ex['label']}" in prompt

    def test_includes_scale_and_metric_id(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)
        prompt = build_rubric_prompt(examples, scale="1-5", metric_id="my_metric")
        assert "1-5" in prompt
        assert "my_metric" in prompt

    def test_includes_clear_instructions(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)
        prompt = build_rubric_prompt(examples, scale="0-1", metric_id="m")
        # Schema instructions
        assert "rubric" in prompt
        assert "scoring_guidance" in prompt
        assert "threshold" in prompt
        # Return contract
        assert "Return ONLY the JSON object" in prompt or "Return a single JSON object" in prompt
        # Must hint at criterion count
        assert "3 to 5" in prompt or "3-5" in prompt

    def test_includes_description_when_provided(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)
        prompt = build_rubric_prompt(
            examples,
            scale="0-1",
            metric_id="m",
            description="A custom description of the metric.",
        )
        assert "A custom description of the metric." in prompt

    def test_rejects_unsupported_scale(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)
        with pytest.raises(ValueError, match=r"Unsupported scale"):
            build_rubric_prompt(examples, scale="0-100", metric_id="m")


# ---------------------------------------------------------------------------
# parse_rubric_response
# ---------------------------------------------------------------------------


class TestParseRubricResponse:
    def test_parses_bare_json(self) -> None:
        rubric = _valid_rubric_dict()
        text = json.dumps(rubric)
        out = parse_rubric_response(text)
        assert out["id"] == rubric["id"]
        assert len(out["config"]["rubric"]) == 4

    def test_parses_markdown_fenced_json(self) -> None:
        rubric = _valid_rubric_dict()
        text = f"Here is the rubric:\n```json\n{json.dumps(rubric)}\n```\nLet me know."
        out = parse_rubric_response(text)
        assert out["id"] == rubric["id"]
        assert out["config"]["scale"] == "0-1"

    def test_parses_plain_fenced_json(self) -> None:
        rubric = _valid_rubric_dict()
        text = f"```\n{json.dumps(rubric)}\n```"
        out = parse_rubric_response(text)
        assert out["id"] == rubric["id"]

    def test_rejects_no_json(self) -> None:
        with pytest.raises(ValueError, match=r"Could not extract a JSON object"):
            parse_rubric_response("I'm sorry, I cannot help with that.")

    def test_rejects_missing_top_level_fields(self) -> None:
        rubric = _valid_rubric_dict()
        del rubric["description"]
        with pytest.raises(ValueError, match=r"missing required field"):
            parse_rubric_response(json.dumps(rubric))

    def test_rejects_missing_config(self) -> None:
        rubric = {"id": "m", "description": "x"}
        with pytest.raises(ValueError, match=r"missing required field"):
            parse_rubric_response(json.dumps(rubric))

    def test_rejects_missing_config_fields(self) -> None:
        rubric = _valid_rubric_dict()
        del rubric["config"]["scale"]
        with pytest.raises(ValueError, match=r"config is missing required field"):
            parse_rubric_response(json.dumps(rubric))

    def test_rejects_empty_rubric_list(self) -> None:
        rubric = _valid_rubric_dict()
        rubric["config"]["rubric"] = []
        with pytest.raises(ValueError, match=r"must be a non-empty list"):
            parse_rubric_response(json.dumps(rubric))

    def test_rejects_non_string_rubric_items(self) -> None:
        rubric = _valid_rubric_dict()
        rubric["config"]["rubric"] = ["good", 42, "also good"]
        with pytest.raises(ValueError, match=r"non-empty strings"):
            parse_rubric_response(json.dumps(rubric))

    def test_defaults_type_to_subjective(self) -> None:
        rubric = _valid_rubric_dict()
        del rubric["type"]
        out = parse_rubric_response(json.dumps(rubric))
        assert out["type"] == "subjective"


# ---------------------------------------------------------------------------
# generate_rubric (with mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateRubric:
    def test_returns_valid_rubric_with_mocked_llm(self) -> None:
        examples = _make_examples(MIN_EXAMPLES + 2)
        mock_response = json.dumps(_valid_rubric_dict(metric_id="helpfulness"))

        calls: list[str] = []

        def mock_llm(prompt: str) -> str:
            calls.append(prompt)
            return mock_response

        result = generate_rubric(
            examples,
            scale="0-1",
            metric_id="helpfulness",
            llm=mock_llm,
        )

        assert len(calls) == 1
        assert '<example index="1">' in calls[0]
        assert "helpfulness" in calls[0]
        assert result["id"] == "helpfulness"
        assert result["config"]["scale"] == "0-1"
        assert len(result["config"]["rubric"]) >= 3

    def test_forces_metric_id_even_if_llm_hallucinates(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)
        bad_rubric = _valid_rubric_dict(metric_id="something_else")

        def mock_llm(prompt: str) -> str:
            return json.dumps(bad_rubric)

        result = generate_rubric(
            examples,
            scale="0-1",
            metric_id="actual_id",
            llm=mock_llm,
        )
        assert result["id"] == "actual_id"

    def test_handles_markdown_fenced_response(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)
        rubric = _valid_rubric_dict()

        def mock_llm(prompt: str) -> str:
            return f"Sure! Here's the rubric:\n\n```json\n{json.dumps(rubric)}\n```\n"

        result = generate_rubric(examples, scale="0-1", metric_id="helpfulness", llm=mock_llm)
        assert result["id"] == "helpfulness"

    def test_raises_on_empty_examples(self) -> None:
        with pytest.raises(ValueError, match=r"No examples"):
            generate_rubric([], scale="0-1", llm=lambda _: "{}")

    def test_raises_on_unparseable_response(self) -> None:
        examples = _make_examples(MIN_EXAMPLES)

        def mock_llm(prompt: str) -> str:
            return "Sorry, I cannot do that."

        with pytest.raises(ValueError, match=r"Could not extract"):
            generate_rubric(examples, scale="0-1", metric_id="m", llm=mock_llm)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _make_cli_namespace(**overrides: Any) -> argparse.Namespace:
    """Build the argparse.Namespace cmd_auto_rubric expects."""
    defaults = dict(
        examples="",
        metric_id="helpfulness",
        description=None,
        scale="0-1",
        provider="gemini",
        output=None,
        api_key=None,
        quiet=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCLI:
    def test_writes_json_file_and_prints_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: write examples JSONL and stub the LLM caller used internally
        examples_path = tmp_path / "examples.jsonl"
        _write_jsonl(examples_path, _make_examples(MIN_EXAMPLES + 1))
        output_path = tmp_path / "out" / "helpfulness.json"

        rubric_dict = _valid_rubric_dict(metric_id="helpfulness")

        def fake_generate_rubric(examples, **kwargs):  # type: ignore[no-untyped-def]
            assert len(examples) == MIN_EXAMPLES + 1
            assert kwargs["metric_id"] == "helpfulness"
            assert kwargs["scale"] == "0-1"
            assert kwargs["provider"] == "gemini"
            return rubric_dict

        monkeypatch.setattr(
            "evalyn_sdk.cli.commands.auto_rubric.generate_rubric",
            fake_generate_rubric,
        )

        args = _make_cli_namespace(
            examples=str(examples_path),
            output=str(output_path),
            metric_id="helpfulness",
        )

        # Act: run the CLI command, capturing stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_auto_rubric(args)
        stdout = buf.getvalue()

        # Assert: file written with correct shape
        assert output_path.exists()
        on_disk = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(on_disk, list)
        assert len(on_disk) == 1
        assert on_disk[0]["id"] == "helpfulness"
        assert on_disk[0]["type"] == "subjective"
        assert len(on_disk[0]["config"]["rubric"]) >= 3

        # Assert: summary contains required pieces
        assert "AUTO-RUBRIC" in stdout
        assert "Examples loaded" in stdout
        assert "Wrote rubric" in stdout
        assert "helpfulness" in stdout

    def test_uses_default_output_path_when_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default path is data/metrics/<id>.json relative to cwd. Run from tmp_path.
        examples_path = tmp_path / "examples.jsonl"
        _write_jsonl(examples_path, _make_examples(MIN_EXAMPLES))

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "evalyn_sdk.cli.commands.auto_rubric.generate_rubric",
            lambda *_args, **_kwargs: _valid_rubric_dict(metric_id="my_metric"),
        )

        args = _make_cli_namespace(
            examples=str(examples_path),
            metric_id="my_metric",
            output=None,
        )

        with redirect_stdout(io.StringIO()):
            cmd_auto_rubric(args)

        expected = tmp_path / "data" / "metrics" / "my_metric.json"
        assert expected.exists()
        on_disk = json.loads(expected.read_text(encoding="utf-8"))
        assert on_disk[0]["id"] == "my_metric"

    def test_register_commands_adds_parser(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        register_commands(sub)

        # Should accept the documented flags without error
        ns = parser.parse_args(
            [
                "auto-rubric",
                "--examples",
                "ex.jsonl",
                "--metric-id",
                "m",
                "--scale",
                "1-5",
                "--provider",
                "openai",
            ]
        )
        assert ns.examples == "ex.jsonl"
        assert ns.metric_id == "m"
        assert ns.scale == "1-5"
        assert ns.provider == "openai"

    def test_cli_rejects_invalid_scale(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        _write_jsonl(examples_path, _make_examples(MIN_EXAMPLES))
        args = _make_cli_namespace(examples=str(examples_path), scale="bogus")

        with pytest.raises(SystemExit):
            with redirect_stdout(io.StringIO()):
                cmd_auto_rubric(args)

    def test_cli_rejects_too_few_examples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        examples_path = tmp_path / "examples.jsonl"
        _write_jsonl(examples_path, _make_examples(MIN_EXAMPLES - 1))
        args = _make_cli_namespace(examples=str(examples_path))

        with pytest.raises(SystemExit):
            with redirect_stdout(io.StringIO()):
                cmd_auto_rubric(args)


# ---------------------------------------------------------------------------
# Sanity: supported scales constant is exposed
# ---------------------------------------------------------------------------


def test_supported_scales_constant() -> None:
    assert "0-1" in SUPPORTED_SCALES
    assert "1-5" in SUPPORTED_SCALES
    assert "1-10" in SUPPORTED_SCALES
