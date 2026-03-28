"""Tests for cli_reference module."""

from __future__ import annotations

import argparse

import pytest

from evalyn_sdk.cli_reference import (
    ArgDoc,
    CommandDoc,
    extract_arguments,
    extract_commands,
    format_markdown,
    format_plain_text,
    generate_reference,
    group_commands,
)


# ---------------------------------------------------------------------------
# Helpers - build small argparse parsers for testing
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    """Build a minimal parser with two subcommands for testing."""
    parser = argparse.ArgumentParser(prog="testcli", description="Test CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    subs = parser.add_subparsers(dest="command")

    run_p = subs.add_parser("run", help="Run an evaluation")
    run_p.add_argument("--dataset", required=True, help="Path to dataset")
    run_p.add_argument("--limit", type=int, default=100, help="Max items")
    run_p.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Output format"
    )

    show_p = subs.add_parser("show", help="Show results")
    show_p.add_argument("name", help="Result name")
    show_p.add_argument("--all", action="store_true", help="Show all details")

    return parser


def _make_parser_with_groups() -> argparse.ArgumentParser:
    """Build a parser with grouped subcommands (via CommandDoc.group)."""
    parser = argparse.ArgumentParser(prog="grouped")
    subs = parser.add_subparsers(dest="command")
    subs.add_parser("list", help="List items")
    subs.add_parser("show", help="Show item")
    subs.add_parser("delete", help="Delete item")
    return parser


def _make_empty_parser() -> argparse.ArgumentParser:
    """Parser with no subcommands."""
    return argparse.ArgumentParser(prog="empty", description="No commands")


def _make_parser_with_epilog() -> argparse.ArgumentParser:
    """Parser with subcommand that has an epilog."""
    parser = argparse.ArgumentParser(prog="example")
    subs = parser.add_subparsers(dest="command")
    p = subs.add_parser(
        "demo",
        help="Demo command",
        epilog="example demo --flag\nexample demo --other",
    )
    p.add_argument("--flag", action="store_true", help="A flag")
    return parser


# ---------------------------------------------------------------------------
# ArgDoc dataclass
# ---------------------------------------------------------------------------


class TestArgDoc:
    def test_as_dict_roundtrip(self):
        arg = ArgDoc(
            flags=["--dataset", "-d"],
            help="Path to dataset",
            default="/tmp",
            required=True,
            choices=["a", "b"],
            type_name="str",
        )
        d = arg.as_dict()
        restored = ArgDoc.from_dict(d)
        assert restored.flags == ["--dataset", "-d"]
        assert restored.help == "Path to dataset"
        assert restored.default == "/tmp"
        assert restored.required is True
        assert restored.choices == ["a", "b"]
        assert restored.type_name == "str"

    def test_as_dict_keys(self):
        arg = ArgDoc(flags=["--x"], help="test")
        d = arg.as_dict()
        assert set(d.keys()) == {
            "flags",
            "help",
            "default",
            "required",
            "choices",
            "type_name",
        }

    def test_from_dict_defaults(self):
        d = {"flags": ["--x"]}
        arg = ArgDoc.from_dict(d)
        assert arg.help == ""
        assert arg.default is None
        assert arg.required is False
        assert arg.choices is None
        assert arg.type_name == ""

    def test_choices_none_in_dict(self):
        arg = ArgDoc(flags=["--x"], help="test", choices=None)
        assert arg.as_dict()["choices"] is None


# ---------------------------------------------------------------------------
# CommandDoc dataclass
# ---------------------------------------------------------------------------


class TestCommandDoc:
    def test_as_dict_roundtrip(self):
        cmd = CommandDoc(
            name="run",
            description="Run eval",
            arguments=[ArgDoc(flags=["--x"], help="flag")],
            examples=["run --x"],
            group="eval",
        )
        d = cmd.as_dict()
        restored = CommandDoc.from_dict(d)
        assert restored.name == "run"
        assert restored.description == "Run eval"
        assert len(restored.arguments) == 1
        assert restored.arguments[0].flags == ["--x"]
        assert restored.examples == ["run --x"]
        assert restored.group == "eval"

    def test_as_dict_keys(self):
        cmd = CommandDoc(name="x", description="y")
        d = cmd.as_dict()
        assert set(d.keys()) == {"name", "description", "arguments", "examples", "group"}

    def test_from_dict_defaults(self):
        d = {"name": "x", "description": "y"}
        cmd = CommandDoc.from_dict(d)
        assert cmd.arguments == []
        assert cmd.examples == []
        assert cmd.group == ""

    def test_empty_name(self):
        d = {}
        cmd = CommandDoc.from_dict(d)
        assert cmd.name == ""
        assert cmd.description == ""


# ---------------------------------------------------------------------------
# extract_arguments
# ---------------------------------------------------------------------------


class TestExtractArguments:
    def test_extracts_optional_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", action="store_true", help="Verbose")
        parser.add_argument("--count", type=int, default=10, help="Count")
        args = extract_arguments(parser)
        # Should have 2 args (help is skipped)
        assert len(args) == 2

    def test_extracts_positional_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("filename", help="Input file")
        args = extract_arguments(parser)
        assert len(args) == 1
        assert args[0].flags == ["filename"]
        assert args[0].required is True

    def test_flag_type_detection(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", action="store_true", help="A flag")
        args = extract_arguments(parser)
        assert args[0].type_name == "flag"

    def test_choices_extraction(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--fmt", choices=["json", "csv"], help="Format")
        args = extract_arguments(parser)
        assert args[0].choices == ["json", "csv"]

    def test_required_flag(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", required=True, help="Required path")
        args = extract_arguments(parser)
        assert args[0].required is True

    def test_default_value(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--limit", type=int, default=50, help="Limit")
        args = extract_arguments(parser)
        assert args[0].default == 50

    def test_type_name_extraction(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num", type=int, help="Number")
        args = extract_arguments(parser)
        assert args[0].type_name == "int"

    def test_no_args_returns_empty(self):
        parser = argparse.ArgumentParser()
        args = extract_arguments(parser)
        assert args == []

    def test_optional_positional(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("name", nargs="?", default="world", help="Name")
        args = extract_arguments(parser)
        assert args[0].required is False

    def test_star_positional(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("files", nargs="*", help="Files")
        args = extract_arguments(parser)
        assert args[0].required is False


# ---------------------------------------------------------------------------
# extract_commands
# ---------------------------------------------------------------------------


class TestExtractCommands:
    def test_extracts_subcommands(self):
        parser = _make_parser()
        cmds = extract_commands(parser)
        names = [c.name for c in cmds]
        assert "run" in names
        assert "show" in names

    def test_command_count(self):
        parser = _make_parser()
        cmds = extract_commands(parser)
        assert len(cmds) == 2

    def test_command_arguments(self):
        parser = _make_parser()
        cmds = extract_commands(parser)
        run_cmd = next(c for c in cmds if c.name == "run")
        arg_flags = [a.flags for a in run_cmd.arguments]
        assert ["--dataset"] in arg_flags
        assert ["--limit"] in arg_flags

    def test_empty_parser(self):
        parser = _make_empty_parser()
        cmds = extract_commands(parser)
        assert cmds == []

    def test_epilog_becomes_examples(self):
        parser = _make_parser_with_epilog()
        cmds = extract_commands(parser)
        demo = cmds[0]
        assert len(demo.examples) == 2
        assert "example demo --flag" in demo.examples

    def test_positional_in_subcommand(self):
        parser = _make_parser()
        cmds = extract_commands(parser)
        show_cmd = next(c for c in cmds if c.name == "show")
        positional = [a for a in show_cmd.arguments if a.flags == ["name"]]
        assert len(positional) == 1
        assert positional[0].required is True


# ---------------------------------------------------------------------------
# group_commands
# ---------------------------------------------------------------------------


class TestGroupCommands:
    def test_empty_group_becomes_general(self):
        cmds = [CommandDoc(name="x", description="y", group="")]
        groups = group_commands(cmds)
        assert "General" in groups
        assert len(groups["General"]) == 1

    def test_named_groups(self):
        cmds = [
            CommandDoc(name="a", description="", group="alpha"),
            CommandDoc(name="b", description="", group="alpha"),
            CommandDoc(name="c", description="", group="beta"),
        ]
        groups = group_commands(cmds)
        assert len(groups["alpha"]) == 2
        assert len(groups["beta"]) == 1

    def test_mixed_groups(self):
        cmds = [
            CommandDoc(name="a", description="", group=""),
            CommandDoc(name="b", description="", group="special"),
        ]
        groups = group_commands(cmds)
        assert "General" in groups
        assert "special" in groups

    def test_empty_list(self):
        groups = group_commands([])
        assert groups == {}


# ---------------------------------------------------------------------------
# format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def test_contains_header(self):
        cmds = [CommandDoc(name="run", description="Run stuff")]
        md = format_markdown(cmds)
        assert "# CLI Reference" in md

    def test_contains_toc(self):
        cmds = [CommandDoc(name="run", description="Run stuff")]
        md = format_markdown(cmds)
        assert "## Table of Contents" in md
        assert "[run]" in md

    def test_contains_command_section(self):
        cmds = [CommandDoc(name="run", description="Run stuff")]
        md = format_markdown(cmds)
        assert "### run" in md
        assert "Run stuff" in md

    def test_contains_flag_table(self):
        cmds = [
            CommandDoc(
                name="run",
                description="Run",
                arguments=[ArgDoc(flags=["--limit"], help="Max", type_name="int")],
            )
        ]
        md = format_markdown(cmds)
        assert "| Flag |" in md
        assert "`--limit`" in md

    def test_contains_examples(self):
        cmds = [
            CommandDoc(
                name="run",
                description="Run",
                examples=["run --limit 10"],
            )
        ]
        md = format_markdown(cmds)
        assert "run --limit 10" in md

    def test_choices_in_help(self):
        cmds = [
            CommandDoc(
                name="x",
                description="y",
                arguments=[
                    ArgDoc(flags=["--fmt"], help="Format", choices=["json", "csv"])
                ],
            )
        ]
        md = format_markdown(cmds)
        assert "json, csv" in md

    def test_empty_commands(self):
        md = format_markdown([])
        assert "# CLI Reference" in md


# ---------------------------------------------------------------------------
# format_plain_text
# ---------------------------------------------------------------------------


class TestFormatPlainText:
    def test_contains_header(self):
        cmds = [CommandDoc(name="run", description="Run stuff")]
        txt = format_plain_text(cmds)
        assert "CLI Reference" in txt

    def test_contains_command_name(self):
        cmds = [CommandDoc(name="run", description="Run stuff")]
        txt = format_plain_text(cmds)
        assert "run" in txt
        assert "Run stuff" in txt

    def test_contains_flag_info(self):
        cmds = [
            CommandDoc(
                name="run",
                description="Run",
                arguments=[ArgDoc(flags=["--limit"], help="Max", type_name="int")],
            )
        ]
        txt = format_plain_text(cmds)
        assert "--limit" in txt
        assert "(int)" in txt

    def test_required_annotation(self):
        cmds = [
            CommandDoc(
                name="x",
                description="y",
                arguments=[
                    ArgDoc(flags=["--path"], help="Path", required=True)
                ],
            )
        ]
        txt = format_plain_text(cmds)
        assert "[required]" in txt

    def test_default_annotation(self):
        cmds = [
            CommandDoc(
                name="x",
                description="y",
                arguments=[
                    ArgDoc(flags=["--n"], help="Count", default=10)
                ],
            )
        ]
        txt = format_plain_text(cmds)
        assert "[default: 10]" in txt

    def test_choices_annotation(self):
        cmds = [
            CommandDoc(
                name="x",
                description="y",
                arguments=[
                    ArgDoc(flags=["--fmt"], help="Format", choices=["a", "b"])
                ],
            )
        ]
        txt = format_plain_text(cmds)
        assert "[choices: a, b]" in txt

    def test_group_header(self):
        cmds = [CommandDoc(name="x", description="y", group="Analysis")]
        txt = format_plain_text(cmds)
        assert "[Analysis]" in txt

    def test_examples_in_output(self):
        cmds = [
            CommandDoc(
                name="run", description="Run", examples=["run --flag"]
            )
        ]
        txt = format_plain_text(cmds)
        assert "run --flag" in txt


# ---------------------------------------------------------------------------
# generate_reference
# ---------------------------------------------------------------------------


class TestGenerateReference:
    def test_markdown_format(self):
        parser = _make_parser()
        ref = generate_reference(parser, fmt="markdown")
        assert "# CLI Reference" in ref
        assert "### run" in ref

    def test_plain_format(self):
        parser = _make_parser()
        ref = generate_reference(parser, fmt="plain")
        assert "CLI Reference" in ref
        assert "run" in ref

    def test_unknown_format_raises(self):
        parser = _make_parser()
        with pytest.raises(ValueError, match="Unknown format"):
            generate_reference(parser, fmt="html")

    def test_default_is_markdown(self):
        parser = _make_parser()
        ref = generate_reference(parser)
        assert "# CLI Reference" in ref

    def test_empty_parser(self):
        parser = _make_empty_parser()
        ref = generate_reference(parser, fmt="markdown")
        assert "# CLI Reference" in ref

    def test_full_roundtrip_extraction(self):
        """Extract commands from parser, verify structure, format to markdown."""
        parser = _make_parser()
        cmds = extract_commands(parser)
        assert len(cmds) == 2
        for cmd in cmds:
            d = cmd.as_dict()
            restored = CommandDoc.from_dict(d)
            assert restored.name == cmd.name
            assert len(restored.arguments) == len(cmd.arguments)
        md = format_markdown(cmds)
        assert len(md) > 100
