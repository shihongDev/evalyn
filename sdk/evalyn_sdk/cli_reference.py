"""CLI reference auto-generation from argparse definitions.

Introspects argparse parsers to produce structured documentation
in markdown or plain text format. Pure Python, no external dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgDoc:
    """Documentation for a single CLI argument."""

    flags: list[str]
    help: str
    default: Any = None
    required: bool = False
    choices: list[str] | None = None
    type_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "flags": list(self.flags),
            "help": self.help,
            "default": self.default,
            "required": self.required,
            "choices": list(self.choices) if self.choices else None,
            "type_name": self.type_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgDoc:
        return cls(
            flags=data.get("flags", []),
            help=data.get("help", ""),
            default=data.get("default"),
            required=data.get("required", False),
            choices=data.get("choices"),
            type_name=data.get("type_name", ""),
        )


@dataclass
class CommandDoc:
    """Documentation for a single CLI command (subcommand)."""

    name: str
    description: str
    arguments: list[ArgDoc] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    group: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [a.as_dict() for a in self.arguments],
            "examples": list(self.examples),
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandDoc:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            arguments=[ArgDoc.from_dict(a) for a in data.get("arguments", [])],
            examples=data.get("examples", []),
            group=data.get("group", ""),
        )


def extract_arguments(parser: argparse.ArgumentParser) -> list[ArgDoc]:
    """Extract argument documentation from a single parser."""
    args: list[ArgDoc] = []
    for action in parser._actions:
        # Skip the default help action and subparsers
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue

        flags = list(action.option_strings) if action.option_strings else []
        if not flags and action.dest != argparse.SUPPRESS:
            # Positional argument
            flags = [action.dest]

        help_text = action.help or ""
        if help_text == argparse.SUPPRESS:
            continue

        default = action.default
        if default is argparse.SUPPRESS:
            default = None

        required = getattr(action, "required", False)
        # Positional arguments are always required unless they have nargs='?'/'*'
        if not action.option_strings:
            required = action.nargs not in ("?", "*") if action.nargs else True

        choices = None
        if action.choices is not None:
            choices = [str(c) for c in action.choices]

        type_name = ""
        if action.type is not None:
            type_name = getattr(action.type, "__name__", str(action.type))
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            type_name = "flag"
        elif isinstance(action, argparse._CountAction):
            type_name = "count"

        args.append(
            ArgDoc(
                flags=flags,
                help=help_text,
                default=default,
                required=required,
                choices=choices,
                type_name=type_name,
            )
        )
    return args


def extract_commands(parser: argparse.ArgumentParser) -> list[CommandDoc]:
    """Introspect an argparse parser to extract all subcommands."""
    commands: list[CommandDoc] = []

    # Find the subparsers action
    subparsers_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break

    if subparsers_action is None:
        return commands

    for name, subparser in subparsers_action.choices.items():
        description = subparser.description or subparser.format_usage().strip()
        # Try to get help from the subparsers action
        if not subparser.description:
            help_text = subparsers_action._name_parser_map.get(name)
            if help_text and hasattr(subparsers_action, "_choices_actions"):
                for choice_action in subparsers_action._choices_actions:
                    if choice_action.dest == name:
                        description = choice_action.help or description
                        break

        arguments = extract_arguments(subparser)

        # Extract examples from epilog if present
        examples: list[str] = []
        if subparser.epilog:
            for line in subparser.epilog.strip().splitlines():
                stripped = line.strip()
                if stripped:
                    examples.append(stripped)

        commands.append(
            CommandDoc(
                name=name,
                description=description,
                arguments=arguments,
                examples=examples,
                group="",
            )
        )

    return commands


def group_commands(commands: list[CommandDoc]) -> dict[str, list[CommandDoc]]:
    """Group commands by their group field.

    Commands with an empty group are placed under "General".
    """
    groups: dict[str, list[CommandDoc]] = {}
    for cmd in commands:
        key = cmd.group if cmd.group else "General"
        groups.setdefault(key, []).append(cmd)
    return groups


def format_markdown(commands: list[CommandDoc]) -> str:
    """Format commands as a Markdown reference with TOC and flag tables."""
    lines: list[str] = []

    lines.append("# CLI Reference")
    lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    grouped = group_commands(commands)
    for group_name, group_cmds in grouped.items():
        lines.append(f"### {group_name}")
        lines.append("")
        for cmd in group_cmds:
            anchor = cmd.name.replace(" ", "-").lower()
            lines.append(f"- [{cmd.name}](#{anchor})")
        lines.append("")

    # Command sections
    lines.append("## Commands")
    lines.append("")
    for cmd in commands:
        lines.append(f"### {cmd.name}")
        lines.append("")
        lines.append(cmd.description)
        lines.append("")

        if cmd.arguments:
            lines.append("| Flag | Type | Required | Default | Description |")
            lines.append("|------|------|----------|---------|-------------|")
            for arg in cmd.arguments:
                flag_str = ", ".join(arg.flags)
                type_str = arg.type_name or "-"
                req_str = "yes" if arg.required else "no"
                default_str = str(arg.default) if arg.default is not None else "-"
                help_str = arg.help
                if arg.choices:
                    help_str += f" (choices: {', '.join(arg.choices)})"
                lines.append(
                    f"| `{flag_str}` | {type_str} | {req_str} | {default_str} | {help_str} |"
                )
            lines.append("")

        if cmd.examples:
            lines.append("**Examples:**")
            lines.append("")
            for ex in cmd.examples:
                lines.append("```")
                lines.append(ex)
                lines.append("```")
            lines.append("")

    return "\n".join(lines)


def format_plain_text(commands: list[CommandDoc]) -> str:
    """Format commands as plain text reference."""
    lines: list[str] = []

    lines.append("CLI Reference")
    lines.append("=" * 60)
    lines.append("")

    grouped = group_commands(commands)
    for group_name, group_cmds in grouped.items():
        lines.append(f"[{group_name}]")
        lines.append("")
        for cmd in group_cmds:
            lines.append(f"  {cmd.name}")
            lines.append(f"    {cmd.description}")
            lines.append("")

            if cmd.arguments:
                for arg in cmd.arguments:
                    flag_str = ", ".join(arg.flags)
                    parts = [f"    {flag_str}"]
                    if arg.type_name:
                        parts.append(f"  ({arg.type_name})")
                    if arg.required:
                        parts.append("  [required]")
                    if arg.default is not None:
                        parts.append(f"  [default: {arg.default}]")
                    if arg.choices:
                        parts.append(f"  [choices: {', '.join(arg.choices)}]")
                    lines.append("".join(parts))
                    if arg.help:
                        lines.append(f"      {arg.help}")
                lines.append("")

            if cmd.examples:
                lines.append("    Examples:")
                for ex in cmd.examples:
                    lines.append(f"      {ex}")
                lines.append("")

    return "\n".join(lines)


def generate_reference(parser: argparse.ArgumentParser, fmt: str = "markdown") -> str:
    """Generate CLI reference documentation from an argparse parser.

    Args:
        parser: The root argparse.ArgumentParser with subcommands.
        fmt: Output format - "markdown" or "plain".

    Returns:
        Formatted reference string.
    """
    commands = extract_commands(parser)
    if fmt == "markdown":
        return format_markdown(commands)
    elif fmt == "plain":
        return format_plain_text(commands)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'markdown' or 'plain'.")
