"""CLI alias support: user-defined command aliases for evalyn CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Alias:
    """A user-defined command alias."""

    name: str
    command: str
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "command": self.command,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Alias:
        return cls(
            name=data["name"],
            command=data["command"],
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Built-in aliases
# ---------------------------------------------------------------------------

BUILTIN_ALIASES: Dict[str, str] = {
    "q": "quickstart",
    "ls": "list-runs",
    "st": "status",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AliasRegistry:
    """Manages a collection of command aliases."""

    def __init__(self) -> None:
        self.aliases: Dict[str, Alias] = {}

    def register(self, alias: Alias) -> None:
        """Register an alias, overwriting any existing alias with the same name."""
        self.aliases[alias.name] = alias

    def unregister(self, name: str) -> bool:
        """Remove an alias by name. Returns True if it existed."""
        if name in self.aliases:
            del self.aliases[name]
            return True
        return False

    def get(self, name: str) -> Optional[Alias]:
        """Look up an alias by name."""
        return self.aliases.get(name)

    def resolve(self, name: str) -> str:
        """Return the full command for an alias, or the name unchanged."""
        alias = self.aliases.get(name)
        if alias is not None:
            return alias.command
        return name

    def list_aliases(self) -> List[Alias]:
        """Return all registered aliases sorted by name."""
        return sorted(self.aliases.values(), key=lambda a: a.name)

    def is_alias(self, name: str) -> bool:
        """Check whether a name is a registered alias."""
        return name in self.aliases


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_aliases_from_config(config: dict) -> AliasRegistry:
    """Load aliases from a config dict's ``aliases`` section.

    Expected format::

        {"aliases": {"q": "quickstart", "fast-eval": "run-eval --workers 8"}}

    Values can be plain command strings (description defaults to empty)
    or dicts with ``command`` and optional ``description`` keys.
    """
    registry = AliasRegistry()
    aliases_section = config.get("aliases", {})
    for name, value in aliases_section.items():
        if isinstance(value, str):
            registry.register(Alias(name=name, command=value))
        elif isinstance(value, dict):
            registry.register(
                Alias(
                    name=name,
                    command=value["command"],
                    description=value.get("description", ""),
                )
            )
    return registry


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------


def parse_alias_command(alias_command: str, extra_args: List[str]) -> List[str]:
    """Split an alias command string and append extra args.

    >>> parse_alias_command("run-eval --workers 8", ["--verbose"])
    ['run-eval', '--workers', '8', '--verbose']
    """
    parts = alias_command.split()
    parts.extend(extra_args)
    return parts


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_alias_list(aliases: List[Alias]) -> str:
    """Format a list of aliases as a human-readable table.

    Returns an empty string when the list is empty.
    """
    if not aliases:
        return ""

    # Compute column widths
    name_width = max(len(a.name) for a in aliases)
    cmd_width = max(len(a.command) for a in aliases)

    lines: List[str] = []
    header = f"{'ALIAS':<{name_width}}  {'COMMAND':<{cmd_width}}  DESCRIPTION"
    lines.append(header)
    lines.append("-" * len(header))
    for a in aliases:
        desc = a.description or ""
        lines.append(f"{a.name:<{name_width}}  {a.command:<{cmd_width}}  {desc}")
    return "\n".join(lines)
