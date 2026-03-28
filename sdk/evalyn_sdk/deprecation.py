from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DeprecationEntry:
    """A single deprecation record mapping old name to new name."""

    old_name: str
    new_name: str
    since_version: str
    removal_version: str
    category: str  # "config", "flag", or "api"
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "old_name": self.old_name,
            "new_name": self.new_name,
            "since_version": self.since_version,
            "removal_version": self.removal_version,
            "category": self.category,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeprecationEntry:
        return cls(
            old_name=data["old_name"],
            new_name=data["new_name"],
            since_version=data["since_version"],
            removal_version=data["removal_version"],
            category=data["category"],
            message=data["message"],
        )


@dataclass
class DeprecationWarning_:
    """A warning produced when a deprecated name is encountered."""

    entry: DeprecationEntry
    context: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.as_dict(),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeprecationWarning_:
        return cls(
            entry=DeprecationEntry.from_dict(data["entry"]),
            context=data["context"],
        )


class DeprecationRegistry:
    """Registry of deprecated names with lookup and migration support."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], List[DeprecationEntry]] = {}

    def register(self, entry: DeprecationEntry) -> None:
        """Register a deprecation entry."""
        key = (entry.old_name, entry.category)
        self._entries.setdefault(key, []).append(entry)

    def check(self, name: str, category: str) -> Optional[DeprecationEntry]:
        """Check if a name is deprecated in a given category.

        Returns the deprecation entry if found, None otherwise.
        """
        key = (name, category)
        entries = self._entries.get(key)
        if entries:
            return entries[-1]
        return None

    def check_config(self, config: dict) -> List[DeprecationWarning_]:
        """Check all keys in a config dict for deprecations.

        Returns a list of warnings for any deprecated keys found.
        """
        warnings: List[DeprecationWarning_] = []
        for key in config:
            entry = self.check(key, "config")
            if entry is not None:
                context = f"Config key '{key}' is deprecated"
                warnings.append(DeprecationWarning_(entry=entry, context=context))
        return warnings

    def get_all(self, category: Optional[str] = None) -> List[DeprecationEntry]:
        """List all deprecation entries, optionally filtered by category."""
        result: List[DeprecationEntry] = []
        for entries in self._entries.values():
            for entry in entries:
                if category is None or entry.category == category:
                    result.append(entry)
        return result

    def migrate_config(self, config: dict) -> Tuple[dict, List[str]]:
        """Migrate a config dict by replacing deprecated keys with new ones.

        Returns (new_config, list_of_changes) where each change is a
        human-readable description of what was migrated.
        """
        new_config: dict = {}
        changes: List[str] = []
        for key, value in config.items():
            entry = self.check(key, "config")
            if entry is not None:
                new_config[entry.new_name] = value
                changes.append(
                    f"Renamed '{entry.old_name}' -> '{entry.new_name}'"
                )
            else:
                new_config[key] = value
        return new_config, changes


def format_warnings(warnings: List[DeprecationWarning_]) -> str:
    """Format a list of deprecation warnings for display."""
    if not warnings:
        return "No deprecation warnings."
    lines: List[str] = []
    lines.append(f"Found {len(warnings)} deprecation warning(s):")
    for w in warnings:
        e = w.entry
        lines.append(
            f"  - {w.context}: use '{e.new_name}' instead "
            f"(deprecated since {e.since_version}, "
            f"removal in {e.removal_version})"
        )
        if e.message:
            lines.append(f"    {e.message}")
    return "\n".join(lines)


def format_migration_report(changes: List[str]) -> str:
    """Format migration changes for display."""
    if not changes:
        return "No migrations needed."
    lines: List[str] = []
    lines.append(f"Applied {len(changes)} migration(s):")
    for change in changes:
        lines.append(f"  - {change}")
    return "\n".join(lines)


# Default registry with example deprecations
DEFAULT_REGISTRY = DeprecationRegistry()

_DEFAULT_ENTRIES = [
    DeprecationEntry(
        old_name="llm_provider",
        new_name="provider",
        since_version="0.8.0",
        removal_version="1.0.0",
        category="config",
        message="The 'llm_provider' key has been renamed to 'provider'.",
    ),
    DeprecationEntry(
        old_name="max_retries",
        new_name="retry_limit",
        since_version="0.9.0",
        removal_version="1.0.0",
        category="config",
        message="The 'max_retries' key has been renamed to 'retry_limit'.",
    ),
    DeprecationEntry(
        old_name="--verbose",
        new_name="--log-level",
        since_version="0.8.0",
        removal_version="1.0.0",
        category="flag",
        message="Use '--log-level debug' instead of '--verbose'.",
    ),
    DeprecationEntry(
        old_name="run_eval",
        new_name="execute_eval",
        since_version="0.7.0",
        removal_version="1.0.0",
        category="api",
        message="The 'run_eval' function has been renamed to 'execute_eval'.",
    ),
    DeprecationEntry(
        old_name="output_dir",
        new_name="results_dir",
        since_version="0.9.0",
        removal_version="1.1.0",
        category="config",
        message="The 'output_dir' key has been renamed to 'results_dir'.",
    ),
]

for _entry in _DEFAULT_ENTRIES:
    DEFAULT_REGISTRY.register(_entry)
