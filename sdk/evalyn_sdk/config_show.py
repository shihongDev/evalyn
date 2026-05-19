"""Display effective merged configuration from all sources.

Merges global, project, env, and flag configs with clear priority
and tracks which source each value came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Priority order (highest to lowest): flag > env > project > global > default
SOURCE_PRIORITY = ["flag", "env", "project", "global", "default"]


@dataclass
class ConfigSource:
    """A single configuration entry with its origin."""

    key: str
    value: Any
    source: str  # "global", "project", "env", "flag", "default"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConfigSource:
        return cls(
            key=data["key"],
            value=data["value"],
            source=data.get("source", "default"),
        )


@dataclass
class MergedConfig:
    """All resolved configuration entries with source tracking."""

    entries: List[ConfigSource] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "sources_used": list(self.sources_used),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MergedConfig:
        return cls(
            entries=[ConfigSource.from_dict(e) for e in data.get("entries", [])],
            sources_used=list(data.get("sources_used", [])),
        )


def merge_configs(
    global_config: Dict[str, Any],
    project_config: Dict[str, Any],
    env_overrides: Dict[str, Any],
    flag_overrides: Dict[str, Any],
    defaults: Dict[str, Any],
) -> MergedConfig:
    """Merge configs with priority: flag > env > project > global > default.

    Each value is tagged with the highest-priority source that provided it.
    """
    # Collect all keys across every source
    all_keys: set[str] = set()
    all_keys.update(defaults.keys())
    all_keys.update(global_config.keys())
    all_keys.update(project_config.keys())
    all_keys.update(env_overrides.keys())
    all_keys.update(flag_overrides.keys())

    # Layers ordered from highest to lowest priority
    layers: List[tuple[str, Dict[str, Any]]] = [
        ("flag", flag_overrides),
        ("env", env_overrides),
        ("project", project_config),
        ("global", global_config),
        ("default", defaults),
    ]

    entries: List[ConfigSource] = []
    sources_used: set[str] = set()

    for key in sorted(all_keys):
        for source_name, layer in layers:
            if key in layer:
                entries.append(ConfigSource(key=key, value=layer[key], source=source_name))
                sources_used.add(source_name)
                break

    return MergedConfig(
        entries=entries,
        sources_used=sorted(sources_used, key=lambda s: SOURCE_PRIORITY.index(s)),
    )


def get_env_overrides(prefix: str = "EVALYN_") -> Dict[str, str]:
    """Read environment variables with the given prefix.

    Strips the prefix to produce the key name, lowercased.
    Example: EVALYN_PROVIDER -> provider
    """
    result: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            stripped = key[len(prefix):].lower()
            if stripped:
                result[stripped] = value
    return result


def get_effective_value(key: str, config: MergedConfig) -> Optional[ConfigSource]:
    """Look up a specific key in the merged config."""
    for entry in config.entries:
        if entry.key == key:
            return entry
    return None


def highlight_source(source: str) -> str:
    """Return a bracketed source label."""
    return f"[{source}]"


def format_config_show(config: MergedConfig) -> str:
    """Format merged config as a human-readable table.

    Shows key, value, and source for each entry.
    """
    if not config.entries:
        return "No configuration entries."

    # Calculate column widths
    key_width = max(len(e.key) for e in config.entries)
    val_width = max(len(str(e.value)) for e in config.entries)
    src_width = max(len(highlight_source(e.source)) for e in config.entries)

    # Minimum widths
    key_width = max(key_width, 3)
    val_width = max(val_width, 5)
    src_width = max(src_width, 6)

    lines: List[str] = []
    header = f"{'Key':<{key_width}}  {'Value':<{val_width}}  {'Source':<{src_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    for entry in config.entries:
        src_label = highlight_source(entry.source)
        lines.append(
            f"{entry.key:<{key_width}}  {str(entry.value):<{val_width}}  {src_label:<{src_width}}"
        )

    lines.append("")
    lines.append(f"Sources used: {', '.join(config.sources_used)}")
    return "\n".join(lines)
