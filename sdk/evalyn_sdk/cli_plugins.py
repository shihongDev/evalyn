"""CLI plugin system for registering custom commands via entry points.

Discovers installed plugins using importlib.metadata entry points and
manages a registry of available CLI command plugins.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any


@dataclass
class PluginInfo:
    """Metadata for a CLI command plugin."""

    name: str
    module_path: str
    description: str = ""
    version: str = ""
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module_path": self.module_path,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginInfo:
        return cls(
            name=data.get("name", ""),
            module_path=data.get("module_path", ""),
            description=data.get("description", ""),
            version=data.get("version", ""),
            enabled=data.get("enabled", True),
        )


class PluginRegistry:
    """Registry for CLI command plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}

    def register(self, info: PluginInfo) -> None:
        """Register a plugin."""
        self._plugins[info.name] = info

    def unregister(self, name: str) -> bool:
        """Remove a plugin. Returns True if found and removed."""
        if name in self._plugins:
            del self._plugins[name]
            return True
        return False

    def get(self, name: str) -> PluginInfo | None:
        """Look up a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[PluginInfo]:
        """Return all registered plugins."""
        return list(self._plugins.values())

    def list_enabled(self) -> list[PluginInfo]:
        """Return only enabled plugins."""
        return [p for p in self._plugins.values() if p.enabled]

    def is_registered(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins


def discover_entry_points(group: str = "evalyn.commands") -> list[PluginInfo]:
    """Discover installed plugins via importlib.metadata entry points.

    Scans the given entry point group for installed packages that
    provide CLI command plugins. Returns empty list if no entry
    points are found or on ImportError.
    """
    plugins: list[PluginInfo] = []
    try:
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.9+ supports .select()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=group)
        elif isinstance(eps, dict):
            group_eps = eps.get(group, [])
        else:
            group_eps = [ep for ep in eps if ep.group == group]

        for ep in group_eps:
            # Try to get the distribution version
            version = ""
            try:
                if hasattr(ep, "dist") and ep.dist is not None:
                    version = ep.dist.version
            except Exception:
                pass

            plugins.append(PluginInfo(
                name=ep.name,
                module_path=ep.value,
                description="",
                version=version,
                enabled=True,
            ))
    except Exception:
        return []

    return plugins


def create_default_registry() -> PluginRegistry:
    """Create a registry populated with discovered entry point plugins."""
    registry = PluginRegistry()
    for plugin in discover_entry_points():
        registry.register(plugin)
    return registry


def format_plugin_list(plugins: list[PluginInfo]) -> str:
    """Format a human-readable table of plugins."""
    if not plugins:
        return "No plugins registered."

    lines: list[str] = []

    # Compute column widths
    name_width = max(len(p.name) for p in plugins)
    name_width = max(name_width, 4)  # min "Name" header width
    version_width = max((len(p.version) for p in plugins), default=7)
    version_width = max(version_width, 7)  # min "Version" header width
    status_width = 8  # "Enabled" / "Disabled"

    header = (
        f"{'Name':<{name_width}}  "
        f"{'Version':<{version_width}}  "
        f"{'Status':<{status_width}}  "
        f"Description"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for p in plugins:
        status = "Enabled" if p.enabled else "Disabled"
        line = (
            f"{p.name:<{name_width}}  "
            f"{p.version:<{version_width}}  "
            f"{status:<{status_width}}  "
            f"{p.description}"
        )
        lines.append(line)

    return "\n".join(lines)
