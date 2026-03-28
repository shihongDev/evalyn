"""Plugin system for third-party metric, instrumentor, and storage backend plugins.

Discovers and manages plugins via entry points. Supports metric, instrumentor,
storage, and analyzer plugin types with a global registry singleton.
"""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


VALID_PLUGIN_TYPES = {"metric", "instrumentor", "storage", "analyzer"}

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+")


@dataclass
class PluginInfo:
    """Metadata for a registered plugin."""

    name: str
    version: str = "0.0.0"
    plugin_type: str = "metric"
    description: str = ""
    author: str = ""
    entry_point: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginInfo:
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            plugin_type=data.get("plugin_type", "metric"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            entry_point=data.get("entry_point", ""),
        )


@dataclass
class PluginRegistry:
    """Registry of discovered and registered plugins."""

    plugins: Dict[str, PluginInfo] = field(default_factory=dict)

    def register(self, info: PluginInfo) -> None:
        """Register a plugin. Raises ValueError if name already registered."""
        if info.name in self.plugins:
            raise ValueError(f"Plugin '{info.name}' is already registered")
        self.plugins[info.name] = info

    def unregister(self, name: str) -> bool:
        """Remove a plugin by name. Returns True if removed, False if not found."""
        if name in self.plugins:
            del self.plugins[name]
            return True
        return False

    def get(self, name: str) -> Optional[PluginInfo]:
        """Look up a plugin by name."""
        return self.plugins.get(name)

    def list_all(self) -> List[PluginInfo]:
        """Return all registered plugins."""
        return list(self.plugins.values())

    def list_by_type(self, plugin_type: str) -> List[PluginInfo]:
        """Filter plugins by type."""
        return [p for p in self.plugins.values() if p.plugin_type == plugin_type]

    def is_registered(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self.plugins

    @property
    def count(self) -> int:
        """Number of registered plugins."""
        return len(self.plugins)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plugins": {name: info.as_dict() for name, info in self.plugins.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginRegistry:
        plugins = {}
        for name, info_data in data.get("plugins", {}).items():
            plugins[name] = PluginInfo.from_dict(info_data)
        return cls(plugins=plugins)


# Module-level singleton registry
_global_registry: Optional[PluginRegistry] = None


def get_global_registry() -> PluginRegistry:
    """Return the module-level singleton registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Clear the global registry. Useful for testing."""
    global _global_registry
    _global_registry = None


def discover_plugins(
    entry_point_group: str = "evalyn.plugins",
) -> List[PluginInfo]:
    """Discover plugins via importlib.metadata entry_points.

    Returns a list of PluginInfo for each entry point found in the group.
    Returns an empty list if no entry points are found or on error.
    """
    found: List[PluginInfo] = []
    try:
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.9+ dict-like
        if hasattr(eps, "select"):
            group_eps = eps.select(group=entry_point_group)
        elif isinstance(eps, dict):
            group_eps = eps.get(entry_point_group, [])
        else:
            group_eps = [ep for ep in eps if ep.group == entry_point_group]

        for ep in group_eps:
            info = PluginInfo(
                name=ep.name,
                entry_point=f"{ep.value}" if hasattr(ep, "value") else str(ep),
            )
            found.append(info)
    except Exception:
        pass
    return found


def validate_plugin(info: PluginInfo) -> Tuple[bool, List[str]]:
    """Validate plugin info.

    Checks:
    - name is non-empty
    - plugin_type is one of the valid types
    - version matches semver-like format (digits.digits.digits)

    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []
    if not info.name or not info.name.strip():
        errors.append("name must be non-empty")
    if info.plugin_type not in VALID_PLUGIN_TYPES:
        errors.append(
            f"plugin_type must be one of {sorted(VALID_PLUGIN_TYPES)}, "
            f"got '{info.plugin_type}'"
        )
    if not _VERSION_PATTERN.match(info.version):
        errors.append(f"version must match X.Y.Z format, got '{info.version}'")
    return (len(errors) == 0, errors)


def create_plugin_info(name: str, plugin_type: str, **kwargs: Any) -> PluginInfo:
    """Factory for creating PluginInfo with defaults."""
    return PluginInfo(name=name, plugin_type=plugin_type, **kwargs)
