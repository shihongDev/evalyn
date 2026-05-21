from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstrumentationTarget:
    module_name: str
    class_name: str = ""
    method_name: str = ""
    enabled: bool = True

    @property
    def full_path(self) -> str:
        """Return 'module.class.method' formatted string."""
        parts = [self.module_name]
        if self.class_name:
            parts.append(self.class_name)
        if self.method_name:
            parts.append(self.method_name)
        return ".".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentationTarget:
        return cls(
            module_name=data["module_name"],
            class_name=data.get("class_name", ""),
            method_name=data.get("method_name", ""),
            enabled=data.get("enabled", True),
        )


@dataclass
class InstrumentationConfig:
    targets: list[InstrumentationTarget] = field(default_factory=list)
    mode: str = "include"  # "include" or "exclude"

    def as_dict(self) -> dict[str, Any]:
        return {
            "targets": [t.as_dict() for t in self.targets],
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentationConfig:
        return cls(
            targets=[InstrumentationTarget.from_dict(t) for t in data.get("targets", [])],
            mode=data.get("mode", "include"),
        )


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def match_target(path: str, target: InstrumentationTarget) -> bool:
    """Check if path matches a target.

    Supports partial matching: if the target only specifies a module_name,
    any path starting with that module is a match.
    """
    if not target.enabled:
        return False

    target_path = target.full_path
    # Exact match, or partial/prefix match at a dot boundary.
    return path == target_path or path.startswith(target_path + ".")


def should_instrument(target_path: str, config: InstrumentationConfig) -> bool:
    """Check if a given path should be instrumented.

    include mode: only paths matching an enabled target are instrumented.
    exclude mode: all paths except those matching a target are instrumented.
    """
    matches = any(match_target(target_path, t) for t in config.targets)

    if config.mode == "include":
        return matches
    # exclude mode
    return not matches


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _parse_path(path: str) -> InstrumentationTarget:
    """Parse a 'module.class.method' string into an InstrumentationTarget."""
    parts = path.split(".")
    if len(parts) >= 3:
        return InstrumentationTarget(
            module_name=parts[0],
            class_name=parts[1],
            method_name=parts[2],
        )
    elif len(parts) == 2:
        return InstrumentationTarget(
            module_name=parts[0],
            class_name=parts[1],
        )
    else:
        return InstrumentationTarget(module_name=parts[0])


def create_include_config(targets: list[str]) -> InstrumentationConfig:
    """Factory: create an include-mode config from dotted path strings."""
    return InstrumentationConfig(
        targets=[_parse_path(t) for t in targets],
        mode="include",
    )


def create_exclude_config(targets: list[str]) -> InstrumentationConfig:
    """Factory: create an exclude-mode config from dotted path strings."""
    return InstrumentationConfig(
        targets=[_parse_path(t) for t in targets],
        mode="exclude",
    )


def list_instrumented_targets(config: InstrumentationConfig, all_targets: list[str]) -> list[str]:
    """Return which targets from all_targets would be instrumented."""
    return [t for t in all_targets if should_instrument(t, config)]
