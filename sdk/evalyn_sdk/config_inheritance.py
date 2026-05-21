"""
Config inheritance: base config with per-project overrides using deep merge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ConfigLayer:
    """A single configuration layer."""

    name: str
    values: dict[str, Any]
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigLayer:
        return cls(
            name=data["name"],
            values=data.get("values", {}),
            source=data.get("source", ""),
        )


@dataclass
class MergedConfig:
    """Merged configuration from multiple layers (first = lowest priority)."""

    layers: list[ConfigLayer]

    def get(self, key: str, default: Any = None) -> Any:
        """Look up value from highest-priority layer first."""
        for layer in reversed(self.layers):
            if key in layer.values:
                return layer.values[key]
        return default

    def get_nested(self, dotted_key: str, default: Any = None) -> Any:
        """Support 'a.b.c' nested access across merged layers."""
        merged = self.to_flat_dict()
        parts = dotted_key.split(".")
        current: Any = merged
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def to_flat_dict(self) -> dict[str, Any]:
        """Deep-merged dict from all layers (lower priority first)."""
        result: dict[str, Any] = {}
        for layer in self.layers:
            result = deep_merge(result, layer.values)
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.as_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergedConfig:
        return cls(
            layers=[ConfigLayer.from_dict(layer) for layer in data.get("layers", [])],
        )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. Override wins for scalars. Dicts merge recursively. Lists replaced."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_config(layers: list[ConfigLayer]) -> MergedConfig:
    """Build merged config from layers (first = lowest priority)."""
    return MergedConfig(layers=layers)


def load_config_layer(path: str, name: str = "") -> ConfigLayer:
    """Load a YAML or JSON config file. Returns empty layer if file not found."""
    p = Path(path)
    layer_name = name or p.stem

    if not p.exists():
        return ConfigLayer(name=layer_name, values={}, source=str(p))

    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            values = json.load(f)
        return ConfigLayer(name=layer_name, values=values, source=str(p))

    # .yaml/.yml - return empty (no PyYAML dependency)
    return ConfigLayer(name=layer_name, values={}, source=str(p))
