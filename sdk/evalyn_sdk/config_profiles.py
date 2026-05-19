"""Configuration profiles: named environment profiles (dev/staging/prod).

Allows defining per-environment overrides in evalyn.yaml:

    profiles:
      dev:
        provider: ollama
        max_workers: 1
        db: data/dev/traces.sqlite
      staging:
        provider: gemini
        max_workers: 4
      prod:
        provider: gemini
        max_workers: 2
        db: data/prod/traces.sqlite

Select via --profile flag or EVALYN_PROFILE env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ConfigProfile:
    """A named configuration profile with environment-specific overrides."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value from this profile's overrides."""
        return self.overrides.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "overrides": dict(self.overrides),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ConfigProfile:
        description = data.pop("description", "") if isinstance(data, dict) else ""
        return cls(name=name, overrides=data, description=description)


def get_active_profile_name() -> str | None:
    """Get the active profile from environment or return None.

    Checks EVALYN_PROFILE env var.
    """
    return os.environ.get("EVALYN_PROFILE")


def load_profiles(config: dict[str, Any]) -> dict[str, ConfigProfile]:
    """Load all profiles from evalyn.yaml config.

    Args:
        config: Parsed evalyn.yaml config dict.

    Returns:
        Dict of profile_name -> ConfigProfile.
    """
    profiles_data = config.get("profiles", {})
    profiles = {}
    for name, data in profiles_data.items():
        if isinstance(data, dict):
            profiles[name] = ConfigProfile.from_dict(name, dict(data))
    return profiles


def resolve_config_with_profile(
    base_config: dict[str, Any],
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Merge a profile's overrides into the base config.

    Resolution order:
    1. Base config (evalyn.yaml top-level)
    2. Profile overrides (evalyn.yaml profiles.<name>.*)
    3. Environment variables (take precedence over everything)

    Args:
        base_config: The full parsed evalyn.yaml.
        profile_name: Profile to apply. If None, checks EVALYN_PROFILE env var.

    Returns:
        Merged config dict with profile overrides applied.
    """
    # Determine active profile
    name = profile_name or get_active_profile_name()
    if not name:
        return dict(base_config)

    profiles = load_profiles(base_config)
    profile = profiles.get(name)
    if not profile:
        return dict(base_config)

    # Deep merge: profile overrides base
    merged = dict(base_config)
    for key, value in profile.overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Merge nested dicts
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    # Mark active profile in merged config
    merged["_active_profile"] = name

    return merged


def list_profiles(config: dict[str, Any]) -> list:
    """List available profile names and descriptions.

    Args:
        config: Parsed evalyn.yaml.

    Returns:
        List of (name, description) tuples.
    """
    profiles = load_profiles(config)
    return [(p.name, p.description) for p in sorted(profiles.values(), key=lambda p: p.name)]
