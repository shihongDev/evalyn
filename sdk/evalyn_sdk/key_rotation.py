"""API key rotation support for graceful key switching.

Pure Python, no external dependencies. Provides key management,
rotation, and environment-based key loading for eval providers.
Never stores or logs full key values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List


def mask_key(key: str) -> str:
    """Mask a key, showing only the last 4 characters.

    Returns '****' followed by the last 4 chars, or the full key
    masked if shorter than 4 characters.
    """
    if len(key) <= 4:
        return "*" * len(key)
    return "****" + key[-4:]


@dataclass
class KeyConfig:
    """Configuration for a provider's API keys."""

    provider: str
    keys: List[str]
    active_index: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to dict (lossless). Use as_safe_dict() for logging."""
        return {
            "provider": self.provider,
            "keys": list(self.keys),
            "active_index": self.active_index,
        }

    def as_safe_dict(self) -> Dict[str, Any]:
        """Serialize with masked key values for safe logging/display."""
        return {
            "provider": self.provider,
            "keys": [mask_key(k) for k in self.keys],
            "active_index": self.active_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyConfig:
        return cls(
            provider=data["provider"],
            keys=list(data.get("keys", [])),
            active_index=data.get("active_index", 0),
        )


@dataclass
class KeyRotationResult:
    """Result of a key rotation operation."""

    provider: str
    old_key_suffix: str
    new_key_suffix: str
    success: bool
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "old_key_suffix": self.old_key_suffix,
            "new_key_suffix": self.new_key_suffix,
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyRotationResult:
        return cls(
            provider=data["provider"],
            old_key_suffix=data["old_key_suffix"],
            new_key_suffix=data["new_key_suffix"],
            success=data["success"],
            error=data.get("error", ""),
        )


class RotationManager:
    """Manages API key rotation across providers."""

    def __init__(self) -> None:
        self._configs: Dict[str, KeyConfig] = {}

    def register_provider(self, provider: str, keys: list[str]) -> None:
        """Register keys for a provider."""
        self._configs[provider] = KeyConfig(
            provider=provider, keys=list(keys), active_index=0
        )

    def get_active_key(self, provider: str) -> str | None:
        """Get the currently active key for a provider."""
        config = self._configs.get(provider)
        if config is None or not config.keys:
            return None
        return config.keys[config.active_index]

    def rotate(self, provider: str) -> KeyRotationResult:
        """Rotate to the next key for a provider, wrapping around."""
        config = self._configs.get(provider)
        if config is None:
            return KeyRotationResult(
                provider=provider,
                old_key_suffix="",
                new_key_suffix="",
                success=False,
                error=f"Provider '{provider}' not registered",
            )
        if not config.keys:
            return KeyRotationResult(
                provider=provider,
                old_key_suffix="",
                new_key_suffix="",
                success=False,
                error=f"No keys registered for '{provider}'",
            )
        if len(config.keys) < 2:
            suffix = config.keys[0][-4:] if config.keys[0] else ""
            return KeyRotationResult(
                provider=provider,
                old_key_suffix=suffix,
                new_key_suffix=suffix,
                success=False,
                error=f"Only one key registered for '{provider}', cannot rotate",
            )

        old_index = config.active_index
        old_key = config.keys[old_index]
        new_index = (old_index + 1) % len(config.keys)
        config.active_index = new_index
        new_key = config.keys[new_index]

        return KeyRotationResult(
            provider=provider,
            old_key_suffix=old_key[-4:] if old_key else "",
            new_key_suffix=new_key[-4:] if new_key else "",
            success=True,
        )

    def handle_auth_error(self, provider: str) -> KeyRotationResult:
        """Auto-rotate on authentication error (401/403)."""
        return self.rotate(provider)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._configs.keys())

    def get_key_status(self, provider: str) -> dict:
        """Return status dict for a provider."""
        config = self._configs.get(provider)
        if config is None:
            return {
                "provider": provider,
                "num_keys": 0,
                "active_index": -1,
                "active_key_suffix": "",
            }
        active_key = (
            config.keys[config.active_index] if config.keys else ""
        )
        return {
            "provider": provider,
            "num_keys": len(config.keys),
            "active_index": config.active_index,
            "active_key_suffix": active_key[-4:] if active_key else "",
        }


def load_keys_from_env(
    provider: str, env_prefix: str = ""
) -> list[str]:
    """Load API keys from environment variables.

    Looks for PROVIDER_API_KEY, then PROVIDER_API_KEY_1,
    PROVIDER_API_KEY_2, etc. If env_prefix is given, uses that
    as the base name instead of deriving from the provider name.
    """
    if env_prefix:
        base = env_prefix
    else:
        base = provider.upper() + "_API_KEY"

    keys: list[str] = []

    # Check the base key (e.g., OPENAI_API_KEY)
    val = os.environ.get(base)
    if val:
        keys.append(val)

    # Check numbered keys (e.g., OPENAI_API_KEY_1, OPENAI_API_KEY_2, ...)
    i = 1
    while True:
        val = os.environ.get(f"{base}_{i}")
        if val:
            keys.append(val)
            i += 1
        else:
            break

    return keys


def format_rotation_status(manager: RotationManager) -> str:
    """Format human-readable status of all providers."""
    providers = manager.list_providers()
    if not providers:
        return "No providers registered."

    lines: list[str] = []
    lines.append("Key Rotation Status")
    lines.append("=" * 40)

    for provider in providers:
        status = manager.get_key_status(provider)
        lines.append(f"Provider: {provider}")
        lines.append(f"  Keys: {status['num_keys']}")
        lines.append(f"  Active index: {status['active_index']}")
        lines.append(
            f"  Active key: ****{status['active_key_suffix']}"
        )
        lines.append("")

    return "\n".join(lines)
