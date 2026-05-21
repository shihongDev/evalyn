"""
Secrets backend integration: load API keys from external secret managers.

Pure Python with no external dependencies. Cloud SDK calls are abstracted
behind a protocol so backends can be swapped without changing callers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SecretReference:
    """Pointer to a secret stored in a specific backend."""

    backend: str  # "env", "aws", "gcp", "vault"
    path: str  # secret path, ARN, or env var name
    key_field: str = ""  # for JSON secrets, which field to extract
    version: str = "latest"

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "path": self.path,
            "key_field": self.key_field,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecretReference:
        return cls(
            backend=data["backend"],
            path=data["path"],
            key_field=data.get("key_field", ""),
            version=data.get("version", "latest"),
        )


@dataclass
class SecretResult:
    """Resolved secret value with metadata."""

    value: str
    source: str  # backend name
    cached: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dict (lossless). Use as_safe_dict() for logging."""
        return {
            "value": self.value,
            "source": self.source,
            "cached": self.cached,
            "error": self.error,
        }

    def as_safe_dict(self) -> dict[str, Any]:
        """Serialize with masked value for safe logging/display."""
        masked = "***" + self.value[-4:] if len(self.value) >= 4 else "***"
        return {
            "value": masked,
            "source": self.source,
            "cached": self.cached,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecretResult:
        return cls(
            value=data.get("value", ""),
            source=data.get("source", ""),
            cached=data.get("cached", False),
            error=data.get("error", ""),
        )


# ---------------------------------------------------------------------------
# Backend Protocol
# ---------------------------------------------------------------------------


class SecretsBackend(ABC):
    """Abstract base for secret resolution backends."""

    @abstractmethod
    def resolve(self, ref: SecretReference) -> SecretResult:
        """Resolve a secret reference to its value."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""


# ---------------------------------------------------------------------------
# Built-in Backends
# ---------------------------------------------------------------------------


class EnvBackend(SecretsBackend):
    """Reads secrets from environment variables."""

    @property
    def name(self) -> str:
        return "env"

    def resolve(self, ref: SecretReference) -> SecretResult:
        value = os.environ.get(ref.path, "")
        if not value:
            return SecretResult(
                value="",
                source=self.name,
                error=f"environment variable '{ref.path}' not set",
            )
        return SecretResult(value=value, source=self.name)


class DictBackend(SecretsBackend):
    """Reads secrets from an in-memory dict. Useful for testing."""

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self._secrets: dict[str, str] = secrets or {}

    @property
    def name(self) -> str:
        return "dict"

    def resolve(self, ref: SecretReference) -> SecretResult:
        value = self._secrets.get(ref.path, "")
        if not value:
            return SecretResult(
                value="",
                source=self.name,
                error=f"key '{ref.path}' not found in dict backend",
            )
        return SecretResult(value=value, source=self.name)


# ---------------------------------------------------------------------------
# Secrets Manager
# ---------------------------------------------------------------------------


class SecretsManager:
    """Registry and dispatcher for secret resolution backends."""

    def __init__(self) -> None:
        self._backends: dict[str, SecretsBackend] = {}

    def register_backend(self, backend: SecretsBackend) -> None:
        """Register a backend by its name."""
        self._backends[backend.name] = backend

    def resolve(self, ref: SecretReference) -> SecretResult:
        """Dispatch to the appropriate backend."""
        backend = self._backends.get(ref.backend)
        if backend is None:
            return SecretResult(
                value="",
                source=ref.backend,
                error=f"no backend registered for '{ref.backend}'",
            )
        return backend.resolve(ref)

    def resolve_all(self, refs: list[SecretReference]) -> list[SecretResult]:
        """Resolve multiple references."""
        return [self.resolve(ref) for ref in refs]

    def list_backends(self) -> list[str]:
        """List registered backend names."""
        return sorted(self._backends.keys())


# ---------------------------------------------------------------------------
# Parsing and Helpers
# ---------------------------------------------------------------------------


def parse_secret_reference(spec: str) -> SecretReference:
    """Parse a secret reference string.

    Formats:
        backend://path
        backend://path#field
        path            (defaults to backend="env")
    """
    if "://" in spec:
        backend, rest = spec.split("://", 1)
    else:
        backend = "env"
        rest = spec

    key_field = ""
    if "#" in rest:
        rest, key_field = rest.rsplit("#", 1)

    return SecretReference(backend=backend, path=rest, key_field=key_field)


def create_default_manager() -> SecretsManager:
    """Create a manager pre-loaded with the EnvBackend."""
    mgr = SecretsManager()
    mgr.register_backend(EnvBackend())
    return mgr


def format_secrets_status(manager: SecretsManager, refs: list[SecretReference]) -> str:
    """Resolve each reference and format a human-readable status report.

    Values are masked - only the last 4 characters are shown.
    """
    lines: list[str] = []
    results = manager.resolve_all(refs)
    for ref, result in zip(refs, results):
        label = f"{ref.backend}://{ref.path}"
        if ref.key_field:
            label += f"#{ref.key_field}"
        if result.error:
            lines.append(f"  {label}: ERROR - {result.error}")
        else:
            masked = result.as_safe_dict()["value"]
            status = "cached" if result.cached else "resolved"
            lines.append(f"  {label}: {status} ({masked})")
    header = f"Secrets status ({len(refs)} references):"
    return "\n".join([header] + lines)
