"""
Configurable embedding model selection for sampling.

Provides a registry of embedding model configurations used by the
sampling pipeline. Pure configuration/metadata - no actual model loading.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""

    model_id: str
    model_name: str
    dimensions: int
    source: str  # "huggingface", "local", or "api"
    max_tokens: int = 512
    cache_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "source": self.source,
            "max_tokens": self.max_tokens,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingModelConfig:
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            dimensions=data["dimensions"],
            source=data["source"],
            max_tokens=data.get("max_tokens", 512),
            cache_key=data.get("cache_key", ""),
        )


# ---------------------------------------------------------------------------
# Builtin models
# ---------------------------------------------------------------------------

BUILTIN_MODELS: dict[str, EmbeddingModelConfig] = {
    "all-MiniLM-L6-v2": EmbeddingModelConfig(
        model_id="all-MiniLM-L6-v2",
        model_name="all-MiniLM-L6-v2",
        dimensions=384,
        source="huggingface",
    ),
    "all-mpnet-base-v2": EmbeddingModelConfig(
        model_id="all-mpnet-base-v2",
        model_name="all-mpnet-base-v2",
        dimensions=768,
        source="huggingface",
    ),
    "bge-small-en-v1.5": EmbeddingModelConfig(
        model_id="bge-small-en-v1.5",
        model_name="bge-small-en-v1.5",
        dimensions=384,
        source="huggingface",
    ),
    "text-embedding-3-small": EmbeddingModelConfig(
        model_id="text-embedding-3-small",
        model_name="text-embedding-3-small",
        dimensions=1536,
        source="api",
    ),
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EmbeddingRegistry:
    """Registry for managing embedding model configurations."""

    def __init__(self) -> None:
        self._models: dict[str, EmbeddingModelConfig] = {}
        self._default_id: str = "all-MiniLM-L6-v2"

    def register(self, config: EmbeddingModelConfig) -> None:
        """Register an embedding model configuration."""
        self._models[config.model_id] = config

    def get(self, model_id: str) -> EmbeddingModelConfig | None:
        """Get a model config by ID, or None if not found."""
        return self._models.get(model_id)

    def get_default(self) -> EmbeddingModelConfig:
        """Return the default model config.

        Falls back to a hardcoded all-MiniLM-L6-v2 config if the
        default ID is not in the registry.
        """
        model = self._models.get(self._default_id)
        if model is not None:
            return model
        return BUILTIN_MODELS["all-MiniLM-L6-v2"]

    def list_models(self) -> list[EmbeddingModelConfig]:
        """Return all registered models as a list."""
        return list(self._models.values())

    def set_default(self, model_id: str) -> None:
        """Set the default model ID."""
        self._default_id = model_id


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_default_registry() -> EmbeddingRegistry:
    """Create a registry populated with all builtin models."""
    registry = EmbeddingRegistry()
    for config in BUILTIN_MODELS.values():
        registry.register(config)
    return registry


def get_cache_key(model_id: str, text: str) -> str:
    """Return a SHA-256 hex digest of model_id + text for cache lookup."""
    raw = model_id + text
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def format_model_catalog(registry: EmbeddingRegistry) -> str:
    """Format a human-readable catalog of registered models."""
    models = registry.list_models()
    if not models:
        return "No models registered."
    lines: list[str] = ["Embedding Models:"]
    for m in models:
        default_marker = " (default)" if m.model_id == registry._default_id else ""
        lines.append(
            f"  {m.model_id}{default_marker} - {m.dimensions}d, "
            f"source={m.source}, max_tokens={m.max_tokens}"
        )
    return "\n".join(lines)
