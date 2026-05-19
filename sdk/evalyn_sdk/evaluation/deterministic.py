"""Deterministic evaluation mode.

Provides utilities to make evaluation runs reproducible by controlling
all sources of randomness: sampling order, metric evaluation order,
and LLM judge temperature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DeterministicConfig:
    """Configuration for deterministic evaluation mode.

    When enabled, enforces:
    - Fixed seed for all random operations (sampling, ordering)
    - Temperature 0 for all LLM judge calls
    - Deterministic item evaluation order (sorted by item ID)
    - Deterministic metric evaluation order (sorted by metric ID)
    """

    enabled: bool = False
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "seed": self.seed}


@dataclass
class RunManifest:
    """Record of everything that could affect evaluation results.

    Stored alongside eval run results for reproducibility verification.
    """

    evalyn_version: str = ""
    python_version: str = ""
    seed: int | None = None
    dataset_hash: str = ""
    metric_hashes: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    provider: str = ""
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "evalyn_version": self.evalyn_version,
            "python_version": self.python_version,
            "seed": self.seed,
            "dataset_hash": self.dataset_hash,
            "metric_hashes": dict(self.metric_hashes),
            "config_hash": self.config_hash,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(
            evalyn_version=data.get("evalyn_version", ""),
            python_version=data.get("python_version", ""),
            seed=data.get("seed"),
            dataset_hash=data.get("dataset_hash", ""),
            metric_hashes=data.get("metric_hashes", {}),
            config_hash=data.get("config_hash", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
        )


def sort_items_deterministically(items: list, seed: int = 42) -> list:
    """Sort dataset items in a deterministic order.

    Uses a seeded hash of each item's ID to produce a stable ordering
    that's independent of insertion order or filesystem order.

    Args:
        items: List of DatasetItem objects.
        seed: Random seed for ordering.

    Returns:
        New list sorted deterministically.
    """
    def sort_key(item):
        # Hash item ID with seed for stable ordering
        h = hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest()
        return h

    return sorted(items, key=sort_key)


def sort_metrics_deterministically(metrics: list) -> list:
    """Sort metrics by ID for deterministic evaluation order.

    Args:
        metrics: List of Metric objects.

    Returns:
        New list sorted by metric spec ID.
    """
    return sorted(metrics, key=lambda m: m.spec.id)


def compute_dataset_hash(items: list) -> str:
    """Compute a deterministic hash of a dataset's contents.

    Args:
        items: List of DatasetItem objects.

    Returns:
        SHA-256 hash (16 chars) of the sorted dataset content.
    """
    content_parts = []
    for item in sorted(items, key=lambda i: i.id):
        content_parts.append(json.dumps({
            "id": item.id,
            "input": item.input,
            "output": item.output,
        }, sort_keys=True, default=str))
    combined = "\n".join(content_parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def build_manifest(
    items: list,
    metrics: list,
    seed: int | None = None,
    provider: str = "",
    model: str = "",
) -> RunManifest:
    """Build a run manifest capturing all parameters that affect results.

    Args:
        items: Dataset items being evaluated.
        metrics: Metrics being applied.
        seed: Random seed (None if not deterministic mode).
        provider: LLM provider name.
        model: Model name.

    Returns:
        RunManifest for this evaluation.
    """
    import sys

    metric_hashes = {}
    for m in metrics:
        if hasattr(m, "spec") and hasattr(m.spec, "version_hash"):
            metric_hashes[m.spec.id] = m.spec.version_hash

    return RunManifest(
        evalyn_version=_get_evalyn_version(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        seed=seed,
        dataset_hash=compute_dataset_hash(items),
        metric_hashes=metric_hashes,
        provider=provider,
        model=model,
    )


def verify_manifest(
    current: RunManifest,
    previous: RunManifest,
) -> list[str]:
    """Compare two manifests and report differences.

    Args:
        current: Current run's manifest.
        previous: Previous run's manifest to compare against.

    Returns:
        List of warning strings describing differences. Empty = identical.
    """
    warnings = []

    if current.evalyn_version != previous.evalyn_version and previous.evalyn_version:
        warnings.append(
            f"Evalyn version changed: {previous.evalyn_version} -> {current.evalyn_version}"
        )

    if current.dataset_hash != previous.dataset_hash and previous.dataset_hash:
        warnings.append(
            f"Dataset content changed (hash: {previous.dataset_hash} -> {current.dataset_hash})"
        )

    if current.provider != previous.provider and previous.provider:
        warnings.append(
            f"Provider changed: {previous.provider} -> {current.provider}"
        )

    if current.model != previous.model and previous.model:
        warnings.append(
            f"Model changed: {previous.model} -> {current.model}"
        )

    # Check metric version changes
    for metric_id, prev_hash in previous.metric_hashes.items():
        curr_hash = current.metric_hashes.get(metric_id)
        if curr_hash and curr_hash != prev_hash:
            warnings.append(
                f"Metric '{metric_id}' definition changed (hash: {prev_hash[:8]}.. -> {curr_hash[:8]}..)"
            )
        elif curr_hash is None:
            warnings.append(f"Metric '{metric_id}' was removed")

    for metric_id in current.metric_hashes:
        if metric_id not in previous.metric_hashes:
            warnings.append(f"Metric '{metric_id}' was added")

    return warnings


def _get_evalyn_version() -> str:
    """Get the installed evalyn version."""
    try:
        from importlib.metadata import version
        return version("evalyn-sdk")
    except Exception:
        return "unknown"
