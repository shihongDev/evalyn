"""
Deterministic seed for exact reproduction of simulation runs.

Pure Python, no external dependencies. Provides seeded RNG wrappers,
checksum computation, and reproducibility verification.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SeedConfig:
    """Configuration for deterministic seed behavior."""

    seed: int
    temperature: float = 0.0
    record_in_metadata: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "record_in_metadata": self.record_in_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SeedConfig:
        return cls(
            seed=data["seed"],
            temperature=data.get("temperature", 0.0),
            record_in_metadata=data.get("record_in_metadata", True),
        )


@dataclass
class ReproducibilityRecord:
    """Immutable record capturing seed, checksum, and context for a run."""

    seed: int
    timestamp: str
    item_count: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "timestamp": self.timestamp,
            "item_count": self.item_count,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReproducibilityRecord:
        return cls(
            seed=data["seed"],
            timestamp=data["timestamp"],
            item_count=data["item_count"],
            checksum=data["checksum"],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Seeded RNG
# ---------------------------------------------------------------------------


class SeededRandom:
    """Thin wrapper around random.Random for deterministic operations."""

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._rng.choice(seq)

    def shuffle(self, lst: List[Any]) -> List[Any]:
        """Return a new shuffled list without mutating the original."""
        copy = list(lst)
        self._rng.shuffle(copy)
        return copy

    def sample(self, population: Sequence[Any], k: int) -> List[Any]:
        return self._rng.sample(population, k)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def get_seed(self) -> int:
        return self._seed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_seeded_rng(seed: "int | SeedConfig") -> SeededRandom:
    """Create a SeededRandom instance from an int seed or SeedConfig."""
    if isinstance(seed, SeedConfig):
        seed = seed.seed
    return SeededRandom(seed)


# ---------------------------------------------------------------------------
# Checksum and Records
# ---------------------------------------------------------------------------


def compute_items_checksum(items: List[Dict[str, Any]]) -> str:
    """SHA-256 of all items concatenated as JSON with sorted keys."""
    parts = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items]
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def create_reproducibility_record(
    seed: int,
    items: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> ReproducibilityRecord:
    """Build a ReproducibilityRecord from seed and items."""
    return ReproducibilityRecord(
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        item_count=len(items),
        checksum=compute_items_checksum(items),
        metadata=metadata or {},
    )


def verify_reproducibility(
    record: ReproducibilityRecord, items: List[Dict[str, Any]]
) -> bool:
    """Return True when the items match the recorded checksum."""
    return compute_items_checksum(items) == record.checksum


def format_reproducibility_info(record: ReproducibilityRecord) -> str:
    """Human-readable seed and checksum information."""
    lines = [
        f"Seed: {record.seed}",
        f"Timestamp: {record.timestamp}",
        f"Items: {record.item_count}",
        f"Checksum: {record.checksum}",
    ]
    if record.metadata:
        lines.append(f"Metadata: {json.dumps(record.metadata, sort_keys=True)}")
    return "\n".join(lines)
