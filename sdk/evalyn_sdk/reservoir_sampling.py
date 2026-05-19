"""Online reservoir sampling for streaming dataset construction.

Uses Algorithm R for uniform sampling from a stream of unknown size.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReservoirConfig:
    """Configuration for reservoir sampling."""

    capacity: int = 100
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReservoirConfig:
        return cls(
            capacity=data.get("capacity", 100),
            seed=data.get("seed", 42),
        )


@dataclass
class ReservoirState:
    """Snapshot of reservoir state at a point in time."""

    items: list[dict[str, Any]] = field(default_factory=list)
    seen_count: int = 0
    capacity: int = 100
    is_full: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "seen_count": self.seen_count,
            "capacity": self.capacity,
            "is_full": self.is_full,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReservoirState:
        return cls(
            items=data.get("items", []),
            seen_count=data.get("seen_count", 0),
            capacity=data.get("capacity", 100),
            is_full=data.get("is_full", False),
        )


class Reservoir:
    """Online reservoir sampler using Algorithm R.

    Maintains a uniform random sample of fixed capacity from an
    arbitrarily large stream. Each item in the stream has equal
    probability (capacity / seen_count) of being in the reservoir.
    """

    def __init__(self, capacity: int = 100, seed: int = 42) -> None:
        self._capacity = capacity
        self._seed = seed
        self._rng = random.Random(seed)
        self._items: list[dict[str, Any]] = []
        self._seen_count = 0

    def add(self, item: dict) -> bool:
        """Add an item using Algorithm R.

        Returns True if the item was kept in the reservoir (either
        appended when not full, or replaced an existing item).
        """
        self._seen_count += 1
        if len(self._items) < self._capacity:
            self._items.append(item)
            return True
        j = self._rng.randint(0, self._seen_count - 1)
        if j < self._capacity:
            self._items[j] = item
            return True
        return False

    def add_batch(self, items: list[dict]) -> int:
        """Add multiple items. Returns count of items kept."""
        count = 0
        for item in items:
            if self.add(item):
                count += 1
        return count

    def get_sample(self) -> list[dict]:
        """Return current reservoir contents as a list copy."""
        return list(self._items)

    def get_state(self) -> ReservoirState:
        """Return a snapshot of the current reservoir state."""
        return ReservoirState(
            items=list(self._items),
            seen_count=self._seen_count,
            capacity=self._capacity,
            is_full=self._seen_count >= self._capacity,
        )

    def size(self) -> int:
        """Current number of items in the reservoir."""
        return len(self._items)

    def seen(self) -> int:
        """Total number of items observed so far."""
        return self._seen_count

    def is_full(self) -> bool:
        """True if total items seen >= capacity."""
        return self._seen_count >= self._capacity

    def reset(self) -> None:
        """Clear reservoir and all counters, re-seed RNG."""
        self._items = []
        self._seen_count = 0
        self._rng = random.Random(self._seed)


def create_reservoir(capacity: int = 100, seed: int = 42) -> Reservoir:
    """Factory function to create a new Reservoir."""
    return Reservoir(capacity=capacity, seed=seed)


def stream_from_file(file_path: str, reservoir: Reservoir) -> ReservoirState:
    """Stream a JSONL file through a reservoir line by line.

    Each non-empty line is parsed as JSON and added to the reservoir.
    Returns the final reservoir state.
    """
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            reservoir.add(item)
    return reservoir.get_state()


def format_reservoir_stats(state: ReservoirState) -> str:
    """Human-readable summary of reservoir state."""
    lines = [
        "Reservoir Statistics",
        f"  Capacity:    {state.capacity}",
        f"  Items held:  {len(state.items)}",
        f"  Items seen:  {state.seen_count}",
        f"  Is full:     {state.is_full}",
    ]
    if state.seen_count > 0:
        ratio = len(state.items) / state.seen_count
        lines.append(f"  Sample rate: {ratio:.4f}")
    return "\n".join(lines)
