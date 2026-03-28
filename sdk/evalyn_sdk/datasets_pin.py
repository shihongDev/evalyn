"""Dataset pinning: lock a dataset version hash for reproducible evaluations.

A pin file (.evalyn-pin) stores the SHA-256 hash of the dataset content.
Before evaluation, the pin can be verified to ensure the dataset hasn't
changed since it was pinned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PIN_FILENAME = ".evalyn-pin"


@dataclass
class DatasetPin:
    """A pin recording a dataset's content hash at a point in time."""

    dataset_hash: str
    item_count: int
    pinned_at: str  # ISO timestamp
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_hash": self.dataset_hash,
            "item_count": self.item_count,
            "pinned_at": self.pinned_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetPin":
        return cls(
            dataset_hash=data["dataset_hash"],
            item_count=data["item_count"],
            pinned_at=data.get("pinned_at", ""),
            description=data.get("description", ""),
        )


def compute_dataset_content_hash(items: List) -> str:
    """Compute a deterministic hash of dataset content.

    Args:
        items: List of DatasetItem objects.

    Returns:
        SHA-256 hash (32 chars) of the sorted dataset.
    """
    parts = []
    for item in sorted(items, key=lambda i: i.id):
        parts.append(json.dumps({
            "id": item.id,
            "input": item.input,
            "output": item.output,
        }, sort_keys=True, default=str))
    content = "\n".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def create_pin(
    items: List,
    dataset_dir: Path,
    description: str = "",
) -> DatasetPin:
    """Create a pin file for a dataset.

    Args:
        items: Dataset items to pin.
        dataset_dir: Directory containing the dataset.
        description: Optional description of why this pin was created.

    Returns:
        The created DatasetPin.
    """
    from .models import now_utc

    content_hash = compute_dataset_content_hash(items)
    pin = DatasetPin(
        dataset_hash=content_hash,
        item_count=len(items),
        pinned_at=now_utc().isoformat(),
        description=description,
    )

    pin_path = Path(dataset_dir) / PIN_FILENAME
    pin_path.write_text(json.dumps(pin.as_dict(), indent=2), encoding="utf-8")

    return pin


def load_pin(dataset_dir: Path) -> Optional[DatasetPin]:
    """Load an existing pin file.

    Args:
        dataset_dir: Directory containing the dataset.

    Returns:
        DatasetPin if found, None if no pin file exists.
    """
    pin_path = Path(dataset_dir) / PIN_FILENAME
    if not pin_path.exists():
        return None
    try:
        data = json.loads(pin_path.read_text(encoding="utf-8"))
        return DatasetPin.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def verify_pin(items: List, pin: DatasetPin) -> bool:
    """Verify that current dataset matches the pinned version.

    Args:
        items: Current dataset items.
        pin: The pin to verify against.

    Returns:
        True if dataset matches pin, False if it has changed.
    """
    current_hash = compute_dataset_content_hash(items)
    return current_hash == pin.dataset_hash


def remove_pin(dataset_dir: Path) -> bool:
    """Remove a pin file.

    Args:
        dataset_dir: Directory containing the dataset.

    Returns:
        True if pin was removed, False if no pin existed.
    """
    pin_path = Path(dataset_dir) / PIN_FILENAME
    if pin_path.exists():
        pin_path.unlink()
        return True
    return False
