from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from .models import DatasetItem


def load_dataset(path: str | Path) -> List[DatasetItem]:
    """
    Load dataset items from a JSON array or JSONL file.
    Each row should contain at least an `inputs` object.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    rows: List[Any] = []
    if not raw:
        return []
    if raw.startswith("["):
        rows = json.loads(raw)
    else:
        for line in raw.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [DatasetItem.from_payload(row) for row in rows]


def save_dataset(items: Iterable[DatasetItem], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.__dict__) + "\n")


def hash_inputs(inputs: Mapping[str, Any]) -> str:
    """Deterministic hash for caching dataset invocations."""
    normalized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
