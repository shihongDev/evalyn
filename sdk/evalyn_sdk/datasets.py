from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional
from datetime import datetime

from .models import DatasetItem, FunctionCall


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


def dataset_from_calls(
    calls: Iterable[FunctionCall],
    *,
    use_only_success: bool = True,
    include_metadata: bool = True,
) -> List[DatasetItem]:
    """
    Build a regression dataset from existing traced calls.
    - uses call.inputs as DatasetItem.inputs
    - uses call.output as expected (baseline)
    - filters errors if use_only_success=True
    """
    items: List[DatasetItem] = []
    for call in calls:
        if use_only_success and call.error:
            continue
        items.append(
            DatasetItem(
                id=call.id,
                inputs=call.inputs,
                expected=call.output,
                metadata={"function": call.function_name} if include_metadata else {},
            )
        )
    return items


def build_dataset_from_storage(
    storage,
    *,
    function_name: Optional[str] = None,
    project_id: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 500,
    success_only: bool = True,
    include_metadata: bool = True,
) -> List[DatasetItem]:
    """
    Build a dataset from stored calls with simple filtering.
    Filters:
      - function_name: exact match on call.function_name
      - project_id: matches call.metadata.get("project_id")
      - since/until: call.started_at within range
      - success_only: skip calls with errors
      - limit: max number of matching calls to include (after filtering)
    """
    calls = storage.list_calls(limit=limit * 5) if storage else []
    items: List[DatasetItem] = []
    for call in calls:
        if success_only and call.error:
            continue
        if function_name and call.function_name != function_name:
            continue
        if project_id and isinstance(call.metadata, dict):
            if call.metadata.get("project_id") != project_id:
                continue
        if since and call.started_at < since:
            continue
        if until and call.started_at > until:
            continue

        meta = {}
        if include_metadata:
            meta = {
                "function": call.function_name,
                "call_id": call.id,
                "started_at": call.started_at.isoformat(),
            }
            if isinstance(call.metadata, dict):
                meta.update(call.metadata)

        items.append(
            DatasetItem(
                id=call.id,
                inputs=call.inputs,
                expected=call.output,
                metadata=meta,
            )
        )
        if len(items) >= limit:
            break
    return items
