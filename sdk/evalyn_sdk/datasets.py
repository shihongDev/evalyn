from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional
import hashlib
from datetime import datetime, timezone

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
            f.write(json.dumps(item.as_dict()) + "\n")


def save_dataset_with_meta(
    items: Iterable[DatasetItem],
    dataset_dir: str | Path,
    meta: dict,
    *,
    dataset_filename: str = "dataset.jsonl",
    meta_filename: str = "meta.json",
) -> Path:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / dataset_filename
    save_dataset(items, dataset_path)

    hasher = hashlib.sha256()
    with dataset_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            hasher.update(chunk)
    meta = dict(meta)
    meta["dataset_file"] = dataset_filename
    meta["meta_file"] = meta_filename
    meta["dataset_hash"] = hasher.hexdigest()

    meta_path = dataset_dir / meta_filename
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset_path


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
    Build a dataset from existing traced calls.

    Dataset structure (4 columns):
    - input: The user/system input to the agent
    - output: The agent/LLM response
    - human_label: None (to be filled by human annotators)
    - metadata: call_id, function name, etc.
    """
    items: List[DatasetItem] = []
    for call in calls:
        if use_only_success and call.error:
            continue
        items.append(DatasetItem.from_call(call))
    return items


def build_dataset_from_storage(
    storage,
    *,
    function_name: Optional[str] = None,
    project_id: Optional[str] = None,
    project_name: Optional[str] = None,
    version: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 500,
    success_only: bool = True,
    include_metadata: bool = True,
) -> List[DatasetItem]:
    """
    Build a dataset from stored calls with simple filtering.

    Dataset structure (4 columns):
    - input: The user/system input to the agent
    - output: The agent/LLM response
    - human_label: None (to be filled by human annotators)
    - metadata: call_id, function name, project info, etc.

    Filters:
      - function_name: exact match on call.function_name
      - project_id/project_name: matches call.metadata.get("project_id") or metadata.get("project_name")
      - version: matches call.metadata.get("version")
      - since/until: call.started_at within range
      - success_only: skip calls with errors
      - limit: max number of matching calls to include (after filtering)
    """
    def _as_aware(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    since = _as_aware(since)
    until = _as_aware(until)

    calls = storage.list_calls(limit=limit * 5) if storage else []
    items: List[DatasetItem] = []
    for call in calls:
        started_at = call.started_at
        if started_at and started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if success_only and call.error:
            continue
        if function_name and call.function_name != function_name:
            continue
        if (project_id or project_name) and isinstance(call.metadata, dict):
            meta_pid = call.metadata.get("project_id") or call.metadata.get("project_name")
            wanted = project_id or project_name
            if meta_pid != wanted:
                continue
        if version and isinstance(call.metadata, dict):
            if call.metadata.get("version") != version:
                continue
        if since and started_at and started_at < since:
            continue
        if until and started_at and started_at > until:
            continue

        # Build metadata
        meta = {}
        if include_metadata:
            meta = {
                "function": call.function_name,
                "call_id": call.id,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "duration_ms": call.duration_ms,
            }
            if isinstance(call.metadata, dict):
                meta.update(call.metadata)

        # Create dataset item with new 4-column structure
        items.append(
            DatasetItem(
                id=call.id,
                input=call.inputs,
                output=call.output,
                human_label=None,
                metadata=meta,
            )
        )
        if len(items) >= limit:
            break
    return items
