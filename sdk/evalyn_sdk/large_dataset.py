"""
Large dataset optimization: streaming, chunking, checkpointing for 10k+ items.

Pure Python - handles large JSONL files without loading everything into memory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ChunkConfig:
    """Configuration for chunked processing."""

    chunk_size: int = 100
    checkpoint_interval: int = 500
    max_memory_mb: float = 0.0  # 0 means no limit
    write_batch_size: int = 50

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "checkpoint_interval": self.checkpoint_interval,
            "max_memory_mb": self.max_memory_mb,
            "write_batch_size": self.write_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkConfig:
        return cls(
            chunk_size=data.get("chunk_size", 100),
            checkpoint_interval=data.get("checkpoint_interval", 500),
            max_memory_mb=data.get("max_memory_mb", 0.0),
            write_batch_size=data.get("write_batch_size", 50),
        )


@dataclass
class CheckpointState:
    """State persisted at checkpoint boundaries."""

    processed_count: int
    total_count: int
    last_item_id: str
    checkpoint_path: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "last_item_id": self.last_item_id,
            "checkpoint_path": self.checkpoint_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls(
            processed_count=data.get("processed_count", 0),
            total_count=data.get("total_count", 0),
            last_item_id=data.get("last_item_id", ""),
            checkpoint_path=data.get("checkpoint_path", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class ProcessingStats:
    """Statistics collected during chunk processing."""

    total_items: int
    processed_items: int
    failed_items: int
    elapsed_seconds: float
    items_per_second: float
    peak_memory_mb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "elapsed_seconds": self.elapsed_seconds,
            "items_per_second": self.items_per_second,
            "peak_memory_mb": self.peak_memory_mb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingStats:
        return cls(
            total_items=data.get("total_items", 0),
            processed_items=data.get("processed_items", 0),
            failed_items=data.get("failed_items", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            items_per_second=data.get("items_per_second", 0.0),
            peak_memory_mb=data.get("peak_memory_mb", 0.0),
        )


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def stream_dataset(file_path: str, chunk_size: int = 100) -> Iterator[list[dict]]:
    """Yield chunks of items from a JSONL file without loading everything.

    Each chunk is a list of dicts parsed from consecutive lines.
    Blank lines and unparseable lines are skipped.
    """
    chunk: list[dict] = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def save_checkpoint(state: CheckpointState) -> None:
    """Write checkpoint state atomically via temp file + rename.

    A direct open("w") would truncate the existing checkpoint to zero bytes
    immediately. If the process crashes during json.dump, the checkpoint is
    lost. Instead we write to a temp file, fsync, then atomic-rename.
    """
    parent = os.path.dirname(state.checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=parent or ".",
        prefix=".checkpoint_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as fh:
            json.dump(state.as_dict(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, state.checkpoint_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_checkpoint(checkpoint_path: str) -> CheckpointState | None:
    """Load a checkpoint from disk. Return None if file does not exist.

    If the checkpoint is corrupted (partial write from a previous crash),
    logs a warning and returns None so the caller can start fresh.
    """
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return CheckpointState.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Checkpoint corrupted, starting fresh: %s", e)
        return None


def should_checkpoint(processed: int, interval: int) -> bool:
    """Return True when processed count hits a checkpoint interval boundary."""
    if interval <= 0:
        return False
    return processed > 0 and processed % interval == 0


def estimate_memory_usage(items: list[dict]) -> float:
    """Approximate memory in MB by recursively sizing objects via sys.getsizeof."""
    total = 0
    seen: set[int] = set()

    def _size(obj: Any) -> int:
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        s = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                s += _size(k)
                s += _size(v)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in obj:
                s += _size(v)
        return s

    for item in items:
        total += _size(item)
    return total / (1024 * 1024)


def process_in_chunks(
    items_iter: Iterator[list[dict]],
    process_fn: Callable[[list[dict]], list[dict]],
    config: ChunkConfig,
    checkpoint_path: str = "",
) -> ProcessingStats:
    """Process an iterator of chunks with checkpointing and stats tracking.

    ``process_fn`` receives a chunk (list of dicts) and returns the processed
    chunk. If it raises, those items are counted as failed and processing
    continues with the next chunk.
    """
    start = time.monotonic()
    processed = 0
    failed = 0
    total = 0
    peak_mb = 0.0
    last_item_id = ""

    # Resume from checkpoint if available
    skip_count = 0
    if checkpoint_path:
        existing = load_checkpoint(checkpoint_path)
        if existing is not None:
            skip_count = existing.processed_count
            processed = existing.processed_count
            last_item_id = existing.last_item_id

    for chunk in items_iter:
        total += len(chunk)

        # Skip chunks already processed (resume support)
        if skip_count > 0:
            if skip_count >= len(chunk):
                skip_count -= len(chunk)
                continue
            chunk = chunk[skip_count:]
            skip_count = 0

        # Memory estimation
        mb = estimate_memory_usage(chunk)
        if mb > peak_mb:
            peak_mb = mb

        # Memory limit enforcement
        if config.max_memory_mb > 0 and mb > config.max_memory_mb:
            # Process in smaller sub-chunks
            sub_size = max(1, len(chunk) // 2)
            for i in range(0, len(chunk), sub_size):
                sub = chunk[i : i + sub_size]
                try:
                    result = process_fn(sub)
                    processed += len(result)
                    if result:
                        last_item_id = str(result[-1].get("id", ""))
                except Exception:
                    failed += len(sub)
        else:
            try:
                result = process_fn(chunk)
                processed += len(result)
                if result:
                    last_item_id = str(result[-1].get("id", ""))
            except Exception:
                failed += len(chunk)

        # Checkpoint
        if checkpoint_path and should_checkpoint(processed, config.checkpoint_interval):
            state = CheckpointState(
                processed_count=processed,
                total_count=total,
                last_item_id=last_item_id,
                checkpoint_path=checkpoint_path,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            save_checkpoint(state)

    elapsed = time.monotonic() - start
    ips = processed / elapsed if elapsed > 0 else 0.0

    return ProcessingStats(
        total_items=total,
        processed_items=processed,
        failed_items=failed,
        elapsed_seconds=round(elapsed, 3),
        items_per_second=round(ips, 2),
        peak_memory_mb=round(peak_mb, 4),
    )


def format_processing_stats(stats: ProcessingStats) -> str:
    """Human-readable processing stats."""
    lines = [
        "Processing Stats",
        f"  Total items:      {stats.total_items}",
        f"  Processed:        {stats.processed_items}",
        f"  Failed:           {stats.failed_items}",
        f"  Elapsed:          {stats.elapsed_seconds:.3f}s",
        f"  Items/sec:        {stats.items_per_second:.2f}",
    ]
    if stats.peak_memory_mb > 0:
        lines.append(f"  Peak memory:      {stats.peak_memory_mb:.4f} MB")
    return "\n".join(lines)
