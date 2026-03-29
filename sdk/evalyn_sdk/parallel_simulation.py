"""
Parallel simulation: generate synthetic data with configurable concurrency.

Uses concurrent.futures for parallel batch generation with progress tracking,
error collection, and result merging. Pure Python - stdlib only.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ParallelConfig:
    """Configuration for parallel data generation."""

    workers: int = 4
    batch_size: int = 10
    total_target: int = 100
    timeout_seconds: float = 300.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workers": self.workers,
            "batch_size": self.batch_size,
            "total_target": self.total_target,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParallelConfig:
        return cls(
            workers=data.get("workers", 4),
            batch_size=data.get("batch_size", 10),
            total_target=data.get("total_target", 100),
            timeout_seconds=data.get("timeout_seconds", 300.0),
        )


@dataclass
class GenerationProgress:
    """Snapshot of generation progress."""

    generated: int
    failed: int
    total_target: int
    elapsed_seconds: float
    items_per_second: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated": self.generated,
            "failed": self.failed,
            "total_target": self.total_target,
            "elapsed_seconds": self.elapsed_seconds,
            "items_per_second": self.items_per_second,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GenerationProgress:
        return cls(
            generated=data["generated"],
            failed=data["failed"],
            total_target=data["total_target"],
            elapsed_seconds=data["elapsed_seconds"],
            items_per_second=data["items_per_second"],
        )


@dataclass
class ParallelResult:
    """Result of a parallel generation run."""

    items: List[Dict[str, Any]] = field(default_factory=list)
    progress: GenerationProgress = field(
        default_factory=lambda: GenerationProgress(0, 0, 0, 0.0, 0.0)
    )
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": list(self.items),
            "progress": self.progress.as_dict(),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParallelResult:
        return cls(
            items=data.get("items", []),
            progress=GenerationProgress.from_dict(data["progress"]),
            errors=data.get("errors", []),
        )


# ---------------------------------------------------------------------------
# Batch Utilities
# ---------------------------------------------------------------------------


def create_batches(total: int, batch_size: int) -> List[int]:
    """Split total into batch sizes. Last batch may be smaller."""
    if total <= 0 or batch_size <= 0:
        return []
    batches: List[int] = []
    remaining = total
    while remaining > 0:
        size = min(batch_size, remaining)
        batches.append(size)
        remaining -= size
    return batches


# ---------------------------------------------------------------------------
# Generation Functions
# ---------------------------------------------------------------------------


def generate_parallel(
    generate_fn: Callable[[int], List[Dict[str, Any]]],
    config: ParallelConfig,
) -> ParallelResult:
    """Run generate_fn in parallel using ThreadPoolExecutor.

    generate_fn takes a batch_size int and returns a list of dicts.
    Collects results, handles exceptions per batch, and tracks progress.
    """
    batches = create_batches(config.total_target, config.batch_size)
    all_items: List[Dict[str, Any]] = []
    errors: List[str] = []
    failed = 0

    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        future_to_batch = {
            executor.submit(generate_fn, bs): bs for bs in batches
        }
        try:
            for future in as_completed(
                future_to_batch, timeout=config.timeout_seconds
            ):
                batch_size = future_to_batch[future]
                try:
                    result = future.result()
                    all_items.extend(result)
                except Exception as exc:
                    failed += batch_size
                    errors.append(f"Batch (size={batch_size}) failed: {exc}")
        except TimeoutError:
            for future, batch_size in future_to_batch.items():
                if not future.done():
                    failed += batch_size
            errors.append(
                f"Timeout after {config.timeout_seconds}s with "
                f"{len(all_items)} items collected"
            )

    elapsed = time.monotonic() - start
    generated = len(all_items)
    ips = generated / elapsed if elapsed > 0 else 0.0

    progress = GenerationProgress(
        generated=generated,
        failed=failed,
        total_target=config.total_target,
        elapsed_seconds=round(elapsed, 4),
        items_per_second=round(ips, 2),
    )

    return ParallelResult(items=all_items, progress=progress, errors=errors)


def generate_sequential(
    generate_fn: Callable[[int], List[Dict[str, Any]]],
    config: ParallelConfig,
) -> ParallelResult:
    """Run generate_fn sequentially - same interface as generate_parallel.

    Useful for comparison, debugging, and deterministic reproduction.
    """
    batches = create_batches(config.total_target, config.batch_size)
    all_items: List[Dict[str, Any]] = []
    errors: List[str] = []
    failed = 0

    start = time.monotonic()

    for bs in batches:
        try:
            result = generate_fn(bs)
            all_items.extend(result)
        except Exception as exc:
            failed += bs
            errors.append(f"Batch (size={bs}) failed: {exc}")

    elapsed = time.monotonic() - start
    generated = len(all_items)
    ips = generated / elapsed if elapsed > 0 else 0.0

    progress = GenerationProgress(
        generated=generated,
        failed=failed,
        total_target=config.total_target,
        elapsed_seconds=round(elapsed, 4),
        items_per_second=round(ips, 2),
    )

    return ParallelResult(items=all_items, progress=progress, errors=errors)


# ---------------------------------------------------------------------------
# Result Merging
# ---------------------------------------------------------------------------


def merge_results(results: List[ParallelResult]) -> ParallelResult:
    """Merge multiple ParallelResults into one."""
    if not results:
        return ParallelResult()

    all_items: List[Dict[str, Any]] = []
    all_errors: List[str] = []
    total_generated = 0
    total_failed = 0
    total_target = 0
    total_elapsed = 0.0

    for r in results:
        all_items.extend(r.items)
        all_errors.extend(r.errors)
        total_generated += r.progress.generated
        total_failed += r.progress.failed
        total_target += r.progress.total_target
        total_elapsed += r.progress.elapsed_seconds

    ips = total_generated / total_elapsed if total_elapsed > 0 else 0.0

    progress = GenerationProgress(
        generated=total_generated,
        failed=total_failed,
        total_target=total_target,
        elapsed_seconds=round(total_elapsed, 4),
        items_per_second=round(ips, 2),
    )

    return ParallelResult(
        items=all_items, progress=progress, errors=all_errors
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_progress_bar(progress: GenerationProgress, width: int = 40) -> str:
    """Render a text progress bar.

    Example: "[=====>                                 ] 50/100 (5.2 items/s)"
    """
    if progress.total_target <= 0:
        fraction = 0.0
    else:
        fraction = min(progress.generated / progress.total_target, 1.0)

    filled = int(width * fraction)
    arrow = ">" if 0 < filled < width else ""
    bar = "=" * max(filled - 1, 0) + arrow
    bar = bar.ljust(width)

    # Special case: fully complete
    if filled >= width:
        bar = "=" * width

    return (
        f"[{bar}] {progress.generated}/{progress.total_target} "
        f"({progress.items_per_second:.1f} items/s)"
    )


def format_parallel_report(result: ParallelResult) -> str:
    """Format a human-readable summary of a parallel generation run."""
    p = result.progress
    lines = [
        "Parallel Generation Report",
        "-" * 30,
        f"Generated: {p.generated}/{p.total_target}",
        f"Failed:    {p.failed}",
        f"Elapsed:   {p.elapsed_seconds:.2f}s",
        f"Throughput: {p.items_per_second:.1f} items/s",
    ]
    if result.errors:
        lines.append(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            lines.append(f"  - {err}")
    return "\n".join(lines)
