"""Execution strategies for evaluation runner.

Provides pluggable execution strategies:
- SequentialStrategy: Simple for-loop with per-item checkpointing
- ParallelStrategy: ThreadPoolExecutor with batch checkpointing
"""

from __future__ import annotations

import itertools
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from .models import DatasetItem, FunctionCall, Metric, MetricResult

logger = logging.getLogger(__name__)

# Progress callback type: (current_item, total_items, metric_id, metric_type)
ProgressCallback = Callable[[int, int, str, str], None]


class ExecutionStrategy(ABC):
    """Abstract base for evaluation execution strategies."""

    def __init__(
        self,
        evaluate_fn: Callable[[Metric, FunctionCall, DatasetItem], MetricResult],
        checkpoint_fn: Optional[Callable[[List[MetricResult], set, str], bool]] = None,
        checkpoint_interval: int = 5,
    ):
        """Initialize strategy.

        Args:
            evaluate_fn: Function to evaluate a single metric
            checkpoint_fn: Optional function to save checkpoint
            checkpoint_interval: Items between checkpoints (sequential only)
        """
        self._evaluate = evaluate_fn
        self._checkpoint = checkpoint_fn
        self._checkpoint_interval = checkpoint_interval

    @abstractmethod
    def execute(
        self,
        prepared: List[Tuple[DatasetItem, FunctionCall]],
        metrics: List[Metric],
        progress_callback: Optional[ProgressCallback],
        run_id: str,
        completed_items: set,
    ) -> List[MetricResult]:
        """Execute evaluation on prepared items.

        Args:
            prepared: List of (item, call) tuples ready for evaluation
            metrics: List of metrics to evaluate
            progress_callback: Optional callback for progress updates
            run_id: Current run ID for checkpointing
            completed_items: Set of already completed item IDs (mutable)

        Returns:
            List of MetricResult from evaluation
        """
        pass


class SequentialStrategy(ExecutionStrategy):
    """Sequential execution with per-item checkpointing."""

    def execute(
        self,
        prepared: List[Tuple[DatasetItem, FunctionCall]],
        metrics: List[Metric],
        progress_callback: Optional[ProgressCallback],
        run_id: str,
        completed_items: set,
    ) -> List[MetricResult]:
        """Execute items sequentially with periodic checkpointing."""
        results: List[MetricResult] = []
        total_evals = len(prepared) * len(metrics)
        current_eval = 0
        items_since_checkpoint = 0

        for item, call in prepared:
            for metric in metrics:
                current_eval += 1
                if progress_callback:
                    progress_callback(
                        current_eval,
                        total_evals,
                        metric.spec.id,
                        metric.spec.type,
                    )
                results.append(self._evaluate(metric, call, item))

            # Mark item as completed and checkpoint
            completed_items.add(item.id)
            items_since_checkpoint += 1

            if self._checkpoint and items_since_checkpoint >= self._checkpoint_interval:
                self._checkpoint(results, completed_items, run_id)
                items_since_checkpoint = 0

        # Final checkpoint
        if self._checkpoint and items_since_checkpoint > 0:
            self._checkpoint(results, completed_items, run_id)

        return results


class ParallelStrategy(ExecutionStrategy):
    """Parallel execution with ThreadPoolExecutor."""

    def __init__(
        self,
        evaluate_fn: Callable[[Metric, FunctionCall, DatasetItem], MetricResult],
        checkpoint_fn: Optional[Callable[[List[MetricResult], set, str], bool]] = None,
        checkpoint_interval: int = 5,
        max_workers: int = 4,
    ):
        super().__init__(evaluate_fn, checkpoint_fn, checkpoint_interval)
        self._max_workers = max(1, min(max_workers, 16))

    def execute(
        self,
        prepared: List[Tuple[DatasetItem, FunctionCall]],
        metrics: List[Metric],
        progress_callback: Optional[ProgressCallback],
        run_id: str,
        completed_items: set,
    ) -> List[MetricResult]:
        """Execute items in parallel using ThreadPoolExecutor."""
        progress_lock = threading.Lock()
        eval_counter = itertools.count(1)
        total_evals = len(prepared) * len(metrics)

        def eval_task(
            metric: Metric, call: FunctionCall, item: DatasetItem
        ) -> Tuple[str, MetricResult]:
            """Task for parallel execution."""
            result = self._evaluate(metric, call, item)
            return (item.id, result)

        # Build all tasks
        tasks = [(metric, call, item) for item, call in prepared for metric in metrics]

        # Execute in parallel
        results_by_item: Dict[str, List[MetricResult]] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(eval_task, m, c, i): (i.id, m.spec.id, m.spec.type)
                for m, c, i in tasks
            }

            for future in as_completed(futures):
                item_id_hint, metric_id, metric_type = futures[future]

                try:
                    item_id, result = future.result()
                    results_by_item[item_id].append(result)
                except Exception as e:
                    logger.warning(f"Evaluation task failed: {e}")
                    result = MetricResult(
                        metric_id=metric_id,
                        call_id=f"error-{item_id_hint}",
                        score=None,
                        passed=False,
                        details={"error": str(e), "error_type": type(e).__name__},
                    )
                    results_by_item[item_id_hint].append(result)

                # Update progress (thread-safe)
                if progress_callback:
                    with progress_lock:
                        current = next(eval_counter)
                        progress_callback(
                            current,
                            total_evals,
                            result.metric_id,
                            metric_type,
                        )

        # Collect results in order and mark completed
        results: List[MetricResult] = []
        for item, call in prepared:
            results.extend(results_by_item[item.id])
            completed_items.add(item.id)

        # Checkpoint after parallel batch
        if self._checkpoint:
            self._checkpoint(results, completed_items, run_id)

        return results


def create_strategy(
    max_workers: int,
    evaluate_fn: Callable[[Metric, FunctionCall, DatasetItem], MetricResult],
    checkpoint_fn: Optional[Callable[[List[MetricResult], set, str], bool]] = None,
    checkpoint_interval: int = 5,
) -> ExecutionStrategy:
    """Factory function to create appropriate strategy.

    Args:
        max_workers: Number of parallel workers (1 = sequential)
        evaluate_fn: Function to evaluate a single metric
        checkpoint_fn: Optional checkpoint save function
        checkpoint_interval: Items between checkpoints

    Returns:
        SequentialStrategy if max_workers <= 1, else ParallelStrategy
    """
    if max_workers <= 1:
        return SequentialStrategy(evaluate_fn, checkpoint_fn, checkpoint_interval)
    return ParallelStrategy(
        evaluate_fn, checkpoint_fn, checkpoint_interval, max_workers
    )


__all__ = [
    "ExecutionStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
    "create_strategy",
    "ProgressCallback",
]
