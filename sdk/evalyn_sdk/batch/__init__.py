"""Backwards compatibility re-export.

This package has moved to evalyn_sdk.evaluation.batch.
Please update your imports to:
    from evalyn_sdk.evaluation.batch import BatchEvaluator
"""

# Re-export everything from the new location for backwards compatibility
from ..evaluation.batch import (
    BatchEvaluator,
    BatchEvalProgress,
    BatchProvider,
    BatchJob,
    BatchResult,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    AnthropicBatchProvider,
    create_batch_provider,
)

__all__ = [
    "BatchEvaluator",
    "BatchEvalProgress",
    "BatchProvider",
    "BatchJob",
    "BatchResult",
    "GeminiBatchProvider",
    "OpenAIBatchProvider",
    "AnthropicBatchProvider",
    "create_batch_provider",
]
