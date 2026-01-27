"""Batch processing for LLM evaluation.

Supports batch APIs from multiple providers for cost-effective large-scale evaluation:
- Gemini: 50% cost reduction, 24-hour turnaround
- OpenAI: 50% cost reduction, 24-hour turnaround
- Anthropic: 50% cost reduction, <24 hour turnaround

Usage:
    from evalyn_sdk.evaluation.batch import BatchEvaluator

    # Create evaluator
    evaluator = BatchEvaluator(provider="gemini")

    # Run batch evaluation
    results = evaluator.evaluate(
        prepared=[(item1, call1), (item2, call2), ...],
        metrics=subjective_metrics,
    )

Low-level provider API:
    from evalyn_sdk.evaluation.batch import create_batch_provider

    provider = create_batch_provider("openai")
    job = provider.submit([{"custom_id": "1", "prompt": "..."}])
    provider.wait(job.id)
    results = provider.get_results(job.id)
"""

from .providers import (
    BatchProvider,
    BatchJob,
    BatchResult,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    AnthropicBatchProvider,
    create_batch_provider,
)
from .evaluator import BatchEvaluator, BatchEvalProgress

__all__ = [
    # High-level API
    "BatchEvaluator",
    "BatchEvalProgress",
    # Low-level providers
    "BatchProvider",
    "BatchJob",
    "BatchResult",
    "GeminiBatchProvider",
    "OpenAIBatchProvider",
    "AnthropicBatchProvider",
    "create_batch_provider",
]
