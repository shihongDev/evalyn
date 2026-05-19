"""Evaluation execution package.

This package contains:
- runner: EvalRunner for executing evaluation pipelines
- execution: Execution strategies (sequential, parallel)
- batch: Batch processing with provider APIs
- units: Evaluation unit discovery and projection
"""

from .batch import (
    AnthropicBatchProvider,
    BatchEvalProgress,
    BatchEvaluator,
    BatchJob,
    BatchProvider,
    BatchResult,
    GeminiBatchProvider,
    OpenAIBatchProvider,
    create_batch_provider,
)
from .execution import (
    ExecutionStrategy,
    ParallelStrategy,
    ProgressCallback,
    SequentialStrategy,
    create_strategy,
)
from .runner import (
    EvalRunner,
    list_eval_runs_json,
    load_eval_run_json,
    save_eval_run_json,
)
from .units import (
    CustomBuilder,
    EvalUnitBuilder,
    MultiTurnBuilder,
    OutcomeBuilder,
    SingleTurnBuilder,
    ToolUseBuilder,
    get_builder_for_type,
    get_builders_for_types,
    get_default_builders,
    project_unit,
)

__all__ = [
    # Runner
    "EvalRunner",
    "save_eval_run_json",
    "load_eval_run_json",
    "list_eval_runs_json",
    # Execution strategies
    "ExecutionStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
    "create_strategy",
    "ProgressCallback",
    # Batch processing
    "BatchEvaluator",
    "BatchEvalProgress",
    "BatchProvider",
    "BatchJob",
    "BatchResult",
    "GeminiBatchProvider",
    "OpenAIBatchProvider",
    "AnthropicBatchProvider",
    "create_batch_provider",
    # Evaluation units
    "EvalUnitBuilder",
    "OutcomeBuilder",
    "SingleTurnBuilder",
    "ToolUseBuilder",
    "MultiTurnBuilder",
    "CustomBuilder",
    "get_default_builders",
    "get_builder_for_type",
    "get_builders_for_types",
    "project_unit",
]
