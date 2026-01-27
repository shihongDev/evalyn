"""Evaluation execution package.

This package contains:
- runner: EvalRunner for executing evaluation pipelines
- execution: Execution strategies (sequential, parallel)
- batch: Batch processing with provider APIs
- units: Evaluation unit discovery and projection
"""

from .runner import (
    EvalRunner,
    save_eval_run_json,
    load_eval_run_json,
    list_eval_runs_json,
)
from .execution import (
    ExecutionStrategy,
    SequentialStrategy,
    ParallelStrategy,
    create_strategy,
    ProgressCallback,
)
from .batch import (
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
from .units import (
    EvalUnitBuilder,
    OutcomeBuilder,
    SingleTurnBuilder,
    ToolUseBuilder,
    MultiTurnBuilder,
    CustomBuilder,
    get_default_builders,
    get_builder_for_type,
    get_builders_for_types,
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
