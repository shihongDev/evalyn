"""CLI utility modules."""

from .ui import Spinner, ProgressIndicator
from .config import (
    load_config,
    get_config_default,
    find_latest_dataset,
    resolve_dataset_path,
    _expand_env_vars,
)
from .loaders import (
    _load_callable,
    _get_module_callables,
    _suggest_similar,
)
from .llm_callers import (
    _parse_json_array,
    _ollama_caller,
    _openai_caller,
    _with_spinner,
    _build_llm_caller,
)
from .dataset_utils import (
    _extract_code_meta,
    _resolve_dataset_and_metrics,
    _dataset_has_reference,
    ProgressBar,
)
from .dataset_resolver import DatasetResolver, DatasetInfo, get_dataset
from .formatters import OutputFormatter, get_formatter
from .pipeline import PipelineStep, PipelineOrchestrator, PipelineState, StepResult
from .pipeline_steps import create_pipeline_steps

__all__ = [
    # UI
    "Spinner",
    "ProgressIndicator",
    # Config
    "load_config",
    "get_config_default",
    "find_latest_dataset",
    "resolve_dataset_path",
    "_expand_env_vars",
    # Loaders
    "_load_callable",
    "_get_module_callables",
    "_suggest_similar",
    # LLM callers
    "_parse_json_array",
    "_ollama_caller",
    "_openai_caller",
    "_with_spinner",
    "_build_llm_caller",
    # Dataset utils
    "_extract_code_meta",
    "_resolve_dataset_and_metrics",
    "_dataset_has_reference",
    "ProgressBar",
    # Dataset resolver
    "DatasetResolver",
    "DatasetInfo",
    "get_dataset",
    # Formatters
    "OutputFormatter",
    "get_formatter",
    # Pipeline
    "PipelineStep",
    "PipelineOrchestrator",
    "PipelineState",
    "StepResult",
    "create_pipeline_steps",
]
