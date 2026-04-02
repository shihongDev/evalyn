__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Eager imports: core functionality needed by all users
# ---------------------------------------------------------------------------

# Core decorators and tracer
from .decorators import eval, configure_tracer, get_default_tracer
from .trace.tracer import EvalTracer, eval_session

# Modules - auto-instrumentation (must load eagerly for sys.meta_path patching)
from . import trace
from .trace import auto_instrument
from .trace.auto_instrument import (
    trace as trace_decorator,
    patch_all,
    is_patched,
)
from .trace.instrumentation.providers._shared import calculate_cost

# Evaluation
from .evaluation.runner import EvalRunner

# Models (lightweight dataclasses)
from .models import (
    Annotation,
    CalibrationRecord,
    DatasetItem,
    EvalRun,
    FunctionCall,
    Metric,
    MetricRegistry,
    MetricResult,
    MetricSpec,
    MetricType,
)

# Datasets
from .datasets import (
    load_dataset,
    save_dataset,
    hash_inputs,
    dataset_from_calls,
    build_dataset_from_storage,
)

# OpenTelemetry configuration
from .trace.otel import configure_otel, configure_default_otel


# ---------------------------------------------------------------------------
# Lazy imports: loaded on first access via __getattr__ (PEP 562)
#
# Defers calibration (9 optimizers), simulation, annotation, judges,
# suggesters, and metric registries to reduce import time and memory
# when users only need core tracing and evaluation.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Judges
    "LLMJudge": (".judges", "LLMJudge"),
    "EchoJudge": (".judges", "EchoJudge"),
    # Objective metrics
    "OBJECTIVE_REGISTRY": (".metrics.objective", "OBJECTIVE_REGISTRY"),
    "exact_match_metric": (".metrics.objective", "exact_match_metric"),
    "latency_metric": (".metrics.objective", "latency_metric"),
    "cost_metric": (".metrics.objective", "cost_metric"),
    "bleu_metric": (".metrics.objective", "bleu_metric"),
    "pass_at_k_metric": (".metrics.objective", "pass_at_k_metric"),
    "json_valid_metric": (".metrics.objective", "json_valid_metric"),
    "regex_match_metric": (".metrics.objective", "regex_match_metric"),
    "token_length_metric": (".metrics.objective", "token_length_metric"),
    "tool_call_count_metric": (".metrics.objective", "tool_call_count_metric"),
    "register_builtin_metrics": (".metrics.objective", "register_builtin_metrics"),
    # Subjective metrics
    "JUDGE_TEMPLATES": (".metrics.subjective", "JUDGE_TEMPLATES"),
    "SUBJECTIVE_REGISTRY": (".metrics.subjective", "SUBJECTIVE_REGISTRY"),
    # Metric factory
    "build_objective_metric": (".metrics.factory", "build_objective_metric"),
    "build_subjective_metric": (".metrics.factory", "build_subjective_metric"),
    "list_template_ids": (".metrics.factory", "list_template_ids"),
    # Suggesters
    "MetricSuggester": (".metrics.suggester", "MetricSuggester"),
    "HeuristicSuggester": (".metrics.suggester", "HeuristicSuggester"),
    "LLMSuggester": (".metrics.suggester", "LLMSuggester"),
    "LLMRegistrySelector": (".metrics.suggester", "LLMRegistrySelector"),
    "DEFAULT_JUDGE_PROMPT": (".metrics.suggester", "DEFAULT_JUDGE_PROMPT"),
    # Annotation module
    "SpanAnnotation": (".annotation", "SpanAnnotation"),
    "AnnotationSpanType": (".annotation", "AnnotationSpanType"),
    "LLMCallAnnotation": (".annotation", "LLMCallAnnotation"),
    "ToolCallAnnotation": (".annotation", "ToolCallAnnotation"),
    "ReasoningAnnotation": (".annotation", "ReasoningAnnotation"),
    "RetrievalAnnotation": (".annotation", "RetrievalAnnotation"),
    "OverallAnnotation": (".annotation", "OverallAnnotation"),
    "extract_spans_from_trace": (".annotation", "extract_spans_from_trace"),
    "get_annotation_prompts": (".annotation", "get_annotation_prompts"),
    "ANNOTATION_SCHEMAS": (".annotation", "ANNOTATION_SCHEMAS"),
    # Calibration module
    "CalibrationEngine": (".calibration", "CalibrationEngine"),
    "AlignmentMetrics": (".calibration", "AlignmentMetrics"),
    "BasicOptimizer": (".calibration", "BasicOptimizer"),
    "PromptOptimizationResult": (".calibration", "PromptOptimizationResult"),
    "ValidationResult": (".calibration", "ValidationResult"),
    "GEPAConfig": (".calibration", "GEPAConfig"),
    "GEPAOptimizer": (".calibration", "GEPAOptimizer"),
    "GEPA_AVAILABLE": (".calibration", "GEPA_AVAILABLE"),
    "APEConfig": (".calibration", "APEConfig"),
    "APEOptimizer": (".calibration", "APEOptimizer"),
    "save_calibration": (".calibration", "save_calibration"),
    "load_optimized_prompt": (".calibration", "load_optimized_prompt"),
    # Simulation module
    "UserSimulator": (".simulation", "UserSimulator"),
    "AgentSimulator": (".simulation", "AgentSimulator"),
    "SimulationConfig": (".simulation", "SimulationConfig"),
    "GeneratedQuery": (".simulation", "GeneratedQuery"),
    "synthetic_dataset": (".simulation", "synthetic_dataset"),
    "simulate_agent": (".simulation", "simulate_agent"),
}

# Submodules that should be importable as `evalyn_sdk.X`
_LAZY_SUBMODULES: dict[str, str] = {
    "annotation": ".annotation",
    "calibration": ".calibration",
    "simulation": ".simulation",
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val  # Cache so __getattr__ is not called again
        return val

    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(_LAZY_SUBMODULES[name], __name__)
        globals()[name] = mod
        return mod

    raise AttributeError(f"module 'evalyn_sdk' has no attribute {name!r}")


def __dir__():
    eager = list(globals().keys())
    return sorted(set(eager + list(_LAZY_IMPORTS) + list(_LAZY_SUBMODULES)))


__all__ = [
    # Modules
    "annotation",
    "calibration",
    "trace",
    "simulation",
    # Auto-instrumentation
    "auto_instrument",
    "trace_decorator",
    "patch_all",
    "is_patched",
    "calculate_cost",
    # Models
    "Annotation",
    "CalibrationRecord",
    "DatasetItem",
    "EvalRun",
    "FunctionCall",
    "MetricResult",
    "MetricSpec",
    "MetricType",
    # Core
    "EvalRunner",
    "EvalTracer",
    "MetricRegistry",
    "Metric",
    # Judges
    "LLMJudge",
    "EchoJudge",
    "JUDGE_TEMPLATES",
    # Suggesters
    "MetricSuggester",
    "HeuristicSuggester",
    "LLMSuggester",
    "LLMRegistrySelector",
    "DEFAULT_JUDGE_PROMPT",
    # Calibration
    "CalibrationEngine",
    "AlignmentMetrics",
    "BasicOptimizer",
    "PromptOptimizationResult",
    "ValidationResult",
    "GEPAConfig",
    "GEPAOptimizer",
    "GEPA_AVAILABLE",
    "APEConfig",
    "APEOptimizer",
    "save_calibration",
    "load_optimized_prompt",
    # Datasets
    "build_dataset_from_storage",
    "load_dataset",
    "save_dataset",
    "hash_inputs",
    "dataset_from_calls",
    # OpenTelemetry
    "configure_otel",
    "configure_default_otel",
    # Tracing
    "eval_session",
    "eval",
    "configure_tracer",
    "get_default_tracer",
    # Objective metrics
    "exact_match_metric",
    "latency_metric",
    "cost_metric",
    "bleu_metric",
    "pass_at_k_metric",
    "json_valid_metric",
    "regex_match_metric",
    "token_length_metric",
    "tool_call_count_metric",
    "register_builtin_metrics",
    # Templates
    "OBJECTIVE_REGISTRY",
    "SUBJECTIVE_REGISTRY",
    "build_objective_metric",
    "build_subjective_metric",
    "list_template_ids",
    # Span annotations
    "SpanAnnotation",
    "AnnotationSpanType",
    "LLMCallAnnotation",
    "ToolCallAnnotation",
    "ReasoningAnnotation",
    "RetrievalAnnotation",
    "OverallAnnotation",
    "extract_spans_from_trace",
    "get_annotation_prompts",
    "ANNOTATION_SCHEMAS",
    # Simulation
    "UserSimulator",
    "AgentSimulator",
    "SimulationConfig",
    "GeneratedQuery",
    "synthetic_dataset",
    "simulate_agent",
]
