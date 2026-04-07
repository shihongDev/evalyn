__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Eager imports: lightweight dataclasses only
# ---------------------------------------------------------------------------

import importlib as _importlib

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


# ---------------------------------------------------------------------------
# Lazy imports: loaded on first access via __getattr__ (PEP 562)
#
# Everything except models is deferred to keep `import evalyn_sdk` fast.
# The CLI and SDK API both benefit: the CLI avoids loading trace providers,
# evaluation runners, etc. just to parse commands, and SDK users only pay
# for the modules they actually use.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core decorators and tracer
    "eval": (".decorators", "eval"),
    "configure_tracer": (".decorators", "configure_tracer"),
    "get_default_tracer": (".decorators", "get_default_tracer"),
    "EvalTracer": (".trace.tracer", "EvalTracer"),
    "eval_session": (".trace.tracer", "eval_session"),
    # Auto-instrumentation
    "trace_decorator": (".trace.auto_instrument", "trace"),
    "patch_all": (".trace.auto_instrument", "patch_all"),
    "is_patched": (".trace.auto_instrument", "is_patched"),
    "calculate_cost": (".trace.instrumentation.providers._shared", "calculate_cost"),
    # Evaluation
    "EvalRunner": (".evaluation.runner", "EvalRunner"),
    # Datasets
    "load_dataset": (".datasets", "load_dataset"),
    "save_dataset": (".datasets", "save_dataset"),
    "hash_inputs": (".datasets", "hash_inputs"),
    "dataset_from_calls": (".datasets", "dataset_from_calls"),
    "build_dataset_from_storage": (".datasets", "build_dataset_from_storage"),
    # OpenTelemetry configuration
    "configure_otel": (".trace.otel", "configure_otel"),
    "configure_default_otel": (".trace.otel", "configure_default_otel"),
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
    "trace": ".trace",
    "auto_instrument": ".trace.auto_instrument",
    "annotation": ".annotation",
    "calibration": ".calibration",
    "simulation": ".simulation",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val  # Cache so __getattr__ is not called again
        return val

    if name in _LAZY_SUBMODULES:
        mod = _importlib.import_module(_LAZY_SUBMODULES[name], __name__)
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
