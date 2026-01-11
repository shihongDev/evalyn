__version__ = "0.0.1"

from .decorators import eval, configure_tracer, get_default_tracer
from .trace.tracer import EvalTracer, eval_session

# Auto-instrumentation (patches LLM libraries on import)
from . import trace
from .trace import auto_instrument
from .trace.auto_instrument import (
    trace as trace_decorator,
    patch_all,
    is_patched,
    calculate_cost,
)
from .runner import EvalRunner
from .models import Metric, MetricRegistry
from .metrics.objective import (
    exact_match_metric,
    latency_metric,
    cost_metric,
    bleu_metric,
    pass_at_k_metric,
    json_valid_metric,
    regex_match_metric,
    token_length_metric,
    tool_call_count_metric,
    register_builtin_metrics,
)
from .metrics.judges import LLMJudge, EchoJudge, JUDGE_TEMPLATES
from .metrics.templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from .metrics.factory import (
    build_objective_metric,
    build_subjective_metric,
    list_template_ids,
)
from .datasets import (
    load_dataset,
    save_dataset,
    hash_inputs,
    dataset_from_calls,
    build_dataset_from_storage,
)
from .metrics.suggester import (
    MetricSuggester,
    HeuristicSuggester,
    LLMSuggester,
    LLMRegistrySelector,
    DEFAULT_JUDGE_PROMPT,
)

# Annotation and calibration
from . import annotation
from .annotation import (
    CalibrationEngine,
    AlignmentMetrics,
    PromptOptimizer,
    PromptOptimizationResult,
    ValidationResult,
    GEPAConfig,
    GEPAOptimizer,
    GEPA_AVAILABLE,
    save_calibration,
    load_optimized_prompt,
    SpanAnnotation,
    SpanType,
    LLMCallAnnotation,
    ToolCallAnnotation,
    ReasoningAnnotation,
    RetrievalAnnotation,
    OverallAnnotation,
    extract_spans_from_trace,
    get_annotation_prompts,
    ANNOTATION_SCHEMAS,
)

# Simulation
from . import simulation
from .simulation import (
    UserSimulator,
    AgentSimulator,
    SimulationConfig,
    GeneratedQuery,
    synthetic_dataset,
    simulate_agent,
    random_prompt_variations,
)
from .trace.otel import configure_otel, configure_default_otel, OTEL_AVAILABLE
from .models import (
    Annotation,
    CalibrationRecord,
    DatasetItem,
    EvalRun,
    FunctionCall,
    MetricResult,
    MetricSpec,
    MetricType,
)

__all__ = [
    # Auto-instrumentation
    "auto_instrument",
    "trace",  # trace module
    "trace_decorator",  # @trace decorator
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
    "PromptOptimizer",
    "PromptOptimizationResult",
    "GEPAConfig",
    "GEPAOptimizer",
    "GEPA_AVAILABLE",
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
    "OTEL_AVAILABLE",
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
    "OBJECTIVE_TEMPLATES",
    "SUBJECTIVE_TEMPLATES",
    "build_objective_metric",
    "build_subjective_metric",
    "list_template_ids",
    # Span annotations
    "SpanAnnotation",
    "SpanType",
    "LLMCallAnnotation",
    "ToolCallAnnotation",
    "ReasoningAnnotation",
    "RetrievalAnnotation",
    "OverallAnnotation",
    "extract_spans_from_trace",
    "get_annotation_prompts",
    "ANNOTATION_SCHEMAS",
    # Modules
    "annotation",
    "simulation",
    # Simulation
    "UserSimulator",
    "AgentSimulator",
    "SimulationConfig",
    "GeneratedQuery",
    "synthetic_dataset",
    "simulate_agent",
    "random_prompt_variations",
]
