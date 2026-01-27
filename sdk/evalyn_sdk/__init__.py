__version__ = "0.0.1"

# Core decorators and tracer
from .decorators import eval, configure_tracer, get_default_tracer
from .trace.tracer import EvalTracer, eval_session

# Modules - auto-instrumentation
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

# Models
from .models import Metric, MetricRegistry

# Objective metrics (must be imported before judges to avoid circular import)
from .metrics.objective import (
    OBJECTIVE_REGISTRY,
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

# Judges (depends on metrics)
from .judges import LLMJudge, EchoJudge

# Subjective metrics
from .metrics.subjective import JUDGE_TEMPLATES, SUBJECTIVE_REGISTRY
from .metrics.factory import (
    build_objective_metric,
    build_subjective_metric,
    list_template_ids,
)

# Datasets
from .datasets import (
    load_dataset,
    save_dataset,
    hash_inputs,
    dataset_from_calls,
    build_dataset_from_storage,
)

# Suggesters
from .metrics.suggester import (
    MetricSuggester,
    HeuristicSuggester,
    LLMSuggester,
    LLMRegistrySelector,
    DEFAULT_JUDGE_PROMPT,
)

# Annotation module
from . import annotation
from .annotation import (
    SpanAnnotation,
    AnnotationSpanType,
    LLMCallAnnotation,
    ToolCallAnnotation,
    ReasoningAnnotation,
    RetrievalAnnotation,
    OverallAnnotation,
    extract_spans_from_trace,
    get_annotation_prompts,
    ANNOTATION_SCHEMAS,
)

# Calibration module
from . import calibration
from .calibration import (
    CalibrationEngine,
    AlignmentMetrics,
    BasicOptimizer,
    PromptOptimizationResult,
    ValidationResult,
    GEPAConfig,
    GEPAOptimizer,
    GEPA_AVAILABLE,
    APEConfig,
    APEOptimizer,
    save_calibration,
    load_optimized_prompt,
)

# Simulation module
from . import simulation
from .simulation import (
    UserSimulator,
    AgentSimulator,
    SimulationConfig,
    GeneratedQuery,
    synthetic_dataset,
    simulate_agent,
)

# OpenTelemetry configuration
from .trace.otel import configure_otel, configure_default_otel

# Additional models
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
    # Modules
    "annotation",
    "calibration",
    "trace",  # trace module
    # Auto-instrumentation
    "auto_instrument",
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
    # Modules
    "simulation",
    # Simulation
    "UserSimulator",
    "AgentSimulator",
    "SimulationConfig",
    "GeneratedQuery",
    "synthetic_dataset",
    "simulate_agent",
]
