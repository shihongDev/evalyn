__version__ = "0.0.1"

from .decorators import eval, configure_tracer, get_default_tracer
from .tracing import EvalTracer, eval_session
from .runner import EvalRunner
from .metrics.registry import MetricRegistry, Metric
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
from .metrics.subjective import (
    subjective_metric,
    tone_metric,
    toxicity_metric,
    DEFAULT_TONE_PROMPT,
    DEFAULT_TOXICITY_PROMPT,
)
from .metrics.templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from .metrics.factory import build_objective_metric, build_subjective_metric, list_template_ids
from .metrics.judges import LLMJudge, EchoJudge, OpenAIJudge, GeminiJudge
from .datasets import load_dataset, save_dataset, hash_inputs, dataset_from_calls, build_dataset_from_storage
from .metrics.suggester import MetricSuggester, HeuristicSuggester, LLMSuggester, LLMRegistrySelector, DEFAULT_JUDGE_PROMPT
from .calibration import CalibrationEngine, AlignmentMetrics, PromptOptimizer, PromptOptimizationResult
from .otel import configure_otel, configure_default_otel, OTEL_AVAILABLE
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
    "Annotation",
    "CalibrationRecord",
    "DatasetItem",
    "EvalRun",
    "FunctionCall",
    "MetricResult",
    "MetricSpec",
    "MetricType",
    "EvalRunner",
    "EvalTracer",
    "MetricRegistry",
    "Metric",
    "LLMJudge",
    "EchoJudge",
    "OpenAIJudge",
    "GeminiJudge",
    "MetricSuggester",
    "HeuristicSuggester",
    "LLMSuggester",
    "LLMRegistrySelector",
    "DEFAULT_JUDGE_PROMPT",
    "CalibrationEngine",
    "AlignmentMetrics",
    "PromptOptimizer",
    "PromptOptimizationResult",
    "build_dataset_from_storage",
    "configure_otel",
    "configure_default_otel",
    "OTEL_AVAILABLE",
    "load_dataset",
    "save_dataset",
    "hash_inputs",
    "dataset_from_calls",
    "eval_session",
    "eval",
    "configure_tracer",
    "get_default_tracer",
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
    "subjective_metric",
    "tone_metric",
    "toxicity_metric",
    "DEFAULT_TONE_PROMPT",
    "DEFAULT_TOXICITY_PROMPT",
    "OBJECTIVE_TEMPLATES",
    "SUBJECTIVE_TEMPLATES",
    "build_objective_metric",
    "build_subjective_metric",
    "list_template_ids",
]
