from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .objective import (
    # Efficiency
    latency_metric,
    cost_metric,
    token_length_metric,
    compression_ratio_metric,
    # Correctness
    bleu_metric,
    pass_at_k_metric,
    levenshtein_similarity_metric,
    cosine_word_overlap_metric,
    # Structure
    json_valid_metric,
    regex_match_metric,
    csv_valid_metric,
    xml_valid_metric,
    json_schema_keys_metric,
    json_types_match_metric,
    json_path_present_metric,
    regex_capture_count_metric,
    syntax_valid_metric,
    code_complexity_metric,
    # Text overlap
    rouge_l_metric,
    rouge_1_metric,
    rouge_2_metric,
    token_overlap_f1_metric,
    jaccard_similarity_metric,
    # Numeric
    numeric_mae_metric,
    numeric_rmse_metric,
    numeric_rel_error_metric,
    numeric_within_tolerance_metric,
    # Output quality
    output_nonempty_metric,
    output_length_range_metric,
    # Readability
    flesch_kincaid_metric,
    sentence_count_metric,
    avg_sentence_length_metric,
    # Diversity
    distinct_1_metric,
    distinct_2_metric,
    vocabulary_richness_metric,
    # Trace-based
    tool_call_count_metric,
    llm_call_count_metric,
    llm_error_rate_metric,
    tool_success_ratio_metric,
    tool_error_count_metric,
    # Grounding
    url_count_metric,
    citation_count_metric,
    markdown_link_count_metric,
    # Format validation
    yaml_valid_metric,
    markdown_structure_metric,
    html_valid_metric,
    sql_valid_metric,
    # Structure detection
    bullet_count_metric,
    heading_count_metric,
    code_block_count_metric,
    table_count_metric,
    paragraph_count_metric,
    word_count_metric,
    # Repetition
    repetition_ratio_metric,
    duplicate_line_ratio_metric,
    # Uncertainty/Confidence
    hedging_count_metric,
    question_count_metric,
    confidence_markers_metric,
    # Code quality
    comment_ratio_metric,
    function_count_metric,
    import_count_metric,
    # Character/Format
    ascii_ratio_metric,
    uppercase_ratio_metric,
    numeric_density_metric,
    whitespace_ratio_metric,
    # Match variants
    prefix_match_metric,
    suffix_match_metric,
    contains_all_metric,
    contains_none_metric,
    # List/Enumeration
    numbered_list_count_metric,
    list_item_count_metric,
    # Response quality
    emoji_count_metric,
    link_density_metric,
)
from ..models import Metric
from .judges import LLMJudge
from .objective import OBJECTIVE_REGISTRY
from .subjective import SUBJECTIVE_REGISTRY


def _tpl_by_id(templates: List[dict]) -> Dict[str, dict]:
    return {t["id"]: t for t in templates}


def _safe_float(value: Any, default: float) -> float:
    """Convert value to float with fallback to default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_rubric(rubric: Any) -> List[str]:
    """Normalize rubric to a list of strings."""
    if not rubric:
        return []
    if isinstance(rubric, str):
        return [rubric]
    return rubric if isinstance(rubric, list) else []


_OBJECTIVE_TPL = _tpl_by_id(OBJECTIVE_REGISTRY)
_SUBJECTIVE_TPL = _tpl_by_id(SUBJECTIVE_REGISTRY)


def _wrap_with_consistency_confidence(
    base_metric: Metric,
    judge: "LLMJudge",
    n_samples: int,
    threshold: float,
) -> Metric:
    """Wrap a subjective metric with self-consistency confidence calculation.

    Runs the judge N times (with temp > 0) and calculates confidence as agreement ratio.
    Higher agreement = higher confidence in the judgment.

    NOTE: This temporarily sets temperature to 0.7 to get diverse samples.
    With temp=0, all samples would be identical.
    """
    from ..confidence import SelfConsistencyConfidence
    from ..models import DatasetItem, FunctionCall, MetricResult
    from .judges import LLMJudge as JudgeClass

    confidence_estimator = SelfConsistencyConfidence(n_samples=n_samples)

    # Create a sampling judge with temp > 0 for diversity
    sampling_judge = JudgeClass(
        name=judge.name,
        prompt=judge.prompt,
        model=judge.model,
        temperature=0.7,  # Need diversity for self-consistency
        api_key=judge._api_key,
        rubric=judge.rubric,
    )

    def handler_with_confidence(call: FunctionCall, item: DatasetItem) -> MetricResult:
        # Run judge multiple times with temp > 0
        results = [sampling_judge.score(call, item) for _ in range(n_samples)]

        # Extract pass/fail verdicts for consistency check
        def extract_verdict(r):
            return r.get("passed")

        confidence_result = confidence_estimator.estimate(
            samples=results,
            answer_extractor=extract_verdict,
        )

        # Use majority vote result
        verdicts = [r.get("passed") for r in results]
        pass_count = sum(1 for v in verdicts if v is True)
        fail_count = sum(1 for v in verdicts if v is False)

        if pass_count > fail_count:
            passed = True
            score = 1.0
        elif fail_count > pass_count:
            passed = False
            score = 0.0
        else:
            # Tie: use first result
            passed = results[0].get("passed")
            score = results[0].get("score", 0.5)

        # Collect reasons from all samples
        reasons = [r.get("reason") for r in results if r.get("reason")]
        majority_reason = reasons[0] if reasons else None

        # Aggregate token usage from all samples
        total_input_tokens = sum(r.get("input_tokens", 0) or 0 for r in results)
        total_output_tokens = sum(r.get("output_tokens", 0) or 0 for r in results)
        model = results[0].get("model") if results else None

        return MetricResult(
            metric_id=base_metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={
                "judge": judge.name,
                "reason": majority_reason,
                "confidence": round(confidence_result.score, 4),
                "confidence_method": "consistency",
                "n_samples": n_samples,
                "pass_count": pass_count,
                "fail_count": fail_count,
            },
            raw_judge={"samples": results, "confidence": confidence_result.details},
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=model,
        )

    return Metric(base_metric.spec, handler_with_confidence)


def _wrap_with_logprobs_confidence(
    base_metric: Metric,
    judge: "LLMJudge",
    threshold: float,
) -> Metric:
    """Wrap a subjective metric with logprobs-based confidence.

    Uses token log probabilities to calculate confidence.
    Only works with providers that support logprobs (openai, ollama).
    """
    from ..models import DatasetItem, FunctionCall, MetricResult

    def handler_with_confidence(call: FunctionCall, item: DatasetItem) -> MetricResult:
        # score_with_confidence returns result dict with confidence included
        result = judge.score_with_confidence(call, item)

        passed = result.get("passed")
        score = result.get("score")
        if score is None and passed is not None:
            score = 1.0 if passed else 0.0

        confidence = result.get("confidence", 0.5)

        return MetricResult(
            metric_id=base_metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={
                "judge": judge.name,
                "reason": result.get("reason"),
                "confidence": round(confidence, 4),
                "confidence_method": "logprobs",
            },
            raw_judge=result,
        )

    return Metric(base_metric.spec, handler_with_confidence)


def _wrap_with_deepconf_confidence(
    base_metric: Metric,
    judge: "LLMJudge",
    threshold: float,
    strategy: str = "bottom10",
) -> Metric:
    """Wrap a subjective metric with DeepConf confidence.

    DeepConf uses specialized aggregation strategies (bottom-10%, tail)
    that better distinguish correct from incorrect reasoning traces.
    Only works with providers that support logprobs (openai, limited ollama).

    References:
        - "Deep Think with Confidence" (arXiv:2508.15260)
        - https://jiaweizzhao.github.io/deepconf/
    """
    from ..models import DatasetItem, FunctionCall, MetricResult

    def handler_with_confidence(call: FunctionCall, item: DatasetItem) -> MetricResult:
        result = judge.score_with_deepconf(call, item, strategy=strategy)

        passed = result.get("passed")
        score = result.get("score")
        if score is None and passed is not None:
            score = 1.0 if passed else 0.0

        confidence = result.get("confidence", 0.5)

        return MetricResult(
            metric_id=base_metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={
                "judge": judge.name,
                "reason": result.get("reason"),
                "confidence": round(confidence, 4),
                "confidence_method": f"deepconf_{strategy}",
            },
            raw_judge=result,
        )

    return Metric(base_metric.spec, handler_with_confidence)


def list_template_ids() -> List[str]:
    """Return all known template ids (objective + subjective)."""
    return sorted(list(_OBJECTIVE_TPL.keys()) + list(_SUBJECTIVE_TPL.keys()))


def _apply_unit_types(metric: Metric, config: Optional[Dict[str, Any]]) -> Metric:
    """Apply unit_types from config to metric's spec if present."""
    if config and "unit_types" in config:
        unit_types = config["unit_types"]
        if isinstance(unit_types, str):
            # Handle comma-separated string
            unit_types = [t.strip() for t in unit_types.split(",") if t.strip()]
        if isinstance(unit_types, list) and unit_types:
            metric.spec.unit_types = unit_types
    return metric


def build_objective_metric(
    metric_id: str, config: Optional[Dict[str, Any]] = None
) -> Metric:
    cfg = config or {}
    builders: Dict[str, Callable[[Dict[str, Any]], Metric]] = {
        # Efficiency
        "latency_ms": lambda c: latency_metric(),
        "cost": lambda c: cost_metric(),
        "token_length": lambda c: token_length_metric(max_chars=c.get("max_chars")),
        "compression_ratio": lambda c: compression_ratio_metric(
            min_ratio=c.get("min_ratio"),
            max_ratio=c.get("max_ratio"),
        ),
        # Correctness
        "bleu": lambda c: bleu_metric(),
        "pass_at_k": lambda c: pass_at_k_metric(
            k=int(c.get("k", 5)),
            candidate_field=str(c.get("candidate_field", "candidates")),
            success_field=str(c.get("success_field", "passed")),
        ),
        "levenshtein_similarity": lambda c: levenshtein_similarity_metric(),
        "cosine_word_overlap": lambda c: cosine_word_overlap_metric(),
        # Structure
        "json_valid": lambda c: json_valid_metric(),
        "regex_match": lambda c: regex_match_metric(pattern=c.get("pattern")),
        "csv_valid": lambda c: csv_valid_metric(dialect=str(c.get("dialect", "excel"))),
        "xml_valid": lambda c: xml_valid_metric(),
        "json_schema_keys": lambda c: json_schema_keys_metric(
            required_keys=c.get("required_keys")
        ),
        "json_types_match": lambda c: json_types_match_metric(schema=c.get("schema")),
        "json_path_present": lambda c: json_path_present_metric(paths=c.get("paths")),
        "regex_capture_count": lambda c: regex_capture_count_metric(
            pattern=c.get("pattern"),
            min_count=int(c.get("min_count", 1)),
        ),
        "syntax_valid": lambda c: syntax_valid_metric(
            language=str(c.get("language", "python"))
        ),
        "code_complexity": lambda c: code_complexity_metric(
            max_lines=c.get("max_lines"),
            max_depth=c.get("max_depth"),
        ),
        # Text overlap
        "rouge_l": lambda c: rouge_l_metric(),
        "rouge_1": lambda c: rouge_1_metric(),
        "rouge_2": lambda c: rouge_2_metric(),
        "token_overlap_f1": lambda c: token_overlap_f1_metric(),
        "jaccard_similarity": lambda c: jaccard_similarity_metric(),
        # Numeric
        "numeric_mae": lambda c: numeric_mae_metric(output_field=c.get("output_field")),
        "numeric_rmse": lambda c: numeric_rmse_metric(
            output_field=c.get("output_field")
        ),
        "numeric_rel_error": lambda c: numeric_rel_error_metric(
            output_field=c.get("output_field")
        ),
        "numeric_within_tolerance": lambda c: numeric_within_tolerance_metric(
            output_field=c.get("output_field"),
            tolerance=float(c.get("tolerance", 0.0)),
        ),
        # Output quality
        "output_nonempty": lambda c: output_nonempty_metric(),
        "output_length_range": lambda c: output_length_range_metric(
            min_chars=int(c.get("min_chars", 0)),
            max_chars=c.get("max_chars"),
        ),
        # Readability
        "flesch_kincaid": lambda c: flesch_kincaid_metric(max_grade=c.get("max_grade")),
        "sentence_count": lambda c: sentence_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "avg_sentence_length": lambda c: avg_sentence_length_metric(
            max_avg=c.get("max_avg")
        ),
        # Diversity
        "distinct_1": lambda c: distinct_1_metric(min_ratio=c.get("min_ratio")),
        "distinct_2": lambda c: distinct_2_metric(min_ratio=c.get("min_ratio")),
        "vocabulary_richness": lambda c: vocabulary_richness_metric(
            min_ratio=c.get("min_ratio")
        ),
        # Trace-based
        "tool_call_count": lambda c: tool_call_count_metric(),
        "llm_call_count": lambda c: llm_call_count_metric(
            request_kind=str(c.get("request_kind", ".request"))
        ),
        "llm_error_rate": lambda c: llm_error_rate_metric(
            request_kind=str(c.get("request_kind", ".request")),
            error_kind=str(c.get("error_kind", ".error")),
        ),
        "tool_success_ratio": lambda c: tool_success_ratio_metric(
            success_kind=str(c.get("success_kind", "tool.success")),
            error_kind=str(c.get("error_kind", "tool.error")),
        ),
        "tool_error_count": lambda c: tool_error_count_metric(
            error_kind=str(c.get("error_kind", "tool.error"))
        ),
        # Grounding
        "url_count": lambda c: url_count_metric(
            pattern=str(c.get("pattern", r"https?://")),
            min_count=int(c.get("min_count", 1)),
        ),
        "citation_count": lambda c: citation_count_metric(
            min_count=int(c.get("min_count", 0))
        ),
        "markdown_link_count": lambda c: markdown_link_count_metric(
            min_count=int(c.get("min_count", 0))
        ),
        # Format validation
        "yaml_valid": lambda c: yaml_valid_metric(),
        "markdown_structure": lambda c: markdown_structure_metric(
            require_heading=c.get("require_heading", False)
        ),
        "html_valid": lambda c: html_valid_metric(),
        "sql_valid": lambda c: sql_valid_metric(),
        # Structure detection
        "bullet_count": lambda c: bullet_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "heading_count": lambda c: heading_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "code_block_count": lambda c: code_block_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "table_count": lambda c: table_count_metric(min_count=c.get("min_count")),
        "paragraph_count": lambda c: paragraph_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "word_count": lambda c: word_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        # Repetition
        "repetition_ratio": lambda c: repetition_ratio_metric(
            n=int(c.get("n", 3)),
            max_ratio=c.get("max_ratio"),
        ),
        "duplicate_line_ratio": lambda c: duplicate_line_ratio_metric(
            max_ratio=c.get("max_ratio")
        ),
        # Uncertainty/Confidence
        "hedging_count": lambda c: hedging_count_metric(max_count=c.get("max_count")),
        "question_count": lambda c: question_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "confidence_markers": lambda c: confidence_markers_metric(),
        # Code quality
        "comment_ratio": lambda c: comment_ratio_metric(min_ratio=c.get("min_ratio")),
        "function_count": lambda c: function_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "import_count": lambda c: import_count_metric(max_count=c.get("max_count")),
        # Character/Format
        "ascii_ratio": lambda c: ascii_ratio_metric(min_ratio=c.get("min_ratio")),
        "uppercase_ratio": lambda c: uppercase_ratio_metric(
            max_ratio=c.get("max_ratio")
        ),
        "numeric_density": lambda c: numeric_density_metric(),
        "whitespace_ratio": lambda c: whitespace_ratio_metric(
            max_ratio=c.get("max_ratio")
        ),
        # Match variants
        "prefix_match": lambda c: prefix_match_metric(prefix=c.get("prefix")),
        "suffix_match": lambda c: suffix_match_metric(suffix=c.get("suffix")),
        "contains_all": lambda c: contains_all_metric(substrings=c.get("substrings")),
        "contains_none": lambda c: contains_none_metric(forbidden=c.get("forbidden")),
        # List/Enumeration
        "numbered_list_count": lambda c: numbered_list_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        "list_item_count": lambda c: list_item_count_metric(
            min_count=c.get("min_count"),
            max_count=c.get("max_count"),
        ),
        # Response quality
        "emoji_count": lambda c: emoji_count_metric(max_count=c.get("max_count")),
        "link_density": lambda c: link_density_metric(max_ratio=c.get("max_ratio")),
    }

    if metric_id not in builders:
        raise KeyError(f"Unknown objective metric id: {metric_id}")
    metric = builders[metric_id](cfg)
    return _apply_unit_types(metric, cfg)


def build_subjective_metric(
    metric_id: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    judge: Optional[LLMJudge] = None,
    description: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: str = "gemini",
    confidence_method: str = "none",
    confidence_samples: int = 3,
) -> Metric:
    """
    Build a subjective metric from template ID OR custom config.

    Works in two modes:
    1. Template mode: metric_id matches a known template, uses template as base
    2. Custom mode: metric_id is custom, uses config directly (for brainstormed metrics)

    The prompt is constructed from:
    1. Template's base prompt (if any)
    2. Config's custom prompt (if provided, overrides template)
    3. Rubric items from config

    Config options:
    - prompt: Custom prompt (overrides template prompt)
    - rubric: List of criteria for PASS/FAIL evaluation
    - threshold: Score threshold for pass (default: 0.7)
    - model: Judge model name (depends on provider)
    - temperature: Judge temperature (default: 0.0)
    - description: Metric description (for custom metrics)
    - api_key: API key for LLM judge (optional, falls back to env var)

    Provider options:
    - provider: "gemini" (default), "openai", or "ollama"

    Confidence options:
    - confidence_method: "none", "consistency", "logprobs", or "deepconf"
      - consistency: Run judge N times with temp>0, measure agreement
      - logprobs: Use token probabilities (openai/ollama only)
      - deepconf: Meta AI's DeepConf with bottom-10% aggregation (openai only)
    - confidence_samples: Number of samples for consistency method (default: 3)
    """
    tpl = _SUBJECTIVE_TPL.get(metric_id)
    cfg = config or {}

    # Merge template config with provided config
    if tpl:
        base_cfg = dict(tpl.get("config") or {})
        cfg = {**base_cfg, **cfg}
        tpl_description = tpl.get("description", "")
        tpl_prompt = tpl.get("prompt", "")
    else:
        # Custom metric (e.g., from brainstorm mode)
        tpl_description = ""
        tpl_prompt = ""

    # Parse threshold with fallback to default
    threshold_f = _safe_float(cfg.get("threshold"), default=0.7)

    # Build prompt from config or template
    custom_prompt = cfg.get("prompt")
    if custom_prompt:
        prompt = str(custom_prompt).strip()
    elif tpl_prompt:
        prompt = str(tpl_prompt).strip()
    else:
        prompt = f"You are an expert evaluator for '{metric_id}'. Evaluate the agent's output."

    # Normalize rubric to list
    rubric = _normalize_rubric(cfg.get("rubric"))

    if rubric:
        rubric_lines = "\n".join(
            [f"- {str(r).strip()}" for r in rubric if str(r).strip()]
        )
        if prompt:
            prompt += "\n\n"
        prompt += "Evaluate using this rubric (PASS only if all criteria met):\n"
        prompt += rubric_lines
    elif not tpl and not custom_prompt:
        # For custom metrics without rubric, add generic evaluation guidance
        prompt += "\n\nReturn PASS if the output is satisfactory, FAIL otherwise."

    # Create judge if not provided
    if judge is None:
        # Default model depends on provider
        default_models = {
            "gemini": "gemini-2.5-flash-lite",
            "openai": "gpt-4o-mini",
            "ollama": "llama3.2",
        }
        model = cfg.get("model", default_models.get(provider, "gemini-2.5-flash-lite"))
        temperature = float(cfg.get("temperature", 0.0))
        judge = LLMJudge(
            name=metric_id,
            prompt=prompt,
            model=model,
            temperature=temperature,
            api_key=api_key,
            provider=provider,
            rubric=rubric if rubric else None,
        )

    # Use provided description, or template description, or config description
    final_description = (
        description
        or cfg.get("description")
        or tpl_description
        or "LLM judge subjective score"
    )

    base_metric = judge.as_metric(
        metric_id=metric_id,
        threshold=threshold_f,
        description=str(final_description),
    )

    # Apply confidence wrapper based on method
    confidence_wrappers = {
        "consistency": lambda: _wrap_with_consistency_confidence(base_metric, judge, confidence_samples, threshold_f),
        "logprobs": lambda: _wrap_with_logprobs_confidence(base_metric, judge, threshold_f),
        "deepconf": lambda: _wrap_with_deepconf_confidence(base_metric, judge, threshold_f),
    }
    wrapper = confidence_wrappers.get(confidence_method)
    metric = wrapper() if wrapper else base_metric

    return _apply_unit_types(metric, cfg)
