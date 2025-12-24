from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .objective import (
    bleu_metric,
    cost_metric,
    csv_valid_metric,
    jaccard_similarity_metric,
    json_path_present_metric,
    json_schema_keys_metric,
    json_types_match_metric,
    json_valid_metric,
    latency_metric,
    llm_call_count_metric,
    llm_error_rate_metric,
    numeric_mae_metric,
    numeric_rel_error_metric,
    numeric_rmse_metric,
    numeric_within_tolerance_metric,
    output_length_range_metric,
    output_nonempty_metric,
    pass_at_k_metric,
    regex_capture_count_metric,
    regex_match_metric,
    rouge_1_metric,
    rouge_2_metric,
    rouge_l_metric,
    token_length_metric,
    token_overlap_f1_metric,
    tool_call_count_metric,
    tool_error_count_metric,
    tool_success_ratio_metric,
    url_count_metric,
    xml_valid_metric,
)
from .registry import Metric
from .subjective import subjective_metric
from .templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from .judges import GeminiJudge, LLMJudge


def _tpl_by_id(templates: List[dict]) -> Dict[str, dict]:
    return {t["id"]: t for t in templates}


_OBJECTIVE_TPL = _tpl_by_id(OBJECTIVE_TEMPLATES)
_SUBJECTIVE_TPL = _tpl_by_id(SUBJECTIVE_TEMPLATES)


def list_template_ids() -> List[str]:
    """Return all known template ids (objective + subjective)."""
    return sorted(list(_OBJECTIVE_TPL.keys()) + list(_SUBJECTIVE_TPL.keys()))


def build_objective_metric(metric_id: str, config: Optional[Dict[str, Any]] = None) -> Metric:
    cfg = config or {}
    builders: Dict[str, Callable[[Dict[str, Any]], Metric]] = {
        "latency_ms": lambda c: latency_metric(),
        "cost": lambda c: cost_metric(),
        "bleu": lambda c: bleu_metric(),
        "pass_at_k": lambda c: pass_at_k_metric(
            k=int(c.get("k", 5)),
            candidate_field=str(c.get("candidate_field", "candidates")),
            success_field=str(c.get("success_field", "passed")),
        ),
        "json_valid": lambda c: json_valid_metric(),
        "regex_match": lambda c: regex_match_metric(pattern=c.get("pattern")),
        "token_length": lambda c: token_length_metric(max_chars=c.get("max_chars")),
        "tool_call_count": lambda c: tool_call_count_metric(),
        "rouge_l": lambda c: rouge_l_metric(),
        "rouge_1": lambda c: rouge_1_metric(),
        "rouge_2": lambda c: rouge_2_metric(),
        "token_overlap_f1": lambda c: token_overlap_f1_metric(),
        "jaccard_similarity": lambda c: jaccard_similarity_metric(),
        "numeric_mae": lambda c: numeric_mae_metric(output_field=c.get("output_field")),
        "numeric_rmse": lambda c: numeric_rmse_metric(output_field=c.get("output_field")),
        "numeric_rel_error": lambda c: numeric_rel_error_metric(output_field=c.get("output_field")),
        "numeric_within_tolerance": lambda c: numeric_within_tolerance_metric(
            output_field=c.get("output_field"),
            tolerance=float(c.get("tolerance", 0.0)),
        ),
        "json_schema_keys": lambda c: json_schema_keys_metric(required_keys=c.get("required_keys")),
        "json_types_match": lambda c: json_types_match_metric(schema=c.get("schema")),
        "json_path_present": lambda c: json_path_present_metric(paths=c.get("paths")),
        "regex_capture_count": lambda c: regex_capture_count_metric(
            pattern=c.get("pattern"),
            min_count=int(c.get("min_count", 1)),
        ),
        "csv_valid": lambda c: csv_valid_metric(dialect=str(c.get("dialect", "excel"))),
        "xml_valid": lambda c: xml_valid_metric(),
        "output_nonempty": lambda c: output_nonempty_metric(),
        "output_length_range": lambda c: output_length_range_metric(
            min_chars=int(c.get("min_chars", 0)),
            max_chars=c.get("max_chars"),
        ),
        "llm_call_count": lambda c: llm_call_count_metric(request_kind=str(c.get("request_kind", ".request"))),
        "llm_error_rate": lambda c: llm_error_rate_metric(
            request_kind=str(c.get("request_kind", ".request")),
            error_kind=str(c.get("error_kind", ".error")),
        ),
        "tool_success_ratio": lambda c: tool_success_ratio_metric(
            success_kind=str(c.get("success_kind", "tool.success")),
            error_kind=str(c.get("error_kind", "tool.error")),
        ),
        "tool_error_count": lambda c: tool_error_count_metric(error_kind=str(c.get("error_kind", "tool.error"))),
        "url_count": lambda c: url_count_metric(
            pattern=str(c.get("pattern", r"https?://")),
            min_count=int(c.get("min_count", 1)),
        ),
    }

    if metric_id not in builders:
        raise KeyError(f"Unknown objective metric id: {metric_id}")
    return builders[metric_id](cfg)


def build_subjective_metric(
    metric_id: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    judge: Optional[LLMJudge] = None,
    description: Optional[str] = None,
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
    - model: Judge model name (default: gemini-2.0-flash-exp)
    - temperature: Judge temperature (default: 0.0)
    - description: Metric description (for custom metrics)
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

    # Parse threshold
    threshold = cfg.get("threshold", 0.7)
    try:
        threshold_f = float(threshold)
    except Exception:
        threshold_f = 0.7

    # Build prompt from config or template
    custom_prompt = cfg.get("prompt")
    if custom_prompt:
        prompt = str(custom_prompt).strip()
    elif tpl_prompt:
        prompt = str(tpl_prompt).strip()
    else:
        # Generate default prompt for custom metrics
        prompt = f"You are an expert evaluator for '{metric_id}'. Evaluate the agent's output."

    # Add rubric if provided
    rubric = cfg.get("rubric") or []
    if isinstance(rubric, str):
        rubric = [rubric]
    if not isinstance(rubric, list):
        rubric = []

    if rubric:
        rubric_lines = "\n".join([f"- {str(r).strip()}" for r in rubric if str(r).strip()])
        if prompt:
            prompt += "\n\n"
        prompt += "Evaluate using this rubric (PASS only if all criteria met):\n"
        prompt += rubric_lines
    elif not tpl and not custom_prompt:
        # For custom metrics without rubric, add generic evaluation guidance
        prompt += "\n\nReturn PASS if the output is satisfactory, FAIL otherwise."

    # Create judge if not provided
    if judge is None:
        model = cfg.get("model", "gemini-2.0-flash-exp")
        temperature = float(cfg.get("temperature", 0.0))
        judge = GeminiJudge(
            name=metric_id,
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

    # Use provided description, or template description, or config description
    final_description = (
        description
        or cfg.get("description")
        or tpl_description
        or "LLM judge subjective score"
    )

    return subjective_metric(
        metric_id=metric_id,
        judge=judge,
        description=str(final_description),
        success_threshold=threshold_f,
        config=cfg,
    )
