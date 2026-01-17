from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models import DatasetItem, FunctionCall, Metric, MetricResult, MetricSpec


def _make_result(
    spec: MetricSpec,
    item: DatasetItem,
    call: FunctionCall,
    score: Optional[float],
    passed: Optional[bool],
    details: Optional[Dict[str, Any]] = None,
) -> MetricResult:
    """Factory for MetricResult with common boilerplate."""
    return MetricResult(
        metric_id=spec.id,
        item_id=item.id,
        call_id=call.id,
        score=score,
        passed=passed,
        details=details or {},
    )


# =============================================================================
# Objective Metric Templates
# =============================================================================
#
# TRACE-COMPATIBLE (works without human labels):
#   - Uses: input, output, trace, metadata
#   - Examples: latency_ms, json_valid, tool_call_count
#
# REFERENCE-BASED (needs human_label.reference):
#   - Uses: output vs human-provided reference text
#   - Examples: rouge_l, bleu, token_overlap_f1
#
# SCOPE defines what part of the trace the metric applies to:
#   - "overall": Evaluates the final function output (default)
#   - "llm_call": Evaluates individual LLM API call outputs
#   - "tool_call": Evaluates individual tool call results
#   - "trace": Aggregates over the entire trace (counts, ratios)

OBJECTIVE_REGISTRY = [
    # === TRACE-COMPATIBLE METRICS (no human labels needed) ===
    {
        "id": "latency_ms",
        "type": "objective",
        "description": "Measure execution latency in milliseconds.",
        "config": {},
        "category": "efficiency",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "cost",
        "type": "objective",
        "description": "Total LLM cost from trace events.",
        "config": {},
        "category": "efficiency",
        "scope": "trace",
        "requires_reference": False,
    },
    {
        "id": "json_valid",
        "type": "objective",
        "description": "Checks whether output parses as JSON.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "regex_match",
        "type": "objective",
        "description": "Checks output against a regex pattern.",
        "config": {"pattern": ""},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "token_length",
        "type": "objective",
        "description": "Checks output length (chars) against a maximum.",
        "config": {"max_chars": None},
        "category": "efficiency",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "tool_call_count",
        "type": "objective",
        "description": "Counts tool-related events in the trace.",
        "config": {},
        "category": "robustness",
        "scope": "trace",
        "requires_reference": False,
    },
    {
        "id": "output_nonempty",
        "type": "objective",
        "description": "PASS if output is not empty/None.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "output_length_range",
        "type": "objective",
        "description": "PASS if output length is within specified range.",
        "config": {"min_chars": 0, "max_chars": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "llm_call_count",
        "type": "objective",
        "description": "Count LLM API calls in trace.",
        "config": {"request_kind": ".request"},
        "category": "efficiency",
        "scope": "trace",
        "requires_reference": False,
    },
    {
        "id": "llm_error_rate",
        "type": "objective",
        "description": "Error rate of LLM calls based on trace events.",
        "config": {"request_kind": ".request", "error_kind": ".error"},
        "category": "robustness",
        "scope": "trace",
        "requires_reference": False,
    },
    {
        "id": "tool_success_ratio",
        "type": "objective",
        "description": "Ratio of successful tool calls to total tool calls.",
        "config": {"success_kind": "tool.success", "error_kind": "tool.error"},
        "category": "robustness",
        "scope": "trace",
        "requires_reference": False,
    },
    {
        "id": "tool_error_count",
        "type": "objective",
        "description": "Count of tool errors in trace.",
        "config": {"error_kind": "tool.error"},
        "category": "robustness",
        "scope": "trace",
        "requires_reference": False,
    },
    {
        "id": "csv_valid",
        "type": "objective",
        "description": "Checks whether output parses as CSV.",
        "config": {"dialect": "excel"},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "xml_valid",
        "type": "objective",
        "description": "Checks whether output parses as XML.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "url_count",
        "type": "objective",
        "description": "Counts URLs in the output (proxy for citations).",
        "config": {"pattern": "https?://", "min_count": 1},
        "category": "grounding",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "json_schema_keys",
        "type": "objective",
        "description": "Check JSON output includes required keys.",
        "config": {"required_keys": []},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "json_types_match",
        "type": "objective",
        "description": "Check JSON key types match expected schema.",
        "config": {"schema": {}},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "json_path_present",
        "type": "objective",
        "description": "Check required JSON paths exist (dot notation).",
        "config": {"paths": []},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "regex_capture_count",
        "type": "objective",
        "description": "Count regex matches and enforce a minimum count.",
        "config": {"pattern": "", "min_count": 1},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "pass_at_k",
        "type": "objective",
        "description": "Probability at least one of top-k candidates succeeds.",
        "config": {"k": 5, "candidate_field": "candidates", "success_field": "passed"},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": False,
    },
    # === REFERENCE-BASED METRICS (need human_label.reference) ===
    {
        "id": "bleu",
        "type": "objective",
        "description": "Text similarity using BLEU (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "rouge_l",
        "type": "objective",
        "description": "ROUGE-L similarity (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "rouge_1",
        "type": "objective",
        "description": "ROUGE-1 unigram overlap (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "rouge_2",
        "type": "objective",
        "description": "ROUGE-2 bigram overlap (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "token_overlap_f1",
        "type": "objective",
        "description": "Token overlap F1 (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "jaccard_similarity",
        "type": "objective",
        "description": "Jaccard similarity (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "numeric_mae",
        "type": "objective",
        "description": "Mean absolute error (needs human_label.reference).",
        "config": {"output_field": None},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "numeric_rmse",
        "type": "objective",
        "description": "Root mean squared error (needs human_label.reference).",
        "config": {"output_field": None},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "numeric_rel_error",
        "type": "objective",
        "description": "Relative error (needs human_label.reference).",
        "config": {"output_field": None},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "numeric_within_tolerance",
        "type": "objective",
        "description": "Pass if error within tolerance (needs human_label.reference).",
        "config": {"output_field": None, "tolerance": 0.0},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
]


# =============================================================================
# Handler Functions
# =============================================================================


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _get_reference(item: DatasetItem) -> str:
    """
    Get reference text for comparison metrics.
    Looks in order:
    1. item.human_label.reference (new 4-column model)
    2. item.expected (backwards compat)
    3. Empty string if not found
    """
    # Try human_label.reference first (new model)
    if item.human_label and isinstance(item.human_label, dict):
        ref = item.human_label.get("reference") or item.human_label.get("expected")
        if ref:
            return _as_text(ref)
    # Fall back to expected (backwards compat)
    if item.expected:
        return _as_text(item.expected)
    return ""


def _get_output(call: FunctionCall, item: DatasetItem) -> str:
    """Get the output text, preferring item.output over call.output."""
    # Use item.output if available (from dataset)
    if item.output is not None:
        return _as_text(item.output)
    # Fall back to call.output (from trace)
    return _as_text(call.output)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _overlap_f1(candidate: str, reference: str, n: int = 1) -> float:
    cand = _tokenize(candidate)
    ref = _tokenize(reference)
    if not cand or not ref:
        return 0.0
    cand_counts = _ngram_counts(cand, n)
    ref_counts = _ngram_counts(ref, n)
    if not cand_counts or not ref_counts:
        return 0.0
    overlap = 0
    for ng, c_count in cand_counts.items():
        overlap += min(c_count, ref_counts.get(ng, 0))
    cand_total = sum(cand_counts.values())
    ref_total = sum(ref_counts.values())
    if cand_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / cand_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    # O(n*m) DP with O(min(n,m)) memory.
    if len(a) < len(b):
        shorter, longer = a, b
    else:
        shorter, longer = b, a
    prev = [0] * (len(shorter) + 1)
    for token in longer:
        cur = [0]
        for j, s_tok in enumerate(shorter, start=1):
            if token == s_tok:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(candidate: str, reference: str) -> float:
    cand = _tokenize(candidate)
    ref = _tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    precision = lcs / len(cand) if cand else 0.0
    recall = lcs / len(ref) if ref else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _parse_json_value(value: Any) -> Tuple[Any, Optional[str]]:
    if value is None:
        return None, "empty"
    if isinstance(value, (dict, list)):
        return value, None
    if isinstance(value, str):
        try:
            return json.loads(value), None
        except Exception as exc:
            return None, str(exc)
    try:
        return json.loads(json.dumps(value)), None
    except Exception as exc:
        return None, str(exc)


def _get_by_path(obj: Any, path: str) -> Tuple[bool, Any]:
    """
    Basic JSON path resolver for dot paths with optional [index], e.g.:
      a.b[0].c
    """
    cur = obj
    if path is None:
        return False, None
    for part in str(path).split("."):
        if part == "":
            continue
        name = part
        idx = None
        if "[" in part and part.endswith("]"):
            name, rest = part.split("[", 1)
            try:
                idx = int(rest[:-1])
            except Exception:
                idx = None
        if name:
            if not isinstance(cur, dict) or name not in cur:
                return False, None
            cur = cur[name]
        if idx is not None:
            if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                return False, None
            cur = cur[idx]
    return True, cur


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _extract_number(value: Any, output_field: Optional[str] = None) -> Optional[float]:
    if output_field and isinstance(value, dict) and output_field in value:
        return _coerce_number(value.get(output_field))
    return _coerce_number(value)


def latency_metric(metric_id: str = "latency_ms") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Latency (ms)",
        type="objective",
        description="Execution time in milliseconds.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        return _make_result(
            spec, item, call, call.duration_ms, None, {"duration_ms": call.duration_ms}
        )

    return Metric(spec, handler)


def exact_match_metric(
    metric_id: str = "exact_match",
    expected_field: str = "expected",
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Exact Match",
        type="objective",
        description="Checks if the output matches the expected value exactly.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        expected = getattr(item, expected_field, None)
        actual = call.output
        passed = actual == expected
        score = 1.0 if passed else 0.0
        return _make_result(
            spec, item, call, score, passed, {"expected": expected, "actual": actual}
        )

    return Metric(spec, handler)


def substring_metric(
    metric_id: str = "substring",
    needle: Optional[str] = None,
    expected_field: str = "expected_substring",
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Substring",
        type="objective",
        description="Checks whether the output contains a target substring.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        target = needle or item.metadata.get(expected_field) or ""
        output_text = call.output or ""
        passed = target in output_text
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {"target": target, "output_excerpt": str(output_text)[:200]},
        )

    return Metric(spec, handler)


def cost_metric(metric_id: str = "cost") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Token/Cost",
        type="objective",
        description="Records cost metadata if present on the call.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        cost_info: Any = call.metadata.get("cost")
        score = cost_info.get("total") if isinstance(cost_info, dict) else None
        return _make_result(spec, item, call, score, None, {"cost": cost_info})

    return Metric(spec, handler)


def register_builtin_metrics(registry) -> None:
    """
    Register Evalyn's built-in objective metrics into a registry.

    Notes:
    - Each metric id is registered independently (e.g., `rouge_1`, `rouge_2`, `rouge_l` are separate).
    - Subjective judge metrics are not registered here because they require an LLM judge configuration.
    """
    builtin_metrics = [
        # Efficiency
        latency_metric(),
        cost_metric(),
        token_length_metric(),
        # Correctness
        bleu_metric(),
        pass_at_k_metric(),
        # Structure / formatting
        json_valid_metric(),
        regex_match_metric(),
        csv_valid_metric(),
        xml_valid_metric(),
        json_schema_keys_metric(),
        json_types_match_metric(),
        json_path_present_metric(),
        regex_capture_count_metric(),
        # Text overlap / similarity
        rouge_l_metric(),
        rouge_1_metric(),
        rouge_2_metric(),
        token_overlap_f1_metric(),
        jaccard_similarity_metric(),
        # Numeric
        numeric_mae_metric(),
        numeric_rmse_metric(),
        numeric_rel_error_metric(),
        numeric_within_tolerance_metric(),
        # Output quality
        output_nonempty_metric(),
        output_length_range_metric(),
        # Trace-based robustness
        tool_call_count_metric(),
        llm_call_count_metric(),
        llm_error_rate_metric(),
        tool_success_ratio_metric(),
        tool_error_count_metric(),
        # Grounding proxies
        url_count_metric(),
    ]

    for metric in builtin_metrics:
        registry.register(metric)


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _simple_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    # Minimal BLEU without brevity penalty to stay dependency-light.
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        cand_ngrams = _ngrams(cand_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        if not cand_ngrams or not ref_ngrams:
            precisions.append(0.0)
            continue
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        match = 0
        for ng in cand_ngrams:
            if ref_counts.get(ng, 0) > 0:
                match += 1
                ref_counts[ng] -= 1
        precisions.append(match / len(cand_ngrams))

    # geometric mean of precisions, guard zeros
    precisions = [p if p > 0 else 1e-9 for p in precisions]
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    # brevity penalty
    bp = (
        1.0
        if len(cand_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    )
    return bp * geo_mean


def bleu_metric(metric_id: str = "bleu") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="BLEU",
        type="objective",
        description="Simple BLEU score between output and expected text.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = item.expected or ""
        candidate = call.output or ""
        score = _simple_bleu(str(candidate), str(reference))
        return _make_result(
            spec,
            item,
            call,
            score,
            None,
            {"candidate": str(candidate)[:200], "reference": str(reference)[:200]},
        )

    return Metric(spec, handler)


def pass_at_k_metric(
    metric_id: str = "pass_at_k",
    k: int = 5,
    candidate_field: str = "candidates",
    success_field: str = "passed",
) -> Metric:
    """
    Computes pass@k over a set of candidate outputs. Expects call.output to be a list of candidate dicts
    or a dict containing the candidate list under `candidate_field`. Each candidate should include a boolean `success_field`.
    """
    spec = MetricSpec(
        id=metric_id,
        name=f"Pass@{k}",
        type="objective",
        description=f"Probability at least one of top-{k} candidates succeeds.",
        config={
            "k": k,
            "candidate_field": candidate_field,
            "success_field": success_field,
        },
    )

    def _extract_candidates(output: Any) -> Sequence[dict]:
        if isinstance(output, dict) and candidate_field in output:
            return output.get(candidate_field, []) or []
        if isinstance(output, list):
            return output
        return []

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        candidates = _extract_candidates(call.output)
        successes = [
            c.get(success_field, False) for c in candidates if isinstance(c, dict)
        ]
        n = len(successes)
        if n == 0:
            return _make_result(spec, item, call, 0.0, False, {"candidates": 0, "k": k})

        k_eff = min(k, n)
        # pass@k estimate: 1 - ((n - c choose k) / (n choose k)) where c = successes count
        c = sum(1 for s in successes if s)
        if c == 0:
            score = 0.0
        else:
            from math import comb

            score = 1.0 - (comb(n - c, k_eff) / comb(n, k_eff))

        return _make_result(
            spec, item, call, score, None, {"candidates": n, "successes": c, "k": k_eff}
        )

    return Metric(spec, handler)


def json_valid_metric(metric_id: str = "json_valid") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="JSON Valid",
        type="objective",
        description="Checks whether output parses as JSON.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        output_text = call.output or ""
        try:
            json.loads(
                output_text if isinstance(output_text, str) else json.dumps(output_text)
            )
            passed = True
        except Exception:
            passed = False
        score = 1.0 if passed else 0.0
        return _make_result(
            spec, item, call, score, passed, {"output_excerpt": str(output_text)[:200]}
        )

    return Metric(spec, handler)


def regex_match_metric(
    metric_id: str = "regex_match", pattern: Optional[str] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Regex Match",
        type="objective",
        description="Checks if output matches a regex pattern.",
        config={"pattern": pattern},
    )
    compiled = re.compile(pattern) if pattern else None

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        patt = compiled or re.compile(item.metadata.get("regex_pattern", "") or "")
        text = str(call.output or "")
        passed = bool(patt.search(text)) if patt.pattern else False
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {"pattern": patt.pattern, "output_excerpt": text[:200]},
        )

    return Metric(spec, handler)


def token_length_metric(
    metric_id: str = "token_length",
    max_chars: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Length Check",
        type="objective",
        description="Checks output length (chars) against a maximum.",
        config={"max_chars": max_chars},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = call.output or ""
        length = len(str(text))
        limit = max_chars or item.metadata.get("max_chars")
        passed = True if not limit else length <= limit
        score = 1.0 if passed else 0.0
        return _make_result(
            spec, item, call, score, passed, {"length": length, "max_chars": limit}
        )

    return Metric(spec, handler)


def tool_call_count_metric(metric_id: str = "tool_call_count") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Tool Call Count",
        type="objective",
        description="Counts tool-related events in the trace.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        events = call.trace or []
        count = sum(1 for ev in events if "tool" in ev.kind.lower())
        return _make_result(
            spec, item, call, float(count), None, {"tool_events": count}
        )

    return Metric(spec, handler)


def rouge_l_metric(metric_id: str = "rouge_l") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="ROUGE-L",
        type="objective",
        description="ROUGE-L F1 (LCS) similarity between output and reference (from human_label.reference).",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)
        if not reference:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {"error": "No reference text (set human_label.reference)"},
            )
        score = _rouge_l_f1(candidate, reference)
        return _make_result(
            spec,
            item,
            call,
            score,
            None,
            {
                "candidate_excerpt": candidate[:200],
                "reference_excerpt": reference[:200],
            },
        )

    return Metric(spec, handler)


def rouge_1_metric(metric_id: str = "rouge_1") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="ROUGE-1",
        type="objective",
        description="ROUGE-1 F1 (unigram overlap) between output and reference.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)
        if not reference:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {"error": "No reference text (set human_label.reference)"},
            )
        score = _overlap_f1(candidate, reference, n=1)
        return _make_result(
            spec,
            item,
            call,
            score,
            None,
            {
                "candidate_excerpt": candidate[:200],
                "reference_excerpt": reference[:200],
            },
        )

    return Metric(spec, handler)


def rouge_2_metric(metric_id: str = "rouge_2") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="ROUGE-2",
        type="objective",
        description="ROUGE-2 F1 (bigram overlap) between output and reference.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)
        if not reference:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {"error": "No reference text (set human_label.reference)"},
            )
        score = _overlap_f1(candidate, reference, n=2)
        return _make_result(
            spec,
            item,
            call,
            score,
            None,
            {
                "candidate_excerpt": candidate[:200],
                "reference_excerpt": reference[:200],
            },
        )

    return Metric(spec, handler)


def token_overlap_f1_metric(metric_id: str = "token_overlap_f1") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Token Overlap F1",
        type="objective",
        description="Token overlap F1 (unigram) between output and reference.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)
        score = _overlap_f1(candidate, reference, n=1)
        return _make_result(
            spec,
            item,
            call,
            score,
            None,
            {
                "candidate_excerpt": candidate[:200],
                "reference_excerpt": reference[:200],
            },
        )

    return Metric(spec, handler)


def jaccard_similarity_metric(metric_id: str = "jaccard_similarity") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Jaccard Similarity",
        type="objective",
        description="Jaccard similarity between token sets of output and reference.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)
        if not reference:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {"error": "No reference text (set human_label.reference)"},
            )
        cand_set = set(_tokenize(candidate))
        ref_set = set(_tokenize(reference))
        if not cand_set or not ref_set:
            score = 0.0
        else:
            score = len(cand_set & ref_set) / len(cand_set | ref_set)
        return _make_result(
            spec,
            item,
            call,
            score,
            None,
            {
                "candidate_excerpt": candidate[:200],
                "reference_excerpt": reference[:200],
            },
        )

    return Metric(spec, handler)


def numeric_mae_metric(
    metric_id: str = "numeric_mae",
    expected_field: str = "expected",
    output_field: Optional[str] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Numeric MAE",
        type="objective",
        description="Mean absolute error for numeric outputs.",
        config={"expected_field": expected_field, "output_field": output_field},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        expected = _extract_number(item.expected, None)
        predicted = _extract_number(call.output, output_field)
        if expected is None or predicted is None:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {
                    "expected": item.expected,
                    "predicted": call.output,
                    "error": "not_numeric",
                },
            )
        err = abs(predicted - expected)
        return _make_result(
            spec,
            item,
            call,
            err,
            None,
            {"expected": expected, "predicted": predicted, "mae": err},
        )

    return Metric(spec, handler)


def numeric_rmse_metric(
    metric_id: str = "numeric_rmse",
    expected_field: str = "expected",
    output_field: Optional[str] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Numeric RMSE",
        type="objective",
        description="Root mean squared error for numeric outputs.",
        config={"expected_field": expected_field, "output_field": output_field},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        expected = _extract_number(item.expected, None)
        predicted = _extract_number(call.output, output_field)
        if expected is None or predicted is None:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {
                    "expected": item.expected,
                    "predicted": call.output,
                    "error": "not_numeric",
                },
            )
        err = predicted - expected
        rmse = math.sqrt(err * err)
        return _make_result(
            spec,
            item,
            call,
            rmse,
            None,
            {"expected": expected, "predicted": predicted, "rmse": rmse},
        )

    return Metric(spec, handler)


def numeric_rel_error_metric(
    metric_id: str = "numeric_rel_error",
    expected_field: str = "expected",
    output_field: Optional[str] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Numeric Relative Error",
        type="objective",
        description="Relative error |pred-expected|/|expected| for numeric outputs.",
        config={"expected_field": expected_field, "output_field": output_field},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        expected = _extract_number(item.expected, None)
        predicted = _extract_number(call.output, output_field)
        if expected is None or predicted is None:
            return _make_result(
                spec,
                item,
                call,
                None,
                None,
                {
                    "expected": item.expected,
                    "predicted": call.output,
                    "error": "not_numeric",
                },
            )
        denom = abs(expected) if abs(expected) > 1e-12 else 1.0
        rel = abs(predicted - expected) / denom
        return _make_result(
            spec,
            item,
            call,
            rel,
            None,
            {"expected": expected, "predicted": predicted, "rel_error": rel},
        )

    return Metric(spec, handler)


def numeric_within_tolerance_metric(
    metric_id: str = "numeric_within_tolerance",
    expected_field: str = "expected",
    output_field: Optional[str] = None,
    tolerance: float = 0.0,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Numeric Within Tolerance",
        type="objective",
        description="Passes if numeric prediction is within tolerance of expected.",
        config={
            "expected_field": expected_field,
            "output_field": output_field,
            "tolerance": tolerance,
        },
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        expected = _extract_number(item.expected, None)
        predicted = _extract_number(call.output, output_field)
        tol = (
            tolerance
            if tolerance is not None
            else float(item.metadata.get("tolerance") or 0.0)
        )
        if expected is None or predicted is None:
            return _make_result(
                spec,
                item,
                call,
                0.0,
                False,
                {
                    "expected": item.expected,
                    "predicted": call.output,
                    "error": "not_numeric",
                },
            )
        err = abs(predicted - expected)
        passed = err <= tol
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {
                "expected": expected,
                "predicted": predicted,
                "tolerance": tol,
                "abs_error": err,
            },
        )

    return Metric(spec, handler)


def json_schema_keys_metric(
    metric_id: str = "json_schema_keys", required_keys: Optional[List[str]] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="JSON Required Keys",
        type="objective",
        description="Checks JSON output includes required keys.",
        config={"required_keys": required_keys or []},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        required = required_keys or item.metadata.get("required_keys") or []
        value, err = _parse_json_value(call.output)
        if err or not isinstance(value, dict):
            return _make_result(
                spec,
                item,
                call,
                0.0,
                False,
                {"error": err or "not_object", "required_keys": required},
            )
        missing = [k for k in required if k not in value]
        passed = len(missing) == 0
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {"missing": missing, "required_keys": required},
        )

    return Metric(spec, handler)


def json_types_match_metric(
    metric_id: str = "json_types_match", schema: Optional[dict] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="JSON Types Match",
        type="objective",
        description="Checks JSON key types match expected schema.",
        config={"schema": schema or {}},
    )

    def _type_ok(value: Any, type_name: str) -> bool:
        t = type_name.lower()
        if t in {"str", "string", "text"}:
            return isinstance(value, str)
        if t in {"int", "integer"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if t in {"float", "number"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t in {"bool", "boolean"}:
            return isinstance(value, bool)
        if t in {"object", "dict"}:
            return isinstance(value, dict)
        if t in {"array", "list"}:
            return isinstance(value, list)
        return True

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        effective_schema = schema or item.metadata.get("schema") or {}
        value, err = _parse_json_value(call.output)
        if err or not isinstance(value, dict):
            return _make_result(
                spec,
                item,
                call,
                0.0,
                False,
                {"error": err or "not_object", "schema": effective_schema},
            )
        mismatches = {}
        for key, type_name in (effective_schema or {}).items():
            if key not in value:
                mismatches[key] = {"expected_type": type_name, "actual": "missing"}
                continue
            if isinstance(type_name, str) and not _type_ok(value.get(key), type_name):
                mismatches[key] = {
                    "expected_type": type_name,
                    "actual_type": type(value.get(key)).__name__,
                }
        passed = not mismatches
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {"mismatches": mismatches, "schema": effective_schema},
        )

    return Metric(spec, handler)


def json_path_present_metric(
    metric_id: str = "json_path_present", paths: Optional[List[str]] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="JSON Path Present",
        type="objective",
        description="Checks required JSON paths exist (dot notation).",
        config={"paths": paths or []},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        effective_paths = paths or item.metadata.get("paths") or []
        value, err = _parse_json_value(call.output)
        if err:
            return _make_result(
                spec, item, call, 0.0, False, {"error": err, "paths": effective_paths}
            )
        missing = []
        for p in effective_paths:
            ok, _ = _get_by_path(value, p)
            if not ok:
                missing.append(p)
        passed = len(missing) == 0
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {"missing": missing, "paths": effective_paths},
        )

    return Metric(spec, handler)


def regex_capture_count_metric(
    metric_id: str = "regex_capture_count",
    pattern: Optional[str] = None,
    min_count: int = 1,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Regex Capture Count",
        type="objective",
        description="Counts regex matches and enforces a minimum count.",
        config={"pattern": pattern or "", "min_count": min_count},
    )
    compiled = re.compile(pattern) if pattern else None

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        patt = compiled or re.compile(item.metadata.get("regex_pattern", "") or "")
        text = _as_text(call.output)
        count = len(patt.findall(text)) if patt.pattern else 0
        required = item.metadata.get("min_count", min_count)
        passed = count >= int(required or 0)
        return _make_result(
            spec,
            item,
            call,
            float(count),
            passed,
            {"pattern": patt.pattern, "count": count, "min_count": required},
        )

    return Metric(spec, handler)


def csv_valid_metric(metric_id: str = "csv_valid", dialect: str = "excel") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="CSV Valid",
        type="objective",
        description="Checks whether output parses as CSV.",
        config={"dialect": dialect},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        import csv
        import io

        text = _as_text(call.output)
        try:
            reader = csv.reader(io.StringIO(text), dialect=dialect)
            rows = list(reader)
            if not rows:
                return _make_result(spec, item, call, 0.0, False, {"rows": 0})
            widths = {len(r) for r in rows}
            passed = len(widths) <= 1
            score = 1.0 if passed else 0.0
            return _make_result(
                spec,
                item,
                call,
                score,
                passed,
                {"rows": len(rows), "widths": sorted(widths)},
            )
        except Exception as exc:
            return _make_result(spec, item, call, 0.0, False, {"error": str(exc)})

    return Metric(spec, handler)


def xml_valid_metric(metric_id: str = "xml_valid") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="XML Valid",
        type="objective",
        description="Checks whether output parses as XML.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        import xml.etree.ElementTree as ET

        text = _as_text(call.output)
        try:
            ET.fromstring(text)
            return _make_result(spec, item, call, 1.0, True, {})
        except Exception as exc:
            return _make_result(spec, item, call, 0.0, False, {"error": str(exc)})

    return Metric(spec, handler)


def output_nonempty_metric(metric_id: str = "output_nonempty") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Output Non-empty",
        type="objective",
        description="Checks that output is present and non-empty.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output).strip()
        passed = len(text) > 0
        score = 1.0 if passed else 0.0
        return _make_result(spec, item, call, score, passed, {"length": len(text)})

    return Metric(spec, handler)


def output_length_range_metric(
    metric_id: str = "output_length_range",
    min_chars: int = 0,
    max_chars: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Output Length Range",
        type="objective",
        description="Checks output length is within [min,max].",
        config={"min_chars": min_chars, "max_chars": max_chars},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        length = len(text)
        min_v = item.metadata.get("min_chars", min_chars)
        max_v = item.metadata.get("max_chars", max_chars)
        passed = length >= int(min_v or 0) and (max_v is None or length <= int(max_v))
        score = 1.0 if passed else 0.0
        return _make_result(
            spec,
            item,
            call,
            score,
            passed,
            {"length": length, "min_chars": min_v, "max_chars": max_v},
        )

    return Metric(spec, handler)


def llm_call_count_metric(
    metric_id: str = "llm_call_count", request_kind: str = ".request"
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="LLM Call Count",
        type="objective",
        description="Counts LLM request events in the trace.",
        config={"request_kind": request_kind},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        rk = str(item.metadata.get("request_kind") or request_kind).lower()
        count = sum(1 for ev in (call.trace or []) if rk in ev.kind.lower())
        return _make_result(
            spec, item, call, float(count), None, {"request_kind": rk, "count": count}
        )

    return Metric(spec, handler)


def llm_error_rate_metric(
    metric_id: str = "llm_error_rate",
    request_kind: str = ".request",
    error_kind: str = ".error",
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="LLM Error Rate",
        type="objective",
        description="Error rate of LLM calls based on trace events.",
        config={"request_kind": request_kind, "error_kind": error_kind},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        rk = str(item.metadata.get("request_kind") or request_kind).lower()
        ek = str(item.metadata.get("error_kind") or error_kind).lower()
        req = sum(1 for ev in (call.trace or []) if rk in ev.kind.lower())
        err = sum(1 for ev in (call.trace or []) if ek in ev.kind.lower())
        rate = (err / req) if req else None
        return _make_result(
            spec,
            item,
            call,
            rate,
            None,
            {"requests": req, "errors": err, "request_kind": rk, "error_kind": ek},
        )

    return Metric(spec, handler)


def tool_success_ratio_metric(
    metric_id: str = "tool_success_ratio",
    success_kind: str = "tool.success",
    error_kind: str = "tool.error",
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Tool Success Ratio",
        type="objective",
        description="Ratio of tool success events to total tool events.",
        config={"success_kind": success_kind, "error_kind": error_kind},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        sk = str(item.metadata.get("success_kind") or success_kind).lower()
        ek = str(item.metadata.get("error_kind") or error_kind).lower()
        successes = sum(1 for ev in (call.trace or []) if sk in ev.kind.lower())
        errors = sum(1 for ev in (call.trace or []) if ek in ev.kind.lower())
        total = successes + errors
        ratio = (successes / total) if total else None
        return _make_result(
            spec,
            item,
            call,
            ratio,
            None,
            {
                "successes": successes,
                "errors": errors,
                "total": total,
                "success_kind": sk,
                "error_kind": ek,
            },
        )

    return Metric(spec, handler)


def tool_error_count_metric(
    metric_id: str = "tool_error_count", error_kind: str = "tool.error"
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Tool Error Count",
        type="objective",
        description="Counts tool error events in the trace.",
        config={"error_kind": error_kind},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        ek = str(item.metadata.get("error_kind") or error_kind).lower()
        count = sum(1 for ev in (call.trace or []) if ek in ev.kind.lower())
        return _make_result(
            spec, item, call, float(count), None, {"error_kind": ek, "count": count}
        )

    return Metric(spec, handler)


def url_count_metric(
    metric_id: str = "url_count", pattern: str = r"https?://", min_count: int = 1
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="URL Count",
        type="objective",
        description="Counts URLs in the output.",
        config={"pattern": pattern, "min_count": min_count},
    )
    compiled = re.compile(pattern)

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        patt = item.metadata.get("pattern") or pattern
        compiled_local = compiled if patt == pattern else re.compile(str(patt))
        text = _as_text(call.output)
        count = len(compiled_local.findall(text))
        min_v = int(item.metadata.get("min_count") or min_count)
        passed = count >= min_v
        return _make_result(
            spec,
            item,
            call,
            float(count),
            passed,
            {"count": count, "min_count": min_v, "pattern": str(patt)},
        )

    return Metric(spec, handler)
