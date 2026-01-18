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
    # === NEW METRICS: Code-specific ===
    {
        "id": "syntax_valid",
        "type": "objective",
        "description": "Checks if code output has valid Python syntax.",
        "config": {"language": "python"},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "code_complexity",
        "type": "objective",
        "description": "Estimates code complexity (line count, nesting depth).",
        "config": {"max_lines": None, "max_depth": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    # === NEW METRICS: Readability ===
    {
        "id": "flesch_kincaid",
        "type": "objective",
        "description": "Flesch-Kincaid readability grade level.",
        "config": {"max_grade": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "sentence_count",
        "type": "objective",
        "description": "Counts sentences in output.",
        "config": {"min_count": None, "max_count": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "avg_sentence_length",
        "type": "objective",
        "description": "Average sentence length in words.",
        "config": {"max_avg": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    # === NEW METRICS: Diversity ===
    {
        "id": "distinct_1",
        "type": "objective",
        "description": "Distinct-1: ratio of unique unigrams to total unigrams.",
        "config": {"min_ratio": None},
        "category": "diversity",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "distinct_2",
        "type": "objective",
        "description": "Distinct-2: ratio of unique bigrams to total bigrams.",
        "config": {"min_ratio": None},
        "category": "diversity",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "vocabulary_richness",
        "type": "objective",
        "description": "Type-token ratio (unique words / total words).",
        "config": {"min_ratio": None},
        "category": "diversity",
        "scope": "overall",
        "requires_reference": False,
    },
    # === NEW METRICS: Compression/Ratio ===
    {
        "id": "compression_ratio",
        "type": "objective",
        "description": "Ratio of output length to input length.",
        "config": {"min_ratio": None, "max_ratio": None},
        "category": "efficiency",
        "scope": "overall",
        "requires_reference": False,
    },
    # === NEW METRICS: Citation/Grounding ===
    {
        "id": "citation_count",
        "type": "objective",
        "description": "Counts citation-like patterns (e.g., [1], (Author, Year)).",
        "config": {"min_count": 0},
        "category": "grounding",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "markdown_link_count",
        "type": "objective",
        "description": "Counts markdown-style links [text](url).",
        "config": {"min_count": 0},
        "category": "grounding",
        "scope": "overall",
        "requires_reference": False,
    },
    # === NEW METRICS: Semantic Similarity (embedding-based) ===
    {
        "id": "levenshtein_similarity",
        "type": "objective",
        "description": "Levenshtein edit distance similarity (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    {
        "id": "cosine_word_overlap",
        "type": "objective",
        "description": "Cosine similarity of word frequency vectors (needs human_label.reference).",
        "config": {},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": True,
    },
    # === MORE METRICS: Format Validation ===
    {
        "id": "yaml_valid",
        "type": "objective",
        "description": "Checks whether output parses as YAML.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "markdown_structure",
        "type": "objective",
        "description": "Validates markdown structure (headings, lists, code blocks).",
        "config": {"require_heading": False},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "html_valid",
        "type": "objective",
        "description": "Checks whether output contains valid HTML tags.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "sql_valid",
        "type": "objective",
        "description": "Basic SQL syntax validation.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Structure Detection ===
    {
        "id": "bullet_count",
        "type": "objective",
        "description": "Counts bullet points in output.",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "heading_count",
        "type": "objective",
        "description": "Counts markdown headings (# style).",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "code_block_count",
        "type": "objective",
        "description": "Counts fenced code blocks (```).",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "table_count",
        "type": "objective",
        "description": "Counts markdown tables.",
        "config": {"min_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "paragraph_count",
        "type": "objective",
        "description": "Counts paragraphs (text blocks separated by blank lines).",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "word_count",
        "type": "objective",
        "description": "Counts words in output.",
        "config": {"min_count": None, "max_count": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Repetition ===
    {
        "id": "repetition_ratio",
        "type": "objective",
        "description": "Detects repeated n-grams (low = less repetitive).",
        "config": {"n": 3, "max_ratio": None},
        "category": "diversity",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "duplicate_line_ratio",
        "type": "objective",
        "description": "Ratio of duplicate lines to total lines.",
        "config": {"max_ratio": None},
        "category": "diversity",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Uncertainty/Confidence ===
    {
        "id": "hedging_count",
        "type": "objective",
        "description": "Counts hedging phrases (maybe, perhaps, might, etc.).",
        "config": {"max_count": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "question_count",
        "type": "objective",
        "description": "Counts questions in output.",
        "config": {"min_count": None, "max_count": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "confidence_markers",
        "type": "objective",
        "description": "Counts confidence markers (certainly, definitely, clearly).",
        "config": {},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Code Quality ===
    {
        "id": "comment_ratio",
        "type": "objective",
        "description": "Ratio of comment lines to code lines.",
        "config": {"min_ratio": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "function_count",
        "type": "objective",
        "description": "Counts function/method definitions in code.",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "import_count",
        "type": "objective",
        "description": "Counts import statements in code.",
        "config": {"max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Character/Format ===
    {
        "id": "ascii_ratio",
        "type": "objective",
        "description": "Ratio of ASCII characters to total characters.",
        "config": {"min_ratio": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "uppercase_ratio",
        "type": "objective",
        "description": "Ratio of uppercase letters to total letters.",
        "config": {"max_ratio": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "numeric_density",
        "type": "objective",
        "description": "Ratio of numeric characters to total characters.",
        "config": {},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "whitespace_ratio",
        "type": "objective",
        "description": "Ratio of whitespace to total characters.",
        "config": {"max_ratio": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Exact Match Variants ===
    {
        "id": "prefix_match",
        "type": "objective",
        "description": "Checks if output starts with expected prefix.",
        "config": {"prefix": None},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "suffix_match",
        "type": "objective",
        "description": "Checks if output ends with expected suffix.",
        "config": {"suffix": None},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "contains_all",
        "type": "objective",
        "description": "Checks if output contains all required substrings.",
        "config": {"substrings": []},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "contains_none",
        "type": "objective",
        "description": "Checks if output contains none of the forbidden substrings.",
        "config": {"forbidden": []},
        "category": "correctness",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: List/Enumeration ===
    {
        "id": "numbered_list_count",
        "type": "objective",
        "description": "Counts numbered list items (1., 2., etc.).",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "list_item_count",
        "type": "objective",
        "description": "Counts total list items (bullets + numbers).",
        "config": {"min_count": None, "max_count": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
    },
    # === MORE METRICS: Response Quality ===
    {
        "id": "emoji_count",
        "type": "objective",
        "description": "Counts emoji characters in output.",
        "config": {"max_count": None},
        "category": "style",
        "scope": "overall",
        "requires_reference": False,
    },
    {
        "id": "link_density",
        "type": "objective",
        "description": "Ratio of link text to total text.",
        "config": {"max_ratio": None},
        "category": "structure",
        "scope": "overall",
        "requires_reference": False,
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


def _extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown fenced code blocks, or return text as-is."""
    code_match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return code_match.group(1) if code_match else text


def _check_min_max_bounds(
    count: int,
    item: DatasetItem,
    min_key: str,
    max_key: str,
    default_min: Optional[int],
    default_max: Optional[int],
) -> Tuple[bool, Optional[int], Optional[int]]:
    """Check if count is within min/max bounds from item metadata or defaults."""
    min_v = item.metadata.get(min_key, default_min)
    max_v = item.metadata.get(max_key, default_max)
    passed = True
    if min_v is not None and count < int(min_v):
        passed = False
    if max_v is not None and count > int(max_v):
        passed = False
    return passed, min_v, max_v


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
        compression_ratio_metric(),
        # Correctness
        bleu_metric(),
        pass_at_k_metric(),
        levenshtein_similarity_metric(),
        cosine_word_overlap_metric(),
        # Structure / formatting
        json_valid_metric(),
        regex_match_metric(),
        csv_valid_metric(),
        xml_valid_metric(),
        json_schema_keys_metric(),
        json_types_match_metric(),
        json_path_present_metric(),
        regex_capture_count_metric(),
        syntax_valid_metric(),
        code_complexity_metric(),
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
        # Readability
        flesch_kincaid_metric(),
        sentence_count_metric(),
        avg_sentence_length_metric(),
        # Diversity
        distinct_1_metric(),
        distinct_2_metric(),
        vocabulary_richness_metric(),
        # Trace-based robustness
        tool_call_count_metric(),
        llm_call_count_metric(),
        llm_error_rate_metric(),
        tool_success_ratio_metric(),
        tool_error_count_metric(),
        # Grounding proxies
        url_count_metric(),
        citation_count_metric(),
        markdown_link_count_metric(),
        # Format validation
        yaml_valid_metric(),
        markdown_structure_metric(),
        html_valid_metric(),
        sql_valid_metric(),
        # Structure detection
        bullet_count_metric(),
        heading_count_metric(),
        code_block_count_metric(),
        table_count_metric(),
        paragraph_count_metric(),
        word_count_metric(),
        # Repetition
        repetition_ratio_metric(),
        duplicate_line_ratio_metric(),
        # Uncertainty/Confidence
        hedging_count_metric(),
        question_count_metric(),
        confidence_markers_metric(),
        # Code quality
        comment_ratio_metric(),
        function_count_metric(),
        import_count_metric(),
        # Character/Format
        ascii_ratio_metric(),
        uppercase_ratio_metric(),
        numeric_density_metric(),
        whitespace_ratio_metric(),
        # Match variants
        prefix_match_metric(),
        suffix_match_metric(),
        contains_all_metric(),
        contains_none_metric(),
        # List/Enumeration
        numbered_list_count_metric(),
        list_item_count_metric(),
        # Response quality
        emoji_count_metric(),
        link_density_metric(),
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


# =============================================================================
# NEW METRICS: Code-specific
# =============================================================================


def syntax_valid_metric(
    metric_id: str = "syntax_valid", language: str = "python"
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Syntax Valid",
        type="objective",
        description="Checks if code output has valid syntax.",
        config={"language": language},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        lang = item.metadata.get("language", language)
        code = _extract_code_from_markdown(text)

        if lang == "python":
            import ast

            try:
                ast.parse(code)
                return _make_result(spec, item, call, 1.0, True, {"language": lang})
            except SyntaxError as e:
                return _make_result(
                    spec, item, call, 0.0, False, {"language": lang, "error": str(e)}
                )
        # For other languages, just check it's not empty
        passed = len(code.strip()) > 0
        return _make_result(
            spec, item, call, 1.0 if passed else 0.0, passed, {"language": lang}
        )

    return Metric(spec, handler)


def code_complexity_metric(
    metric_id: str = "code_complexity",
    max_lines: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Code Complexity",
        type="objective",
        description="Estimates code complexity metrics.",
        config={"max_lines": max_lines, "max_depth": max_depth},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        code = _extract_code_from_markdown(text)

        lines = code.strip().split("\n")
        line_count = len(lines)

        # Estimate nesting depth by counting leading whitespace
        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                max_indent = max(max_indent, indent // 4)  # Assume 4-space indent

        limit_lines = item.metadata.get("max_lines", max_lines)
        limit_depth = item.metadata.get("max_depth", max_depth)

        passed = True
        if limit_lines and line_count > limit_lines:
            passed = False
        if limit_depth and max_indent > limit_depth:
            passed = False

        return _make_result(
            spec,
            item,
            call,
            float(line_count),
            passed,
            {
                "line_count": line_count,
                "max_nesting_depth": max_indent,
                "max_lines": limit_lines,
                "max_depth": limit_depth,
            },
        )

    return Metric(spec, handler)


# =============================================================================
# NEW METRICS: Readability
# =============================================================================


def _count_syllables(word: str) -> int:
    """Simple syllable counter for English words."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent e
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def flesch_kincaid_metric(
    metric_id: str = "flesch_kincaid", max_grade: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Flesch-Kincaid Grade",
        type="objective",
        description="Flesch-Kincaid readability grade level.",
        config={"max_grade": max_grade},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        sentences = _split_sentences(text)
        words = _tokenize(text)

        if not sentences or not words:
            return _make_result(
                spec, item, call, None, None, {"error": "insufficient_text"}
            )

        total_syllables = sum(_count_syllables(w) for w in words)
        avg_sentence_len = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        # Flesch-Kincaid Grade Level formula
        grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59
        grade = max(0, grade)

        limit = item.metadata.get("max_grade", max_grade)
        passed = True if limit is None else grade <= float(limit)

        return _make_result(
            spec,
            item,
            call,
            grade,
            passed,
            {
                "grade_level": round(grade, 2),
                "sentences": len(sentences),
                "words": len(words),
                "syllables": total_syllables,
                "max_grade": limit,
            },
        )

    return Metric(spec, handler)


def sentence_count_metric(
    metric_id: str = "sentence_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Sentence Count",
        type="objective",
        description="Counts sentences in output.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        sentences = _split_sentences(text)
        count = len(sentences)

        passed, min_v, max_v = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )

        return _make_result(
            spec,
            item,
            call,
            float(count),
            passed,
            {"count": count, "min_count": min_v, "max_count": max_v},
        )

    return Metric(spec, handler)


def avg_sentence_length_metric(
    metric_id: str = "avg_sentence_length", max_avg: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Avg Sentence Length",
        type="objective",
        description="Average sentence length in words.",
        config={"max_avg": max_avg},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        sentences = _split_sentences(text)
        words = _tokenize(text)

        if not sentences:
            return _make_result(spec, item, call, None, None, {"error": "no_sentences"})

        avg_len = len(words) / len(sentences)
        limit = item.metadata.get("max_avg", max_avg)
        passed = True if limit is None else avg_len <= float(limit)

        return _make_result(
            spec,
            item,
            call,
            avg_len,
            passed,
            {
                "avg_length": round(avg_len, 2),
                "sentences": len(sentences),
                "words": len(words),
                "max_avg": limit,
            },
        )

    return Metric(spec, handler)


# =============================================================================
# NEW METRICS: Diversity
# =============================================================================


def distinct_1_metric(
    metric_id: str = "distinct_1", min_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Distinct-1",
        type="objective",
        description="Ratio of unique unigrams to total unigrams.",
        config={"min_ratio": min_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        tokens = _tokenize(text)

        if not tokens:
            return _make_result(spec, item, call, 0.0, None, {"error": "no_tokens"})

        unique = len(set(tokens))
        total = len(tokens)
        ratio = unique / total

        limit = item.metadata.get("min_ratio", min_ratio)
        passed = True if limit is None else ratio >= float(limit)

        return _make_result(
            spec,
            item,
            call,
            ratio,
            passed,
            {
                "unique": unique,
                "total": total,
                "ratio": round(ratio, 4),
                "min_ratio": limit,
            },
        )

    return Metric(spec, handler)


def distinct_2_metric(
    metric_id: str = "distinct_2", min_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Distinct-2",
        type="objective",
        description="Ratio of unique bigrams to total bigrams.",
        config={"min_ratio": min_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        tokens = _tokenize(text)

        if len(tokens) < 2:
            return _make_result(
                spec, item, call, 0.0, None, {"error": "insufficient_tokens"}
            )

        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        unique = len(set(bigrams))
        total = len(bigrams)
        ratio = unique / total

        limit = item.metadata.get("min_ratio", min_ratio)
        passed = True if limit is None else ratio >= float(limit)

        return _make_result(
            spec,
            item,
            call,
            ratio,
            passed,
            {
                "unique": unique,
                "total": total,
                "ratio": round(ratio, 4),
                "min_ratio": limit,
            },
        )

    return Metric(spec, handler)


def vocabulary_richness_metric(
    metric_id: str = "vocabulary_richness", min_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Vocabulary Richness",
        type="objective",
        description="Type-token ratio (unique words / total words).",
        config={"min_ratio": min_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        tokens = _tokenize(text)

        if not tokens:
            return _make_result(spec, item, call, 0.0, None, {"error": "no_tokens"})

        unique = len(set(tokens))
        total = len(tokens)
        ratio = unique / total

        limit = item.metadata.get("min_ratio", min_ratio)
        passed = True if limit is None else ratio >= float(limit)

        return _make_result(
            spec,
            item,
            call,
            ratio,
            passed,
            {
                "unique_words": unique,
                "total_words": total,
                "ttr": round(ratio, 4),
                "min_ratio": limit,
            },
        )

    return Metric(spec, handler)


# =============================================================================
# NEW METRICS: Compression/Ratio
# =============================================================================


def compression_ratio_metric(
    metric_id: str = "compression_ratio",
    min_ratio: Optional[float] = None,
    max_ratio: Optional[float] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Compression Ratio",
        type="objective",
        description="Ratio of output length to input length.",
        config={"min_ratio": min_ratio, "max_ratio": max_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        output_text = _as_text(call.output)
        input_text = _as_text(item.input or item.inputs)

        output_len = len(output_text)
        input_len = len(input_text) if input_text else 1

        ratio = output_len / input_len

        min_v = item.metadata.get("min_ratio", min_ratio)
        max_v = item.metadata.get("max_ratio", max_ratio)

        passed = True
        if min_v is not None and ratio < float(min_v):
            passed = False
        if max_v is not None and ratio > float(max_v):
            passed = False

        return _make_result(
            spec,
            item,
            call,
            ratio,
            passed,
            {
                "output_chars": output_len,
                "input_chars": input_len,
                "ratio": round(ratio, 4),
                "min_ratio": min_v,
                "max_ratio": max_v,
            },
        )

    return Metric(spec, handler)


# =============================================================================
# NEW METRICS: Citation/Grounding
# =============================================================================


def citation_count_metric(
    metric_id: str = "citation_count", min_count: int = 0
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Citation Count",
        type="objective",
        description="Counts citation-like patterns.",
        config={"min_count": min_count},
    )
    # Patterns: [1], [2,3], (Author, 2024), (Smith et al., 2023)
    patterns = [
        r"\[\d+(?:,\s*\d+)*\]",  # [1], [1,2,3]
        r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)",  # (Author, 2024), (Smith et al., 2023)
    ]
    combined = re.compile("|".join(patterns))

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        matches = combined.findall(text)
        count = len(matches)

        min_v = int(item.metadata.get("min_count", min_count))
        passed = count >= min_v

        return _make_result(
            spec,
            item,
            call,
            float(count),
            passed,
            {"count": count, "min_count": min_v, "citations": matches[:10]},
        )

    return Metric(spec, handler)


def markdown_link_count_metric(
    metric_id: str = "markdown_link_count", min_count: int = 0
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Markdown Link Count",
        type="objective",
        description="Counts markdown-style links [text](url).",
        config={"min_count": min_count},
    )
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        matches = pattern.findall(text)
        count = len(matches)

        min_v = int(item.metadata.get("min_count", min_count))
        passed = count >= min_v

        return _make_result(
            spec,
            item,
            call,
            float(count),
            passed,
            {"count": count, "min_count": min_v, "links": matches[:10]},
        )

    return Metric(spec, handler)


# =============================================================================
# NEW METRICS: Semantic Similarity
# =============================================================================


def levenshtein_similarity_metric(metric_id: str = "levenshtein_similarity") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Levenshtein Similarity",
        type="objective",
        description="Normalized Levenshtein edit distance similarity.",
        config={},
    )

    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)

        if not reference:
            return _make_result(
                spec, item, call, None, None, {"error": "No reference text"}
            )

        distance = _levenshtein_distance(candidate.lower(), reference.lower())
        max_len = max(len(candidate), len(reference), 1)
        similarity = 1.0 - (distance / max_len)

        return _make_result(
            spec,
            item,
            call,
            similarity,
            None,
            {"distance": distance, "similarity": round(similarity, 4)},
        )

    return Metric(spec, handler)


def cosine_word_overlap_metric(metric_id: str = "cosine_word_overlap") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Cosine Word Overlap",
        type="objective",
        description="Cosine similarity of word frequency vectors.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        reference = _get_reference(item)
        candidate = _get_output(call, item)

        if not reference:
            return _make_result(
                spec, item, call, None, None, {"error": "No reference text"}
            )

        cand_tokens = _tokenize(candidate)
        ref_tokens = _tokenize(reference)

        if not cand_tokens or not ref_tokens:
            return _make_result(spec, item, call, 0.0, None, {"error": "empty_text"})

        # Build word frequency vectors
        all_words = set(cand_tokens) | set(ref_tokens)
        cand_freq = {w: cand_tokens.count(w) for w in all_words}
        ref_freq = {w: ref_tokens.count(w) for w in all_words}

        # Compute cosine similarity
        dot_product = sum(cand_freq[w] * ref_freq[w] for w in all_words)
        cand_norm = math.sqrt(sum(v * v for v in cand_freq.values()))
        ref_norm = math.sqrt(sum(v * v for v in ref_freq.values()))

        if cand_norm == 0 or ref_norm == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (cand_norm * ref_norm)

        return _make_result(
            spec,
            item,
            call,
            similarity,
            None,
            {"similarity": round(similarity, 4)},
        )

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Format Validation
# =============================================================================


def yaml_valid_metric(metric_id: str = "yaml_valid") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="YAML Valid",
        type="objective",
        description="Checks whether output parses as YAML.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        import yaml

        text = _as_text(call.output)
        try:
            yaml.safe_load(text)
            return _make_result(spec, item, call, 1.0, True, {})
        except Exception as exc:
            return _make_result(spec, item, call, 0.0, False, {"error": str(exc)})

    return Metric(spec, handler)


def markdown_structure_metric(
    metric_id: str = "markdown_structure", require_heading: bool = False
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Markdown Structure",
        type="objective",
        description="Validates markdown structure.",
        config={"require_heading": require_heading},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        headings = len(re.findall(r"^#{1,6}\s+", text, re.MULTILINE))
        bullets = len(re.findall(r"^[\s]*[-*+]\s+", text, re.MULTILINE))
        code_blocks = len(re.findall(r"```", text)) // 2
        links = len(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text))

        req_heading = item.metadata.get("require_heading", require_heading)
        passed = True
        if req_heading and headings == 0:
            passed = False

        return _make_result(
            spec,
            item,
            call,
            float(headings + bullets + code_blocks + links),
            passed,
            {
                "headings": headings,
                "bullets": bullets,
                "code_blocks": code_blocks,
                "links": links,
            },
        )

    return Metric(spec, handler)


def html_valid_metric(metric_id: str = "html_valid") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="HTML Valid",
        type="objective",
        description="Checks whether output contains valid HTML structure.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        from html.parser import HTMLParser

        text = _as_text(call.output)

        class TagValidator(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []
                self.valid = True

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)

            def handle_endtag(self, tag):
                if self.tags and self.tags[-1] == tag:
                    self.tags.pop()
                elif tag not in ["br", "hr", "img", "input", "meta", "link"]:
                    self.valid = False

        try:
            parser = TagValidator()
            parser.feed(text)
            passed = parser.valid and len(parser.tags) == 0
            return _make_result(spec, item, call, 1.0 if passed else 0.0, passed, {})
        except Exception as exc:
            return _make_result(spec, item, call, 0.0, False, {"error": str(exc)})

    return Metric(spec, handler)


def sql_valid_metric(metric_id: str = "sql_valid") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="SQL Valid",
        type="objective",
        description="Basic SQL syntax validation.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output).strip().upper()
        # Basic SQL keyword check
        sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "WITH",
        ]
        has_keyword = any(text.startswith(kw) for kw in sql_keywords)
        # Check for basic structure
        has_structure = (
            ("SELECT" in text and "FROM" in text)
            or ("INSERT" in text and "INTO" in text)
            or ("UPDATE" in text and "SET" in text)
            or ("DELETE" in text and "FROM" in text)
            or ("CREATE" in text)
            or ("DROP" in text)
            or ("ALTER" in text)
            or ("WITH" in text and "SELECT" in text)
        )
        passed = has_keyword and has_structure
        return _make_result(
            spec,
            item,
            call,
            1.0 if passed else 0.0,
            passed,
            {"has_sql_structure": passed},
        )

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Structure Detection
# =============================================================================


def bullet_count_metric(
    metric_id: str = "bullet_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Bullet Count",
        type="objective",
        description="Counts bullet points.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = len(re.findall(r"^[\s]*[-*+]\s+", text, re.MULTILINE))
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def heading_count_metric(
    metric_id: str = "heading_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Heading Count",
        type="objective",
        description="Counts markdown headings.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = len(re.findall(r"^#{1,6}\s+", text, re.MULTILINE))
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def code_block_count_metric(
    metric_id: str = "code_block_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Code Block Count",
        type="objective",
        description="Counts fenced code blocks.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = len(re.findall(r"```", text)) // 2
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def table_count_metric(
    metric_id: str = "table_count", min_count: Optional[int] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Table Count",
        type="objective",
        description="Counts markdown tables.",
        config={"min_count": min_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        # Count table separator lines (|---|---|)
        count = len(re.findall(r"^\|[-:| ]+\|$", text, re.MULTILINE))
        min_v = item.metadata.get("min_count", min_count)
        passed = True if min_v is None else count >= int(min_v)
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def paragraph_count_metric(
    metric_id: str = "paragraph_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Paragraph Count",
        type="objective",
        description="Counts paragraphs.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        count = len(paragraphs)
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def word_count_metric(
    metric_id: str = "word_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Word Count",
        type="objective",
        description="Counts words in output.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        words = _tokenize(text)
        count = len(words)
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Repetition
# =============================================================================


def repetition_ratio_metric(
    metric_id: str = "repetition_ratio", n: int = 3, max_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Repetition Ratio",
        type="objective",
        description="Ratio of repeated n-grams.",
        config={"n": n, "max_ratio": max_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        tokens = _tokenize(text)
        n_val = item.metadata.get("n", n)

        if len(tokens) < n_val:
            return _make_result(spec, item, call, 0.0, True, {"ratio": 0.0})

        ngrams = [tuple(tokens[i : i + n_val]) for i in range(len(tokens) - n_val + 1)]
        if not ngrams:
            return _make_result(spec, item, call, 0.0, True, {"ratio": 0.0})

        unique = len(set(ngrams))
        total = len(ngrams)
        ratio = 1.0 - (unique / total)  # Higher = more repetitive

        max_r = item.metadata.get("max_ratio", max_ratio)
        passed = True if max_r is None else ratio <= float(max_r)

        return _make_result(
            spec, item, call, ratio, passed, {"ratio": round(ratio, 4), "n": n_val}
        )

    return Metric(spec, handler)


def duplicate_line_ratio_metric(
    metric_id: str = "duplicate_line_ratio", max_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Duplicate Line Ratio",
        type="objective",
        description="Ratio of duplicate lines.",
        config={"max_ratio": max_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            return _make_result(spec, item, call, 0.0, True, {"ratio": 0.0})

        unique = len(set(lines))
        total = len(lines)
        ratio = 1.0 - (unique / total)

        max_r = item.metadata.get("max_ratio", max_ratio)
        passed = True if max_r is None else ratio <= float(max_r)

        return _make_result(
            spec, item, call, ratio, passed, {"ratio": round(ratio, 4), "lines": total}
        )

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Uncertainty/Confidence
# =============================================================================


def hedging_count_metric(
    metric_id: str = "hedging_count", max_count: Optional[int] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Hedging Count",
        type="objective",
        description="Counts hedging phrases.",
        config={"max_count": max_count},
    )
    hedging_words = [
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\bprobably\b",
        r"\bi think\b",
        r"\bi believe\b",
        r"\bit seems\b",
        r"\bappears to\b",
        r"\bseems like\b",
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\buncertain\b",
        r"\blikely\b",
        r"\bunlikely\b",
    ]
    pattern = re.compile("|".join(hedging_words), re.IGNORECASE)

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = len(pattern.findall(text))
        max_v = item.metadata.get("max_count", max_count)
        passed = True if max_v is None else count <= int(max_v)
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def question_count_metric(
    metric_id: str = "question_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Question Count",
        type="objective",
        description="Counts questions in output.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = text.count("?")
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def confidence_markers_metric(metric_id: str = "confidence_markers") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Confidence Markers",
        type="objective",
        description="Counts confidence markers.",
        config={},
    )
    confidence_words = [
        r"\bcertainly\b",
        r"\bdefinitely\b",
        r"\bclearly\b",
        r"\bobviously\b",
        r"\bwithout doubt\b",
        r"\bundoubtedly\b",
        r"\babsolutely\b",
        r"\bno question\b",
        r"\bfor sure\b",
        r"\bguaranteed\b",
    ]
    pattern = re.compile("|".join(confidence_words), re.IGNORECASE)

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = len(pattern.findall(text))
        return _make_result(spec, item, call, float(count), None, {"count": count})

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Code Quality
# =============================================================================


def comment_ratio_metric(
    metric_id: str = "comment_ratio", min_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Comment Ratio",
        type="objective",
        description="Ratio of comment lines to code lines.",
        config={"min_ratio": min_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        code = _extract_code_from_markdown(text)

        lines = [line.strip() for line in code.split("\n") if line.strip()]
        if not lines:
            return _make_result(spec, item, call, 0.0, None, {"ratio": 0.0})

        comment_lines = sum(
            1 for line in lines if line.startswith("#") or line.startswith("//")
        )
        ratio = comment_lines / len(lines)

        min_r = item.metadata.get("min_ratio", min_ratio)
        passed = True if min_r is None else ratio >= float(min_r)

        return _make_result(
            spec,
            item,
            call,
            ratio,
            passed,
            {
                "comment_lines": comment_lines,
                "total_lines": len(lines),
                "ratio": round(ratio, 4),
            },
        )

    return Metric(spec, handler)


def function_count_metric(
    metric_id: str = "function_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Function Count",
        type="objective",
        description="Counts function definitions.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        # Python: def/async def, JS: function, Arrow functions
        patterns = [
            r"\bdef\s+\w+\s*\(",
            r"\basync\s+def\s+\w+\s*\(",
            r"\bfunction\s+\w*\s*\(",
            r"\bconst\s+\w+\s*=\s*\([^)]*\)\s*=>",
        ]
        count = sum(len(re.findall(p, text)) for p in patterns)
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )

        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def import_count_metric(
    metric_id: str = "import_count", max_count: Optional[int] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Import Count",
        type="objective",
        description="Counts import statements.",
        config={"max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        # Python imports, JS imports/require
        patterns = [
            r"^import\s+",
            r"^from\s+\S+\s+import\s+",
            r"\brequire\s*\(",
        ]
        count = sum(len(re.findall(p, text, re.MULTILINE)) for p in patterns)

        max_v = item.metadata.get("max_count", max_count)
        passed = True if max_v is None else count <= int(max_v)

        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Character/Format
# =============================================================================


def ascii_ratio_metric(
    metric_id: str = "ascii_ratio", min_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="ASCII Ratio",
        type="objective",
        description="Ratio of ASCII characters.",
        config={"min_ratio": min_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        if not text:
            return _make_result(spec, item, call, 1.0, True, {"ratio": 1.0})

        ascii_count = sum(1 for c in text if ord(c) < 128)
        ratio = ascii_count / len(text)

        min_r = item.metadata.get("min_ratio", min_ratio)
        passed = True if min_r is None else ratio >= float(min_r)

        return _make_result(spec, item, call, ratio, passed, {"ratio": round(ratio, 4)})

    return Metric(spec, handler)


def uppercase_ratio_metric(
    metric_id: str = "uppercase_ratio", max_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Uppercase Ratio",
        type="objective",
        description="Ratio of uppercase letters.",
        config={"max_ratio": max_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return _make_result(spec, item, call, 0.0, True, {"ratio": 0.0})

        upper_count = sum(1 for c in letters if c.isupper())
        ratio = upper_count / len(letters)

        max_r = item.metadata.get("max_ratio", max_ratio)
        passed = True if max_r is None else ratio <= float(max_r)

        return _make_result(spec, item, call, ratio, passed, {"ratio": round(ratio, 4)})

    return Metric(spec, handler)


def numeric_density_metric(metric_id: str = "numeric_density") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Numeric Density",
        type="objective",
        description="Ratio of numeric characters.",
        config={},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        if not text:
            return _make_result(spec, item, call, 0.0, None, {"ratio": 0.0})

        digit_count = sum(1 for c in text if c.isdigit())
        ratio = digit_count / len(text)

        return _make_result(spec, item, call, ratio, None, {"ratio": round(ratio, 4)})

    return Metric(spec, handler)


def whitespace_ratio_metric(
    metric_id: str = "whitespace_ratio", max_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Whitespace Ratio",
        type="objective",
        description="Ratio of whitespace characters.",
        config={"max_ratio": max_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        if not text:
            return _make_result(spec, item, call, 0.0, True, {"ratio": 0.0})

        ws_count = sum(1 for c in text if c.isspace())
        ratio = ws_count / len(text)

        max_r = item.metadata.get("max_ratio", max_ratio)
        passed = True if max_r is None else ratio <= float(max_r)

        return _make_result(spec, item, call, ratio, passed, {"ratio": round(ratio, 4)})

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Match Variants
# =============================================================================


def prefix_match_metric(
    metric_id: str = "prefix_match", prefix: Optional[str] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Prefix Match",
        type="objective",
        description="Checks if output starts with expected prefix.",
        config={"prefix": prefix},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        expected = prefix or item.metadata.get("prefix", "")
        passed = text.startswith(expected) if expected else True
        return _make_result(
            spec, item, call, 1.0 if passed else 0.0, passed, {"prefix": expected}
        )

    return Metric(spec, handler)


def suffix_match_metric(
    metric_id: str = "suffix_match", suffix: Optional[str] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Suffix Match",
        type="objective",
        description="Checks if output ends with expected suffix.",
        config={"suffix": suffix},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        expected = suffix or item.metadata.get("suffix", "")
        passed = text.endswith(expected) if expected else True
        return _make_result(
            spec, item, call, 1.0 if passed else 0.0, passed, {"suffix": expected}
        )

    return Metric(spec, handler)


def contains_all_metric(
    metric_id: str = "contains_all", substrings: Optional[List[str]] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Contains All",
        type="objective",
        description="Checks if output contains all required substrings.",
        config={"substrings": substrings or []},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output).lower()
        required = substrings or item.metadata.get("substrings", [])
        missing = [s for s in required if s.lower() not in text]
        passed = len(missing) == 0
        return _make_result(
            spec,
            item,
            call,
            1.0 if passed else 0.0,
            passed,
            {"required": required, "missing": missing},
        )

    return Metric(spec, handler)


def contains_none_metric(
    metric_id: str = "contains_none", forbidden: Optional[List[str]] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Contains None",
        type="objective",
        description="Checks if output contains none of the forbidden substrings.",
        config={"forbidden": forbidden or []},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output).lower()
        blocked = forbidden or item.metadata.get("forbidden", [])
        found = [s for s in blocked if s.lower() in text]
        passed = len(found) == 0
        return _make_result(
            spec,
            item,
            call,
            1.0 if passed else 0.0,
            passed,
            {"forbidden": blocked, "found": found},
        )

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: List/Enumeration
# =============================================================================


def numbered_list_count_metric(
    metric_id: str = "numbered_list_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Numbered List Count",
        type="objective",
        description="Counts numbered list items.",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        count = len(re.findall(r"^\s*\d+[.)]\s+", text, re.MULTILINE))
        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )
        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def list_item_count_metric(
    metric_id: str = "list_item_count",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="List Item Count",
        type="objective",
        description="Counts total list items (bullets + numbers).",
        config={"min_count": min_count, "max_count": max_count},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        bullets = len(re.findall(r"^[\s]*[-*+]\s+", text, re.MULTILINE))
        numbered = len(re.findall(r"^\s*\d+[.)]\s+", text, re.MULTILINE))
        count = bullets + numbered

        passed, _, _ = _check_min_max_bounds(
            count, item, "min_count", "max_count", min_count, max_count
        )

        return _make_result(
            spec,
            item,
            call,
            float(count),
            passed,
            {"bullets": bullets, "numbered": numbered, "total": count},
        )

    return Metric(spec, handler)


# =============================================================================
# MORE METRICS: Response Quality
# =============================================================================


def emoji_count_metric(
    metric_id: str = "emoji_count", max_count: Optional[int] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Emoji Count",
        type="objective",
        description="Counts emoji characters.",
        config={"max_count": max_count},
    )
    # Simplified emoji pattern
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f700-\U0001f77f"  # alchemical
        "\U0001f780-\U0001f7ff"  # geometric
        "\U0001f800-\U0001f8ff"  # supplemental arrows
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U0001fa00-\U0001fa6f"  # chess
        "\U0001fa70-\U0001faff"  # symbols
        "\U00002702-\U000027b0"  # dingbats
        "]+",
        flags=re.UNICODE,
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        matches = emoji_pattern.findall(text)
        count = sum(len(m) for m in matches)

        max_v = item.metadata.get("max_count", max_count)
        passed = True if max_v is None else count <= int(max_v)

        return _make_result(spec, item, call, float(count), passed, {"count": count})

    return Metric(spec, handler)


def link_density_metric(
    metric_id: str = "link_density", max_ratio: Optional[float] = None
) -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Link Density",
        type="objective",
        description="Ratio of link text to total text.",
        config={"max_ratio": max_ratio},
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        text = _as_text(call.output)
        if not text:
            return _make_result(spec, item, call, 0.0, True, {"ratio": 0.0})

        # Find all URLs and markdown links
        url_pattern = r"https?://\S+"
        md_link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        urls = re.findall(url_pattern, text)
        md_links = re.findall(md_link_pattern, text)

        link_chars = sum(len(u) for u in urls) + sum(
            len(t) + len(u) for t, u in md_links
        )
        ratio = link_chars / len(text)

        max_r = item.metadata.get("max_ratio", max_ratio)
        passed = True if max_r is None else ratio <= float(max_r)

        return _make_result(spec, item, call, ratio, passed, {"ratio": round(ratio, 4)})

    return Metric(spec, handler)
