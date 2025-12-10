from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

from .registry import Metric
from ..models import DatasetItem, FunctionCall, MetricResult, MetricSpec


def latency_metric(metric_id: str = "latency_ms") -> Metric:
    spec = MetricSpec(
        id=metric_id,
        name="Latency (ms)",
        type="objective",
        description="Execution time in milliseconds.",
    )

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=call.duration_ms,
            passed=None,
            details={"duration_ms": call.duration_ms},
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
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={"expected": expected, "actual": actual},
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
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={"target": target, "output_excerpt": str(output_text)[:200]},
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
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=cost_info.get("total") if isinstance(cost_info, dict) else None,
            passed=None,
            details={"cost": cost_info},
        )

    return Metric(spec, handler)


def register_builtin_metrics(registry) -> None:
    registry.register(latency_metric())
    registry.register(exact_match_metric())
    registry.register(cost_metric())
    registry.register(bleu_metric())
    registry.register(pass_at_k_metric())


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
    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
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
        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=None,
            details={"candidate": str(candidate)[:200], "reference": str(reference)[:200]},
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
        config={"k": k, "candidate_field": candidate_field, "success_field": success_field},
    )

    def _extract_candidates(output: Any) -> Sequence[dict]:
        if isinstance(output, dict) and candidate_field in output:
            return output.get(candidate_field, []) or []
        if isinstance(output, list):
            return output
        return []

    def handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
        candidates = _extract_candidates(call.output)
        successes = [c.get(success_field, False) for c in candidates if isinstance(c, dict)]
        n = len(successes)
        if n == 0:
            return MetricResult(
                metric_id=spec.id,
                item_id=item.id,
                call_id=call.id,
                score=0.0,
                passed=False,
                details={"candidates": 0, "k": k},
            )

        k_eff = min(k, n)
        # pass@k estimate: 1 - ((n - c choose k) / (n choose k)) where c = successes count
        c = sum(1 for s in successes if s)
        if c == 0:
            score = 0.0
        else:
            from math import comb

            score = 1.0 - (comb(n - c, k_eff) / comb(n, k_eff))

        return MetricResult(
            metric_id=spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=None,
            details={"candidates": n, "successes": c, "k": k_eff},
        )

    return Metric(spec, handler)
