"""Extract span-metric attribution links from judge results."""

from __future__ import annotations

from .models import MetricResult, SpanMetricLink, _default_id


def extract_span_metric_links(result: MetricResult, run_id: str) -> list[SpanMetricLink]:
    """Extract SpanMetricLinks from a MetricResult's raw_judge data.

    Looks for a "span_attribution" array in the raw_judge dict,
    where each element has span_id, relevance, and reason.

    Returns empty list if no attribution data is present (e.g. objective metrics).
    """
    if not result.raw_judge:
        return []

    attributions = result.raw_judge.get("span_attribution", [])
    if not attributions:
        return []

    metric_result_id = f"{result.metric_id}:{result.item_id}:{result.call_id}"
    links = []
    for attr in attributions:
        span_id = attr.get("span_id")
        if not span_id:
            continue
        links.append(
            SpanMetricLink(
                id=_default_id(),
                metric_result_id=metric_result_id,
                span_id=span_id,
                relevance=float(attr.get("relevance", 0.0)),
                reason=attr.get("reason", ""),
                run_id=run_id,
            )
        )
    return links
