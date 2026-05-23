"""
Pytest plugin for evalyn.

Lets engineering teams gate tests in their existing pytest CI on evalyn
evaluation thresholds. Two equivalent forms:

Marker form (primary):

    import pytest

    @pytest.mark.evalyn(
        dataset="data/my_project/dataset.jsonl",
        metrics=["helpfulness_accuracy", "toxicity_safety"],
        thresholds={"helpfulness_accuracy": 0.8, "toxicity_safety": 0.95},
    )
    def test_agent_quality():
        ...

Helper form (explicit, inside the test body):

    from evalyn_sdk.pytest_plugin import evaluate_and_assert

    def test_agent_quality():
        evaluate_and_assert(
            dataset="data/my_project/dataset.jsonl",
            metrics=["helpfulness_accuracy", "toxicity_safety"],
            thresholds={"helpfulness_accuracy": 0.8, "toxicity_safety": 0.95},
        )

A test passes if every metric listed in ``thresholds`` has an actual mean
score (``avg_score``) greater than or equal to the configured threshold.
On failure pytest prints a readable per-metric breakdown.

The plugin is auto-registered through the ``pytest11`` entry point in
``sdk/pyproject.toml``. ``pytest`` itself is only imported here, never as
a hard runtime dependency of the SDK.
"""

from __future__ import annotations

from typing import Any

import pytest

from . import api as _api

_MARKER_NAME = "evalyn"
_MARKER_DOC = (
    "evalyn(dataset, metrics, thresholds, provider='gemini', max_workers=1, "
    "name=None, tags=None): run an evalyn evaluation before the test body, "
    "and assert that each metric listed in ``thresholds`` has an average "
    "score >= the configured threshold. Fails the test with a per-metric "
    "breakdown when any threshold is breached."
)


class EvalynThresholdError(AssertionError):
    """Raised when an evalyn evaluation fails one or more threshold checks."""


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``evalyn`` marker so ``--strict-markers`` is happy."""
    config.addinivalue_line("markers", f"{_MARKER_NAME}: {_MARKER_DOC}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Run the evalyn evaluation before the test body executes.

    When a test carries ``@pytest.mark.evalyn(...)`` we resolve the marker
    kwargs, run :func:`evalyn_sdk.api.evaluate`, and assert thresholds. If
    the threshold check fails (or the eval itself raises) we raise inside
    the wrapper so the test is reported as failed with the underlying
    cause visible. The original test body still runs after a successful
    eval, so users can layer extra assertions on top.
    """
    marker = item.get_closest_marker(_MARKER_NAME)
    if marker is not None:
        kwargs = _resolve_marker_kwargs(marker)
        # Run the eval up-front. If any of the pre-flight checks or the eval
        # itself raises, we let it propagate as the test failure cause.
        _run_eval_and_assert(**kwargs)
    yield


def evaluate_and_assert(
    *,
    dataset: Any,
    metrics: Any,
    thresholds: dict[str, float],
    provider: str = "gemini",
    max_workers: int = 1,
    name: str | None = None,
    tags: list[str] | None = None,
):
    """Run an evalyn evaluation and assert per-metric thresholds.

    Public helper for the explicit form. Returns the underlying
    :class:`evalyn_sdk.api.EvalResult` on success so the caller can layer
    additional assertions.

    Raises :class:`EvalynThresholdError` (an ``AssertionError`` subclass)
    if any threshold is breached, so pytest reports the test as failed.
    """
    return _run_eval_and_assert(
        dataset=dataset,
        metrics=metrics,
        thresholds=thresholds,
        provider=provider,
        max_workers=max_workers,
        name=name,
        tags=tags,
    )


def _resolve_marker_kwargs(marker: pytest.Mark) -> dict[str, Any]:
    """Pull eval kwargs out of a ``pytest.mark.evalyn(...)`` invocation.

    Positional args are not supported; everything must be keyword. We
    surface a clear error message if the user forgot the required
    ``dataset`` or ``thresholds`` keys before paying the cost of the
    eval run.
    """
    if marker.args:
        raise EvalynThresholdError(
            "evalyn marker only accepts keyword arguments "
            "(e.g. @pytest.mark.evalyn(dataset=..., metrics=..., thresholds=...))"
        )

    kwargs = dict(marker.kwargs)
    if "dataset" not in kwargs:
        raise EvalynThresholdError(
            "evalyn marker is missing required 'dataset' kwarg"
        )
    if "thresholds" not in kwargs:
        raise EvalynThresholdError(
            "evalyn marker is missing required 'thresholds' kwarg"
        )
    return kwargs


def _run_eval_and_assert(
    *,
    dataset: Any,
    metrics: Any,
    thresholds: dict[str, float],
    provider: str = "gemini",
    max_workers: int = 1,
    name: str | None = None,
    tags: list[str] | None = None,
):
    """Shared core: validate inputs, run eval, assert thresholds."""
    if not isinstance(thresholds, dict) or not thresholds:
        raise EvalynThresholdError(
            "evalyn 'thresholds' must be a non-empty dict mapping "
            "metric_id -> minimum required mean score"
        )

    # Pre-flight: every threshold metric must appear in the configured
    # metrics list (when metrics is given as a list of IDs). Catches
    # typos like thresholds={"helpfullness": 0.8} early, before paying
    # for an eval that can never satisfy the threshold.
    metric_ids = _collect_metric_ids(metrics)
    if metric_ids is not None:
        missing_in_metrics = [m for m in thresholds if m not in metric_ids]
        if missing_in_metrics:
            raise EvalynThresholdError(
                "evalyn thresholds reference metric(s) not in the metrics "
                f"list: {sorted(missing_in_metrics)}. "
                f"Configured metrics: {sorted(metric_ids)}"
            )

    try:
        result = _api.evaluate(
            dataset=dataset,
            metrics=metrics,
            provider=provider,
            max_workers=max_workers,
            name=name,
            tags=tags,
        )
    except Exception as exc:  # noqa: BLE001 - surface the underlying error
        raise EvalynThresholdError(
            f"evalyn evaluation raised {type(exc).__name__}: {exc}"
        ) from exc

    breaches, missing_from_results = _check_thresholds(result.metric_summary, thresholds)

    if missing_from_results:
        raise EvalynThresholdError(
            "evalyn thresholds reference metric(s) absent from eval results: "
            f"{sorted(missing_from_results)}. "
            f"Metrics in result: {sorted(result.metric_summary)}"
        )

    if breaches:
        raise EvalynThresholdError(_format_breach_message(breaches))

    return result


def _collect_metric_ids(metrics: Any) -> set[str] | None:
    """Best-effort: pull metric IDs out of the user's ``metrics`` arg.

    Returns ``None`` when we cannot statically determine the metric IDs
    (e.g. a path to a metrics JSON file, or pre-built Metric objects we
    don't want to introspect here). In that case the pre-flight check
    is skipped and we fall back to the post-eval check against the
    actual result summary.
    """
    if metrics is None:
        return None
    if isinstance(metrics, list) and metrics and all(isinstance(m, str) for m in metrics):
        return set(metrics)
    return None


def _check_thresholds(
    metric_summary: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Compare actual mean scores against required thresholds.

    Returns a tuple of (breaches, missing). ``breaches`` is a list of
    (metric_id, actual_mean, required) for metrics that fell short.
    ``missing`` lists metric IDs that were requested for thresholding
    but never appeared in the eval result.
    """
    breaches: list[tuple[str, float, float]] = []
    missing: list[str] = []
    for metric_id, required in thresholds.items():
        if metric_id not in metric_summary:
            missing.append(metric_id)
            continue
        actual = metric_summary[metric_id].get("avg_score")
        if actual is None:
            missing.append(metric_id)
            continue
        if actual < required:
            breaches.append((metric_id, float(actual), float(required)))
    return breaches, missing


def _format_breach_message(breaches: list[tuple[str, float, float]]) -> str:
    """Human-readable breakdown shown in the pytest failure report."""
    header = f"evalyn threshold check failed for {len(breaches)} metric(s):"
    lines = [header]
    for metric_id, actual, required in breaches:
        lines.append(
            f"  - {metric_id}: actual_mean={actual:.4f} < required={required:.4f}"
        )
    return "\n".join(lines)


__all__ = [
    "EvalynThresholdError",
    "evaluate_and_assert",
    "pytest_configure",
    "pytest_runtest_call",
]
