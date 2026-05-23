"""
Meta-tests for the evalyn pytest plugin.

The plugin is registered via the ``pytest11`` entry point in
``sdk/pyproject.toml``. These tests exercise both forms - the
``@pytest.mark.evalyn`` marker and the explicit ``evaluate_and_assert``
helper - through pytest's own ``pytester`` fixture, which spins up a
nested pytest session in a temporary directory and lets us assert on
the resulting exit code and stdout.

We rely on the deterministic ``output_nonempty`` and ``token_length``
objective metrics so the meta-tests never need an LLM. ``token_length``
in particular lets us force partial failures by setting
``metadata.max_chars`` per item.
"""

from __future__ import annotations

import textwrap

import pytest

# The pytester fixture is shipped with pytest itself and provides an
# isolated, throwaway pytest invocation. Enable it for this module.
pytest_plugins = ["pytester"]


def _src(body: str) -> str:
    """Dedent + lstrip a test-body literal so pytester.makepyfile gets a
    cleanly column-zero Python module (no leading blank line, no indent
    inherited from the surrounding triple-quoted string)."""
    return textwrap.dedent(body).lstrip("\n")


# ---------------------------------------------------------------------------
# Marker form
# ---------------------------------------------------------------------------


def test_marker_passes_when_thresholds_met(pytester: pytest.Pytester) -> None:
    """A marker whose thresholds are all met should let the test pass."""
    pytester.makepyfile(_src(
        """
        import pytest

        PASSING_DATASET = [
            {"id": "a", "input": {"q": "hi"}, "output": "hello world", "metadata": {}},
            {"id": "b", "input": {"q": "bye"}, "output": "goodbye now", "metadata": {}},
            {"id": "c", "input": {"q": "ok"}, "output": "all good here", "metadata": {}},
        ]

        @pytest.mark.evalyn(
            dataset=PASSING_DATASET,
            metrics=["output_nonempty"],
            thresholds={"output_nonempty": 0.9},
        )
        def test_quality():
            pass
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


def test_marker_fails_with_breach_breakdown(pytester: pytest.Pytester) -> None:
    """A breached threshold should fail the test and print a per-metric
    breakdown including the actual and required values."""
    pytester.makepyfile(_src(
        """
        import pytest

        MIXED_DATASET = [
            {"id": "a", "input": {"q": "hi"}, "output": "hi", "metadata": {"max_chars": 5}},
            {"id": "b", "input": {"q": "ok"}, "output": "okay", "metadata": {"max_chars": 5}},
            {"id": "c", "input": {"q": "long"}, "output": "this is way too long",
             "metadata": {"max_chars": 5}},
        ]

        @pytest.mark.evalyn(
            dataset=MIXED_DATASET,
            metrics=["token_length"],
            thresholds={"token_length": 0.9},
        )
        def test_quality():
            pass
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    # The breakdown should mention the metric, the actual mean, and the
    # required threshold in a structured form.
    result.stdout.fnmatch_lines(
        [
            "*evalyn threshold check failed for 1 metric(s):*",
            "*token_length: actual_mean=0.6667 < required=0.9000*",
        ]
    )


def test_marker_is_registered(pytester: pytest.Pytester) -> None:
    """``pytest --markers`` should list the ``evalyn`` marker so users
    running with ``--strict-markers`` don't get spurious warnings."""
    pytester.makepyfile("def test_noop(): pass")
    result = pytester.runpytest_subprocess("--markers")
    result.stdout.fnmatch_lines(["@pytest.mark.evalyn*"])


def test_marker_missing_dataset_fails_clearly(pytester: pytest.Pytester) -> None:
    """Forgetting the ``dataset`` kwarg must produce a clear error, not a
    cryptic crash inside the eval runner."""
    pytester.makepyfile(_src(
        """
        import pytest

        @pytest.mark.evalyn(
            metrics=["output_nonempty"],
            thresholds={"output_nonempty": 0.9},
        )
        def test_quality():
            pass
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*missing required 'dataset' kwarg*"])


def test_marker_threshold_typo_caught_preflight(pytester: pytest.Pytester) -> None:
    """A threshold key that is not in the configured metrics list should
    be caught before the eval runs, to surface typos early."""
    pytester.makepyfile(_src(
        """
        import pytest

        PASSING_DATASET = [
            {"id": "a", "input": {"q": "hi"}, "output": "hello world", "metadata": {}},
            {"id": "b", "input": {"q": "bye"}, "output": "goodbye now", "metadata": {}},
        ]

        @pytest.mark.evalyn(
            dataset=PASSING_DATASET,
            metrics=["output_nonempty"],
            thresholds={"output_nonemptyy": 0.9},
        )
        def test_quality():
            pass
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        ["*thresholds reference metric(s) not in the metrics list*"]
    )


def test_marker_propagates_eval_error(pytester: pytest.Pytester) -> None:
    """If the eval itself raises (e.g. dataset path missing), the test
    should fail with the underlying error message visible."""
    pytester.makepyfile(_src(
        """
        import pytest

        @pytest.mark.evalyn(
            dataset="/nonexistent/path/dataset.jsonl",
            metrics=["output_nonempty"],
            thresholds={"output_nonempty": 0.9},
        )
        def test_quality():
            pass
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*FileNotFoundError*"])


# ---------------------------------------------------------------------------
# Helper form
# ---------------------------------------------------------------------------


def test_evaluate_and_assert_passes(pytester: pytest.Pytester) -> None:
    """The explicit helper should be importable and pass when thresholds
    are met."""
    pytester.makepyfile(_src(
        """
        from evalyn_sdk.pytest_plugin import evaluate_and_assert

        PASSING_DATASET = [
            {"id": "a", "input": {"q": "hi"}, "output": "hello world", "metadata": {}},
            {"id": "b", "input": {"q": "bye"}, "output": "goodbye now", "metadata": {}},
        ]

        def test_quality():
            result = evaluate_and_assert(
                dataset=PASSING_DATASET,
                metrics=["output_nonempty"],
                thresholds={"output_nonempty": 0.9},
            )
            assert result.metric_summary["output_nonempty"]["avg_score"] == 1.0
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(passed=1)


def test_evaluate_and_assert_fails(pytester: pytest.Pytester) -> None:
    """The explicit helper should raise an AssertionError on breach so
    pytest reports the test as failed with the breakdown."""
    pytester.makepyfile(_src(
        """
        from evalyn_sdk.pytest_plugin import evaluate_and_assert

        MIXED_DATASET = [
            {"id": "a", "input": {"q": "hi"}, "output": "hi", "metadata": {"max_chars": 5}},
            {"id": "b", "input": {"q": "ok"}, "output": "okay", "metadata": {"max_chars": 5}},
            {"id": "c", "input": {"q": "long"}, "output": "this is way too long",
             "metadata": {"max_chars": 5}},
        ]

        def test_quality():
            evaluate_and_assert(
                dataset=MIXED_DATASET,
                metrics=["token_length"],
                thresholds={"token_length": 0.9},
            )
        """
    ))
    result = pytester.runpytest_subprocess("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*evalyn threshold check failed for 1 metric(s):*",
            "*token_length: actual_mean=0.6667 < required=0.9000*",
        ]
    )
