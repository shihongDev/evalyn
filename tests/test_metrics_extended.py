"""Tests for metric suggester and subjective template validation."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from evalyn_sdk.models import FunctionCall, TraceEvent, MetricSpec
from evalyn_sdk.metrics.suggester import HeuristicSuggester


# ---------------------------------------------------------------------------
# HeuristicSuggester
# ---------------------------------------------------------------------------


def _make_call(output, trace_events=None, error=None) -> FunctionCall:
    """Build a FunctionCall with given output and optional trace events."""
    return FunctionCall(
        id="call-1",
        function_name="test_fn",
        inputs={"query": "test"},
        output=output,
        error=error,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        duration_ms=1000.0,
        session_id="sess-1",
        trace=trace_events or [],
        metadata={},
    )


def _dummy_fn():
    """Dummy target function for suggestion."""
    pass


class TestHeuristicSuggesterBasic:
    """Test HeuristicSuggester always-present suggestions."""

    def test_always_includes_latency(self):
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn)
        ids = [sp.id for sp in specs]
        assert "latency_ms" in ids

    def test_always_includes_output_nonempty(self):
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn)
        ids = [sp.id for sp in specs]
        assert "output_nonempty" in ids

    def test_always_includes_helpfulness(self):
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn)
        ids = [sp.id for sp in specs]
        assert "helpfulness_accuracy" in ids

    def test_no_traces_still_works(self):
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn, traces=None)
        assert len(specs) >= 3  # latency, output_nonempty, helpfulness


class TestHeuristicSuggesterConditional:
    """Test conditional metric suggestions based on trace content."""

    def test_json_valid_for_dict_output(self):
        calls = [_make_call(output={"key": "value"})]
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "json_valid" in ids

    def test_json_valid_for_list_output(self):
        calls = [_make_call(output=[1, 2, 3])]
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "json_valid" in ids

    def test_no_json_valid_for_string_output(self):
        calls = [_make_call(output="just a string")]
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "json_valid" not in ids

    def test_tool_call_count_when_tool_events_present(self):
        events = [
            TraceEvent(
                kind="tool_call",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                detail={"tool": "search"},
            )
        ]
        calls = [_make_call(output="result", trace_events=events)]
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "tool_call_count" in ids

    def test_no_tool_call_count_without_tool_events(self):
        events = [
            TraceEvent(
                kind="llm_response",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                detail={"model": "gpt-4"},
            )
        ]
        calls = [_make_call(output="result", trace_events=events)]
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "tool_call_count" not in ids


class TestHeuristicSuggesterReference:
    """Test reference-based metric suggestions."""

    def test_token_overlap_when_has_reference_and_text_output(self):
        calls = [_make_call(output="Python is a language")]
        s = HeuristicSuggester(has_reference=True)
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "token_overlap_f1" in ids

    def test_no_token_overlap_without_reference_flag(self):
        calls = [_make_call(output="Python is a language")]
        s = HeuristicSuggester(has_reference=False)
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "token_overlap_f1" not in ids

    def test_no_token_overlap_for_dict_output_even_with_reference(self):
        calls = [_make_call(output={"structured": True})]
        s = HeuristicSuggester(has_reference=True)
        specs = s.suggest(_dummy_fn, traces=calls)
        ids = [sp.id for sp in specs]
        assert "token_overlap_f1" not in ids


class TestHeuristicSuggesterSpecTypes:
    """Test that returned specs have correct types."""

    def test_all_specs_are_metric_spec(self):
        calls = [
            _make_call(output="text response"),
            _make_call(output={"json": True}),
        ]
        s = HeuristicSuggester(has_reference=True)
        specs = s.suggest(_dummy_fn, traces=calls)
        for sp in specs:
            assert isinstance(sp, MetricSpec)
            assert sp.id
            assert sp.name
            assert sp.type in ("objective", "subjective")

    def test_helpfulness_is_subjective(self):
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn)
        helpfulness = [sp for sp in specs if sp.id == "helpfulness_accuracy"][0]
        assert helpfulness.type == "subjective"

    def test_latency_is_objective(self):
        s = HeuristicSuggester()
        specs = s.suggest(_dummy_fn)
        latency = [sp for sp in specs if sp.id == "latency_ms"][0]
        assert latency.type == "objective"


# ---------------------------------------------------------------------------
# Subjective template validation
# ---------------------------------------------------------------------------

from evalyn_sdk.metrics.subjective import SUBJECTIVE_REGISTRY, CATEGORIES


class TestSubjectiveTemplateStructure:
    """Validate that all subjective templates have required fields."""

    REQUIRED_KEYS = {"id", "type", "description", "config"}
    REQUIRED_CONFIG_KEYS = {"rubric"}

    def test_registry_not_empty(self):
        assert len(SUBJECTIVE_REGISTRY) > 0

    def test_all_templates_have_required_keys(self):
        for tpl in SUBJECTIVE_REGISTRY:
            missing = self.REQUIRED_KEYS - set(tpl.keys())
            assert not missing, f"Template {tpl.get('id', '?')} missing: {missing}"

    def test_all_templates_are_subjective_type(self):
        for tpl in SUBJECTIVE_REGISTRY:
            assert tpl["type"] == "subjective", (
                f"Template {tpl['id']} has type={tpl['type']}"
            )

    def test_all_configs_have_rubric(self):
        for tpl in SUBJECTIVE_REGISTRY:
            config = tpl.get("config", {})
            assert "rubric" in config, f"Template {tpl['id']} missing rubric in config"
            assert isinstance(config["rubric"], list), (
                f"Template {tpl['id']} rubric is not a list"
            )
            assert len(config["rubric"]) >= 1, (
                f"Template {tpl['id']} rubric is empty"
            )

    def test_no_duplicate_ids(self):
        ids = [tpl["id"] for tpl in SUBJECTIVE_REGISTRY]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[i for i in ids if ids.count(i) > 1]}"

    def test_all_ids_are_snake_case(self):
        for tpl in SUBJECTIVE_REGISTRY:
            assert re.match(r"^[a-z][a-z0-9_]*$", tpl["id"]), (
                f"Template {tpl['id']} is not snake_case"
            )


class TestSubjectiveCategories:
    """Validate category definitions."""

    def test_categories_not_empty(self):
        assert len(CATEGORIES) > 0

    def test_all_templates_have_valid_category(self):
        for tpl in SUBJECTIVE_REGISTRY:
            if "category" in tpl:
                assert tpl["category"] in CATEGORIES, (
                    f"Template {tpl['id']} has unknown category: {tpl['category']}"
                )
