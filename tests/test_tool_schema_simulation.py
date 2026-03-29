"""Tests for the tool schema simulation module."""

from __future__ import annotations

import random

from evalyn_sdk.tool_schema_simulation import (
    SimulatedToolInput,
    ToolCoverageResult,
    ToolParameter,
    ToolSchema,
    compute_tool_coverage,
    format_coverage_report,
    generate_multi_tool_query,
    generate_tool_query,
    generate_tool_suite,
    suggest_missing_queries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(name: str, desc: str = "do something", params: list[ToolParameter] | None = None) -> ToolSchema:
    if params is None:
        params = [ToolParameter(name="query", param_type="string")]
    return ToolSchema(tool_name=name, description=desc, parameters=params)


def _search_schema() -> ToolSchema:
    return _make_schema(
        "search",
        "search the web",
        [ToolParameter(name="query", param_type="string", description="search query")],
    )


def _calc_schema() -> ToolSchema:
    return _make_schema(
        "calculator",
        "perform calculations",
        [
            ToolParameter(name="expression", param_type="string"),
            ToolParameter(name="precision", param_type="int", required=False),
        ],
    )


def _weather_schema() -> ToolSchema:
    return _make_schema(
        "weather",
        "get weather forecast",
        [
            ToolParameter(name="city", param_type="string"),
            ToolParameter(name="days", param_type="int"),
        ],
    )


# ---------------------------------------------------------------------------
# ToolParameter tests
# ---------------------------------------------------------------------------


class TestToolParameter:
    def test_defaults(self):
        p = ToolParameter(name="x", param_type="string")
        assert p.required is True
        assert p.description == ""

    def test_custom(self):
        p = ToolParameter(name="count", param_type="int", required=False, description="number of items")
        assert p.param_type == "int"
        assert p.required is False

    def test_as_dict(self):
        p = ToolParameter(name="q", param_type="string", description="query")
        d = p.as_dict()
        assert d["name"] == "q"
        assert d["param_type"] == "string"
        assert d["required"] is True

    def test_from_dict(self):
        d = {"name": "flag", "param_type": "bool", "required": False, "description": "toggle"}
        p = ToolParameter.from_dict(d)
        assert p.name == "flag"
        assert p.param_type == "bool"
        assert p.required is False

    def test_from_dict_defaults(self):
        d = {"name": "x", "param_type": "float"}
        p = ToolParameter.from_dict(d)
        assert p.required is True
        assert p.description == ""

    def test_roundtrip(self):
        p = ToolParameter(name="items", param_type="array", required=True, description="list of items")
        restored = ToolParameter.from_dict(p.as_dict())
        assert restored.name == p.name
        assert restored.param_type == p.param_type
        assert restored.required == p.required
        assert restored.description == p.description


# ---------------------------------------------------------------------------
# ToolSchema tests
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_basic(self):
        s = _search_schema()
        assert s.tool_name == "search"
        assert len(s.parameters) == 1

    def test_as_dict(self):
        s = _calc_schema()
        d = s.as_dict()
        assert d["tool_name"] == "calculator"
        assert len(d["parameters"]) == 2
        assert d["parameters"][0]["name"] == "expression"

    def test_from_dict(self):
        d = {
            "tool_name": "translate",
            "description": "translate text",
            "parameters": [
                {"name": "text", "param_type": "string"},
                {"name": "target_lang", "param_type": "string"},
            ],
        }
        s = ToolSchema.from_dict(d)
        assert s.tool_name == "translate"
        assert len(s.parameters) == 2

    def test_from_dict_no_params(self):
        d = {"tool_name": "ping", "description": "check connectivity"}
        s = ToolSchema.from_dict(d)
        assert s.parameters == []

    def test_roundtrip(self):
        s = _weather_schema()
        restored = ToolSchema.from_dict(s.as_dict())
        assert restored.tool_name == s.tool_name
        assert len(restored.parameters) == len(s.parameters)


# ---------------------------------------------------------------------------
# ToolCoverageResult tests
# ---------------------------------------------------------------------------


class TestToolCoverageResult:
    def test_basic(self):
        r = ToolCoverageResult(
            total_tools=3, exercised_tools=2, coverage_rate=2 / 3,
            uncovered_tools=["weather"], per_tool_counts={"search": 3, "calc": 2, "weather": 0},
        )
        assert r.coverage_rate < 1.0
        assert r.uncovered_tools == ["weather"]

    def test_as_dict(self):
        r = ToolCoverageResult(total_tools=1, exercised_tools=1, coverage_rate=1.0, per_tool_counts={"a": 5})
        d = r.as_dict()
        assert d["total_tools"] == 1
        assert d["per_tool_counts"]["a"] == 5

    def test_from_dict(self):
        d = {
            "total_tools": 2, "exercised_tools": 1, "coverage_rate": 0.5,
            "uncovered_tools": ["b"], "per_tool_counts": {"a": 3, "b": 0},
        }
        r = ToolCoverageResult.from_dict(d)
        assert r.uncovered_tools == ["b"]

    def test_roundtrip(self):
        r = ToolCoverageResult(
            total_tools=4, exercised_tools=3, coverage_rate=0.75,
            uncovered_tools=["d"], per_tool_counts={"a": 1, "b": 2, "c": 3, "d": 0},
        )
        restored = ToolCoverageResult.from_dict(r.as_dict())
        assert restored.total_tools == r.total_tools
        assert restored.uncovered_tools == r.uncovered_tools


# ---------------------------------------------------------------------------
# SimulatedToolInput tests
# ---------------------------------------------------------------------------


class TestSimulatedToolInput:
    def test_basic(self):
        inp = SimulatedToolInput(input_text="search for cats", target_tool="search")
        assert inp.expected_params == {}
        assert inp.tags == []

    def test_as_dict(self):
        inp = SimulatedToolInput(
            input_text="calc 2+2", target_tool="calculator",
            expected_params={"expression": "2+2"}, tags=["math"],
        )
        d = inp.as_dict()
        assert d["target_tool"] == "calculator"
        assert d["expected_params"]["expression"] == "2+2"

    def test_from_dict(self):
        d = {
            "input_text": "hello", "target_tool": "greeting",
            "expected_params": {}, "tags": ["test"],
        }
        inp = SimulatedToolInput.from_dict(d)
        assert inp.target_tool == "greeting"

    def test_roundtrip(self):
        inp = SimulatedToolInput(
            input_text="query", target_tool="search",
            expected_params={"q": "test"}, tags=["a", "b"],
        )
        restored = SimulatedToolInput.from_dict(inp.as_dict())
        assert restored.input_text == inp.input_text
        assert restored.tags == inp.tags


# ---------------------------------------------------------------------------
# generate_tool_query tests
# ---------------------------------------------------------------------------


class TestGenerateToolQuery:
    def test_returns_simulated_input(self):
        schema = _search_schema()
        result = generate_tool_query(schema, rng=random.Random(42))
        assert isinstance(result, SimulatedToolInput)

    def test_target_tool_matches(self):
        schema = _search_schema()
        result = generate_tool_query(schema, rng=random.Random(42))
        assert result.target_tool == "search"

    def test_contains_tool_name_in_query(self):
        schema = _search_schema()
        result = generate_tool_query(schema, rng=random.Random(42))
        assert "search" in result.input_text.lower()

    def test_required_params_in_expected(self):
        schema = _weather_schema()
        result = generate_tool_query(schema, rng=random.Random(42))
        assert "city" in result.expected_params
        assert "days" in result.expected_params

    def test_optional_params_excluded(self):
        schema = _calc_schema()
        result = generate_tool_query(schema, rng=random.Random(42))
        assert "expression" in result.expected_params
        assert "precision" not in result.expected_params

    def test_tags_include_tool_name(self):
        schema = _search_schema()
        result = generate_tool_query(schema, rng=random.Random(42))
        assert "search" in result.tags
        assert "single-tool" in result.tags

    def test_deterministic_with_seed(self):
        schema = _search_schema()
        r1 = generate_tool_query(schema, rng=random.Random(99))
        r2 = generate_tool_query(schema, rng=random.Random(99))
        assert r1.input_text == r2.input_text

    def test_no_rng_works(self):
        schema = _search_schema()
        result = generate_tool_query(schema)
        assert result.target_tool == "search"

    def test_no_params_schema(self):
        schema = ToolSchema(tool_name="ping", description="check connectivity", parameters=[])
        result = generate_tool_query(schema, rng=random.Random(1))
        assert result.expected_params == {}


# ---------------------------------------------------------------------------
# generate_tool_suite tests
# ---------------------------------------------------------------------------


class TestGenerateToolSuite:
    def test_correct_count(self):
        schemas = [_search_schema(), _calc_schema()]
        results = generate_tool_suite(schemas, queries_per_tool=3)
        assert len(results) == 6

    def test_single_query_per_tool(self):
        schemas = [_search_schema()]
        results = generate_tool_suite(schemas, queries_per_tool=1)
        assert len(results) == 1
        assert results[0].target_tool == "search"

    def test_all_tools_covered(self):
        schemas = [_search_schema(), _calc_schema(), _weather_schema()]
        results = generate_tool_suite(schemas, queries_per_tool=2)
        tools = {r.target_tool for r in results}
        assert tools == {"search", "calculator", "weather"}

    def test_deterministic(self):
        schemas = [_search_schema(), _calc_schema()]
        r1 = generate_tool_suite(schemas, queries_per_tool=2)
        r2 = generate_tool_suite(schemas, queries_per_tool=2)
        assert [r.input_text for r in r1] == [r.input_text for r in r2]


# ---------------------------------------------------------------------------
# compute_tool_coverage tests
# ---------------------------------------------------------------------------


class TestComputeToolCoverage:
    def test_full_coverage(self):
        schemas = [_search_schema(), _calc_schema()]
        inputs = [
            SimulatedToolInput(input_text="q1", target_tool="search"),
            SimulatedToolInput(input_text="q2", target_tool="calculator"),
        ]
        result = compute_tool_coverage(inputs, schemas)
        assert result.coverage_rate == 1.0
        assert result.uncovered_tools == []

    def test_partial_coverage(self):
        schemas = [_search_schema(), _calc_schema(), _weather_schema()]
        inputs = [SimulatedToolInput(input_text="q", target_tool="search")]
        result = compute_tool_coverage(inputs, schemas)
        assert result.exercised_tools == 1
        assert result.total_tools == 3
        assert len(result.uncovered_tools) == 2

    def test_no_inputs(self):
        schemas = [_search_schema()]
        result = compute_tool_coverage([], schemas)
        assert result.exercised_tools == 0
        assert result.coverage_rate == 0.0

    def test_multi_tool_input_counted(self):
        schemas = [_search_schema(), _calc_schema()]
        inputs = [
            SimulatedToolInput(input_text="q", target_tool="search, calculator"),
        ]
        result = compute_tool_coverage(inputs, schemas)
        assert result.coverage_rate == 1.0
        assert result.per_tool_counts["search"] == 1
        assert result.per_tool_counts["calculator"] == 1

    def test_per_tool_counts(self):
        schemas = [_search_schema()]
        inputs = [
            SimulatedToolInput(input_text="q1", target_tool="search"),
            SimulatedToolInput(input_text="q2", target_tool="search"),
            SimulatedToolInput(input_text="q3", target_tool="search"),
        ]
        result = compute_tool_coverage(inputs, schemas)
        assert result.per_tool_counts["search"] == 3

    def test_no_schemas(self):
        result = compute_tool_coverage([], [])
        assert result.total_tools == 0
        assert result.coverage_rate == 0.0


# ---------------------------------------------------------------------------
# generate_multi_tool_query tests
# ---------------------------------------------------------------------------


class TestGenerateMultiToolQuery:
    def test_two_tools(self):
        schemas = [_search_schema(), _calc_schema()]
        result = generate_multi_tool_query(schemas, n_tools=2, rng=random.Random(42))
        assert isinstance(result, SimulatedToolInput)
        assert "multi-tool" in result.tags

    def test_target_contains_both_tools(self):
        schemas = [_search_schema(), _calc_schema()]
        result = generate_multi_tool_query(schemas, n_tools=2, rng=random.Random(42))
        targets = [t.strip() for t in result.target_tool.split(",")]
        assert len(targets) == 2

    def test_three_tools(self):
        schemas = [_search_schema(), _calc_schema(), _weather_schema()]
        result = generate_multi_tool_query(schemas, n_tools=3, rng=random.Random(42))
        targets = [t.strip() for t in result.target_tool.split(",")]
        assert len(targets) == 3

    def test_n_tools_clamped_to_available(self):
        schemas = [_search_schema()]
        result = generate_multi_tool_query(schemas, n_tools=5, rng=random.Random(42))
        assert result.target_tool == "search"

    def test_deterministic_with_seed(self):
        schemas = [_search_schema(), _calc_schema(), _weather_schema()]
        r1 = generate_multi_tool_query(schemas, n_tools=2, rng=random.Random(7))
        r2 = generate_multi_tool_query(schemas, n_tools=2, rng=random.Random(7))
        assert r1.input_text == r2.input_text

    def test_no_rng_works(self):
        schemas = [_search_schema(), _calc_schema()]
        result = generate_multi_tool_query(schemas)
        assert "multi-tool" in result.tags


# ---------------------------------------------------------------------------
# format_coverage_report tests
# ---------------------------------------------------------------------------


class TestFormatCoverageReport:
    def test_basic_report(self):
        r = ToolCoverageResult(
            total_tools=3, exercised_tools=2, coverage_rate=2 / 3,
            uncovered_tools=["weather"],
            per_tool_counts={"calculator": 2, "search": 3, "weather": 0},
        )
        report = format_coverage_report(r)
        assert "Tool Coverage Report" in report
        assert "66.67%" in report
        assert "weather" in report

    def test_full_coverage_report(self):
        r = ToolCoverageResult(
            total_tools=2, exercised_tools=2, coverage_rate=1.0,
            uncovered_tools=[], per_tool_counts={"a": 1, "b": 1},
        )
        report = format_coverage_report(r)
        assert "100.00%" in report
        assert "Uncovered" not in report

    def test_empty_tools(self):
        r = ToolCoverageResult(
            total_tools=0, exercised_tools=0, coverage_rate=0.0,
            uncovered_tools=[], per_tool_counts={},
        )
        report = format_coverage_report(r)
        assert "Total tools: 0" in report


# ---------------------------------------------------------------------------
# suggest_missing_queries tests
# ---------------------------------------------------------------------------


class TestSuggestMissingQueries:
    def test_all_covered_returns_empty(self):
        r = ToolCoverageResult(
            total_tools=2, exercised_tools=2, coverage_rate=1.0,
            uncovered_tools=[], per_tool_counts={"search": 1, "calc": 1},
        )
        schemas = [_search_schema(), _calc_schema()]
        suggestions = suggest_missing_queries(r, schemas)
        assert suggestions == []

    def test_uncovered_gets_suggestions(self):
        r = ToolCoverageResult(
            total_tools=2, exercised_tools=1, coverage_rate=0.5,
            uncovered_tools=["calculator"],
            per_tool_counts={"search": 3, "calculator": 0},
        )
        schemas = [_search_schema(), _calc_schema()]
        suggestions = suggest_missing_queries(r, schemas)
        assert len(suggestions) == 1
        assert suggestions[0].target_tool == "calculator"

    def test_multiple_uncovered(self):
        r = ToolCoverageResult(
            total_tools=3, exercised_tools=1, coverage_rate=1 / 3,
            uncovered_tools=["calculator", "weather"],
            per_tool_counts={"search": 1, "calculator": 0, "weather": 0},
        )
        schemas = [_search_schema(), _calc_schema(), _weather_schema()]
        suggestions = suggest_missing_queries(r, schemas)
        assert len(suggestions) == 2
        tool_names = {s.target_tool for s in suggestions}
        assert tool_names == {"calculator", "weather"}

    def test_suggestions_are_valid_inputs(self):
        r = ToolCoverageResult(
            total_tools=1, exercised_tools=0, coverage_rate=0.0,
            uncovered_tools=["search"],
            per_tool_counts={"search": 0},
        )
        schemas = [_search_schema()]
        suggestions = suggest_missing_queries(r, schemas)
        assert len(suggestions) == 1
        assert suggestions[0].input_text  # non-empty
        assert suggestions[0].target_tool == "search"

    def test_deterministic(self):
        r = ToolCoverageResult(
            total_tools=1, exercised_tools=0, coverage_rate=0.0,
            uncovered_tools=["search"],
            per_tool_counts={"search": 0},
        )
        schemas = [_search_schema()]
        s1 = suggest_missing_queries(r, schemas)
        s2 = suggest_missing_queries(r, schemas)
        assert s1[0].input_text == s2[0].input_text
