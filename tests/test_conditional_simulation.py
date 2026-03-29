"""Tests for conditional_simulation module."""

from __future__ import annotations

import random

import pytest

from evalyn_sdk.conditional_simulation import (
    EDGE_CONDITIONS,
    ConditionalConfig,
    ConditionalInput,
    EdgeCondition,
    format_conditional_report,
    generate_all_conditions,
    generate_conditional_inputs,
    generate_empty,
    generate_html_tags,
    generate_max_length,
    generate_mixed_language,
    generate_null_like,
    generate_special_chars,
    generate_sql_injection,
    generate_unicode_cjk,
    generate_unicode_emoji,
    generate_whitespace_only,
)


# ---------------------------------------------------------------------------
# EdgeCondition dataclass
# ---------------------------------------------------------------------------


class TestEdgeCondition:
    def test_basic_fields(self):
        ec = EdgeCondition(
            condition_id="test",
            name="Test",
            description="A test",
            generator="gen_fn",
        )
        assert ec.condition_id == "test"
        assert ec.severity == "medium"

    def test_custom_severity(self):
        ec = EdgeCondition(
            condition_id="x", name="X", description="x", generator="g",
            severity="critical",
        )
        assert ec.severity == "critical"

    def test_as_dict(self):
        ec = EdgeCondition(
            condition_id="a", name="A", description="d", generator="g",
            severity="high",
        )
        d = ec.as_dict()
        assert d["condition_id"] == "a"
        assert d["severity"] == "high"
        assert set(d.keys()) == {"condition_id", "name", "description", "generator", "severity"}

    def test_from_dict(self):
        d = {
            "condition_id": "b",
            "name": "B",
            "description": "desc",
            "generator": "gen",
            "severity": "low",
        }
        ec = EdgeCondition.from_dict(d)
        assert ec.condition_id == "b"
        assert ec.severity == "low"

    def test_from_dict_default_severity(self):
        d = {"condition_id": "c", "name": "C", "description": "d", "generator": "g"}
        ec = EdgeCondition.from_dict(d)
        assert ec.severity == "medium"

    def test_roundtrip(self):
        ec = EdgeCondition(
            condition_id="rt", name="RT", description="round trip",
            generator="gen", severity="critical",
        )
        assert EdgeCondition.from_dict(ec.as_dict()) == ec


# ---------------------------------------------------------------------------
# ConditionalInput dataclass
# ---------------------------------------------------------------------------


class TestConditionalInput:
    def test_basic(self):
        ci = ConditionalInput(input_text="hello")
        assert ci.input_text == "hello"
        assert ci.conditions_applied == []
        assert ci.tags == []

    def test_with_fields(self):
        ci = ConditionalInput(
            input_text="x", conditions_applied=["a", "b"], tags=["high"],
        )
        assert ci.conditions_applied == ["a", "b"]

    def test_as_dict(self):
        ci = ConditionalInput(
            input_text="t", conditions_applied=["c"], tags=["low"],
        )
        d = ci.as_dict()
        assert d["input_text"] == "t"
        assert d["conditions_applied"] == ["c"]

    def test_from_dict(self):
        d = {"input_text": "x", "conditions_applied": ["a"], "tags": ["b"]}
        ci = ConditionalInput.from_dict(d)
        assert ci.input_text == "x"

    def test_from_dict_defaults(self):
        ci = ConditionalInput.from_dict({"input_text": "y"})
        assert ci.conditions_applied == []
        assert ci.tags == []

    def test_roundtrip(self):
        ci = ConditionalInput(
            input_text="data", conditions_applied=["x"], tags=["medium"],
        )
        assert ConditionalInput.from_dict(ci.as_dict()) == ci


# ---------------------------------------------------------------------------
# ConditionalConfig dataclass
# ---------------------------------------------------------------------------


class TestConditionalConfig:
    def test_defaults(self):
        cc = ConditionalConfig()
        assert cc.conditions == []
        assert cc.base_text == ""
        assert cc.combinatorial is False

    def test_as_dict(self):
        cc = ConditionalConfig(
            conditions=["a"], base_text="base", combinatorial=True,
        )
        d = cc.as_dict()
        assert d["combinatorial"] is True
        assert d["base_text"] == "base"

    def test_from_dict(self):
        d = {"conditions": ["x"], "base_text": "b", "combinatorial": True}
        cc = ConditionalConfig.from_dict(d)
        assert cc.combinatorial is True

    def test_from_dict_defaults(self):
        cc = ConditionalConfig.from_dict({})
        assert cc.conditions == []
        assert cc.combinatorial is False

    def test_roundtrip(self):
        cc = ConditionalConfig(
            conditions=["a", "b"], base_text="hello", combinatorial=True,
        )
        assert ConditionalConfig.from_dict(cc.as_dict()) == cc


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------


class TestGenerators:
    def test_generate_empty(self):
        assert generate_empty() == ""

    def test_generate_null_like_with_rng(self):
        rng = random.Random(42)
        result = generate_null_like(rng)
        assert result in ("null", "None", "undefined")

    def test_generate_null_like_without_rng(self):
        result = generate_null_like()
        assert result in ("null", "None", "undefined")

    def test_generate_null_like_deterministic(self):
        r1 = generate_null_like(random.Random(99))
        r2 = generate_null_like(random.Random(99))
        assert r1 == r2

    def test_generate_max_length(self):
        result = generate_max_length()
        assert len(result) == 10000
        assert set(result) == {"a"}

    def test_generate_unicode_cjk(self):
        result = generate_unicode_cjk()
        assert len(result) > 0
        # Check for CJK chars
        assert "\u4f60" in result  # Chinese
        assert "\u3053" in result  # Japanese

    def test_generate_unicode_emoji(self):
        result = generate_unicode_emoji()
        assert "[emoji]" in result
        assert "[heart]" in result

    def test_generate_mixed_language(self):
        result = generate_mixed_language()
        assert "Hello" in result
        assert "\u4e16\u754c" in result  # Chinese
        assert "\u043f\u0440\u0438\u0432\u0435\u0442" in result  # Russian

    def test_generate_special_chars(self):
        result = generate_special_chars()
        assert "!" in result
        assert "@" in result
        assert "#" in result

    def test_generate_whitespace_only(self):
        result = generate_whitespace_only()
        assert result.strip() == ""
        assert "\t" in result
        assert "\n" in result

    def test_generate_sql_injection(self):
        result = generate_sql_injection()
        assert "DROP TABLE" in result
        assert "--" in result

    def test_generate_html_tags(self):
        result = generate_html_tags()
        assert "<script>" in result
        assert "alert" in result


# ---------------------------------------------------------------------------
# EDGE_CONDITIONS registry
# ---------------------------------------------------------------------------


class TestEdgeConditionsRegistry:
    def test_has_all_ten_conditions(self):
        expected = {
            "empty", "null_like", "max_length", "unicode_cjk",
            "unicode_emoji", "mixed_language", "special_chars",
            "whitespace_only", "sql_injection", "html_tags",
        }
        assert set(EDGE_CONDITIONS.keys()) == expected

    def test_all_have_valid_generators(self):
        from evalyn_sdk.conditional_simulation import _GENERATORS
        for cid, cond in EDGE_CONDITIONS.items():
            assert cond.generator in _GENERATORS, f"{cid} generator missing"

    def test_condition_ids_match_keys(self):
        for key, cond in EDGE_CONDITIONS.items():
            assert key == cond.condition_id


# ---------------------------------------------------------------------------
# generate_conditional_inputs
# ---------------------------------------------------------------------------


class TestGenerateConditionalInputs:
    def test_single_condition(self):
        cfg = ConditionalConfig(conditions=["empty"])
        results = generate_conditional_inputs(cfg)
        assert len(results) == 1
        assert results[0].conditions_applied == ["empty"]
        assert results[0].input_text == ""

    def test_multiple_conditions(self):
        cfg = ConditionalConfig(conditions=["empty", "max_length"])
        results = generate_conditional_inputs(cfg)
        assert len(results) == 2

    def test_with_base_text(self):
        cfg = ConditionalConfig(conditions=["empty"], base_text="PREFIX")
        results = generate_conditional_inputs(cfg)
        assert results[0].input_text.startswith("PREFIX ")

    def test_invalid_condition_skipped(self):
        cfg = ConditionalConfig(conditions=["nonexistent", "empty"])
        results = generate_conditional_inputs(cfg)
        assert len(results) == 1
        assert results[0].conditions_applied == ["empty"]

    def test_all_invalid_conditions(self):
        cfg = ConditionalConfig(conditions=["fake1", "fake2"])
        results = generate_conditional_inputs(cfg)
        assert len(results) == 0

    def test_empty_conditions(self):
        cfg = ConditionalConfig(conditions=[])
        results = generate_conditional_inputs(cfg)
        assert len(results) == 0

    def test_combinatorial_false(self):
        cfg = ConditionalConfig(
            conditions=["empty", "max_length"], combinatorial=False,
        )
        results = generate_conditional_inputs(cfg)
        assert len(results) == 2

    def test_combinatorial_true_two_conditions(self):
        cfg = ConditionalConfig(
            conditions=["empty", "max_length"], combinatorial=True,
        )
        results = generate_conditional_inputs(cfg)
        # 2 single + 1 combo = 3
        assert len(results) == 3
        combo = results[2]
        assert "combinatorial" in combo.tags
        assert sorted(combo.conditions_applied) == ["empty", "max_length"]

    def test_combinatorial_three_conditions(self):
        cfg = ConditionalConfig(
            conditions=["empty", "max_length", "sql_injection"],
            combinatorial=True,
        )
        results = generate_conditional_inputs(cfg)
        # 3 single + 3 combos = 6
        assert len(results) == 6

    def test_combinatorial_single_condition_no_combos(self):
        cfg = ConditionalConfig(
            conditions=["empty"], combinatorial=True,
        )
        results = generate_conditional_inputs(cfg)
        assert len(results) == 1

    def test_tags_include_severity(self):
        cfg = ConditionalConfig(conditions=["sql_injection"])
        results = generate_conditional_inputs(cfg)
        assert "critical" in results[0].tags

    def test_deterministic_with_rng(self):
        cfg = ConditionalConfig(conditions=["null_like"])
        r1 = generate_conditional_inputs(cfg, rng=random.Random(42))
        r2 = generate_conditional_inputs(cfg, rng=random.Random(42))
        assert r1[0].input_text == r2[0].input_text

    def test_combinatorial_base_text(self):
        cfg = ConditionalConfig(
            conditions=["empty", "special_chars"],
            base_text="BASE",
            combinatorial=True,
        )
        results = generate_conditional_inputs(cfg)
        combo = [r for r in results if "combinatorial" in r.tags]
        assert len(combo) == 1
        assert combo[0].input_text.startswith("BASE ")

    def test_combinatorial_severity_picks_higher(self):
        # sql_injection=critical, unicode_emoji=low - combo should be critical
        cfg = ConditionalConfig(
            conditions=["unicode_emoji", "sql_injection"],
            combinatorial=True,
        )
        results = generate_conditional_inputs(cfg)
        combo = [r for r in results if "combinatorial" in r.tags][0]
        assert "critical" in combo.tags


# ---------------------------------------------------------------------------
# generate_all_conditions
# ---------------------------------------------------------------------------


class TestGenerateAllConditions:
    def test_returns_all_conditions(self):
        results = generate_all_conditions()
        assert len(results) == len(EDGE_CONDITIONS)

    def test_with_base_text(self):
        results = generate_all_conditions(base_text="CTX")
        for r in results:
            assert r.input_text.startswith("CTX ")

    def test_no_combinatorial(self):
        results = generate_all_conditions()
        for r in results:
            assert "combinatorial" not in r.tags

    def test_each_condition_covered(self):
        results = generate_all_conditions()
        covered = set()
        for r in results:
            covered.update(r.conditions_applied)
        assert covered == set(EDGE_CONDITIONS.keys())


# ---------------------------------------------------------------------------
# format_conditional_report
# ---------------------------------------------------------------------------


class TestFormatConditionalReport:
    def test_empty_inputs(self):
        report = format_conditional_report([])
        assert "Total inputs generated: 0" in report

    def test_basic_report(self):
        inputs = generate_all_conditions()
        report = format_conditional_report(inputs)
        assert "Conditional Simulation Report" in report
        assert f"Total inputs generated: {len(inputs)}" in report

    def test_conditions_covered(self):
        inputs = [ConditionalInput(input_text="x", conditions_applied=["empty"], tags=["high"])]
        report = format_conditional_report(inputs)
        assert "Empty String: 1 input(s)" in report

    def test_combinatorial_count(self):
        inputs = [
            ConditionalInput(input_text="a", conditions_applied=["empty"], tags=["high"]),
            ConditionalInput(
                input_text="b",
                conditions_applied=["empty", "max_length"],
                tags=["high", "combinatorial"],
            ),
        ]
        report = format_conditional_report(inputs)
        assert "Combinatorial inputs: 1" in report

    def test_severity_breakdown(self):
        inputs = [
            ConditionalInput(input_text="a", conditions_applied=["sql_injection"], tags=["critical"]),
            ConditionalInput(input_text="b", conditions_applied=["empty"], tags=["high"]),
        ]
        report = format_conditional_report(inputs)
        assert "critical: 1" in report
        assert "high: 1" in report

    def test_unknown_condition_in_report(self):
        inputs = [ConditionalInput(input_text="x", conditions_applied=["unknown_cond"], tags=["medium"])]
        report = format_conditional_report(inputs)
        # Unknown conditions still show up by their ID
        assert "unknown_cond: 1 input(s)" in report
