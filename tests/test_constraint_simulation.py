"""Tests for constraint_simulation module."""

from __future__ import annotations

import pytest

from evalyn_sdk.constraint_simulation import (
    Constraint,
    ConstraintCheckResult,
    ConstraintSet,
    check_constraint,
    check_constraints,
    count_passing,
    filter_by_constraints,
    format_constraint_summary,
    generate_constrained_text,
    parse_constraint,
    parse_constraint_string,
)


# ---------------------------------------------------------------------------
# Constraint dataclass
# ---------------------------------------------------------------------------


class TestConstraint:
    def test_basic_fields(self):
        c = Constraint(field="age", operator="gt", value=18)
        assert c.field == "age"
        assert c.operator == "gt"
        assert c.value == 18

    def test_as_dict(self):
        c = Constraint(field="name", operator="eq", value="alice")
        d = c.as_dict()
        assert d == {"field": "name", "operator": "eq", "value": "alice"}

    def test_from_dict(self):
        c = Constraint.from_dict({"field": "score", "operator": "gte", "value": 0.5})
        assert c.field == "score"
        assert c.operator == "gte"
        assert c.value == 0.5

    def test_roundtrip(self):
        original = Constraint(field="x", operator="ne", value=42)
        restored = Constraint.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# ConstraintSet dataclass
# ---------------------------------------------------------------------------


class TestConstraintSet:
    def test_defaults(self):
        cs = ConstraintSet()
        assert cs.constraints == []
        assert cs.logic == "AND"

    def test_as_dict(self):
        cs = ConstraintSet(
            constraints=[Constraint("a", "eq", 1)],
            logic="OR",
        )
        d = cs.as_dict()
        assert d["logic"] == "OR"
        assert len(d["constraints"]) == 1

    def test_from_dict(self):
        data = {
            "constraints": [{"field": "x", "operator": "gt", "value": 5}],
            "logic": "AND",
        }
        cs = ConstraintSet.from_dict(data)
        assert len(cs.constraints) == 1
        assert cs.logic == "AND"

    def test_roundtrip(self):
        cs = ConstraintSet(
            constraints=[Constraint("a", "eq", 1), Constraint("b", "lt", 10)],
            logic="OR",
        )
        restored = ConstraintSet.from_dict(cs.as_dict())
        assert restored == cs


# ---------------------------------------------------------------------------
# ConstraintCheckResult dataclass
# ---------------------------------------------------------------------------


class TestConstraintCheckResult:
    def test_basic(self):
        r = ConstraintCheckResult(item_id="item1", passed=True, all_constraints=3)
        assert r.passed is True
        assert r.failed_constraints == []

    def test_as_dict(self):
        r = ConstraintCheckResult(
            item_id="x",
            passed=False,
            failed_constraints=["age gt 18"],
            all_constraints=2,
        )
        d = r.as_dict()
        assert d["item_id"] == "x"
        assert d["passed"] is False
        assert d["failed_constraints"] == ["age gt 18"]

    def test_from_dict(self):
        data = {
            "item_id": "y",
            "passed": True,
            "failed_constraints": [],
            "all_constraints": 1,
        }
        r = ConstraintCheckResult.from_dict(data)
        assert r.item_id == "y"
        assert r.passed is True

    def test_roundtrip(self):
        original = ConstraintCheckResult("z", False, ["a eq b"], 3)
        restored = ConstraintCheckResult.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# parse_constraint
# ---------------------------------------------------------------------------


class TestParseConstraint:
    def test_eq(self):
        c = parse_constraint("name=alice")
        assert c.field == "name"
        assert c.operator == "eq"
        assert c.value == "alice"

    def test_ne(self):
        c = parse_constraint("status!=active")
        assert c.operator == "ne"
        assert c.value == "active"

    def test_gt_numeric(self):
        c = parse_constraint("age>18")
        assert c.operator == "gt"
        assert c.value == 18

    def test_lt_numeric(self):
        c = parse_constraint("score<0.5")
        assert c.operator == "lt"
        assert c.value == 0.5

    def test_gte(self):
        c = parse_constraint("count>=10")
        assert c.operator == "gte"
        assert c.value == 10

    def test_lte(self):
        c = parse_constraint("priority<=3")
        assert c.operator == "lte"
        assert c.value == 3

    def test_contains(self):
        c = parse_constraint("text~=hello")
        assert c.operator == "contains"
        assert c.value == "hello"

    def test_not_contains(self):
        c = parse_constraint("body!~spam")
        assert c.operator == "not_contains"
        assert c.value == "spam"

    def test_whitespace_handling(self):
        c = parse_constraint("  field = value  ")
        assert c.field == "field"
        assert c.value == "value"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_constraint("no_operator_here")


# ---------------------------------------------------------------------------
# parse_constraint_string
# ---------------------------------------------------------------------------


class TestParseConstraintString:
    def test_single(self):
        cs = parse_constraint_string("age>18")
        assert len(cs.constraints) == 1
        assert cs.logic == "AND"

    def test_and(self):
        cs = parse_constraint_string("age>18 AND name=alice")
        assert len(cs.constraints) == 2
        assert cs.logic == "AND"

    def test_or(self):
        cs = parse_constraint_string("status=active OR status=pending")
        assert len(cs.constraints) == 2
        assert cs.logic == "OR"

    def test_empty(self):
        cs = parse_constraint_string("")
        assert len(cs.constraints) == 0

    def test_triple_and(self):
        cs = parse_constraint_string("a=1 AND b=2 AND c=3")
        assert len(cs.constraints) == 3
        assert cs.logic == "AND"


# ---------------------------------------------------------------------------
# check_constraint
# ---------------------------------------------------------------------------


class TestCheckConstraint:
    def test_eq_match(self):
        assert check_constraint({"x": "hello"}, Constraint("x", "eq", "hello")) is True

    def test_eq_no_match(self):
        assert check_constraint({"x": "hello"}, Constraint("x", "eq", "world")) is False

    def test_ne_match(self):
        assert check_constraint({"x": "a"}, Constraint("x", "ne", "b")) is True

    def test_gt_numeric(self):
        assert check_constraint({"score": 0.8}, Constraint("score", "gt", 0.5)) is True
        assert check_constraint({"score": 0.3}, Constraint("score", "gt", 0.5)) is False

    def test_lt_numeric(self):
        assert check_constraint({"val": 2}, Constraint("val", "lt", 5)) is True
        assert check_constraint({"val": 10}, Constraint("val", "lt", 5)) is False

    def test_gte(self):
        assert check_constraint({"n": 5}, Constraint("n", "gte", 5)) is True
        assert check_constraint({"n": 4}, Constraint("n", "gte", 5)) is False

    def test_lte(self):
        assert check_constraint({"n": 5}, Constraint("n", "lte", 5)) is True
        assert check_constraint({"n": 6}, Constraint("n", "lte", 5)) is False

    def test_contains(self):
        assert check_constraint({"t": "hello world"}, Constraint("t", "contains", "world")) is True
        assert check_constraint({"t": "hello"}, Constraint("t", "contains", "world")) is False

    def test_contains_case_insensitive(self):
        assert check_constraint({"t": "Hello World"}, Constraint("t", "contains", "hello")) is True

    def test_not_contains(self):
        assert check_constraint({"t": "hello"}, Constraint("t", "not_contains", "world")) is True
        assert check_constraint({"t": "hello world"}, Constraint("t", "not_contains", "world")) is False

    def test_missing_field(self):
        assert check_constraint({}, Constraint("x", "eq", 1)) is False

    def test_numeric_string(self):
        assert check_constraint({"age": "25"}, Constraint("age", "gt", 18)) is True

    def test_eq_numeric_coercion(self):
        assert check_constraint({"val": "5"}, Constraint("val", "eq", 5)) is True


# ---------------------------------------------------------------------------
# check_constraints
# ---------------------------------------------------------------------------


class TestCheckConstraints:
    def test_and_all_pass(self):
        cs = ConstraintSet([Constraint("a", "eq", 1), Constraint("b", "eq", 2)], "AND")
        item = {"id": "i1", "a": 1, "b": 2}
        result = check_constraints(item, cs)
        assert result.passed is True
        assert result.failed_constraints == []
        assert result.all_constraints == 2

    def test_and_one_fails(self):
        cs = ConstraintSet([Constraint("a", "eq", 1), Constraint("b", "eq", 99)], "AND")
        item = {"id": "i1", "a": 1, "b": 2}
        result = check_constraints(item, cs)
        assert result.passed is False
        assert len(result.failed_constraints) == 1

    def test_or_one_passes(self):
        cs = ConstraintSet([Constraint("a", "eq", 1), Constraint("b", "eq", 99)], "OR")
        item = {"id": "i2", "a": 1, "b": 2}
        result = check_constraints(item, cs)
        assert result.passed is True

    def test_or_none_pass(self):
        cs = ConstraintSet([Constraint("a", "eq", 99), Constraint("b", "eq", 99)], "OR")
        item = {"id": "i3", "a": 1, "b": 2}
        result = check_constraints(item, cs)
        assert result.passed is False

    def test_empty_constraints(self):
        cs = ConstraintSet([], "AND")
        result = check_constraints({"id": "i4"}, cs)
        assert result.passed is True

    def test_item_id_fallback(self):
        cs = ConstraintSet([], "AND")
        result = check_constraints({"no_id": True}, cs)
        assert result.item_id == "unknown"


# ---------------------------------------------------------------------------
# filter_by_constraints
# ---------------------------------------------------------------------------


class TestFilterByConstraints:
    def test_filters_correctly(self):
        items = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.3},
            {"id": "3", "score": 0.7},
        ]
        cs = ConstraintSet([Constraint("score", "gt", 0.5)])
        result = filter_by_constraints(items, cs)
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "1" in ids
        assert "3" in ids

    def test_empty_items(self):
        cs = ConstraintSet([Constraint("x", "eq", 1)])
        assert filter_by_constraints([], cs) == []

    def test_no_constraints(self):
        items = [{"id": "a"}, {"id": "b"}]
        cs = ConstraintSet()
        assert len(filter_by_constraints(items, cs)) == 2


# ---------------------------------------------------------------------------
# generate_constrained_text
# ---------------------------------------------------------------------------


class TestGenerateConstrainedText:
    def test_min_length_gte(self):
        cs = ConstraintSet([Constraint("length", "gte", 50)])
        result = generate_constrained_text("short", cs)
        assert len(result) >= 50

    def test_max_length_lte(self):
        cs = ConstraintSet([Constraint("max_length", "lte", 10)])
        result = generate_constrained_text("this is a long text that should be truncated", cs)
        assert len(result) <= 10

    def test_topic_contains(self):
        cs = ConstraintSet([Constraint("topic", "contains", "python")])
        result = generate_constrained_text("hello world", cs)
        assert "python" in result.lower()

    def test_topic_not_contains(self):
        cs = ConstraintSet([Constraint("topic", "not_contains", "hello")])
        result = generate_constrained_text("hello world", cs)
        assert "hello" not in result.lower()

    def test_contains_prepend(self):
        cs = ConstraintSet([Constraint("content", "contains", "keyword")])
        result = generate_constrained_text("base text", cs)
        assert "keyword" in result.lower()

    def test_not_contains_removal(self):
        cs = ConstraintSet([Constraint("content", "not_contains", "bad")])
        result = generate_constrained_text("this is bad content", cs)
        assert "bad" not in result.lower()

    def test_length_eq(self):
        cs = ConstraintSet([Constraint("length", "eq", 20)])
        result = generate_constrained_text("hello world test", cs)
        assert len(result) == 20

    def test_no_constraints(self):
        result = generate_constrained_text("hello", ConstraintSet())
        assert result == "hello"


# ---------------------------------------------------------------------------
# format_constraint_summary
# ---------------------------------------------------------------------------


class TestFormatConstraintSummary:
    def test_empty(self):
        result = format_constraint_summary(ConstraintSet())
        assert "No constraints" in result

    def test_with_constraints(self):
        cs = ConstraintSet([
            Constraint("age", "gt", 18),
            Constraint("name", "eq", "alice"),
        ])
        result = format_constraint_summary(cs)
        assert "AND logic" in result
        assert "age" in result
        assert "name" in result
        assert "2 constraint(s)" in result

    def test_or_logic(self):
        cs = ConstraintSet([Constraint("a", "eq", 1)], logic="OR")
        result = format_constraint_summary(cs)
        assert "OR logic" in result


# ---------------------------------------------------------------------------
# count_passing
# ---------------------------------------------------------------------------


class TestCountPassing:
    def test_count(self):
        items = [
            {"id": "1", "v": 10},
            {"id": "2", "v": 20},
            {"id": "3", "v": 30},
        ]
        cs = ConstraintSet([Constraint("v", "gte", 20)])
        assert count_passing(items, cs) == 2

    def test_zero(self):
        items = [{"id": "1", "v": 1}]
        cs = ConstraintSet([Constraint("v", "gt", 100)])
        assert count_passing(items, cs) == 0

    def test_all(self):
        items = [{"id": "1", "v": 5}, {"id": "2", "v": 10}]
        cs = ConstraintSet([Constraint("v", "gt", 0)])
        assert count_passing(items, cs) == 2
