"""Tests for DAG-based deterministic scoring."""

import pytest
from evalyn_sdk.evaluation.dag_metric import (
    DAGNode,
    DAGMetric,
    DAGResult,
    _evaluate_condition,
    evaluate_dag,
    build_simple_dag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(output="", inp=""):
    return {
        "input": inp,
        "output": output,
        "output_length": len(output),
        "input_length": len(inp),
    }


def _leaf(node_id, score):
    return DAGNode(id=node_id, condition="", score=score)


def _decision(node_id, condition, true_id, false_id, score=None):
    return DAGNode(
        id=node_id, condition=condition,
        true_branch=true_id, false_branch=false_id, score=score,
    )


# ---------------------------------------------------------------------------
# DAGNode
# ---------------------------------------------------------------------------

class TestDAGNode:
    def test_as_dict(self):
        node = DAGNode(id="n1", condition='len(output) > 5', true_branch="n2",
                       false_branch="n3", score=None, label="check length")
        d = node.as_dict()
        assert d["id"] == "n1"
        assert d["true_branch"] == "n2"
        assert d["label"] == "check length"

    def test_from_dict_roundtrip(self):
        node = DAGNode(id="n1", condition='"hello" in output',
                       true_branch="t", false_branch="f", score=0.5, label="kw")
        rebuilt = DAGNode.from_dict(node.as_dict())
        assert rebuilt.id == node.id
        assert rebuilt.condition == node.condition
        assert rebuilt.true_branch == node.true_branch
        assert rebuilt.score == node.score


# ---------------------------------------------------------------------------
# DAGMetric serialization
# ---------------------------------------------------------------------------

class TestDAGMetricSerialization:
    def test_as_dict_from_dict_roundtrip(self):
        dag = DAGMetric(
            name="test_metric",
            nodes={
                "root": _decision("root", 'len(output) > 3', "yes", "no"),
                "yes": _leaf("yes", 1.0),
                "no": _leaf("no", 0.0),
            },
            root_id="root",
            description="a test dag",
        )
        d = dag.as_dict()
        rebuilt = DAGMetric.from_dict(d)
        assert rebuilt.name == "test_metric"
        assert rebuilt.root_id == "root"
        assert rebuilt.description == "a test dag"
        assert set(rebuilt.nodes.keys()) == {"root", "yes", "no"}
        assert rebuilt.nodes["yes"].score == 1.0

    def test_empty_nodes(self):
        dag = DAGMetric(name="empty")
        d = dag.as_dict()
        rebuilt = DAGMetric.from_dict(d)
        assert rebuilt.nodes == {}


# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------

class TestConditionParsing:
    def test_len_output_gt(self):
        assert _evaluate_condition('len(output) > 5', _ctx(output="abcdef")) is True
        assert _evaluate_condition('len(output) > 5', _ctx(output="abc")) is False

    def test_len_output_lt(self):
        assert _evaluate_condition('len(output) < 3', _ctx(output="ab")) is True
        assert _evaluate_condition('len(output) < 3', _ctx(output="abcd")) is False

    def test_len_input_gte(self):
        assert _evaluate_condition('len(input) >= 2', _ctx(inp="ab")) is True
        assert _evaluate_condition('len(input) >= 2', _ctx(inp="a")) is False

    def test_output_length_var(self):
        ctx = {"input": "", "output": "hello", "output_length": 5, "input_length": 0}
        assert _evaluate_condition('output_length > 3', ctx) is True
        assert _evaluate_condition('output_length > 10', ctx) is False

    def test_input_length_var(self):
        ctx = {"input": "hi", "output": "", "output_length": 0, "input_length": 2}
        assert _evaluate_condition('input_length == 2', ctx) is True

    def test_keyword_in_output(self):
        assert _evaluate_condition('"hello" in output', _ctx(output="say Hello world")) is True
        assert _evaluate_condition('"goodbye" in output', _ctx(output="hello")) is False

    def test_keyword_not_in_output(self):
        assert _evaluate_condition('"error" not in output', _ctx(output="all good")) is True
        assert _evaluate_condition('"error" not in output', _ctx(output="got an Error")) is False

    def test_keyword_in_input(self):
        assert _evaluate_condition('"python" in input', _ctx(inp="Learn Python")) is True

    def test_keyword_not_in_input(self):
        assert _evaluate_condition('"java" not in input', _ctx(inp="Learn Python")) is True
        assert _evaluate_condition('"python" not in input', _ctx(inp="Learn Python")) is False

    def test_invalid_condition_returns_false(self):
        assert _evaluate_condition('some_random_thing > 5', _ctx()) is False
        assert _evaluate_condition('', _ctx()) is False
        assert _evaluate_condition('not a real condition', _ctx()) is False


# ---------------------------------------------------------------------------
# Single node DAG (leaf)
# ---------------------------------------------------------------------------

class TestSingleNodeDAG:
    def test_leaf_returns_score(self):
        dag = DAGMetric(
            name="leaf_only",
            nodes={"leaf": _leaf("leaf", 0.75)},
            root_id="leaf",
        )
        result = evaluate_dag(dag, _ctx())
        assert result.score == 0.75
        assert result.path == ["leaf"]
        assert result.decisions == []

    def test_leaf_none_score_defaults_zero(self):
        dag = DAGMetric(
            name="none_score",
            nodes={"leaf": DAGNode(id="leaf", condition="", score=None)},
            root_id="leaf",
        )
        result = evaluate_dag(dag, _ctx())
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Two-level DAG
# ---------------------------------------------------------------------------

class TestTwoLevelDAG:
    def _make_dag(self):
        return DAGMetric(
            name="two_level",
            nodes={
                "root": _decision("root", 'len(output) > 5', "pass", "fail"),
                "pass": _leaf("pass", 1.0),
                "fail": _leaf("fail", 0.0),
            },
            root_id="root",
        )

    def test_true_branch(self):
        dag = self._make_dag()
        result = evaluate_dag(dag, _ctx(output="long enough text"))
        assert result.score == 1.0
        assert result.path == ["root", "pass"]
        assert len(result.decisions) == 1
        assert result.decisions[0][1] is True

    def test_false_branch(self):
        dag = self._make_dag()
        result = evaluate_dag(dag, _ctx(output="short"))
        assert result.score == 0.0
        assert result.path == ["root", "fail"]
        assert result.decisions[0][1] is False


# ---------------------------------------------------------------------------
# Deep DAG (3+ levels)
# ---------------------------------------------------------------------------

class TestDeepDAG:
    def test_three_level_all_true(self):
        dag = DAGMetric(
            name="deep",
            nodes={
                "n1": _decision("n1", 'len(output) > 3', "n2", "fail1"),
                "n2": _decision("n2", '"good" in output', "n3", "fail2"),
                "n3": _decision("n3", 'len(output) > 10', "pass", "partial"),
                "fail1": _leaf("fail1", 0.0),
                "fail2": _leaf("fail2", 0.25),
                "partial": _leaf("partial", 0.5),
                "pass": _leaf("pass", 1.0),
            },
            root_id="n1",
        )
        result = evaluate_dag(dag, _ctx(output="this is a good long text"))
        assert result.score == 1.0
        assert result.path == ["n1", "n2", "n3", "pass"]
        assert len(result.decisions) == 3

    def test_three_level_middle_fail(self):
        dag = DAGMetric(
            name="deep",
            nodes={
                "n1": _decision("n1", 'len(output) > 3', "n2", "fail1"),
                "n2": _decision("n2", '"good" in output', "n3", "fail2"),
                "n3": _decision("n3", 'len(output) > 10', "pass", "partial"),
                "fail1": _leaf("fail1", 0.0),
                "fail2": _leaf("fail2", 0.25),
                "partial": _leaf("partial", 0.5),
                "pass": _leaf("pass", 1.0),
            },
            root_id="n1",
        )
        # output > 3 but "good" not in output
        result = evaluate_dag(dag, _ctx(output="this is bad text"))
        assert result.score == 0.25
        assert "fail2" in result.path


# ---------------------------------------------------------------------------
# Path and decision tracking
# ---------------------------------------------------------------------------

class TestPathTracking:
    def test_path_records_all_visited_nodes(self):
        dag = DAGMetric(
            name="path_test",
            nodes={
                "a": _decision("a", 'len(output) > 0', "b", "x"),
                "b": _decision("b", 'len(output) > 100', "c", "y"),
                "x": _leaf("x", 0.0),
                "y": _leaf("y", 0.5),
                "c": _leaf("c", 1.0),
            },
            root_id="a",
        )
        result = evaluate_dag(dag, _ctx(output="hello"))
        assert result.path == ["a", "b", "y"]

    def test_decisions_record_condition_and_outcome(self):
        dag = DAGMetric(
            name="decisions_test",
            nodes={
                "a": _decision("a", '"yes" in output', "b", "no_leaf"),
                "b": _leaf("b", 1.0),
                "no_leaf": _leaf("no_leaf", 0.0),
            },
            root_id="a",
        )
        result = evaluate_dag(dag, _ctx(output="YES please"))
        assert result.decisions == [('"yes" in output', True)]


# ---------------------------------------------------------------------------
# DAGResult serialization
# ---------------------------------------------------------------------------

class TestDAGResult:
    def test_as_dict(self):
        r = DAGResult(
            metric_name="test",
            score=0.8,
            path=["a", "b", "c"],
            decisions=[('len(output) > 5', True), ('"ok" in output', False)],
        )
        d = r.as_dict()
        assert d["metric_name"] == "test"
        assert d["score"] == 0.8
        assert d["path"] == ["a", "b", "c"]
        assert d["decisions"] == [['len(output) > 5', True], ['"ok" in output', False]]

    def test_metric_name_from_dag(self):
        dag = DAGMetric(
            name="my_dag",
            nodes={"leaf": _leaf("leaf", 0.5)},
            root_id="leaf",
        )
        result = evaluate_dag(dag, _ctx())
        assert result.metric_name == "my_dag"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_node_reference(self):
        """Node references a next_id that does not exist in the DAG."""
        dag = DAGMetric(
            name="missing_ref",
            nodes={
                "root": _decision("root", 'len(output) > 0', "nonexistent", "also_missing", score=0.3),
            },
            root_id="root",
        )
        result = evaluate_dag(dag, _ctx(output="hello"))
        # Should fall back to root's score since target is missing
        assert result.score == 0.3
        assert result.path == ["root"]

    def test_empty_dag(self):
        dag = DAGMetric(name="empty", nodes={}, root_id="root")
        result = evaluate_dag(dag, _ctx())
        assert result.score == 0.0
        assert result.path == []

    def test_empty_root_id(self):
        dag = DAGMetric(name="no_root", nodes={"a": _leaf("a", 1.0)}, root_id="")
        result = evaluate_dag(dag, _ctx())
        assert result.score == 0.0
        assert result.path == []


# ---------------------------------------------------------------------------
# build_simple_dag
# ---------------------------------------------------------------------------

class TestBuildSimpleDAG:
    def test_single_condition(self):
        dag = build_simple_dag("simple", [('len(output) > 5', 1.0, 0.0)])
        assert dag.name == "simple"
        assert dag.root_id == "cond_0"
        # Should have: cond_0, true_0, false_0
        assert "cond_0" in dag.nodes
        assert "true_0" in dag.nodes
        assert "false_0" in dag.nodes

    def test_single_condition_true_path(self):
        dag = build_simple_dag("s", [('len(output) > 5', 1.0, 0.0)])
        result = evaluate_dag(dag, _ctx(output="long enough"))
        assert result.score == 1.0

    def test_single_condition_false_path(self):
        dag = build_simple_dag("s", [('len(output) > 5', 1.0, 0.0)])
        result = evaluate_dag(dag, _ctx(output="abc"))
        assert result.score == 0.0

    def test_two_conditions_chain(self):
        dag = build_simple_dag("chain", [
            ('len(output) > 3', 1.0, 0.0),
            ('"good" in output', 0.9, 0.1),
        ])
        # First true, second true -> 0.9
        result = evaluate_dag(dag, _ctx(output="this is good"))
        assert result.score == 0.9

    def test_two_conditions_first_fails(self):
        dag = build_simple_dag("chain", [
            ('len(output) > 100', 1.0, 0.0),
            ('"good" in output', 0.9, 0.1),
        ])
        result = evaluate_dag(dag, _ctx(output="short"))
        assert result.score == 0.0  # false_0

    def test_two_conditions_second_fails(self):
        dag = build_simple_dag("chain", [
            ('len(output) > 3', 1.0, 0.0),
            ('"good" in output', 0.9, 0.1),
        ])
        result = evaluate_dag(dag, _ctx(output="this is bad"))
        assert result.score == 0.1  # false_1

    def test_empty_conditions(self):
        dag = build_simple_dag("empty", [])
        assert dag.root_id == "leaf"
        result = evaluate_dag(dag, _ctx())
        assert result.score == 0.0

    def test_roundtrip_serialization(self):
        dag = build_simple_dag("rt", [
            ('len(output) > 5', 1.0, 0.0),
            ('"ok" in output', 0.8, 0.2),
        ])
        d = dag.as_dict()
        rebuilt = DAGMetric.from_dict(d)
        # Verify same result
        ctx = _ctx(output="this is ok text")
        r1 = evaluate_dag(dag, ctx)
        r2 = evaluate_dag(rebuilt, ctx)
        assert r1.score == r2.score
        assert r1.path == r2.path
