"""Tests for the SAMMO-style structural prompt optimization module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.prompt_optimization import (
    PromptSection,
    PromptDAG,
    MutationOp,
    OptimizationResult,
    build_prompt_dag,
    render_prompt,
    apply_drop_mutation,
    apply_reorder_mutation,
    apply_merge_mutation,
    apply_split_mutation,
    estimate_cost,
    topological_sort,
    detect_cycles,
    format_dag_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(sid: str, content: str = "", **kwargs) -> PromptSection:
    return PromptSection(
        section_id=sid,
        section_type=kwargs.get("section_type", "instruction"),
        content=content or f"Content of {sid}",
        weight=kwargs.get("weight", 1.0),
        optional=kwargs.get("optional", False),
    )


def _simple_dag() -> PromptDAG:
    """Three-section DAG: A -> B -> C."""
    return build_prompt_dag(
        sections=[_section("A"), _section("B"), _section("C")],
        edges=[("A", "B"), ("B", "C")],
    )


# ---------------------------------------------------------------------------
# PromptSection
# ---------------------------------------------------------------------------


class TestPromptSection:
    def test_basic_construction(self):
        s = PromptSection(
            section_id="s1", section_type="instruction", content="Do X"
        )
        assert s.section_id == "s1"
        assert s.weight == 1.0
        assert s.optional is False

    def test_as_dict(self):
        s = _section("s1", "hello", weight=0.5, optional=True)
        d = s.as_dict()
        assert d["section_id"] == "s1"
        assert d["content"] == "hello"
        assert d["weight"] == 0.5
        assert d["optional"] is True

    def test_from_dict_minimal(self):
        s = PromptSection.from_dict({
            "section_id": "x",
            "section_type": "context",
            "content": "ctx",
        })
        assert s.weight == 1.0
        assert s.optional is False

    def test_from_dict_full(self):
        s = PromptSection.from_dict({
            "section_id": "x",
            "section_type": "examples",
            "content": "ex",
            "weight": 2.0,
            "optional": True,
        })
        assert s.weight == 2.0
        assert s.optional is True

    def test_roundtrip(self):
        original = _section("rt", "data", weight=0.3, optional=True,
                            section_type="rubric")
        restored = PromptSection.from_dict(original.as_dict())
        assert restored.section_id == original.section_id
        assert restored.content == original.content
        assert restored.weight == original.weight


# ---------------------------------------------------------------------------
# PromptDAG
# ---------------------------------------------------------------------------


class TestPromptDAG:
    def test_as_dict(self):
        dag = _simple_dag()
        d = dag.as_dict()
        assert len(d["sections"]) == 3
        assert len(d["edges"]) == 2
        assert d["edges"][0] == ["A", "B"]

    def test_from_dict(self):
        dag = _simple_dag()
        d = dag.as_dict()
        restored = PromptDAG.from_dict(d)
        assert len(restored.sections) == 3
        assert restored.edges[0] == ("A", "B")

    def test_roundtrip(self):
        dag = _simple_dag()
        dag.metadata = {"version": 1}
        restored = PromptDAG.from_dict(dag.as_dict())
        assert restored.metadata["version"] == 1
        assert len(restored.sections) == len(dag.sections)

    def test_empty_dag(self):
        dag = PromptDAG()
        assert dag.sections == []
        assert dag.edges == []
        assert dag.metadata == {}


# ---------------------------------------------------------------------------
# MutationOp
# ---------------------------------------------------------------------------


class TestMutationOp:
    def test_basic(self):
        m = MutationOp(op_type="drop", target_section="s1")
        assert m.parameters == {}

    def test_as_dict(self):
        m = MutationOp(
            op_type="split", target_section="s2",
            parameters={"split_point": 10},
        )
        d = m.as_dict()
        assert d["op_type"] == "split"
        assert d["parameters"]["split_point"] == 10

    def test_from_dict(self):
        m = MutationOp.from_dict({
            "op_type": "merge",
            "target_section": "combined",
            "parameters": {"ids": ["a", "b"]},
        })
        assert m.op_type == "merge"
        assert m.parameters["ids"] == ["a", "b"]

    def test_roundtrip(self):
        original = MutationOp(op_type="reorder", target_section="all",
                              parameters={"order": [1, 2, 3]})
        restored = MutationOp.from_dict(original.as_dict())
        assert restored.op_type == original.op_type
        assert restored.parameters == original.parameters


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


class TestOptimizationResult:
    def test_as_dict(self):
        dag = _simple_dag()
        m = MutationOp(op_type="drop", target_section="B")
        r = OptimizationResult(
            original=dag, mutated=dag, mutation=m,
            score_delta=0.1, cost_delta=-0.05,
        )
        d = r.as_dict()
        assert d["score_delta"] == 0.1
        assert d["cost_delta"] == -0.05
        assert "sections" in d["original"]

    def test_from_dict(self):
        dag = _simple_dag()
        m = MutationOp(op_type="drop", target_section="B")
        r = OptimizationResult(original=dag, mutated=dag, mutation=m)
        restored = OptimizationResult.from_dict(r.as_dict())
        assert restored.score_delta == 0.0
        assert len(restored.original.sections) == 3


# ---------------------------------------------------------------------------
# detect_cycles
# ---------------------------------------------------------------------------


class TestDetectCycles:
    def test_no_edges(self):
        assert detect_cycles([]) is False

    def test_linear_chain(self):
        assert detect_cycles([("A", "B"), ("B", "C")]) is False

    def test_simple_cycle(self):
        assert detect_cycles([("A", "B"), ("B", "A")]) is True

    def test_self_loop(self):
        assert detect_cycles([("A", "A")]) is True

    def test_complex_dag(self):
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        assert detect_cycles(edges) is False

    def test_complex_cycle(self):
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        assert detect_cycles(edges) is True

    def test_diamond_no_cycle(self):
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        assert detect_cycles(edges) is False


# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_linear_chain(self):
        dag = _simple_dag()
        order = topological_sort(dag)
        assert order == ["A", "B", "C"]

    def test_no_edges(self):
        dag = build_prompt_dag([_section("X"), _section("Y"), _section("Z")])
        order = topological_sort(dag)
        # Should preserve original order
        assert order == ["X", "Y", "Z"]

    def test_diamond(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B"), _section("C"), _section("D")],
            edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
        order = topological_sort(dag)
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_reverse_listed_order(self):
        dag = build_prompt_dag(
            [_section("C"), _section("B"), _section("A")],
            edges=[("A", "B"), ("B", "C")],
        )
        order = topological_sort(dag)
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")


# ---------------------------------------------------------------------------
# build_prompt_dag
# ---------------------------------------------------------------------------


class TestBuildPromptDag:
    def test_basic_build(self):
        dag = build_prompt_dag([_section("A"), _section("B")])
        assert len(dag.sections) == 2
        assert dag.edges == []

    def test_with_edges(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B")],
            edges=[("A", "B")],
        )
        assert dag.edges == [("A", "B")]

    def test_invalid_edge_source(self):
        with pytest.raises(ValueError, match="edge source"):
            build_prompt_dag([_section("A")], edges=[("X", "A")])

    def test_invalid_edge_target(self):
        with pytest.raises(ValueError, match="edge target"):
            build_prompt_dag([_section("A")], edges=[("A", "X")])

    def test_cycle_rejected(self):
        with pytest.raises(ValueError, match="cycle"):
            build_prompt_dag(
                [_section("A"), _section("B")],
                edges=[("A", "B"), ("B", "A")],
            )

    def test_none_edges(self):
        dag = build_prompt_dag([_section("A")], edges=None)
        assert dag.edges == []


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_simple_render(self):
        dag = build_prompt_dag([
            _section("A", "First section"),
            _section("B", "Second section"),
        ])
        text = render_prompt(dag)
        assert "First section" in text
        assert "Second section" in text

    def test_dependency_order(self):
        dag = build_prompt_dag(
            [_section("B", "BBB"), _section("A", "AAA")],
            edges=[("A", "B")],
        )
        text = render_prompt(dag)
        assert text.index("AAA") < text.index("BBB")

    def test_sections_separated_by_double_newline(self):
        dag = build_prompt_dag([_section("X", "X"), _section("Y", "Y")])
        text = render_prompt(dag)
        assert "\n\n" in text

    def test_single_section(self):
        dag = build_prompt_dag([_section("only", "only content")])
        assert render_prompt(dag) == "only content"


# ---------------------------------------------------------------------------
# apply_drop_mutation
# ---------------------------------------------------------------------------


class TestApplyDropMutation:
    def test_drop_optional_section(self):
        dag = build_prompt_dag([
            _section("A"),
            _section("B", optional=True),
            _section("C"),
        ])
        result = apply_drop_mutation(dag, "B")
        ids = [s.section_id for s in result.sections]
        assert "B" not in ids
        assert len(result.sections) == 2

    def test_drop_removes_edges(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B", optional=True), _section("C")],
            edges=[("A", "B"), ("B", "C")],
        )
        result = apply_drop_mutation(dag, "B")
        for src, dst in result.edges:
            assert src != "B" and dst != "B"

    def test_drop_non_optional_raises(self):
        dag = build_prompt_dag([_section("A")])
        with pytest.raises(ValueError, match="not optional"):
            apply_drop_mutation(dag, "A")

    def test_drop_nonexistent_raises(self):
        dag = build_prompt_dag([_section("A")])
        with pytest.raises(ValueError, match="not found"):
            apply_drop_mutation(dag, "Z")

    def test_drop_preserves_other_edges(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B", optional=True), _section("C"), _section("D")],
            edges=[("A", "B"), ("A", "C"), ("C", "D")],
        )
        result = apply_drop_mutation(dag, "B")
        assert ("A", "C") in result.edges
        assert ("C", "D") in result.edges


# ---------------------------------------------------------------------------
# apply_reorder_mutation
# ---------------------------------------------------------------------------


class TestApplyReorderMutation:
    def test_basic_reorder(self):
        dag = build_prompt_dag([_section("A"), _section("B"), _section("C")])
        result = apply_reorder_mutation(dag, ["C", "B", "A"])
        ids = [s.section_id for s in result.sections]
        assert ids == ["C", "B", "A"]

    def test_reorder_mismatched_ids(self):
        dag = build_prompt_dag([_section("A"), _section("B")])
        with pytest.raises(ValueError, match="exactly"):
            apply_reorder_mutation(dag, ["A", "C"])

    def test_reorder_missing_id(self):
        dag = build_prompt_dag([_section("A"), _section("B")])
        with pytest.raises(ValueError):
            apply_reorder_mutation(dag, ["A"])

    def test_reorder_preserves_edges(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B")],
            edges=[("A", "B")],
        )
        result = apply_reorder_mutation(dag, ["B", "A"])
        assert ("A", "B") in result.edges


# ---------------------------------------------------------------------------
# apply_merge_mutation
# ---------------------------------------------------------------------------


class TestApplyMergeMutation:
    def test_basic_merge(self):
        dag = build_prompt_dag([
            _section("A", "Part 1"),
            _section("B", "Part 2"),
            _section("C", "Part 3"),
        ])
        result = apply_merge_mutation(dag, ["A", "B"], "AB")
        ids = [s.section_id for s in result.sections]
        assert "AB" in ids
        assert "A" not in ids
        assert "B" not in ids
        assert "C" in ids

    def test_merge_content_concatenated(self):
        dag = build_prompt_dag([
            _section("A", "Hello"),
            _section("B", "World"),
        ])
        result = apply_merge_mutation(dag, ["A", "B"], "AB")
        merged = [s for s in result.sections if s.section_id == "AB"][0]
        assert "Hello" in merged.content
        assert "World" in merged.content

    def test_merge_inherits_first_type(self):
        dag = build_prompt_dag([
            PromptSection(section_id="A", section_type="context", content="a"),
            PromptSection(section_id="B", section_type="examples", content="b"),
        ])
        result = apply_merge_mutation(dag, ["A", "B"], "AB")
        merged = [s for s in result.sections if s.section_id == "AB"][0]
        assert merged.section_type == "context"

    def test_merge_rewrites_edges(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B"), _section("C")],
            edges=[("A", "B"), ("B", "C")],
        )
        result = apply_merge_mutation(dag, ["A", "B"], "AB")
        # Edge from AB to C should exist
        assert ("AB", "C") in result.edges

    def test_merge_nonexistent_raises(self):
        dag = build_prompt_dag([_section("A")])
        with pytest.raises(ValueError, match="not found"):
            apply_merge_mutation(dag, ["A", "Z"], "AZ")

    def test_merge_removes_self_loops(self):
        dag = build_prompt_dag(
            [_section("A"), _section("B")],
            edges=[("A", "B")],
        )
        result = apply_merge_mutation(dag, ["A", "B"], "AB")
        for src, dst in result.edges:
            assert src != dst


# ---------------------------------------------------------------------------
# apply_split_mutation
# ---------------------------------------------------------------------------


class TestApplySplitMutation:
    def test_basic_split(self):
        dag = build_prompt_dag([_section("A", "HelloWorld")])
        result = apply_split_mutation(dag, "A", 5)
        ids = [s.section_id for s in result.sections]
        assert "A_a" in ids
        assert "A_b" in ids
        assert "A" not in ids

    def test_split_content(self):
        dag = build_prompt_dag([_section("A", "HelloWorld")])
        result = apply_split_mutation(dag, "A", 5)
        section_map = {s.section_id: s for s in result.sections}
        assert section_map["A_a"].content == "Hello"
        assert section_map["A_b"].content == "World"

    def test_split_creates_internal_edge(self):
        dag = build_prompt_dag([_section("A", "HelloWorld")])
        result = apply_split_mutation(dag, "A", 5)
        assert ("A_a", "A_b") in result.edges

    def test_split_rewrites_incoming_edges(self):
        dag = build_prompt_dag(
            [_section("X", "xx"), _section("A", "HelloWorld")],
            edges=[("X", "A")],
        )
        result = apply_split_mutation(dag, "A", 5)
        # Incoming to A should become incoming to A_a
        assert ("X", "A_a") in result.edges

    def test_split_rewrites_outgoing_edges(self):
        dag = build_prompt_dag(
            [_section("A", "HelloWorld"), _section("Y", "yy")],
            edges=[("A", "Y")],
        )
        result = apply_split_mutation(dag, "A", 5)
        # Outgoing from A should become outgoing from A_b
        assert ("A_b", "Y") in result.edges

    def test_split_nonexistent_raises(self):
        dag = build_prompt_dag([_section("A", "text")])
        with pytest.raises(ValueError, match="not found"):
            apply_split_mutation(dag, "Z", 2)

    def test_split_at_zero_raises(self):
        dag = build_prompt_dag([_section("A", "text")])
        with pytest.raises(ValueError, match="split_point"):
            apply_split_mutation(dag, "A", 0)

    def test_split_at_end_raises(self):
        dag = build_prompt_dag([_section("A", "text")])
        with pytest.raises(ValueError, match="split_point"):
            apply_split_mutation(dag, "A", 4)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_basic_cost(self):
        dag = build_prompt_dag([_section("A", "12345")])
        cost = estimate_cost(dag, cost_per_char=0.01)
        assert abs(cost - 0.05) < 1e-9

    def test_weight_affects_cost(self):
        dag = build_prompt_dag([_section("A", "12345", weight=2.0)])
        cost = estimate_cost(dag, cost_per_char=0.01)
        assert abs(cost - 0.10) < 1e-9

    def test_multiple_sections(self):
        dag = build_prompt_dag([
            _section("A", "12345"),
            _section("B", "67890"),
        ])
        cost = estimate_cost(dag, cost_per_char=0.01)
        assert abs(cost - 0.10) < 1e-9

    def test_empty_dag(self):
        dag = PromptDAG()
        assert estimate_cost(dag) == 0.0

    def test_default_cost_per_char(self):
        dag = build_prompt_dag([_section("A", "x" * 100)])
        cost = estimate_cost(dag)
        assert abs(cost - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# format_dag_summary
# ---------------------------------------------------------------------------


class TestFormatDagSummary:
    def test_contains_section_count(self):
        dag = _simple_dag()
        summary = format_dag_summary(dag)
        assert "3 sections" in summary

    def test_contains_edge_count(self):
        dag = _simple_dag()
        summary = format_dag_summary(dag)
        assert "2 edges" in summary

    def test_contains_section_ids(self):
        dag = _simple_dag()
        summary = format_dag_summary(dag)
        assert "[A]" in summary
        assert "[B]" in summary
        assert "[C]" in summary

    def test_contains_estimated_cost(self):
        dag = _simple_dag()
        summary = format_dag_summary(dag)
        assert "Estimated cost" in summary

    def test_optional_flag_shown(self):
        dag = build_prompt_dag([_section("A", optional=True)])
        summary = format_dag_summary(dag)
        assert "(optional)" in summary

    def test_edge_listing(self):
        dag = _simple_dag()
        summary = format_dag_summary(dag)
        assert "A -> B" in summary
        assert "B -> C" in summary

    def test_no_edges_section(self):
        dag = build_prompt_dag([_section("A")])
        summary = format_dag_summary(dag)
        assert "Edges" not in summary
