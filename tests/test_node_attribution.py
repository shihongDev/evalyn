"""Tests for node-level metric attribution."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.node_attribution import (
    AttributionReport,
    AttributionResult,
    NodeScore,
    attribute_to_nodes,
    build_attribution_report,
    find_failure_hotspots,
    suggest_fixes,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _node(node_id: str, name: str, metric: str, score: float, passed: bool) -> dict:
    return {
        "node_id": node_id,
        "node_name": name,
        "metric_id": metric,
        "score": score,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# attribute_to_nodes
# ---------------------------------------------------------------------------

class TestAttributeToNodes:
    def test_single_failure(self):
        """One failing node should get contribution=1.0."""
        nodes = [
            _node("n1", "generate", "accuracy", 0.3, False),
            _node("n2", "validate", "accuracy", 1.0, True),
        ]
        result = attribute_to_nodes("item-1", 0.65, nodes)
        assert result.item_id == "item-1"
        assert result.overall_score == 0.65
        assert len(result.node_scores) == 2
        # generate has deficit 0.7, validate has deficit 0.0
        gen = result.node_scores[0]
        assert gen.node_name == "generate"
        assert gen.contribution == 1.0
        val = result.node_scores[1]
        assert val.contribution == 0.0

    def test_all_pass(self):
        """When all nodes score 1.0, contributions should all be 0.0."""
        nodes = [
            _node("n1", "a", "m", 1.0, True),
            _node("n2", "b", "m", 1.0, True),
        ]
        result = attribute_to_nodes("item-2", 1.0, nodes)
        for ns in result.node_scores:
            assert ns.contribution == 0.0

    def test_multiple_failures(self):
        """Multiple failing nodes should split contribution proportionally."""
        nodes = [
            _node("n1", "a", "m", 0.5, False),  # deficit 0.5
            _node("n2", "b", "m", 0.0, False),  # deficit 1.0
            _node("n3", "c", "m", 1.0, True),   # deficit 0.0
        ]
        result = attribute_to_nodes("item-3", 0.5, nodes)
        contribs = [ns.contribution for ns in result.node_scores]
        # total deficit = 1.5
        assert abs(contribs[0] - 0.5 / 1.5) < 1e-9
        assert abs(contribs[1] - 1.0 / 1.5) < 1e-9
        assert contribs[2] == 0.0

    def test_primary_failure_node_correct(self):
        """Primary failure should be the node with the lowest score."""
        nodes = [
            _node("n1", "ok_node", "m", 0.8, True),
            _node("n2", "bad_node", "m", 0.1, False),
            _node("n3", "mid_node", "m", 0.5, False),
        ]
        result = attribute_to_nodes("item-4", 0.4, nodes)
        assert result.primary_failure_node == "bad_node"

    def test_contribution_sums_to_one(self):
        """Contributions across all nodes should sum to approximately 1.0."""
        nodes = [
            _node("n1", "a", "m", 0.2, False),
            _node("n2", "b", "m", 0.4, False),
            _node("n3", "c", "m", 0.6, False),
            _node("n4", "d", "m", 0.8, True),
        ]
        result = attribute_to_nodes("item-5", 0.5, nodes)
        total = sum(ns.contribution for ns in result.node_scores)
        assert abs(total - 1.0) < 1e-9

    def test_empty_nodes(self):
        """Empty node list should return result with no scores."""
        result = attribute_to_nodes("item-6", 0.5, [])
        assert result.node_scores == []
        assert result.primary_failure_node is None

    def test_node_score_fields(self):
        """NodeScore should carry all fields from input."""
        nodes = [_node("n99", "my_node", "metric_x", 0.7, True)]
        result = attribute_to_nodes("item-7", 0.7, nodes)
        ns = result.node_scores[0]
        assert ns.node_id == "n99"
        assert ns.node_name == "my_node"
        assert ns.metric_id == "metric_x"
        assert ns.score == 0.7
        assert ns.passed is True


# ---------------------------------------------------------------------------
# build_attribution_report
# ---------------------------------------------------------------------------

class TestBuildAttributionReport:
    def test_hotspots_counted(self):
        """Failure hotspots should count failed nodes across items."""
        r1 = attribute_to_nodes("i1", 0.5, [
            _node("n1", "gen", "m", 0.3, False),
            _node("n2", "val", "m", 1.0, True),
        ])
        r2 = attribute_to_nodes("i2", 0.4, [
            _node("n1", "gen", "m", 0.2, False),
            _node("n2", "val", "m", 0.5, False),
        ])
        report = build_attribution_report([r1, r2])
        assert report.failure_hotspots["gen"] == 2
        assert report.failure_hotspots["val"] == 1
        assert report.total_items == 2

    def test_empty_results(self):
        """Empty results list produces an empty report."""
        report = build_attribution_report([])
        assert report.results == []
        assert report.failure_hotspots == {}
        assert report.total_items == 0

    def test_all_passing(self):
        """All-passing nodes produce no hotspots."""
        r = attribute_to_nodes("i1", 1.0, [
            _node("n1", "a", "m", 1.0, True),
            _node("n2", "b", "m", 1.0, True),
        ])
        report = build_attribution_report([r])
        assert report.failure_hotspots == {}


# ---------------------------------------------------------------------------
# find_failure_hotspots
# ---------------------------------------------------------------------------

class TestFindFailureHotspots:
    def test_min_failures_filter(self):
        """Only nodes with >= min_failures should be returned."""
        report = AttributionReport(
            failure_hotspots={"a": 5, "b": 1, "c": 3},
            total_items=10,
        )
        result = find_failure_hotspots(report, min_failures=2)
        names = [name for name, _ in result]
        assert "a" in names
        assert "c" in names
        assert "b" not in names

    def test_sorted_descending(self):
        """Results should be sorted by count descending."""
        report = AttributionReport(
            failure_hotspots={"x": 2, "y": 5, "z": 3},
            total_items=10,
        )
        result = find_failure_hotspots(report, min_failures=1)
        counts = [c for _, c in result]
        assert counts == [5, 3, 2]

    def test_empty_hotspots(self):
        """No hotspots should return empty list."""
        report = AttributionReport(failure_hotspots={}, total_items=0)
        assert find_failure_hotspots(report) == []

    def test_default_min_failures(self):
        """Default min_failures=2 should filter out nodes with count=1."""
        report = AttributionReport(
            failure_hotspots={"a": 1, "b": 2},
            total_items=5,
        )
        result = find_failure_hotspots(report)
        assert len(result) == 1
        assert result[0] == ("b", 2)


# ---------------------------------------------------------------------------
# suggest_fixes
# ---------------------------------------------------------------------------

class TestSuggestFixes:
    def test_generates_suggestions(self):
        """Should generate a suggestion for each hotspot node."""
        report = AttributionReport(
            failure_hotspots={"bad_node": 5, "ok_node": 1},
            total_items=10,
        )
        suggestions = suggest_fixes(report)
        assert len(suggestions) == 2
        assert "bad_node" in suggestions[0]
        assert "5 failures" in suggestions[0]

    def test_sorted_by_count(self):
        """Suggestions should be ordered by failure count descending."""
        report = AttributionReport(
            failure_hotspots={"x": 2, "y": 10, "z": 5},
            total_items=20,
        )
        suggestions = suggest_fixes(report)
        assert "y" in suggestions[0]
        assert "z" in suggestions[1]
        assert "x" in suggestions[2]

    def test_empty_hotspots(self):
        """No hotspots should return empty list."""
        report = AttributionReport(failure_hotspots={}, total_items=0)
        assert suggest_fixes(report) == []

    def test_suggestion_text(self):
        """Suggestion text should mention 'review prompt/logic'."""
        report = AttributionReport(
            failure_hotspots={"node_a": 3},
            total_items=5,
        )
        suggestions = suggest_fixes(report)
        assert "review prompt/logic" in suggestions[0]


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------

class TestFormatText:
    def test_format_includes_header(self):
        report = AttributionReport(total_items=3)
        text = report.format_text()
        assert "NODE ATTRIBUTION REPORT" in text
        assert "Total items: 3" in text

    def test_format_includes_hotspots(self):
        report = AttributionReport(
            failure_hotspots={"node_x": 4},
            total_items=5,
        )
        text = report.format_text()
        assert "FAILURE HOTSPOTS" in text
        assert "node_x: 4 failures" in text

    def test_format_includes_item_details(self):
        r = attribute_to_nodes("item-1", 0.5, [
            _node("n1", "gen", "acc", 0.3, False),
        ])
        report = build_attribution_report([r])
        text = report.format_text()
        assert "item-1" in text
        assert "gen" in text
        assert "FAIL" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------

class TestRoundtrips:
    def test_node_score_as_dict(self):
        ns = NodeScore("n1", "gen", "acc", 0.5, False, 0.8)
        d = ns.as_dict()
        assert d["node_id"] == "n1"
        assert d["contribution"] == 0.8

    def test_attribution_result_roundtrip(self):
        original = attribute_to_nodes("item-rt", 0.6, [
            _node("n1", "a", "m", 0.4, False),
            _node("n2", "b", "m", 0.9, True),
        ])
        d = original.as_dict()
        restored = AttributionResult.from_dict(d)
        assert restored.item_id == original.item_id
        assert restored.overall_score == original.overall_score
        assert restored.primary_failure_node == original.primary_failure_node
        assert len(restored.node_scores) == len(original.node_scores)
        for orig_ns, rest_ns in zip(original.node_scores, restored.node_scores):
            assert abs(orig_ns.contribution - rest_ns.contribution) < 1e-9

    def test_attribution_report_roundtrip(self):
        r = attribute_to_nodes("item-1", 0.5, [
            _node("n1", "gen", "m", 0.3, False),
        ])
        original = build_attribution_report([r])
        d = original.as_dict()
        restored = AttributionReport.from_dict(d)
        assert restored.total_items == original.total_items
        assert restored.failure_hotspots == original.failure_hotspots
        assert len(restored.results) == len(original.results)
