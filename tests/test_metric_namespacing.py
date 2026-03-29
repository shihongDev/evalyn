"""Tests for metric namespacing module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.metric_namespacing import (
    Namespace,
    NamespacedMetric,
    NamespaceRegistry,
    format_namespace_tree,
    parse_full_id,
)


# ---------------------------------------------------------------------------
# NamespacedMetric dataclass
# ---------------------------------------------------------------------------


class TestNamespacedMetric:
    def test_basic_creation(self):
        m = NamespacedMetric(namespace="team_a", metric_id="accuracy", full_id="team_a/accuracy")
        assert m.namespace == "team_a"
        assert m.metric_id == "accuracy"
        assert m.full_id == "team_a/accuracy"
        assert m.description == ""

    def test_with_description(self):
        m = NamespacedMetric(namespace="ns", metric_id="f1", full_id="ns/f1", description="F1 score")
        assert m.description == "F1 score"

    def test_as_dict(self):
        m = NamespacedMetric(namespace="ns", metric_id="acc", full_id="ns/acc", description="d")
        d = m.as_dict()
        assert d == {"namespace": "ns", "metric_id": "acc", "full_id": "ns/acc", "description": "d"}

    def test_from_dict(self):
        d = {"namespace": "ns", "metric_id": "acc", "full_id": "ns/acc", "description": "d"}
        m = NamespacedMetric.from_dict(d)
        assert m.namespace == "ns"
        assert m.metric_id == "acc"

    def test_from_dict_no_description(self):
        d = {"namespace": "ns", "metric_id": "acc", "full_id": "ns/acc"}
        m = NamespacedMetric.from_dict(d)
        assert m.description == ""

    def test_roundtrip(self):
        m = NamespacedMetric(namespace="x", metric_id="y", full_id="x/y", description="z")
        m2 = NamespacedMetric.from_dict(m.as_dict())
        assert m == m2


# ---------------------------------------------------------------------------
# Namespace dataclass
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_basic_creation(self):
        ns = Namespace(name="team_a")
        assert ns.name == "team_a"
        assert ns.description == ""
        assert ns.metrics == []

    def test_with_metrics(self):
        ns = Namespace(name="ns", metrics=["accuracy", "f1"])
        assert len(ns.metrics) == 2

    def test_as_dict(self):
        ns = Namespace(name="ns", description="d", metrics=["a", "b"])
        d = ns.as_dict()
        assert d == {"name": "ns", "description": "d", "metrics": ["a", "b"]}

    def test_from_dict(self):
        d = {"name": "ns", "description": "d", "metrics": ["a"]}
        ns = Namespace.from_dict(d)
        assert ns.name == "ns"
        assert ns.metrics == ["a"]

    def test_from_dict_defaults(self):
        d = {"name": "ns"}
        ns = Namespace.from_dict(d)
        assert ns.description == ""
        assert ns.metrics == []

    def test_roundtrip(self):
        ns = Namespace(name="ns", description="d", metrics=["a", "b"])
        ns2 = Namespace.from_dict(ns.as_dict())
        assert ns == ns2

    def test_metrics_list_is_independent(self):
        """from_dict should create a new list, not share the original."""
        original = ["a", "b"]
        ns = Namespace.from_dict({"name": "ns", "metrics": original})
        ns.metrics.append("c")
        assert len(original) == 2


# ---------------------------------------------------------------------------
# parse_full_id
# ---------------------------------------------------------------------------


class TestParseFullId:
    def test_with_namespace(self):
        assert parse_full_id("team/accuracy") == ("team", "accuracy")

    def test_no_namespace(self):
        assert parse_full_id("accuracy") == ("default", "accuracy")

    def test_multiple_slashes(self):
        ns, metric = parse_full_id("a/b/c")
        assert ns == "a"
        assert metric == "b/c"

    def test_empty_metric(self):
        ns, metric = parse_full_id("ns/")
        assert ns == "ns"
        assert metric == ""

    def test_empty_string(self):
        assert parse_full_id("") == ("default", "")


# ---------------------------------------------------------------------------
# NamespaceRegistry
# ---------------------------------------------------------------------------


class TestNamespaceRegistryBasics:
    def test_empty_registry(self):
        reg = NamespaceRegistry()
        assert reg.list_namespaces() == []
        assert reg.list_metrics() == []

    def test_register_namespace(self):
        reg = NamespaceRegistry()
        reg.register_namespace("team_a", "Team A metrics")
        namespaces = reg.list_namespaces()
        assert len(namespaces) == 1
        assert namespaces[0].name == "team_a"
        assert namespaces[0].description == "Team A metrics"

    def test_register_namespace_updates_description(self):
        reg = NamespaceRegistry()
        reg.register_namespace("ns", "old")
        reg.register_namespace("ns", "new")
        assert reg.list_namespaces()[0].description == "new"

    def test_register_multiple_namespaces(self):
        reg = NamespaceRegistry()
        reg.register_namespace("a")
        reg.register_namespace("b")
        assert len(reg.list_namespaces()) == 2


class TestNamespaceRegistryAddMetric:
    def test_add_metric(self):
        reg = NamespaceRegistry()
        m = reg.add_metric("team_a", "accuracy", "Accuracy metric")
        assert m.full_id == "team_a/accuracy"
        assert m.description == "Accuracy metric"

    def test_auto_creates_namespace(self):
        reg = NamespaceRegistry()
        reg.add_metric("new_ns", "metric1")
        namespaces = reg.list_namespaces()
        assert len(namespaces) == 1
        assert namespaces[0].name == "new_ns"

    def test_metric_added_to_namespace_list(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        reg.add_metric("ns", "f1")
        ns = reg.list_namespaces()[0]
        assert "accuracy" in ns.metrics
        assert "f1" in ns.metrics

    def test_no_duplicate_metric_in_namespace(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        reg.add_metric("ns", "accuracy", "updated desc")
        ns = reg.list_namespaces()[0]
        assert ns.metrics.count("accuracy") == 1

    def test_add_metric_returns_namespaced_metric(self):
        reg = NamespaceRegistry()
        m = reg.add_metric("ns", "acc")
        assert isinstance(m, NamespacedMetric)
        assert m.namespace == "ns"
        assert m.metric_id == "acc"


class TestNamespaceRegistryResolve:
    def test_resolve_existing(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        m = reg.resolve("ns/accuracy")
        assert m is not None
        assert m.metric_id == "accuracy"

    def test_resolve_missing(self):
        reg = NamespaceRegistry()
        assert reg.resolve("ns/missing") is None

    def test_resolve_default_namespace(self):
        reg = NamespaceRegistry()
        reg.add_metric("default", "accuracy")
        m = reg.resolve("accuracy")
        assert m is not None
        assert m.namespace == "default"

    def test_resolve_wrong_namespace(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns_a", "accuracy")
        assert reg.resolve("ns_b/accuracy") is None


class TestNamespaceRegistrySearch:
    def test_search_by_substring(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        reg.add_metric("ns", "f1_accuracy")
        reg.add_metric("ns", "loss")
        results = reg.search("accuracy")
        assert len(results) == 2

    def test_search_no_match(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        results = reg.search("zzz")
        assert results == []

    def test_search_with_namespace_filter(self):
        reg = NamespaceRegistry()
        reg.add_metric("a", "accuracy")
        reg.add_metric("b", "accuracy")
        results = reg.search("accuracy", namespace="a")
        assert len(results) == 1
        assert results[0].namespace == "a"

    def test_search_empty_query(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        reg.add_metric("ns", "f1")
        results = reg.search("")
        assert len(results) == 2


class TestNamespaceRegistryListMetrics:
    def test_list_all(self):
        reg = NamespaceRegistry()
        reg.add_metric("a", "m1")
        reg.add_metric("b", "m2")
        metrics = reg.list_metrics()
        assert len(metrics) == 2

    def test_list_by_namespace(self):
        reg = NamespaceRegistry()
        reg.add_metric("a", "m1")
        reg.add_metric("a", "m2")
        reg.add_metric("b", "m3")
        metrics = reg.list_metrics(namespace="a")
        assert len(metrics) == 2
        assert all(m.namespace == "a" for m in metrics)

    def test_list_empty_namespace(self):
        reg = NamespaceRegistry()
        reg.register_namespace("empty")
        metrics = reg.list_metrics(namespace="empty")
        assert metrics == []


class TestNamespaceRegistryCompare:
    def test_compare_across_namespaces(self):
        reg = NamespaceRegistry()
        reg.add_metric("team_a", "accuracy")
        reg.add_metric("team_b", "accuracy")
        reg.add_metric("team_a", "f1")
        results = reg.compare_across_namespaces("accuracy")
        assert len(results) == 2
        namespaces = {m.namespace for m in results}
        assert namespaces == {"team_a", "team_b"}

    def test_compare_no_match(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        results = reg.compare_across_namespaces("missing")
        assert results == []

    def test_compare_single_namespace(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "accuracy")
        results = reg.compare_across_namespaces("accuracy")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# format_namespace_tree
# ---------------------------------------------------------------------------


class TestFormatNamespaceTree:
    def test_empty_registry(self):
        reg = NamespaceRegistry()
        text = format_namespace_tree(reg)
        assert text == ""

    def test_single_namespace_single_metric(self):
        reg = NamespaceRegistry()
        reg.add_metric("team_a", "accuracy")
        text = format_namespace_tree(reg)
        assert "team_a/" in text
        assert "accuracy" in text

    def test_multiple_namespaces(self):
        reg = NamespaceRegistry()
        reg.add_metric("a", "m1")
        reg.add_metric("b", "m2")
        text = format_namespace_tree(reg)
        assert "a/" in text
        assert "b/" in text

    def test_namespace_description(self):
        reg = NamespaceRegistry()
        reg.register_namespace("ns", "My namespace")
        reg.add_metric("ns", "acc")
        text = format_namespace_tree(reg)
        assert "My namespace" in text

    def test_metric_description(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "acc", "Accuracy metric")
        text = format_namespace_tree(reg)
        assert "Accuracy metric" in text

    def test_tree_connectors(self):
        reg = NamespaceRegistry()
        reg.add_metric("ns", "m1")
        reg.add_metric("ns", "m2")
        text = format_namespace_tree(reg)
        assert "|--" in text
        assert "`--" in text
