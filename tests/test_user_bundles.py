"""Tests for the user bundles module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.user_bundles import (
    BUILTIN_BUNDLES,
    BundleRegistry,
    MetricBundle,
    format_bundle_detail,
    format_bundle_list,
    load_bundles_from_config,
)


# ---------------------------------------------------------------------------
# MetricBundle
# ---------------------------------------------------------------------------


class TestMetricBundle:
    def test_basic_fields(self):
        b = MetricBundle(
            bundle_id="test",
            name="Test",
            description="A test bundle",
            metric_ids=["m1", "m2"],
            tags=["t1"],
        )
        assert b.bundle_id == "test"
        assert b.name == "Test"
        assert b.description == "A test bundle"
        assert b.metric_ids == ["m1", "m2"]
        assert b.tags == ["t1"]

    def test_defaults(self):
        b = MetricBundle(bundle_id="x", name="X", description="")
        assert b.metric_ids == []
        assert b.tags == []

    def test_as_dict(self):
        b = MetricBundle(
            bundle_id="b1", name="B1", description="desc",
            metric_ids=["a"], tags=["t"],
        )
        d = b.as_dict()
        assert d["bundle_id"] == "b1"
        assert d["metric_ids"] == ["a"]
        assert d["tags"] == ["t"]

    def test_from_dict(self):
        d = {
            "bundle_id": "b2",
            "name": "B2",
            "description": "d",
            "metric_ids": ["x", "y"],
            "tags": ["tag"],
        }
        b = MetricBundle.from_dict(d)
        assert b.bundle_id == "b2"
        assert b.metric_ids == ["x", "y"]

    def test_from_dict_defaults(self):
        d = {"bundle_id": "b3", "name": "B3"}
        b = MetricBundle.from_dict(d)
        assert b.description == ""
        assert b.metric_ids == []
        assert b.tags == []

    def test_roundtrip(self):
        original = MetricBundle(
            bundle_id="rt",
            name="Roundtrip",
            description="test",
            metric_ids=["m1", "m2"],
            tags=["a", "b"],
        )
        restored = MetricBundle.from_dict(original.as_dict())
        assert restored == original

    def test_as_dict_copies_lists(self):
        b = MetricBundle(bundle_id="x", name="X", description="", metric_ids=["m1"])
        d = b.as_dict()
        d["metric_ids"].append("m2")
        assert b.metric_ids == ["m1"]


# ---------------------------------------------------------------------------
# BundleRegistry - register / unregister / get
# ---------------------------------------------------------------------------


class TestBundleRegistryBasic:
    def test_empty_registry(self):
        reg = BundleRegistry()
        assert reg.list_bundles() == []

    def test_register_and_get(self):
        reg = BundleRegistry()
        b = MetricBundle(bundle_id="x", name="X", description="")
        reg.register(b)
        assert reg.get("x") is b

    def test_get_missing_returns_none(self):
        reg = BundleRegistry()
        assert reg.get("nope") is None

    def test_unregister_existing(self):
        reg = BundleRegistry()
        b = MetricBundle(bundle_id="x", name="X", description="")
        reg.register(b)
        assert reg.unregister("x") is True
        assert reg.get("x") is None

    def test_unregister_missing(self):
        reg = BundleRegistry()
        assert reg.unregister("nope") is False

    def test_register_replaces(self):
        reg = BundleRegistry()
        b1 = MetricBundle(bundle_id="x", name="X1", description="")
        b2 = MetricBundle(bundle_id="x", name="X2", description="")
        reg.register(b1)
        reg.register(b2)
        assert reg.get("x").name == "X2"


# ---------------------------------------------------------------------------
# BundleRegistry - list_bundles
# ---------------------------------------------------------------------------


class TestBundleRegistryList:
    def test_list_sorted(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(bundle_id="c", name="C", description=""))
        reg.register(MetricBundle(bundle_id="a", name="A", description=""))
        reg.register(MetricBundle(bundle_id="b", name="B", description=""))
        ids = [b.bundle_id for b in reg.list_bundles()]
        assert ids == ["a", "b", "c"]

    def test_list_empty(self):
        reg = BundleRegistry()
        assert reg.list_bundles() == []


# ---------------------------------------------------------------------------
# BundleRegistry - search
# ---------------------------------------------------------------------------


class TestBundleRegistrySearch:
    def _make_registry(self) -> BundleRegistry:
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="safety", name="Safety", description="Content safety",
            metric_ids=[], tags=["safety", "moderation"],
        ))
        reg.register(MetricBundle(
            bundle_id="quality", name="Quality", description="Output quality",
            metric_ids=[], tags=["quality"],
        ))
        return reg

    def test_search_by_name(self):
        reg = self._make_registry()
        results = reg.search("Safety")
        assert len(results) == 1
        assert results[0].bundle_id == "safety"

    def test_search_case_insensitive(self):
        reg = self._make_registry()
        results = reg.search("safety")
        assert len(results) == 1

    def test_search_by_description(self):
        reg = self._make_registry()
        results = reg.search("Content")
        assert len(results) == 1

    def test_search_by_tag(self):
        reg = self._make_registry()
        results = reg.search("moderation")
        assert len(results) == 1

    def test_search_no_match(self):
        reg = self._make_registry()
        results = reg.search("zzz_no_match")
        assert results == []

    def test_search_partial(self):
        reg = self._make_registry()
        results = reg.search("afe")
        assert len(results) == 1
        assert results[0].bundle_id == "safety"


# ---------------------------------------------------------------------------
# BundleRegistry - get_metrics
# ---------------------------------------------------------------------------


class TestBundleRegistryGetMetrics:
    def test_get_metrics(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="x", name="X", description="",
            metric_ids=["m1", "m2"],
        ))
        assert reg.get_metrics("x") == ["m1", "m2"]

    def test_get_metrics_missing(self):
        reg = BundleRegistry()
        assert reg.get_metrics("nope") == []

    def test_get_metrics_returns_copy(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="x", name="X", description="",
            metric_ids=["m1"],
        ))
        ids = reg.get_metrics("x")
        ids.append("m2")
        assert reg.get_metrics("x") == ["m1"]


# ---------------------------------------------------------------------------
# BundleRegistry - compose
# ---------------------------------------------------------------------------


class TestBundleRegistryCompose:
    def test_compose_two(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="a", name="A", description="",
            metric_ids=["m1", "m2"], tags=["t1"],
        ))
        reg.register(MetricBundle(
            bundle_id="b", name="B", description="",
            metric_ids=["m2", "m3"], tags=["t2"],
        ))
        result = reg.compose(["a", "b"])
        assert result.metric_ids == ["m1", "m2", "m3"]
        assert result.tags == ["t1", "t2"]

    def test_compose_deduplicates_metrics(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="a", name="A", description="",
            metric_ids=["m1", "m2"],
        ))
        reg.register(MetricBundle(
            bundle_id="b", name="B", description="",
            metric_ids=["m1", "m2"],
        ))
        result = reg.compose(["a", "b"])
        assert result.metric_ids == ["m1", "m2"]

    def test_compose_deduplicates_tags(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="a", name="A", description="", tags=["t1", "t2"],
        ))
        reg.register(MetricBundle(
            bundle_id="b", name="B", description="", tags=["t2", "t3"],
        ))
        result = reg.compose(["a", "b"])
        assert result.tags == ["t1", "t2", "t3"]

    def test_compose_missing_bundle_skipped(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="a", name="A", description="",
            metric_ids=["m1"],
        ))
        result = reg.compose(["a", "nonexistent"])
        assert result.metric_ids == ["m1"]

    def test_compose_empty_list(self):
        reg = BundleRegistry()
        result = reg.compose([])
        assert result.metric_ids == []
        assert result.name == "Composite"

    def test_compose_bundle_id_format(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(bundle_id="a", name="A", description=""))
        reg.register(MetricBundle(bundle_id="b", name="B", description=""))
        result = reg.compose(["a", "b"])
        assert result.bundle_id == "a+b"

    def test_compose_name_format(self):
        reg = BundleRegistry()
        reg.register(MetricBundle(bundle_id="a", name="Alpha", description=""))
        reg.register(MetricBundle(bundle_id="b", name="Beta", description=""))
        result = reg.compose(["a", "b"])
        assert result.name == "Alpha + Beta"


# ---------------------------------------------------------------------------
# BundleRegistry - recommend
# ---------------------------------------------------------------------------


class TestBundleRegistryRecommend:
    def _make_registry(self) -> BundleRegistry:
        reg = BundleRegistry()
        reg.register(MetricBundle(
            bundle_id="safety", name="Safety", description="",
            metric_ids=[], tags=["safety", "moderation"],
        ))
        reg.register(MetricBundle(
            bundle_id="agent", name="Agent", description="",
            metric_ids=[], tags=["agent", "tool", "reasoning"],
        ))
        reg.register(MetricBundle(
            bundle_id="quality", name="Quality", description="",
            metric_ids=[], tags=["quality"],
        ))
        return reg

    def test_recommend_single_match(self):
        reg = self._make_registry()
        results = reg.recommend(["quality"])
        assert len(results) == 1
        assert results[0].bundle_id == "quality"

    def test_recommend_multiple_matches(self):
        reg = self._make_registry()
        results = reg.recommend(["agent", "tool"])
        # agent matches 2 tags, should be first
        assert results[0].bundle_id == "agent"

    def test_recommend_no_match(self):
        reg = self._make_registry()
        results = reg.recommend(["zzz"])
        assert results == []

    def test_recommend_case_insensitive(self):
        reg = self._make_registry()
        results = reg.recommend(["SAFETY"])
        assert len(results) == 1
        assert results[0].bundle_id == "safety"

    def test_recommend_sorted_by_match_count(self):
        reg = self._make_registry()
        # agent has 3 tags, safety has 2, quality has 1
        # If we match "agent" + "tool" + "safety": agent=2 matches, safety=1 match
        results = reg.recommend(["agent", "tool", "safety"])
        assert results[0].bundle_id == "agent"
        assert results[1].bundle_id == "safety"


# ---------------------------------------------------------------------------
# load_bundles_from_config
# ---------------------------------------------------------------------------


class TestLoadBundlesFromConfig:
    def test_basic_load(self):
        config = {
            "bundles": [
                {
                    "bundle_id": "custom",
                    "name": "Custom",
                    "description": "A custom bundle",
                    "metric_ids": ["m1"],
                    "tags": ["t1"],
                },
            ]
        }
        reg = load_bundles_from_config(config)
        assert reg.get("custom") is not None
        assert reg.get("custom").metric_ids == ["m1"]

    def test_empty_config(self):
        reg = load_bundles_from_config({})
        assert reg.list_bundles() == []

    def test_empty_bundles_list(self):
        reg = load_bundles_from_config({"bundles": []})
        assert reg.list_bundles() == []

    def test_multiple_bundles(self):
        config = {
            "bundles": [
                {"bundle_id": "a", "name": "A", "description": ""},
                {"bundle_id": "b", "name": "B", "description": ""},
            ]
        }
        reg = load_bundles_from_config(config)
        assert len(reg.list_bundles()) == 2


# ---------------------------------------------------------------------------
# BUILTIN_BUNDLES
# ---------------------------------------------------------------------------


class TestBuiltinBundles:
    def test_has_four_bundles(self):
        assert len(BUILTIN_BUNDLES.list_bundles()) == 4

    def test_safety_bundle(self):
        b = BUILTIN_BUNDLES.get("safety")
        assert b is not None
        assert "toxicity" in b.metric_ids
        assert "injection" in b.metric_ids
        assert "pii" in b.metric_ids

    def test_quality_bundle(self):
        b = BUILTIN_BUNDLES.get("quality")
        assert b is not None
        assert "helpfulness" in b.metric_ids

    def test_accuracy_bundle(self):
        b = BUILTIN_BUNDLES.get("accuracy")
        assert b is not None
        assert "hallucination" in b.metric_ids

    def test_agent_bundle(self):
        b = BUILTIN_BUNDLES.get("agent")
        assert b is not None
        assert "tool_call" in b.metric_ids
        assert "goal_completion" in b.metric_ids


# ---------------------------------------------------------------------------
# format_bundle_list
# ---------------------------------------------------------------------------


class TestFormatBundleList:
    def test_empty_list(self):
        result = format_bundle_list([])
        assert "No bundles found" in result

    def test_contains_bundle_id(self):
        bundles = [MetricBundle(
            bundle_id="test", name="Test", description="desc",
            metric_ids=["m1"], tags=["t1"],
        )]
        result = format_bundle_list(bundles)
        assert "[test]" in result

    def test_contains_metrics(self):
        bundles = [MetricBundle(
            bundle_id="x", name="X", description="d",
            metric_ids=["alpha", "beta"], tags=[],
        )]
        result = format_bundle_list(bundles)
        assert "alpha" in result
        assert "beta" in result

    def test_contains_tags(self):
        bundles = [MetricBundle(
            bundle_id="x", name="X", description="d",
            metric_ids=[], tags=["tag1", "tag2"],
        )]
        result = format_bundle_list(bundles)
        assert "tag1" in result


# ---------------------------------------------------------------------------
# format_bundle_detail
# ---------------------------------------------------------------------------


class TestFormatBundleDetail:
    def test_contains_name(self):
        b = MetricBundle(
            bundle_id="test", name="Test Bundle", description="desc",
            metric_ids=["m1"], tags=["t1"],
        )
        result = format_bundle_detail(b)
        assert "Test Bundle" in result

    def test_contains_description(self):
        b = MetricBundle(
            bundle_id="test", name="T", description="A detailed description",
        )
        result = format_bundle_detail(b)
        assert "A detailed description" in result

    def test_contains_metric_ids(self):
        b = MetricBundle(
            bundle_id="test", name="T", description="",
            metric_ids=["alpha", "beta"],
        )
        result = format_bundle_detail(b)
        assert "alpha" in result
        assert "beta" in result

    def test_metric_count(self):
        b = MetricBundle(
            bundle_id="test", name="T", description="",
            metric_ids=["m1", "m2", "m3"],
        )
        result = format_bundle_detail(b)
        assert "Metrics (3)" in result

    def test_no_tags(self):
        b = MetricBundle(
            bundle_id="test", name="T", description="", tags=[],
        )
        result = format_bundle_detail(b)
        assert "none" in result

    def test_with_tags(self):
        b = MetricBundle(
            bundle_id="test", name="T", description="",
            tags=["safety", "agent"],
        )
        result = format_bundle_detail(b)
        assert "safety" in result
        assert "agent" in result
