"""Tests for deterministic evaluation mode."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.evaluation.deterministic import (
    DeterministicConfig,
    RunManifest,
    sort_items_deterministically,
    sort_metrics_deterministically,
    compute_dataset_hash,
    build_manifest,
    verify_manifest,
)
from evalyn_sdk.models import DatasetItem, Metric, MetricSpec, MetricResult


def _make_item(item_id: str) -> DatasetItem:
    return DatasetItem(id=item_id, input={"q": item_id}, output=f"answer-{item_id}")


def _make_metric(metric_id: str) -> Metric:
    spec = MetricSpec(id=metric_id, name=metric_id, type="objective")
    return Metric(spec, lambda c, i: MetricResult(
        metric_id=metric_id, item_id=i.id, call_id=c.id,
        score=1.0, passed=True, details={},
    ))


class TestDeterministicConfig:
    def test_default_disabled(self):
        cfg = DeterministicConfig()
        assert cfg.enabled is False
        assert cfg.seed == 42

    def test_as_dict(self):
        cfg = DeterministicConfig(enabled=True, seed=123)
        assert cfg.as_dict() == {"enabled": True, "seed": 123}


class TestSortItemsDeterministically:
    def test_produces_stable_order(self):
        items = [_make_item("c"), _make_item("a"), _make_item("b")]
        sorted1 = sort_items_deterministically(items, seed=42)
        sorted2 = sort_items_deterministically(items, seed=42)
        assert [i.id for i in sorted1] == [i.id for i in sorted2]

    def test_different_seed_different_order(self):
        items = [_make_item(f"item-{i}") for i in range(20)]
        sorted1 = sort_items_deterministically(items, seed=1)
        sorted2 = sort_items_deterministically(items, seed=999)
        # With 20 items, different seeds should produce different orders
        ids1 = [i.id for i in sorted1]
        ids2 = [i.id for i in sorted2]
        assert ids1 != ids2  # extremely unlikely to be equal

    def test_same_items_regardless_of_input_order(self):
        items_a = [_make_item("x"), _make_item("y"), _make_item("z")]
        items_b = [_make_item("z"), _make_item("x"), _make_item("y")]
        sorted_a = sort_items_deterministically(items_a, seed=42)
        sorted_b = sort_items_deterministically(items_b, seed=42)
        assert [i.id for i in sorted_a] == [i.id for i in sorted_b]

    def test_empty_items(self):
        assert sort_items_deterministically([], seed=42) == []

    def test_single_item(self):
        items = [_make_item("only")]
        result = sort_items_deterministically(items)
        assert len(result) == 1
        assert result[0].id == "only"


class TestSortMetricsDeterministically:
    def test_sorts_by_id(self):
        metrics = [_make_metric("z_metric"), _make_metric("a_metric"), _make_metric("m_metric")]
        sorted_m = sort_metrics_deterministically(metrics)
        assert [m.spec.id for m in sorted_m] == ["a_metric", "m_metric", "z_metric"]

    def test_empty(self):
        assert sort_metrics_deterministically([]) == []


class TestComputeDatasetHash:
    def test_deterministic(self):
        items = [_make_item("a"), _make_item("b")]
        h1 = compute_dataset_hash(items)
        h2 = compute_dataset_hash(items)
        assert h1 == h2

    def test_order_independent(self):
        items_a = [_make_item("a"), _make_item("b")]
        items_b = [_make_item("b"), _make_item("a")]
        assert compute_dataset_hash(items_a) == compute_dataset_hash(items_b)

    def test_different_content_different_hash(self):
        items_a = [_make_item("a")]
        items_b = [_make_item("b")]
        assert compute_dataset_hash(items_a) != compute_dataset_hash(items_b)

    def test_hash_is_16_chars(self):
        items = [_make_item("test")]
        assert len(compute_dataset_hash(items)) == 16

    def test_empty(self):
        h = compute_dataset_hash([])
        assert len(h) == 16


class TestRunManifest:
    def test_as_dict_roundtrip(self):
        m = RunManifest(
            evalyn_version="0.15.0",
            python_version="3.13.5",
            seed=42,
            dataset_hash="abc123",
            metric_hashes={"help": "hash1"},
            provider="gemini",
            model="gemini-2.5-flash-lite",
        )
        d = m.as_dict()
        restored = RunManifest.from_dict(d)
        assert restored.evalyn_version == "0.15.0"
        assert restored.seed == 42
        assert restored.metric_hashes == {"help": "hash1"}

    def test_from_dict_defaults(self):
        m = RunManifest.from_dict({})
        assert m.evalyn_version == ""
        assert m.seed is None
        assert m.metric_hashes == {}


class TestBuildManifest:
    def test_basic(self):
        items = [_make_item("a"), _make_item("b")]
        metrics = [_make_metric("m1")]
        m = build_manifest(items, metrics, seed=42, provider="gemini")
        assert m.seed == 42
        assert m.provider == "gemini"
        assert m.dataset_hash != ""
        assert "m1" in m.metric_hashes
        assert m.python_version != ""

    def test_without_seed(self):
        m = build_manifest([], [], seed=None)
        assert m.seed is None


class TestVerifyManifest:
    def test_identical_manifests(self):
        m = RunManifest(evalyn_version="1.0", dataset_hash="abc", provider="gemini")
        assert verify_manifest(m, m) == []

    def test_version_change(self):
        prev = RunManifest(evalyn_version="1.0")
        curr = RunManifest(evalyn_version="2.0")
        warnings = verify_manifest(curr, prev)
        assert len(warnings) == 1
        assert "version changed" in warnings[0].lower()

    def test_dataset_change(self):
        prev = RunManifest(dataset_hash="abc123")
        curr = RunManifest(dataset_hash="def456")
        warnings = verify_manifest(curr, prev)
        assert any("dataset" in w.lower() for w in warnings)

    def test_provider_change(self):
        prev = RunManifest(provider="gemini")
        curr = RunManifest(provider="openai")
        warnings = verify_manifest(curr, prev)
        assert any("provider" in w.lower() for w in warnings)

    def test_metric_modified(self):
        prev = RunManifest(metric_hashes={"help": "hash_v1"})
        curr = RunManifest(metric_hashes={"help": "hash_v2"})
        warnings = verify_manifest(curr, prev)
        assert any("help" in w and "changed" in w for w in warnings)

    def test_metric_added(self):
        prev = RunManifest(metric_hashes={"help": "h1"})
        curr = RunManifest(metric_hashes={"help": "h1", "safety": "h2"})
        warnings = verify_manifest(curr, prev)
        assert any("safety" in w and "added" in w for w in warnings)

    def test_metric_removed(self):
        prev = RunManifest(metric_hashes={"help": "h1", "safety": "h2"})
        curr = RunManifest(metric_hashes={"help": "h1"})
        warnings = verify_manifest(curr, prev)
        assert any("safety" in w and "removed" in w for w in warnings)

    def test_no_warnings_on_empty_prev(self):
        prev = RunManifest()
        curr = RunManifest(evalyn_version="2.0", provider="gemini")
        warnings = verify_manifest(curr, prev)
        assert warnings == []  # empty previous = no comparison baseline
