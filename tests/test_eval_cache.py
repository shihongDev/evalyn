"""Tests for evaluation caching."""

import json
import pytest
from pathlib import Path
from evalyn_sdk.evaluation.cache import EvalCache, CacheStats


class TestCacheKey:
    """Test cache key generation."""

    def test_deterministic(self):
        k1 = EvalCache.make_key({"q": "test"}, "answer", "helpfulness", "abc123")
        k2 = EvalCache.make_key({"q": "test"}, "answer", "helpfulness", "abc123")
        assert k1 == k2

    def test_different_input_different_key(self):
        k1 = EvalCache.make_key({"q": "test1"}, "answer", "helpfulness", "abc123")
        k2 = EvalCache.make_key({"q": "test2"}, "answer", "helpfulness", "abc123")
        assert k1 != k2

    def test_different_output_different_key(self):
        k1 = EvalCache.make_key({"q": "test"}, "answer1", "helpfulness", "abc123")
        k2 = EvalCache.make_key({"q": "test"}, "answer2", "helpfulness", "abc123")
        assert k1 != k2

    def test_different_metric_different_key(self):
        k1 = EvalCache.make_key({"q": "test"}, "answer", "helpfulness", "abc123")
        k2 = EvalCache.make_key({"q": "test"}, "answer", "toxicity", "abc123")
        assert k1 != k2

    def test_different_config_hash_different_key(self):
        k1 = EvalCache.make_key({"q": "test"}, "answer", "helpfulness", "v1hash")
        k2 = EvalCache.make_key({"q": "test"}, "answer", "helpfulness", "v2hash")
        assert k1 != k2

    def test_different_provider_different_key(self):
        k1 = EvalCache.make_key({"q": "test"}, "answer", "help", "h", provider="gemini")
        k2 = EvalCache.make_key({"q": "test"}, "answer", "help", "h", provider="openai")
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = EvalCache.make_key({"q": "test"}, "answer", "help", "h", model="gpt-4")
        k2 = EvalCache.make_key({"q": "test"}, "answer", "help", "h", model="gpt-3.5")
        assert k1 != k2

    def test_key_is_32_chars(self):
        k = EvalCache.make_key({"q": "test"}, "answer", "helpfulness", "abc")
        assert len(k) == 32

    def test_handles_none_values(self):
        k = EvalCache.make_key(None, None, "test", "hash")
        assert len(k) == 32


class TestCacheGetPut:
    """Test basic get/put operations."""

    def test_miss_returns_none(self):
        cache = EvalCache(enabled=True)
        assert cache.get("nonexistent") is None

    def test_put_then_get(self):
        cache = EvalCache(enabled=True)
        result = {"metric_id": "help", "score": 0.9, "passed": True}
        cache.put("key1", result)
        assert cache.get("key1") == result

    def test_multiple_entries(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.put("k2", {"score": 0.9})
        assert cache.get("k1")["score"] == 0.8
        assert cache.get("k2")["score"] == 0.9

    def test_overwrite(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"score": 0.5})
        cache.put("k1", {"score": 0.9})
        assert cache.get("k1")["score"] == 0.9

    def test_size(self):
        cache = EvalCache(enabled=True)
        assert cache.size == 0
        cache.put("k1", {"score": 0.8})
        assert cache.size == 1
        cache.put("k2", {"score": 0.9})
        assert cache.size == 2


class TestCacheDisabled:
    """Test behavior when cache is disabled."""

    def test_disabled_get_returns_none(self):
        cache = EvalCache(enabled=False)
        cache._cache["key1"] = {"score": 0.9}
        assert cache.get("key1") is None

    def test_disabled_put_does_nothing(self):
        cache = EvalCache(enabled=False)
        cache.put("key1", {"score": 0.9})
        assert cache.size == 0

    def test_disabled_save_does_nothing(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = EvalCache(path=path, enabled=False)
        cache.save()
        assert not path.exists()


class TestCacheStats:
    """Test cache statistics tracking."""

    def test_initial_stats(self):
        cache = EvalCache(enabled=True)
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.stores == 0
        assert cache.stats.hit_rate == 0.0

    def test_stats_on_miss(self):
        cache = EvalCache(enabled=True)
        cache.get("missing")
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_stats_on_hit(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.get("k1")
        assert cache.stats.hits == 1
        assert cache.stats.stores == 1

    def test_hit_rate(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.get("k1")       # hit
        cache.get("k1")       # hit
        cache.get("missing")  # miss
        assert cache.stats.hit_rate == pytest.approx(2/3, abs=0.01)

    def test_stats_as_dict(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.get("k1")
        d = cache.stats.as_dict()
        assert d["hits"] == 1
        assert d["misses"] == 0
        assert d["stores"] == 1
        assert d["total"] == 1

    def test_disabled_cache_counts_misses(self):
        cache = EvalCache(enabled=False)
        cache.get("anything")
        assert cache.stats.misses == 1


class TestCachePersistence:
    """Test save/load from disk."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "cache.json"

        # Save
        cache1 = EvalCache(path=path, enabled=True)
        cache1.put("k1", {"score": 0.8, "metric_id": "help"})
        cache1.put("k2", {"score": 0.9, "metric_id": "safety"})
        cache1.save()

        assert path.exists()

        # Load
        cache2 = EvalCache(path=path, enabled=True)
        assert cache2.size == 2
        assert cache2.get("k1")["score"] == 0.8
        assert cache2.get("k2")["score"] == 0.9

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cache = EvalCache(path=path, enabled=True)
        assert cache.size == 0

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{")
        cache = EvalCache(path=path, enabled=True)
        assert cache.size == 0  # gracefully handles corruption

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "cache.json"
        cache = EvalCache(path=path, enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.save()
        assert path.exists()


class TestCacheClear:
    """Test cache clearing."""

    def test_clear_memory(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.put("k2", {"score": 0.9})
        cache.clear()
        assert cache.size == 0
        assert cache.get("k1") is None

    def test_clear_with_file(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = EvalCache(path=path, enabled=True)
        cache.put("k1", {"score": 0.8})
        cache.save()
        assert path.exists()
        cache.clear()
        assert not path.exists()


class TestCacheInvalidation:
    """Test metric-specific cache invalidation."""

    def test_invalidate_metric(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"metric_id": "help", "score": 0.8})
        cache.put("k2", {"metric_id": "safety", "score": 0.9})
        cache.put("k3", {"metric_id": "help", "score": 0.7})

        removed = cache.invalidate_metric("help")
        assert removed == 2
        assert cache.size == 1
        assert cache.get("k2")["metric_id"] == "safety"

    def test_invalidate_nonexistent_metric(self):
        cache = EvalCache(enabled=True)
        cache.put("k1", {"metric_id": "help", "score": 0.8})
        removed = cache.invalidate_metric("nonexistent")
        assert removed == 0
        assert cache.size == 1

    def test_invalidate_empty_cache(self):
        cache = EvalCache(enabled=True)
        removed = cache.invalidate_metric("help")
        assert removed == 0
