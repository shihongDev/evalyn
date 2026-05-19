"""Evaluation caching: skip re-computing unchanged metric/item pairs.

Content-addressable cache keyed by hash of (item content + metric identity).
Stores MetricResult objects and returns them on cache hit, avoiding redundant
LLM judge calls for identical evaluations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache usage during an evaluation run."""

    hits: int = 0
    misses: int = 0
    stores: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


class EvalCache:
    """Content-addressable cache for evaluation results.

    Cache key = hash(item_input + item_output + metric_id + metric_config_hash + provider + model)

    The cache is stored as a JSON file on disk, loaded into memory on init.
    Thread-safe for reads (dict lookups are atomic in CPython).

    Usage:
        cache = EvalCache(path=".evalyn/eval_cache.json")
        key = cache.make_key(item, metric_spec, provider, model)
        result = cache.get(key)
        if result is None:
            result = evaluate(...)
            cache.put(key, result)
        cache.save()  # persist to disk
    """

    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled
        self._cache: dict[str, dict[str, Any]] = {}
        self._stats = CacheStats()

        if self.enabled and self.path and self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug("Loaded eval cache with %d entries", len(self._cache))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load eval cache: %s", e)
                self._cache = {}

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def make_key(
        item_input: Any,
        item_output: Any,
        metric_id: str,
        config_hash: str,
        provider: str = "",
        model: str = "",
    ) -> str:
        """Generate a deterministic cache key from evaluation parameters.

        Args:
            item_input: The item's input data.
            item_output: The item's output data.
            metric_id: Metric identifier.
            config_hash: MetricSpec.version_hash (captures prompt/rubric/threshold).
            provider: LLM provider name (for subjective metrics).
            model: Model name (for subjective metrics).

        Returns:
            SHA-256 hash string (32 chars).
        """
        content = json.dumps({
            "input": item_input,
            "output": item_output,
            "metric_id": metric_id,
            "config_hash": config_hash,
            "provider": provider,
            "model": model,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached result.

        Returns:
            Cached MetricResult as dict, or None on miss.
        """
        if not self.enabled:
            self._stats.misses += 1
            return None

        result = self._cache.get(key)
        if result is not None:
            self._stats.hits += 1
        else:
            self._stats.misses += 1
        return result

    def put(self, key: str, result: dict[str, Any]) -> None:
        """Store a result in the cache."""
        if not self.enabled:
            return
        self._cache[key] = result
        self._stats.stores += 1

    def save(self) -> None:
        """Persist cache to disk."""
        if not self.enabled or not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, default=str)
            logger.debug("Saved eval cache with %d entries", len(self._cache))
        except OSError as e:
            logger.warning("Could not save eval cache: %s", e)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        if self.path and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass

    def invalidate_metric(self, metric_id: str) -> int:
        """Remove all cached entries for a specific metric.

        Useful when a metric's definition changes.

        Returns:
            Number of entries removed.
        """
        # We can't efficiently filter by metric_id from the hash key alone,
        # so we inspect the stored result's metric_id field.
        to_remove = [
            k for k, v in self._cache.items()
            if isinstance(v, dict) and v.get("metric_id") == metric_id
        ]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)
