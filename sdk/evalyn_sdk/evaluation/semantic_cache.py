"""Semantic caching for LLM judge calls.

Content-addressable cache that also supports fuzzy matching for
similar-but-not-identical inputs using text similarity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    """A cached judge response."""

    key: str
    input_text: str
    output_text: str
    result: Dict[str, Any]
    hit_count: int = 0


@dataclass
class SemanticCacheStats:
    """Statistics for semantic cache usage."""

    exact_hits: int = 0
    fuzzy_hits: int = 0
    misses: int = 0
    stores: int = 0

    @property
    def total_lookups(self) -> int:
        return self.exact_hits + self.fuzzy_hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        return (self.exact_hits + self.fuzzy_hits) / total if total > 0 else 0.0

    @property
    def api_calls_saved(self) -> int:
        return self.exact_hits + self.fuzzy_hits

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exact_hits": self.exact_hits,
            "fuzzy_hits": self.fuzzy_hits,
            "misses": self.misses,
            "stores": self.stores,
            "hit_rate": round(self.hit_rate, 4),
            "api_calls_saved": self.api_calls_saved,
        }


class SemanticCache:
    """Cache for LLM judge calls with exact and fuzzy matching.

    Exact matching uses content hash. Fuzzy matching uses SequenceMatcher
    to find similar cached entries when exact match misses.

    Usage:
        cache = SemanticCache(fuzzy_threshold=0.95)
        result = cache.get(prompt, input_text, output_text)
        if result is None:
            result = call_llm_judge(...)
            cache.put(prompt, input_text, output_text, result)
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.95,
        fuzzy_enabled: bool = True,
        max_entries: int = 10000,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_enabled = fuzzy_enabled
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = SemanticCacheStats()

    @property
    def stats(self) -> SemanticCacheStats:
        return self._stats

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def _make_key(prompt: str, input_text: str, output_text: str) -> str:
        content = json.dumps({
            "prompt": prompt,
            "input": input_text,
            "output": output_text,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self,
        prompt: str,
        input_text: str,
        output_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached result.

        Tries exact match first, then fuzzy match if enabled.

        Returns:
            Cached result dict, or None on miss.
        """
        key = self._make_key(prompt, input_text, output_text)

        # Exact match
        entry = self._cache.get(key)
        if entry is not None:
            entry.hit_count += 1
            self._stats.exact_hits += 1
            return entry.result

        # Fuzzy match
        if self.fuzzy_enabled and self._cache:
            combined = input_text + " " + output_text
            best_match = None
            best_sim = 0.0

            for cached_entry in self._cache.values():
                cached_combined = cached_entry.input_text + " " + cached_entry.output_text
                sim = SequenceMatcher(None, combined, cached_combined).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_match = cached_entry

            if best_match and best_sim >= self.fuzzy_threshold:
                best_match.hit_count += 1
                self._stats.fuzzy_hits += 1
                return best_match.result

        self._stats.misses += 1
        return None

    def put(
        self,
        prompt: str,
        input_text: str,
        output_text: str,
        result: Dict[str, Any],
    ) -> None:
        """Store a result in the cache."""
        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        key = self._make_key(prompt, input_text, output_text)
        self._cache[key] = CacheEntry(
            key=key,
            input_text=input_text,
            output_text=output_text,
            result=result,
        )
        self._stats.stores += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def _evict_oldest(self) -> None:
        """Remove the least-used entry."""
        if not self._cache:
            return
        min_key = min(self._cache, key=lambda k: self._cache[k].hit_count)
        del self._cache[min_key]
