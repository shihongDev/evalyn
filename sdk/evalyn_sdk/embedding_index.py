"""Dataset embedding index for fast similarity search.

Pre-compute and store word-set feature indices for lightweight similarity
lookups. Pure Python - uses Jaccard similarity on word-set features as an
embedding proxy (no neural embeddings required).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def extract_features(text: str) -> list[str]:
    """Extract lowercased word features from text.

    Words are alphanumeric tokens with minimum length 2, returned sorted.
    """
    words = _WORD_RE.findall(text.lower())
    return sorted(set(w for w in words if len(w) >= 2))


def compute_feature_hash(features: list[str]) -> str:
    """Compute SHA256 hash of joined features."""
    joined = " ".join(features)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class EmbeddingEntry:
    """A single entry in the embedding index."""

    item_id: str
    features: list[str]
    feature_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "features": self.features,
            "feature_hash": self.feature_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingEntry:
        return cls(
            item_id=data["item_id"],
            features=data.get("features", []),
            feature_hash=data.get("feature_hash", ""),
        )


def _jaccard_similarity(a: set, b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class EmbeddingIndex:
    """In-memory word-set feature index for fast similarity search."""

    def __init__(self) -> None:
        self._entries: dict[str, EmbeddingEntry] = {}

    def add(self, item_id: str, text: str) -> None:
        """Extract word features from text and store under item_id."""
        features = extract_features(text)
        fhash = compute_feature_hash(features)
        self._entries[item_id] = EmbeddingEntry(
            item_id=item_id,
            features=features,
            feature_hash=fhash,
        )

    def add_batch(self, items: dict[str, str]) -> int:
        """Add multiple items at once. Returns the count of items added."""
        for item_id, text in items.items():
            self.add(item_id, text)
        return len(items)

    def get(self, item_id: str) -> EmbeddingEntry | None:
        """Retrieve an entry by item_id, or None if not found."""
        return self._entries.get(item_id)

    def size(self) -> int:
        """Return the number of entries in the index."""
        return len(self._entries)

    def nearest_neighbors(
        self, query_text: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """Find k nearest neighbors by Jaccard similarity.

        Returns a list of (item_id, similarity) tuples sorted by
        descending similarity.
        """
        query_features = set(extract_features(query_text))
        scored: list[tuple[str, float]] = []
        for entry in self._entries.values():
            sim = _jaccard_similarity(query_features, set(entry.features))
            scored.append((entry.item_id, sim))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def find_duplicates(
        self, threshold: float = 0.95
    ) -> list[tuple[str, str, float]]:
        """Find pairs of entries with Jaccard similarity >= threshold.

        Returns a list of (item_id_a, item_id_b, similarity) tuples
        sorted by descending similarity.
        """
        ids = sorted(self._entries.keys())
        results: list[tuple[str, str, float]] = []
        for i in range(len(ids)):
            entry_a = self._entries[ids[i]]
            set_a = set(entry_a.features)
            for j in range(i + 1, len(ids)):
                entry_b = self._entries[ids[j]]
                sim = _jaccard_similarity(set_a, set(entry_b.features))
                if sim >= threshold:
                    results.append((ids[i], ids[j], sim))
        results.sort(key=lambda x: (-x[2], x[0], x[1]))
        return results

    def save(self, path: str) -> None:
        """Save the index as JSON to the given file path."""
        data = {
            "entries": [e.as_dict() for e in self._entries.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> EmbeddingIndex:
        """Load an index from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        index = cls()
        for entry_data in data.get("entries", []):
            entry = EmbeddingEntry.from_dict(entry_data)
            index._entries[entry.item_id] = entry
        return index

    def load(self, path: str) -> None:
        """Load entries from a JSON file into this index (instance method)."""
        loaded = self.load_from_file(path)
        self._entries = loaded._entries


def build_index_from_dataset(items: dict[str, str]) -> EmbeddingIndex:
    """Convenience function to build an index from a dict of id -> text."""
    index = EmbeddingIndex()
    index.add_batch(items)
    return index


def format_index_stats(index: EmbeddingIndex) -> str:
    """Format human-readable stats about the index."""
    n = index.size()
    if n == 0:
        return "Empty index (0 entries)."

    total_features = 0
    min_features = float("inf")
    max_features = 0
    for entry in index._entries.values():
        count = len(entry.features)
        total_features += count
        if count < min_features:
            min_features = count
        if count > max_features:
            max_features = count

    avg_features = total_features / n

    lines = [
        "Embedding Index Stats:",
        f"  Entries:      {n}",
        f"  Total features: {total_features}",
        f"  Avg features: {avg_features:.1f}",
        f"  Min features: {min_features}",
        f"  Max features: {max_features}",
    ]
    return "\n".join(lines)
