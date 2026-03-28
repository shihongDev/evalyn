"""
Multi-DB queries: query across prod and test databases simultaneously.

Provides a manager for multiple database sources and utilities for
cross-database filtering, comparison, and result merging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DatabaseSource:
    """A single database source."""

    name: str
    path: str = ""
    db_type: str = "sqlite"
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "db_type": self.db_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatabaseSource:
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            db_type=data.get("db_type", "sqlite"),
            description=data.get("description", ""),
        )


@dataclass
class MultiDBConfig:
    """Configuration for multi-database queries."""

    sources: List[DatabaseSource] = field(default_factory=list)
    default_source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sources": [s.as_dict() for s in self.sources],
            "default_source": self.default_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiDBConfig:
        sources = [
            DatabaseSource.from_dict(s) for s in data.get("sources", [])
        ]
        return cls(
            sources=sources,
            default_source=data.get("default_source", ""),
        )


@dataclass
class CrossDBQueryResult:
    """Result from querying a single database source."""

    source: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    count: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "records": self.records,
            "count": self.count,
            "duration_ms": self.duration_ms,
        }


@dataclass
class MultiDBReport:
    """Aggregated report across multiple database sources."""

    results: List[CrossDBQueryResult] = field(default_factory=list)
    total_records: int = 0
    sources_queried: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_records": self.total_records,
            "sources_queried": self.sources_queried,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiDBReport:
        results = [
            CrossDBQueryResult(
                source=r.get("source", ""),
                records=r.get("records", []),
                count=r.get("count", 0),
                duration_ms=r.get("duration_ms", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            total_records=data.get("total_records", 0),
            sources_queried=data.get("sources_queried", []),
        )

    def format_text(self) -> str:
        lines = [
            "Multi-DB Report",
            "-" * 40,
            f"  Sources queried: {', '.join(self.sources_queried) or 'none'}",
            f"  Total records: {self.total_records}",
        ]
        for r in self.results:
            lines.append(f"  [{r.source}] {r.count} records ({r.duration_ms:.1f} ms)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MultiDBManager:
    """Manage and query across multiple database sources."""

    def __init__(self, config: MultiDBConfig) -> None:
        self._config = config
        self._sources: List[DatabaseSource] = list(config.sources)

    def add_source(self, source: DatabaseSource) -> None:
        """Add a database source."""
        self._sources.append(source)

    def remove_source(self, name: str) -> bool:
        """Remove a source by name. Returns True if found and removed."""
        for i, s in enumerate(self._sources):
            if s.name == name:
                self._sources.pop(i)
                return True
        return False

    def list_sources(self) -> List[DatabaseSource]:
        """Return all registered sources."""
        return list(self._sources)

    def query_all(
        self,
        filter_fn: Callable[[Dict[str, Any]], bool],
        records_by_source: Dict[str, List[Dict[str, Any]]],
    ) -> MultiDBReport:
        """Query all sources with a filter function.

        records_by_source maps source name to its records. The filter_fn
        is applied to each record to decide inclusion.
        """
        results: List[CrossDBQueryResult] = []
        total = 0
        queried: List[str] = []

        for source in self._sources:
            records = records_by_source.get(source.name, [])
            start = time.monotonic()
            matched = [r for r in records if filter_fn(r)]
            elapsed_ms = (time.monotonic() - start) * 1000.0

            result = CrossDBQueryResult(
                source=source.name,
                records=matched,
                count=len(matched),
                duration_ms=elapsed_ms,
            )
            results.append(result)
            total += len(matched)
            queried.append(source.name)

        return MultiDBReport(
            results=results,
            total_records=total,
            sources_queried=queried,
        )

    def compare_sources(
        self,
        records_by_source: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Compare record counts, overlap, and unique records per source.

        Uses the "id" field in records to determine overlap.
        """
        source_names = [s.name for s in self._sources]
        counts: Dict[str, int] = {}
        id_sets: Dict[str, set] = {}

        for name in source_names:
            records = records_by_source.get(name, [])
            counts[name] = len(records)
            id_sets[name] = {r.get("id") for r in records if "id" in r}

        # Compute pairwise overlap
        all_ids: set = set()
        for ids in id_sets.values():
            all_ids |= ids

        # Common across ALL sources
        if id_sets:
            common = set.intersection(*id_sets.values()) if id_sets else set()
        else:
            common = set()

        # Unique per source (present only in that source)
        unique: Dict[str, List[str]] = {}
        for name in source_names:
            other_ids: set = set()
            for other_name, ids in id_sets.items():
                if other_name != name:
                    other_ids |= ids
            unique[name] = sorted(id_sets.get(name, set()) - other_ids)

        return {
            "counts": counts,
            "common_ids": sorted(common),
            "unique_per_source": unique,
        }

    def merge_results(
        self,
        results: List[CrossDBQueryResult],
    ) -> List[Dict[str, Any]]:
        """Merge records from all results, adding a _source tag to each."""
        merged: List[Dict[str, Any]] = []
        for result in results:
            for record in result.records:
                tagged = dict(record)
                tagged["_source"] = result.source
                merged.append(tagged)
        return merged


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_multi_db(sources: List[Dict[str, str]]) -> MultiDBManager:
    """Factory: create a MultiDBManager from a list of source dicts."""
    db_sources = [DatabaseSource.from_dict(s) for s in sources]
    config = MultiDBConfig(sources=db_sources)
    return MultiDBManager(config)


def find_common_records(
    records_a: List[Dict],
    records_b: List[Dict],
    key: str = "id",
) -> List[str]:
    """Find record IDs (by key) present in both lists."""
    ids_a = {r.get(key) for r in records_a if key in r}
    ids_b = {r.get(key) for r in records_b if key in r}
    return sorted(ids_a & ids_b)
