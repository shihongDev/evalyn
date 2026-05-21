"""Dataset lineage: track which traces and runs produced each dataset item."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LineageRecord:
    """Single provenance record for a dataset item."""

    item_id: str
    source_type: str = "trace"  # "trace" / "run" / "manual" / "import"
    source_id: str = ""
    created_at: str = ""
    parent_item_id: str = ""
    transformation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "created_at": self.created_at,
            "parent_item_id": self.parent_item_id,
            "transformation": self.transformation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageRecord:
        return cls(
            item_id=data.get("item_id", ""),
            source_type=data.get("source_type", "trace"),
            source_id=data.get("source_id", ""),
            created_at=data.get("created_at", ""),
            parent_item_id=data.get("parent_item_id", ""),
            transformation=data.get("transformation", ""),
        )


@dataclass
class LineageChain:
    """Full ancestry chain for a single dataset item."""

    item_id: str
    records: list[LineageRecord] = field(default_factory=list)
    depth: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "records": [r.as_dict() for r in self.records],
            "depth": self.depth,
        }

    def format_text(self) -> str:
        lines = [f"Lineage for {self.item_id} (depth {self.depth}):"]
        for i, rec in enumerate(self.records):
            indent = "  " * i
            source = f"{rec.source_type}"
            if rec.source_id:
                source += f":{rec.source_id}"
            suffix = f" [{rec.transformation}]" if rec.transformation else ""
            lines.append(f"{indent}-> {rec.item_id} ({source}){suffix}")
        return "\n".join(lines)


@dataclass
class LineageReport:
    """Aggregate lineage report across all items."""

    chains: list[LineageChain] = field(default_factory=list)
    total_items: int = 0
    source_distribution: dict[str, int] = field(default_factory=dict)
    orphaned_items: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chains": [c.as_dict() for c in self.chains],
            "total_items": self.total_items,
            "source_distribution": dict(self.source_distribution),
            "orphaned_items": list(self.orphaned_items),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageReport:
        chains = [
            LineageChain(
                item_id=c["item_id"],
                records=[LineageRecord.from_dict(r) for r in c.get("records", [])],
                depth=c.get("depth", 0),
            )
            for c in data.get("chains", [])
        ]
        return cls(
            chains=chains,
            total_items=data.get("total_items", 0),
            source_distribution=data.get("source_distribution", {}),
            orphaned_items=data.get("orphaned_items", []),
        )

    def format_text(self) -> str:
        lines = [
            f"Lineage Report ({self.total_items} items)",
        ]
        if self.source_distribution:
            lines.append("Source distribution:")
            for src, count in sorted(self.source_distribution.items()):
                lines.append(f"  {src}: {count}")
        if self.orphaned_items:
            lines.append(f"Orphaned items: {', '.join(self.orphaned_items)}")
        for chain in self.chains:
            lines.append("")
            lines.append(chain.format_text())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def record_lineage(
    item_id: str,
    source_type: str,
    source_id: str = "",
    parent_id: str = "",
    transformation: str = "",
) -> LineageRecord:
    """Create a lineage record with a UTC timestamp."""
    return LineageRecord(
        item_id=item_id,
        source_type=source_type,
        source_id=source_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_item_id=parent_id,
        transformation=transformation,
    )


def build_lineage_chain(item_id: str, records: list[LineageRecord]) -> LineageChain:
    """Follow parent_item_id chain to build full lineage for an item."""
    by_item: dict[str, LineageRecord] = {}
    for rec in records:
        by_item[rec.item_id] = rec

    chain: list[LineageRecord] = []
    current = item_id
    seen: set[str] = set()

    while current in by_item and current not in seen:
        seen.add(current)
        rec = by_item[current]
        chain.append(rec)
        if not rec.parent_item_id:
            break
        current = rec.parent_item_id

    return LineageChain(
        item_id=item_id,
        records=chain,
        depth=len(chain),
    )


def build_lineage_report(records: list[LineageRecord]) -> LineageReport:
    """Aggregate lineage across all items into a report."""
    item_ids = list({rec.item_id for rec in records})
    chains = [build_lineage_chain(iid, records) for iid in sorted(item_ids)]
    dist = compute_source_distribution(records)
    orphaned = find_orphaned_items(item_ids, records)

    return LineageReport(
        chains=chains,
        total_items=len(item_ids),
        source_distribution=dist,
        orphaned_items=orphaned,
    )


def find_orphaned_items(item_ids: list[str], records: list[LineageRecord]) -> list[str]:
    """Return item IDs that have no lineage records."""
    recorded = {rec.item_id for rec in records}
    return sorted(iid for iid in item_ids if iid not in recorded)


def render_lineage_tree(chain: LineageChain) -> str:
    """Render an ASCII tree of a lineage chain."""
    if not chain.records:
        return f"{chain.item_id} (no lineage)"

    lines: list[str] = []
    for i, rec in enumerate(chain.records):
        prefix = "  " * i
        connector = "+-" if i == 0 else "`-"
        source = rec.source_type
        if rec.source_id:
            source += f":{rec.source_id}"
        suffix = f" [{rec.transformation}]" if rec.transformation else ""
        lines.append(f"{prefix}{connector} {rec.item_id} ({source}){suffix}")

    return "\n".join(lines)


def compute_source_distribution(
    records: list[LineageRecord],
) -> dict[str, int]:
    """Count records by source_type."""
    dist: dict[str, int] = {}
    for rec in records:
        dist[rec.source_type] = dist.get(rec.source_type, 0) + 1
    return dist
