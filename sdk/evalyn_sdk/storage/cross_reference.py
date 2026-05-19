"""Storage cross-reference report: show relationships between stored entities.

Provides dataclasses and pure functions to build entity summaries,
detect relations, find orphans, and render Mermaid diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EntityRelation:
    """A directional relationship between two entity types."""

    source_type: str
    target_type: str
    relation: str = "references"
    count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "target_type": self.target_type,
            "relation": self.relation,
            "count": self.count,
        }


@dataclass
class EntitySummary:
    """Summary of a single entity type and its references."""

    entity_type: str
    count: int = 0
    referenced_by: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "count": self.count,
            "referenced_by": list(self.referenced_by),
            "references": list(self.references),
        }


@dataclass
class CrossReferenceReport:
    """Full cross-reference report across all entity types."""

    entities: List[EntitySummary] = field(default_factory=list)
    relations: List[EntityRelation] = field(default_factory=list)
    total_entities: int = 0
    orphaned_entities: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.as_dict() for e in self.entities],
            "relations": [r.as_dict() for r in self.relations],
            "total_entities": self.total_entities,
            "orphaned_entities": list(self.orphaned_entities),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CrossReferenceReport:
        entities = [
            EntitySummary(
                entity_type=e["entity_type"],
                count=e.get("count", 0),
                referenced_by=e.get("referenced_by", []),
                references=e.get("references", []),
            )
            for e in data.get("entities", [])
        ]
        relations = [
            EntityRelation(
                source_type=r["source_type"],
                target_type=r["target_type"],
                relation=r.get("relation", "references"),
                count=r.get("count", 0),
            )
            for r in data.get("relations", [])
        ]
        return cls(
            entities=entities,
            relations=relations,
            total_entities=data.get("total_entities", 0),
            orphaned_entities=data.get("orphaned_entities", []),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Cross-Reference Report")
        lines.append("=" * 40)
        lines.append(f"Total entity types: {len(self.entities)}")
        lines.append(f"Total entities: {self.total_entities}")
        if self.entities:
            lines.append("")
            lines.append("Entities:")
            for e in self.entities:
                lines.append(f"  {e.entity_type}: {e.count} records")
                if e.references:
                    lines.append(f"    references: {', '.join(e.references)}")
                if e.referenced_by:
                    lines.append(f"    referenced by: {', '.join(e.referenced_by)}")
        if self.relations:
            lines.append("")
            lines.append("Relations:")
            for r in self.relations:
                lines.append(
                    f"  {r.source_type} --{r.relation}--> {r.target_type} ({r.count})"
                )
        if self.orphaned_entities:
            lines.append("")
            lines.append("Orphaned entities:")
            for name in self.orphaned_entities:
                lines.append(f"  - {name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_entity_summary(
    entity_type: str,
    records: List[Dict[str, Any]],
    reference_fields: Dict[str, str] | None = None,
) -> EntitySummary:
    """Analyze one entity type.

    reference_fields maps field_name -> target_entity_type. For each field
    present in the records, the entity is considered to reference the target
    entity type.
    """
    if reference_fields is None:
        reference_fields = {}

    refs: Set[str] = set()
    for field_name, target_type in reference_fields.items():
        for record in records:
            if field_name in record and record[field_name] is not None:
                refs.add(target_type)
                break

    return EntitySummary(
        entity_type=entity_type,
        count=len(records),
        references=sorted(refs),
    )


def detect_relations(
    entities: Dict[str, List[Dict[str, Any]]],
    reference_map: Dict[str, Dict[str, str]],
) -> List[EntityRelation]:
    """Detect all relations across entity types.

    reference_map: entity_type -> {field_name -> target_entity_type}.
    Counts how many records in the source entity reference each target.
    """
    relations: List[EntityRelation] = []

    for source_type, field_map in sorted(reference_map.items()):
        records = entities.get(source_type, [])
        target_counts: Dict[str, int] = {}

        for field_name, target_type in field_map.items():
            count = 0
            for record in records:
                if field_name in record and record[field_name] is not None:
                    count += 1
            if count > 0:
                target_counts[target_type] = (
                    target_counts.get(target_type, 0) + count
                )

        for target_type, count in sorted(target_counts.items()):
            relations.append(
                EntityRelation(
                    source_type=source_type,
                    target_type=target_type,
                    relation="references",
                    count=count,
                )
            )

    return relations


def find_orphaned_entities(
    entities: Dict[str, List[Dict[str, Any]]],
    reference_map: Dict[str, Dict[str, str]],
) -> List[str]:
    """Find entity types with no references to or from other entity types."""
    all_types = set(entities.keys())

    # Types that reference something
    sources: Set[str] = set()
    # Types that are referenced by something
    targets: Set[str] = set()

    for source_type, field_map in reference_map.items():
        records = entities.get(source_type, [])
        for field_name, target_type in field_map.items():
            for record in records:
                if field_name in record and record[field_name] is not None:
                    sources.add(source_type)
                    targets.add(target_type)
                    break

    connected = sources | targets
    orphaned = sorted(all_types - connected)
    return orphaned


def build_cross_reference_report(
    entities: Dict[str, List[Dict[str, Any]]],
    reference_map: Dict[str, Dict[str, str]],
) -> CrossReferenceReport:
    """Build a full cross-reference report."""
    relations = detect_relations(entities, reference_map)
    orphaned = find_orphaned_entities(entities, reference_map)

    # Build referenced_by map
    referenced_by_map: Dict[str, Set[str]] = {}
    for rel in relations:
        referenced_by_map.setdefault(rel.target_type, set()).add(rel.source_type)

    summaries: List[EntitySummary] = []
    for entity_type in sorted(entities.keys()):
        records = entities[entity_type]
        ref_fields = reference_map.get(entity_type, {})
        summary = build_entity_summary(entity_type, records, ref_fields)
        summary.referenced_by = sorted(referenced_by_map.get(entity_type, set()))
        summaries.append(summary)

    total = sum(len(records) for records in entities.values())

    return CrossReferenceReport(
        entities=summaries,
        relations=relations,
        total_entities=total,
        orphaned_entities=orphaned,
    )


def render_relation_diagram(report: CrossReferenceReport) -> str:
    """Render a Mermaid ER diagram from the report."""
    lines: List[str] = []
    lines.append("erDiagram")

    if not report.relations and not report.entities:
        return "\n".join(lines)

    # Emit entity blocks
    for entity in report.entities:
        lines.append(f"    {entity.entity_type} {{")
        lines.append(f"        int count \"{entity.count}\"")
        lines.append("    }")

    # Emit relations
    for rel in report.relations:
        label = rel.relation
        lines.append(
            f"    {rel.source_type} }}o--o{{ {rel.target_type} : \"{label}\""
        )

    return "\n".join(lines)
