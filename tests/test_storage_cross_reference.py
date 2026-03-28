"""
Tests for evalyn_sdk.storage.cross_reference.

Run with: uv run pytest tests/test_storage_cross_reference.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.cross_reference import (
    CrossReferenceReport,
    EntityRelation,
    EntitySummary,
    build_cross_reference_report,
    build_entity_summary,
    detect_relations,
    find_orphaned_entities,
    render_relation_diagram,
)


# ---------------------------------------------------------------------------
# EntityRelation
# ---------------------------------------------------------------------------


class TestEntityRelation:
    def test_as_dict(self):
        rel = EntityRelation(
            source_type="order",
            target_type="user",
            relation="references",
            count=5,
        )
        d = rel.as_dict()
        assert d["source_type"] == "order"
        assert d["target_type"] == "user"
        assert d["relation"] == "references"
        assert d["count"] == 5

    def test_defaults(self):
        rel = EntityRelation(source_type="a", target_type="b")
        assert rel.relation == "references"
        assert rel.count == 0


# ---------------------------------------------------------------------------
# EntitySummary
# ---------------------------------------------------------------------------


class TestEntitySummary:
    def test_as_dict(self):
        es = EntitySummary(
            entity_type="user",
            count=10,
            referenced_by=["order"],
            references=["role"],
        )
        d = es.as_dict()
        assert d["entity_type"] == "user"
        assert d["count"] == 10
        assert d["referenced_by"] == ["order"]
        assert d["references"] == ["role"]

    def test_defaults(self):
        es = EntitySummary(entity_type="item")
        assert es.count == 0
        assert es.referenced_by == []
        assert es.references == []


# ---------------------------------------------------------------------------
# CrossReferenceReport
# ---------------------------------------------------------------------------


class TestCrossReferenceReport:
    def test_as_dict_from_dict_roundtrip(self):
        report = CrossReferenceReport(
            entities=[
                EntitySummary(entity_type="user", count=5, references=["role"]),
                EntitySummary(entity_type="role", count=2, referenced_by=["user"]),
            ],
            relations=[
                EntityRelation(source_type="user", target_type="role", count=5),
            ],
            total_entities=7,
            orphaned_entities=["log"],
        )
        d = report.as_dict()
        restored = CrossReferenceReport.from_dict(d)
        assert len(restored.entities) == 2
        assert restored.entities[0].entity_type == "user"
        assert restored.entities[0].references == ["role"]
        assert len(restored.relations) == 1
        assert restored.relations[0].source_type == "user"
        assert restored.total_entities == 7
        assert restored.orphaned_entities == ["log"]

    def test_from_dict_defaults(self):
        report = CrossReferenceReport.from_dict({})
        assert report.entities == []
        assert report.relations == []
        assert report.total_entities == 0
        assert report.orphaned_entities == []

    def test_format_text_basic(self):
        report = CrossReferenceReport(
            entities=[
                EntitySummary(
                    entity_type="user", count=10,
                    references=["role"], referenced_by=["order"],
                ),
            ],
            relations=[
                EntityRelation(source_type="order", target_type="user", count=5),
            ],
            total_entities=10,
            orphaned_entities=["log"],
        )
        text = report.format_text()
        assert "Cross-Reference Report" in text
        assert "Total entities: 10" in text
        assert "user: 10 records" in text
        assert "references: role" in text
        assert "referenced by: order" in text
        assert "order --references--> user (5)" in text
        assert "Orphaned entities:" in text
        assert "- log" in text

    def test_format_text_empty(self):
        report = CrossReferenceReport()
        text = report.format_text()
        assert "Cross-Reference Report" in text
        assert "Total entities: 0" in text


# ---------------------------------------------------------------------------
# build_entity_summary
# ---------------------------------------------------------------------------


class TestBuildEntitySummary:
    def test_counts_records(self):
        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        summary = build_entity_summary("user", records)
        assert summary.entity_type == "user"
        assert summary.count == 3

    def test_detects_references(self):
        records = [
            {"id": 1, "role_id": 10},
            {"id": 2, "role_id": 20},
        ]
        summary = build_entity_summary(
            "user", records, reference_fields={"role_id": "role"}
        )
        assert "role" in summary.references

    def test_no_reference_fields(self):
        records = [{"id": 1}]
        summary = build_entity_summary("item", records)
        assert summary.references == []

    def test_reference_field_not_in_records(self):
        records = [{"id": 1}]
        summary = build_entity_summary(
            "user", records, reference_fields={"role_id": "role"}
        )
        assert summary.references == []

    def test_reference_field_none_value(self):
        records = [{"id": 1, "role_id": None}]
        summary = build_entity_summary(
            "user", records, reference_fields={"role_id": "role"}
        )
        assert summary.references == []

    def test_empty_records(self):
        summary = build_entity_summary("empty", [])
        assert summary.count == 0
        assert summary.references == []


# ---------------------------------------------------------------------------
# detect_relations
# ---------------------------------------------------------------------------


class TestDetectRelations:
    def test_finds_relations(self):
        entities = {
            "order": [
                {"id": 1, "user_id": 10},
                {"id": 2, "user_id": 20},
            ],
            "user": [{"id": 10}, {"id": 20}],
        }
        reference_map = {"order": {"user_id": "user"}}
        relations = detect_relations(entities, reference_map)
        assert len(relations) == 1
        assert relations[0].source_type == "order"
        assert relations[0].target_type == "user"
        assert relations[0].count == 2

    def test_no_matching_records(self):
        entities = {"order": [{"id": 1}], "user": [{"id": 10}]}
        reference_map = {"order": {"user_id": "user"}}
        relations = detect_relations(entities, reference_map)
        assert relations == []

    def test_multiple_reference_fields(self):
        entities = {
            "order": [
                {"id": 1, "user_id": 10, "product_id": 100},
            ],
            "user": [{"id": 10}],
            "product": [{"id": 100}],
        }
        reference_map = {"order": {"user_id": "user", "product_id": "product"}}
        relations = detect_relations(entities, reference_map)
        assert len(relations) == 2
        targets = {r.target_type for r in relations}
        assert targets == {"user", "product"}

    def test_empty_entities(self):
        relations = detect_relations({}, {})
        assert relations == []


# ---------------------------------------------------------------------------
# find_orphaned_entities
# ---------------------------------------------------------------------------


class TestFindOrphanedEntities:
    def test_finds_orphans(self):
        entities = {
            "order": [{"id": 1, "user_id": 10}],
            "user": [{"id": 10}],
            "log": [{"id": 1}],
        }
        reference_map = {"order": {"user_id": "user"}}
        orphaned = find_orphaned_entities(entities, reference_map)
        assert "log" in orphaned
        assert "order" not in orphaned
        assert "user" not in orphaned

    def test_no_orphans(self):
        entities = {
            "order": [{"id": 1, "user_id": 10}],
            "user": [{"id": 10}],
        }
        reference_map = {"order": {"user_id": "user"}}
        orphaned = find_orphaned_entities(entities, reference_map)
        assert orphaned == []

    def test_all_orphans(self):
        entities = {"a": [{"id": 1}], "b": [{"id": 2}]}
        reference_map: dict = {}
        orphaned = find_orphaned_entities(entities, reference_map)
        assert sorted(orphaned) == ["a", "b"]

    def test_empty(self):
        orphaned = find_orphaned_entities({}, {})
        assert orphaned == []


# ---------------------------------------------------------------------------
# build_cross_reference_report
# ---------------------------------------------------------------------------


class TestBuildCrossReferenceReport:
    def test_full_report(self):
        entities = {
            "order": [
                {"id": 1, "user_id": 10},
                {"id": 2, "user_id": 20},
            ],
            "user": [{"id": 10}, {"id": 20}],
            "log": [{"id": 1}],
        }
        reference_map = {"order": {"user_id": "user"}}
        report = build_cross_reference_report(entities, reference_map)
        assert report.total_entities == 5
        assert len(report.entities) == 3
        assert len(report.relations) == 1
        assert "log" in report.orphaned_entities

    def test_referenced_by_populated(self):
        entities = {
            "order": [{"id": 1, "user_id": 10}],
            "user": [{"id": 10}],
        }
        reference_map = {"order": {"user_id": "user"}}
        report = build_cross_reference_report(entities, reference_map)
        user_summary = next(e for e in report.entities if e.entity_type == "user")
        assert "order" in user_summary.referenced_by

    def test_empty_entities(self):
        report = build_cross_reference_report({}, {})
        assert report.total_entities == 0
        assert report.entities == []
        assert report.relations == []

    def test_entity_summaries_sorted(self):
        entities = {
            "zebra": [{"id": 1}],
            "alpha": [{"id": 2}],
            "middle": [{"id": 3}],
        }
        report = build_cross_reference_report(entities, {})
        types = [e.entity_type for e in report.entities]
        assert types == ["alpha", "middle", "zebra"]


# ---------------------------------------------------------------------------
# render_relation_diagram
# ---------------------------------------------------------------------------


class TestRenderRelationDiagram:
    def test_mermaid_output(self):
        report = CrossReferenceReport(
            entities=[
                EntitySummary(entity_type="user", count=5),
                EntitySummary(entity_type="order", count=10),
            ],
            relations=[
                EntityRelation(source_type="order", target_type="user", count=10),
            ],
        )
        diagram = render_relation_diagram(report)
        assert diagram.startswith("erDiagram")
        assert "user" in diagram
        assert "order" in diagram
        assert "references" in diagram

    def test_empty_report(self):
        report = CrossReferenceReport()
        diagram = render_relation_diagram(report)
        assert diagram == "erDiagram"

    def test_entity_counts_in_diagram(self):
        report = CrossReferenceReport(
            entities=[EntitySummary(entity_type="item", count=42)],
        )
        diagram = render_relation_diagram(report)
        assert "42" in diagram
        assert "item" in diagram

    def test_multiple_relations(self):
        report = CrossReferenceReport(
            entities=[
                EntitySummary(entity_type="order", count=1),
                EntitySummary(entity_type="user", count=1),
                EntitySummary(entity_type="product", count=1),
            ],
            relations=[
                EntityRelation(source_type="order", target_type="user", count=1),
                EntityRelation(source_type="order", target_type="product", count=1),
            ],
        )
        diagram = render_relation_diagram(report)
        lines = diagram.split("\n")
        relation_lines = [l for l in lines if "--" in l]
        assert len(relation_lines) == 2
