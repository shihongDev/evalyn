"""Tests for annotation_delegation module."""

from __future__ import annotations

import pytest

from evalyn_sdk.annotation_delegation import (
    AnnotatorProfile,
    Assignment,
    DelegationManager,
    DelegationRule,
    compute_workload_balance,
    format_delegation_dashboard,
)


# ---------------------------------------------------------------------------
# AnnotatorProfile
# ---------------------------------------------------------------------------


class TestAnnotatorProfile:
    def test_create_defaults(self):
        p = AnnotatorProfile(annotator_id="a1", name="Alice")
        assert p.expertise_tags == []
        assert p.max_daily_items == 50
        assert p.items_assigned == 0
        assert p.items_completed == 0

    def test_create_with_values(self):
        p = AnnotatorProfile(
            annotator_id="a2",
            name="Bob",
            expertise_tags=["nlp", "safety"],
            max_daily_items=30,
            items_assigned=5,
            items_completed=2,
        )
        assert p.name == "Bob"
        assert p.expertise_tags == ["nlp", "safety"]
        assert p.max_daily_items == 30

    def test_as_dict(self):
        p = AnnotatorProfile(annotator_id="a1", name="Alice", expertise_tags=["nlp"])
        d = p.as_dict()
        assert d["annotator_id"] == "a1"
        assert d["name"] == "Alice"
        assert d["expertise_tags"] == ["nlp"]
        assert d["max_daily_items"] == 50

    def test_from_dict(self):
        d = {"annotator_id": "a1", "name": "Alice", "expertise_tags": ["nlp"]}
        p = AnnotatorProfile.from_dict(d)
        assert p.annotator_id == "a1"
        assert p.expertise_tags == ["nlp"]

    def test_from_dict_defaults(self):
        d = {"annotator_id": "a1", "name": "Alice"}
        p = AnnotatorProfile.from_dict(d)
        assert p.expertise_tags == []
        assert p.max_daily_items == 50

    def test_roundtrip(self):
        p = AnnotatorProfile(
            annotator_id="x",
            name="X",
            expertise_tags=["a", "b"],
            max_daily_items=10,
            items_assigned=3,
            items_completed=1,
        )
        p2 = AnnotatorProfile.from_dict(p.as_dict())
        assert p2.annotator_id == p.annotator_id
        assert p2.expertise_tags == p.expertise_tags
        assert p2.items_assigned == p.items_assigned


# ---------------------------------------------------------------------------
# DelegationRule
# ---------------------------------------------------------------------------


class TestDelegationRule:
    def test_create_defaults(self):
        r = DelegationRule(rule_id="r1", item_tag="safety")
        assert r.required_expertise == []
        assert r.priority == 0

    def test_as_dict(self):
        r = DelegationRule(
            rule_id="r1",
            item_tag="safety",
            required_expertise=["safety"],
            priority=5,
        )
        d = r.as_dict()
        assert d["rule_id"] == "r1"
        assert d["priority"] == 5

    def test_from_dict(self):
        d = {
            "rule_id": "r1",
            "item_tag": "safety",
            "required_expertise": ["safety"],
            "priority": 3,
        }
        r = DelegationRule.from_dict(d)
        assert r.rule_id == "r1"
        assert r.priority == 3

    def test_roundtrip(self):
        r = DelegationRule(
            rule_id="r1",
            item_tag="nlp",
            required_expertise=["nlp", "linguistics"],
            priority=10,
        )
        r2 = DelegationRule.from_dict(r.as_dict())
        assert r2.rule_id == r.rule_id
        assert r2.required_expertise == r.required_expertise


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


class TestAssignment:
    def test_create_defaults(self):
        a = Assignment(item_id="i1", annotator_id="a1", assigned_at="2026-01-01T00:00:00+00:00")
        assert a.status == "pending"
        assert a.rule_id == ""

    def test_as_dict(self):
        a = Assignment(
            item_id="i1",
            annotator_id="a1",
            assigned_at="2026-01-01T00:00:00+00:00",
            status="completed",
            rule_id="r1",
        )
        d = a.as_dict()
        assert d["item_id"] == "i1"
        assert d["status"] == "completed"

    def test_from_dict(self):
        d = {
            "item_id": "i1",
            "annotator_id": "a1",
            "assigned_at": "2026-01-01T00:00:00+00:00",
            "status": "in_progress",
        }
        a = Assignment.from_dict(d)
        assert a.status == "in_progress"

    def test_roundtrip(self):
        a = Assignment(
            item_id="i1",
            annotator_id="a1",
            assigned_at="2026-01-01T00:00:00+00:00",
            status="pending",
            rule_id="r2",
        )
        a2 = Assignment.from_dict(a.as_dict())
        assert a2.item_id == a.item_id
        assert a2.rule_id == a.rule_id


# ---------------------------------------------------------------------------
# DelegationManager - registration and rules
# ---------------------------------------------------------------------------


def _make_manager() -> DelegationManager:
    """Helper: manager with 3 annotators and 2 rules."""
    m = DelegationManager()
    m.register_annotator(
        AnnotatorProfile("a1", "Alice", expertise_tags=["nlp", "safety"], max_daily_items=10)
    )
    m.register_annotator(
        AnnotatorProfile("a2", "Bob", expertise_tags=["nlp"], max_daily_items=10)
    )
    m.register_annotator(
        AnnotatorProfile("a3", "Carol", expertise_tags=["safety", "coding"], max_daily_items=10)
    )
    m.add_rule(DelegationRule("r1", "safety", required_expertise=["safety"], priority=5))
    m.add_rule(DelegationRule("r2", "nlp", required_expertise=["nlp"], priority=3))
    return m


class TestDelegationManagerBasic:
    def test_register_annotator(self):
        m = DelegationManager()
        m.register_annotator(AnnotatorProfile("a1", "Alice"))
        assert "a1" in m.annotators

    def test_add_rule_sorted(self):
        m = DelegationManager()
        m.add_rule(DelegationRule("r1", "a", priority=1))
        m.add_rule(DelegationRule("r2", "b", priority=10))
        m.add_rule(DelegationRule("r3", "c", priority=5))
        assert [r.rule_id for r in m.rules] == ["r2", "r3", "r1"]


# ---------------------------------------------------------------------------
# DelegationManager - assignment
# ---------------------------------------------------------------------------


class TestDelegationManagerAssign:
    def test_assign_item_expertise_match(self):
        m = _make_manager()
        a = m.assign_item("item1", ["safety"])
        assert a is not None
        # Alice has both safety and nlp; Carol has safety and coding.
        # Both have overlap=1 for safety. Tie-break by items_assigned (both 0).
        # Either Alice or Carol is valid.
        assert a.annotator_id in ("a1", "a3")
        assert a.status == "pending"

    def test_assign_item_increments_count(self):
        m = _make_manager()
        m.assign_item("item1", ["nlp"])
        profile = m.annotators[m.assignments[-1].annotator_id]
        assert profile.items_assigned == 1

    def test_assign_item_no_capacity(self):
        m = DelegationManager()
        m.register_annotator(AnnotatorProfile("a1", "Alice", max_daily_items=0))
        result = m.assign_item("item1", ["anything"])
        assert result is None

    def test_assign_item_no_annotators(self):
        m = DelegationManager()
        result = m.assign_item("item1", ["x"])
        assert result is None

    def test_assign_item_no_rules_picks_least_loaded(self):
        m = DelegationManager()
        p1 = AnnotatorProfile("a1", "Alice", max_daily_items=10, items_assigned=5)
        p2 = AnnotatorProfile("a2", "Bob", max_daily_items=10, items_assigned=1)
        m.register_annotator(p1)
        m.register_annotator(p2)
        a = m.assign_item("item1", ["unknown_tag"])
        assert a is not None
        assert a.annotator_id == "a2"  # Bob has fewer items

    def test_assign_item_sets_rule_id(self):
        m = _make_manager()
        a = m.assign_item("item1", ["safety"])
        assert a is not None
        assert a.rule_id == "r1"

    def test_assign_item_no_matching_rule(self):
        m = _make_manager()
        a = m.assign_item("item1", ["unrelated_tag"])
        assert a is not None
        assert a.rule_id == ""

    def test_assign_item_workload_balance(self):
        """After assigning many items, the load should spread across annotators."""
        m = _make_manager()
        for i in range(9):
            m.assign_item(f"item{i}", ["nlp"])
        # Alice(nlp,safety) and Bob(nlp) should get items; Carol lacks nlp
        counts = {aid: p.items_assigned for aid, p in m.annotators.items()}
        # Verify that no single annotator has all 9
        assert counts["a1"] < 9
        assert counts["a2"] < 9

    def test_assign_batch(self):
        m = _make_manager()
        items = [("i1", ["safety"]), ("i2", ["nlp"]), ("i3", ["coding"])]
        results = m.assign_batch(items)
        assert len(results) == 3

    def test_assign_batch_partial(self):
        """If capacity runs out, batch returns only successful assignments."""
        m = DelegationManager()
        m.register_annotator(AnnotatorProfile("a1", "Alice", max_daily_items=2))
        items = [("i1", []), ("i2", []), ("i3", []), ("i4", [])]
        results = m.assign_batch(items)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# DelegationManager - workload and queries
# ---------------------------------------------------------------------------


class TestDelegationManagerWorkload:
    def test_get_annotator_workload(self):
        m = _make_manager()
        m.assign_item("item1", ["nlp"])
        aid = m.assignments[-1].annotator_id
        wl = m.get_annotator_workload(aid)
        assert wl["assigned"] == 1
        assert wl["completed"] == 0
        assert wl["remaining_capacity"] == 9

    def test_get_annotator_workload_unknown(self):
        m = DelegationManager()
        wl = m.get_annotator_workload("nonexistent")
        assert wl["assigned"] == 0

    def test_get_assignments_all(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        m.assign_item("i2", ["safety"])
        assert len(m.get_assignments()) == 2

    def test_get_assignments_by_annotator(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        m.assign_item("i2", ["nlp"])
        aid = m.assignments[0].annotator_id
        filtered = m.get_assignments(annotator_id=aid)
        assert all(a.annotator_id == aid for a in filtered)

    def test_get_assignments_by_status(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        m.assign_item("i2", ["nlp"])
        m.complete_assignment("i1")
        completed = m.get_assignments(status="completed")
        assert len(completed) == 1
        assert completed[0].item_id == "i1"

    def test_get_assignments_combined_filter(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        m.assign_item("i2", ["nlp"])
        m.complete_assignment("i1")
        aid = m.assignments[0].annotator_id
        result = m.get_assignments(annotator_id=aid, status="completed")
        assert all(a.status == "completed" and a.annotator_id == aid for a in result)


# ---------------------------------------------------------------------------
# DelegationManager - complete
# ---------------------------------------------------------------------------


class TestDelegationManagerComplete:
    def test_complete_assignment(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        assert m.complete_assignment("i1") is True
        assert m.assignments[0].status == "completed"

    def test_complete_updates_profile(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        aid = m.assignments[0].annotator_id
        m.complete_assignment("i1")
        assert m.annotators[aid].items_completed == 1

    def test_complete_nonexistent(self):
        m = _make_manager()
        assert m.complete_assignment("nonexistent") is False

    def test_complete_already_completed(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        m.complete_assignment("i1")
        # Second call should return False
        assert m.complete_assignment("i1") is False


# ---------------------------------------------------------------------------
# DelegationManager - rebalance
# ---------------------------------------------------------------------------


class TestDelegationManagerRebalance:
    def test_rebalance_redistributes(self):
        m = DelegationManager()
        m.register_annotator(AnnotatorProfile("a1", "Alice", max_daily_items=10))
        m.register_annotator(AnnotatorProfile("a2", "Bob", max_daily_items=10))
        # Force all items onto a1
        m.annotators["a2"].max_daily_items = 0
        for i in range(5):
            m.assign_item(f"item{i}", [])
        assert m.annotators["a1"].items_assigned == 5
        # Restore capacity and rebalance
        m.annotators["a2"].max_daily_items = 10
        items = [(f"item{i}", []) for i in range(5)]
        new_assignments = m.rebalance(items)
        assert len(new_assignments) == 5
        # Now items should be spread
        assert m.annotators["a1"].items_assigned < 5 or m.annotators["a2"].items_assigned > 0

    def test_rebalance_preserves_completed(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        m.assign_item("i2", ["nlp"])
        m.complete_assignment("i1")
        new = m.rebalance([("i2", ["nlp"])])
        # i1 should still be in assignments (completed)
        completed = [a for a in m.assignments if a.item_id == "i1"]
        assert len(completed) == 1
        assert completed[0].status == "completed"

    def test_rebalance_empty(self):
        m = _make_manager()
        result = m.rebalance([])
        assert result == []


# ---------------------------------------------------------------------------
# compute_workload_balance
# ---------------------------------------------------------------------------


class TestComputeWorkloadBalance:
    def test_empty_manager(self):
        m = DelegationManager()
        stats = compute_workload_balance(m)
        assert stats["min"] == 0
        assert stats["gini"] == 0.0

    def test_uniform_workload(self):
        m = DelegationManager()
        for i in range(3):
            p = AnnotatorProfile(f"a{i}", f"N{i}", max_daily_items=10, items_assigned=5)
            m.register_annotator(p)
        stats = compute_workload_balance(m)
        assert stats["min"] == 5
        assert stats["max"] == 5
        assert stats["mean"] == 5.0
        assert stats["gini"] == 0.0

    def test_skewed_workload(self):
        m = DelegationManager()
        m.register_annotator(AnnotatorProfile("a1", "A", items_assigned=10))
        m.register_annotator(AnnotatorProfile("a2", "B", items_assigned=0))
        stats = compute_workload_balance(m)
        assert stats["min"] == 0
        assert stats["max"] == 10
        assert stats["mean"] == 5.0
        assert stats["gini"] > 0

    def test_all_zero(self):
        m = DelegationManager()
        m.register_annotator(AnnotatorProfile("a1", "A"))
        m.register_annotator(AnnotatorProfile("a2", "B"))
        stats = compute_workload_balance(m)
        assert stats["gini"] == 0.0


# ---------------------------------------------------------------------------
# format_delegation_dashboard
# ---------------------------------------------------------------------------


class TestFormatDelegationDashboard:
    def test_contains_header(self):
        m = _make_manager()
        dashboard = format_delegation_dashboard(m)
        assert "Delegation Dashboard" in dashboard

    def test_contains_annotator_names(self):
        m = _make_manager()
        dashboard = format_delegation_dashboard(m)
        assert "Alice" in dashboard
        assert "Bob" in dashboard
        assert "Carol" in dashboard

    def test_shows_assigned_counts(self):
        m = _make_manager()
        m.assign_item("i1", ["nlp"])
        dashboard = format_delegation_dashboard(m)
        # At least one row should show a non-zero assigned count
        assert "1" in dashboard

    def test_empty_manager(self):
        m = DelegationManager()
        dashboard = format_delegation_dashboard(m)
        assert "Delegation Dashboard" in dashboard
