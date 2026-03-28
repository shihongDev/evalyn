"""Tests for golden set management."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_golden import (
    GoldenItem,
    GoldenSet,
    GoldenSetComparison,
    add_to_golden_set,
    bump_version,
    compare_golden_sets,
    create_golden_set,
    filter_golden_set,
    remove_from_golden_set,
    verify_item,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    id: str,
    input_text: str = "question",
    expected_output: str = "answer",
    category: str = "",
    tags: list = [],
    verified: bool = False,
) -> GoldenItem:
    return GoldenItem(
        id=id,
        input_text=input_text,
        expected_output=expected_output,
        category=category,
        tags=list(tags),
        verified=verified,
    )


def _make_set(name: str = "test", items: list = []) -> GoldenSet:
    return GoldenSet(name=name, items=list(items))


# ---------------------------------------------------------------------------
# create_golden_set
# ---------------------------------------------------------------------------


class TestCreateGoldenSet:
    def test_from_dicts(self):
        dicts = [
            {"id": "g1", "input_text": "q1", "expected_output": "a1"},
            {"id": "g2", "input_text": "q2", "expected_output": "a2"},
        ]
        gs = create_golden_set("benchmark", dicts, description="first version")
        assert gs.name == "benchmark"
        assert gs.count == 2
        assert gs.version == 1
        assert gs.description == "first version"
        assert gs.created_at  # non-empty timestamp

    def test_empty_items(self):
        gs = create_golden_set("empty", [])
        assert gs.count == 0

    def test_items_are_golden_items(self):
        dicts = [{"id": "g1", "input_text": "q", "expected_output": "a"}]
        gs = create_golden_set("test", dicts)
        assert isinstance(gs.items[0], GoldenItem)


# ---------------------------------------------------------------------------
# add_to_golden_set
# ---------------------------------------------------------------------------


class TestAddToGoldenSet:
    def test_adds_item(self):
        gs = _make_set("s", [_make_item("a")])
        new = add_to_golden_set(gs, _make_item("b"))
        assert new.count == 2
        assert new.items[-1].id == "b"

    def test_immutable(self):
        original_item = _make_item("a")
        gs = _make_set("s", [original_item])
        new = add_to_golden_set(gs, _make_item("b"))
        # Original unchanged
        assert gs.count == 1
        assert new.count == 2

    def test_preserves_metadata(self):
        gs = GoldenSet(name="s", version=3, description="desc", created_at="ts")
        new = add_to_golden_set(gs, _make_item("x"))
        assert new.version == 3
        assert new.description == "desc"
        assert new.created_at == "ts"


# ---------------------------------------------------------------------------
# remove_from_golden_set
# ---------------------------------------------------------------------------


class TestRemoveFromGoldenSet:
    def test_removes_item(self):
        gs = _make_set("s", [_make_item("a"), _make_item("b")])
        new = remove_from_golden_set(gs, "a")
        assert new.count == 1
        assert new.items[0].id == "b"

    def test_immutable(self):
        gs = _make_set("s", [_make_item("a"), _make_item("b")])
        new = remove_from_golden_set(gs, "a")
        assert gs.count == 2
        assert new.count == 1

    def test_nonexistent_id(self):
        gs = _make_set("s", [_make_item("a")])
        new = remove_from_golden_set(gs, "missing")
        assert new.count == 1


# ---------------------------------------------------------------------------
# verify_item
# ---------------------------------------------------------------------------


class TestVerifyItem:
    def test_marks_verified(self):
        gs = _make_set("s", [_make_item("a")])
        new = verify_item(gs, "a", verified_by="alice")
        assert new.items[0].verified is True
        assert new.items[0].verified_by == "alice"

    def test_immutable(self):
        gs = _make_set("s", [_make_item("a")])
        new = verify_item(gs, "a", verified_by="bob")
        assert gs.items[0].verified is False
        assert new.items[0].verified is True

    def test_nonexistent_id_unchanged(self):
        gs = _make_set("s", [_make_item("a")])
        new = verify_item(gs, "missing")
        assert new.count == 1
        assert new.items[0].verified is False


# ---------------------------------------------------------------------------
# compare_golden_sets
# ---------------------------------------------------------------------------


class TestCompareGoldenSets:
    def test_added(self):
        a = _make_set("a", [_make_item("1")])
        b = _make_set("b", [_make_item("1"), _make_item("2")])
        cmp = compare_golden_sets(a, b)
        assert cmp.added_ids == ["2"]
        assert cmp.removed_ids == []
        assert cmp.modified_ids == []

    def test_removed(self):
        a = _make_set("a", [_make_item("1"), _make_item("2")])
        b = _make_set("b", [_make_item("1")])
        cmp = compare_golden_sets(a, b)
        assert cmp.added_ids == []
        assert cmp.removed_ids == ["2"]

    def test_modified(self):
        a = _make_set("a", [_make_item("1", input_text="old")])
        b = _make_set("b", [_make_item("1", input_text="new")])
        cmp = compare_golden_sets(a, b)
        assert cmp.modified_ids == ["1"]
        assert cmp.added_ids == []
        assert cmp.removed_ids == []

    def test_identical_sets(self):
        items = [_make_item("1"), _make_item("2")]
        a = _make_set("a", items)
        b = _make_set("b", items)
        cmp = compare_golden_sets(a, b)
        assert cmp.added_ids == []
        assert cmp.removed_ids == []
        assert cmp.modified_ids == []

    def test_mixed_changes(self):
        a = _make_set(
            "a",
            [
                _make_item("keep"),
                _make_item("remove_me"),
                _make_item("change", input_text="v1"),
            ],
        )
        b = _make_set(
            "b",
            [
                _make_item("keep"),
                _make_item("new_one"),
                _make_item("change", input_text="v2"),
            ],
        )
        cmp = compare_golden_sets(a, b)
        assert "new_one" in cmp.added_ids
        assert "remove_me" in cmp.removed_ids
        assert "change" in cmp.modified_ids

    def test_empty_sets(self):
        cmp = compare_golden_sets(_make_set("a"), _make_set("b"))
        assert cmp.added_ids == []
        assert cmp.removed_ids == []
        assert cmp.modified_ids == []


# ---------------------------------------------------------------------------
# filter_golden_set
# ---------------------------------------------------------------------------


class TestFilterGoldenSet:
    def test_by_category(self):
        gs = _make_set(
            "s",
            [
                _make_item("a", category="math"),
                _make_item("b", category="science"),
                _make_item("c", category="math"),
            ],
        )
        filtered = filter_golden_set(gs, category="math")
        assert filtered.count == 2
        assert all(item.category == "math" for item in filtered.items)

    def test_by_tags(self):
        gs = _make_set(
            "s",
            [
                _make_item("a", tags=["easy", "core"]),
                _make_item("b", tags=["hard"]),
                _make_item("c", tags=["core"]),
            ],
        )
        filtered = filter_golden_set(gs, tags=["core"])
        assert filtered.count == 2

    def test_verified_only(self):
        gs = _make_set(
            "s",
            [
                _make_item("a", verified=True),
                _make_item("b", verified=False),
            ],
        )
        filtered = filter_golden_set(gs, verified_only=True)
        assert filtered.count == 1
        assert filtered.items[0].id == "a"

    def test_no_filters_returns_all(self):
        gs = _make_set("s", [_make_item("a"), _make_item("b")])
        filtered = filter_golden_set(gs)
        assert filtered.count == 2

    def test_combined_filters(self):
        gs = _make_set(
            "s",
            [
                _make_item("a", category="math", tags=["core"], verified=True),
                _make_item("b", category="math", tags=["extra"], verified=True),
                _make_item("c", category="science", tags=["core"], verified=True),
                _make_item("d", category="math", tags=["core"], verified=False),
            ],
        )
        filtered = filter_golden_set(
            gs, category="math", tags=["core"], verified_only=True
        )
        assert filtered.count == 1
        assert filtered.items[0].id == "a"


# ---------------------------------------------------------------------------
# bump_version
# ---------------------------------------------------------------------------


class TestBumpVersion:
    def test_increments(self):
        gs = _make_set("s")
        assert gs.version == 1
        bumped = bump_version(gs)
        assert bumped.version == 2

    def test_immutable(self):
        gs = _make_set("s")
        bumped = bump_version(gs)
        assert gs.version == 1
        assert bumped.version == 2

    def test_preserves_items(self):
        gs = _make_set("s", [_make_item("a")])
        bumped = bump_version(gs)
        assert bumped.count == 1
        assert bumped.items[0].id == "a"


# ---------------------------------------------------------------------------
# GoldenSet properties
# ---------------------------------------------------------------------------


class TestGoldenSetProperties:
    def test_count(self):
        gs = _make_set("s", [_make_item("a"), _make_item("b")])
        assert gs.count == 2

    def test_verified_count(self):
        gs = _make_set(
            "s",
            [
                _make_item("a", verified=True),
                _make_item("b", verified=False),
                _make_item("c", verified=True),
            ],
        )
        assert gs.verified_count == 2

    def test_verification_rate(self):
        gs = _make_set(
            "s",
            [
                _make_item("a", verified=True),
                _make_item("b", verified=False),
            ],
        )
        assert gs.verification_rate == 0.5

    def test_verification_rate_empty(self):
        gs = _make_set("s")
        assert gs.verification_rate == 0.0

    def test_format_text(self):
        gs = _make_set(
            "s",
            [_make_item("a", verified=True), _make_item("b")],
        )
        gs = GoldenSet(
            name="benchmark",
            items=gs.items,
            version=2,
            description="test set",
        )
        text = gs.format_text()
        assert "benchmark" in text
        assert "v2" in text
        assert "2 items" in text
        assert "test set" in text


# ---------------------------------------------------------------------------
# GoldenSetComparison format_text
# ---------------------------------------------------------------------------


class TestGoldenSetComparisonFormatText:
    def test_format_text_with_changes(self):
        cmp = GoldenSetComparison(
            set_a_name="old",
            set_b_name="new",
            added_ids=["x"],
            removed_ids=["y"],
            modified_ids=["z"],
        )
        text = cmp.format_text()
        assert "old" in text
        assert "new" in text
        assert "Added: 1" in text
        assert "Removed: 1" in text
        assert "Modified: 1" in text
        assert "+ x" in text
        assert "- y" in text
        assert "~ z" in text

    def test_format_text_no_changes(self):
        cmp = GoldenSetComparison(set_a_name="a", set_b_name="b")
        text = cmp.format_text()
        assert "Added: 0" in text


# ---------------------------------------------------------------------------
# Serialization roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_golden_item_roundtrip(self):
        item = GoldenItem(
            id="g1",
            input_text="question",
            expected_output="answer",
            category="math",
            tags=["core", "v2"],
            verified=True,
            verified_by="alice",
        )
        restored = GoldenItem.from_dict(item.as_dict())
        assert restored.id == item.id
        assert restored.input_text == item.input_text
        assert restored.expected_output == item.expected_output
        assert restored.category == item.category
        assert restored.tags == item.tags
        assert restored.verified == item.verified
        assert restored.verified_by == item.verified_by

    def test_golden_set_roundtrip(self):
        gs = create_golden_set(
            "bench",
            [
                {"id": "g1", "input_text": "q", "expected_output": "a", "category": "math"},
                {"id": "g2", "input_text": "q2", "expected_output": "a2"},
            ],
            description="test",
        )
        restored = GoldenSet.from_dict(gs.as_dict())
        assert restored.name == gs.name
        assert restored.count == gs.count
        assert restored.version == gs.version
        assert restored.description == gs.description
        assert restored.created_at == gs.created_at

    def test_golden_set_comparison_as_dict(self):
        cmp = GoldenSetComparison(
            set_a_name="a",
            set_b_name="b",
            added_ids=["x"],
            removed_ids=["y"],
            modified_ids=["z"],
        )
        d = cmp.as_dict()
        assert d["set_a_name"] == "a"
        assert d["added_ids"] == ["x"]
        assert d["removed_ids"] == ["y"]
        assert d["modified_ids"] == ["z"]

    def test_golden_item_from_dict_defaults(self):
        item = GoldenItem.from_dict(
            {"id": "x", "input_text": "q", "expected_output": "a"}
        )
        assert item.category == ""
        assert item.tags == []
        assert item.verified is False
        assert item.verified_by == ""

    def test_golden_set_from_dict_defaults(self):
        gs = GoldenSet.from_dict({"name": "test"})
        assert gs.items == []
        assert gs.version == 1
        assert gs.description == ""
        assert gs.created_at == ""
