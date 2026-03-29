"""Tests for annotation UX improvements module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.annotation_ux import (
    DEFAULT_KEYMAP,
    AnnotationItem,
    BatchAnnotation,
    KeyAction,
    UndoStack,
    create_batch,
    format_annotation_prompt,
    format_keymap_help,
    get_batch_progress,
    process_key_action,
    undo_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(**overrides) -> AnnotationItem:
    defaults = {
        "item_id": "item-001",
        "input_text": "What is Python?",
        "output_text": "Python is a programming language.",
        "metric_scores": {"helpfulness": 0.9, "safety": 1.0},
        "status": "pending",
    }
    defaults.update(overrides)
    return AnnotationItem(**defaults)


def _make_items(n: int) -> list[AnnotationItem]:
    return [
        _make_item(item_id=f"item-{i:03d}")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# KeyAction dataclass
# ---------------------------------------------------------------------------


class TestKeyAction:
    def test_as_dict_roundtrip(self):
        ka = KeyAction(key="y", action="pass", value=None)
        d = ka.as_dict()
        ka2 = KeyAction.from_dict(d)
        assert ka2.key == "y"
        assert ka2.action == "pass"
        assert ka2.value is None

    def test_with_value(self):
        ka = KeyAction(key="3", action="confidence", value=3)
        d = ka.as_dict()
        ka2 = KeyAction.from_dict(d)
        assert ka2.value == 3

    def test_default_value_is_none(self):
        ka = KeyAction(key="q", action="quit")
        assert ka.value is None


# ---------------------------------------------------------------------------
# AnnotationItem dataclass
# ---------------------------------------------------------------------------


class TestAnnotationItem:
    def test_as_dict_roundtrip(self):
        item = _make_item()
        d = item.as_dict()
        item2 = AnnotationItem.from_dict(d)
        assert item2.item_id == "item-001"
        assert item2.input_text == "What is Python?"
        assert item2.status == "pending"

    def test_default_status_is_pending(self):
        item = AnnotationItem.from_dict({
            "item_id": "x",
            "input_text": "in",
            "output_text": "out",
        })
        assert item.status == "pending"

    def test_metric_scores_default_empty(self):
        item = AnnotationItem.from_dict({
            "item_id": "x",
            "input_text": "in",
            "output_text": "out",
        })
        assert item.metric_scores == {}

    def test_metric_scores_preserved(self):
        item = _make_item()
        d = item.as_dict()
        assert d["metric_scores"]["helpfulness"] == 0.9


# ---------------------------------------------------------------------------
# BatchAnnotation dataclass
# ---------------------------------------------------------------------------


class TestBatchAnnotation:
    def test_as_dict_roundtrip(self):
        items = _make_items(3)
        batch = BatchAnnotation(
            items=items,
            batch_id="batch-1",
            batch_size=3,
            annotations={"item-000": 1.0},
        )
        d = batch.as_dict()
        batch2 = BatchAnnotation.from_dict(d)
        assert batch2.batch_id == "batch-1"
        assert batch2.batch_size == 3
        assert len(batch2.items) == 3
        assert batch2.annotations["item-000"] == 1.0

    def test_empty_annotations_default(self):
        batch = BatchAnnotation(
            items=[],
            batch_id="b",
            batch_size=0,
        )
        assert batch.annotations == {}

    def test_from_dict_missing_items(self):
        batch = BatchAnnotation.from_dict({
            "batch_id": "b",
            "batch_size": 0,
        })
        assert batch.items == []


# ---------------------------------------------------------------------------
# DEFAULT_KEYMAP
# ---------------------------------------------------------------------------


class TestDefaultKeymap:
    def test_has_expected_keys(self):
        expected = {"y", "n", "1", "2", "3", "4", "5", "s", "u", "q"}
        assert set(DEFAULT_KEYMAP.keys()) == expected

    def test_y_is_pass(self):
        assert DEFAULT_KEYMAP["y"].action == "pass"

    def test_n_is_fail(self):
        assert DEFAULT_KEYMAP["n"].action == "fail"

    def test_confidence_values(self):
        for i in range(1, 6):
            ka = DEFAULT_KEYMAP[str(i)]
            assert ka.action == "confidence"
            assert ka.value == i

    def test_s_is_skip(self):
        assert DEFAULT_KEYMAP["s"].action == "skip"

    def test_u_is_undo(self):
        assert DEFAULT_KEYMAP["u"].action == "undo"

    def test_q_is_quit(self):
        assert DEFAULT_KEYMAP["q"].action == "quit"


# ---------------------------------------------------------------------------
# UndoStack
# ---------------------------------------------------------------------------


class TestUndoStack:
    def test_empty_initially(self):
        stack = UndoStack()
        assert stack.is_empty()
        assert stack.size() == 0

    def test_push_and_pop(self):
        stack = UndoStack()
        stack.push("item-1", "pending", None)
        assert stack.size() == 1
        entry = stack.pop()
        assert entry == ("item-1", "pending", None)
        assert stack.is_empty()

    def test_pop_empty_returns_none(self):
        stack = UndoStack()
        assert stack.pop() is None

    def test_lifo_order(self):
        stack = UndoStack()
        stack.push("a", "pending", None)
        stack.push("b", "annotated", 1.0)
        entry = stack.pop()
        assert entry[0] == "b"
        entry = stack.pop()
        assert entry[0] == "a"

    def test_size_tracks_correctly(self):
        stack = UndoStack()
        stack.push("a", "pending", None)
        stack.push("b", "pending", None)
        assert stack.size() == 2
        stack.pop()
        assert stack.size() == 1

    def test_push_with_label(self):
        stack = UndoStack()
        stack.push("item-1", "annotated", 3.0)
        entry = stack.pop()
        assert entry == ("item-1", "annotated", 3.0)


# ---------------------------------------------------------------------------
# process_key_action
# ---------------------------------------------------------------------------


class TestProcessKeyAction:
    def test_pass_marks_annotated(self):
        item = _make_item()
        stack = UndoStack()
        result = process_key_action(DEFAULT_KEYMAP["y"], item, stack)
        assert result.status == "annotated"

    def test_fail_marks_annotated(self):
        item = _make_item()
        stack = UndoStack()
        result = process_key_action(DEFAULT_KEYMAP["n"], item, stack)
        assert result.status == "annotated"

    def test_confidence_marks_annotated(self):
        item = _make_item()
        stack = UndoStack()
        result = process_key_action(DEFAULT_KEYMAP["3"], item, stack)
        assert result.status == "annotated"

    def test_skip_marks_skipped(self):
        item = _make_item()
        stack = UndoStack()
        result = process_key_action(DEFAULT_KEYMAP["s"], item, stack)
        assert result.status == "skipped"

    def test_pushes_to_undo_stack(self):
        item = _make_item()
        stack = UndoStack()
        process_key_action(DEFAULT_KEYMAP["y"], item, stack)
        assert stack.size() == 1

    def test_undo_state_captures_previous_status(self):
        item = _make_item(status="pending")
        stack = UndoStack()
        process_key_action(DEFAULT_KEYMAP["y"], item, stack)
        entry = stack.pop()
        assert entry[0] == "item-001"
        assert entry[1] == "pending"  # previous status

    def test_returns_same_item_object(self):
        item = _make_item()
        stack = UndoStack()
        result = process_key_action(DEFAULT_KEYMAP["y"], item, stack)
        assert result is item


# ---------------------------------------------------------------------------
# undo_action
# ---------------------------------------------------------------------------


class TestUndoAction:
    def test_reverts_status(self):
        item = _make_item(status="pending")
        items = {item.item_id: item}
        stack = UndoStack()
        process_key_action(DEFAULT_KEYMAP["y"], item, stack)
        assert item.status == "annotated"
        reverted_id = undo_action(stack, items)
        assert reverted_id == "item-001"
        assert item.status == "pending"

    def test_empty_stack_returns_none(self):
        stack = UndoStack()
        result = undo_action(stack, {})
        assert result is None

    def test_missing_item_does_not_crash(self):
        stack = UndoStack()
        stack.push("nonexistent", "pending", None)
        result = undo_action(stack, {})
        assert result == "nonexistent"

    def test_multiple_undos(self):
        items_list = _make_items(2)
        items = {i.item_id: i for i in items_list}
        stack = UndoStack()
        process_key_action(DEFAULT_KEYMAP["y"], items_list[0], stack)
        process_key_action(DEFAULT_KEYMAP["s"], items_list[1], stack)
        assert items_list[0].status == "annotated"
        assert items_list[1].status == "skipped"
        undo_action(stack, items)
        assert items_list[1].status == "pending"
        undo_action(stack, items)
        assert items_list[0].status == "pending"


# ---------------------------------------------------------------------------
# create_batch
# ---------------------------------------------------------------------------


class TestCreateBatch:
    def test_single_batch(self):
        items = _make_items(3)
        batches = create_batch(items, batch_size=5)
        assert len(batches) == 1
        assert len(batches[0].items) == 3
        assert batches[0].batch_size == 3

    def test_multiple_batches(self):
        items = _make_items(7)
        batches = create_batch(items, batch_size=3)
        assert len(batches) == 3
        assert len(batches[0].items) == 3
        assert len(batches[1].items) == 3
        assert len(batches[2].items) == 1

    def test_exact_division(self):
        items = _make_items(6)
        batches = create_batch(items, batch_size=3)
        assert len(batches) == 2
        assert all(len(b.items) == 3 for b in batches)

    def test_empty_items(self):
        batches = create_batch([], batch_size=5)
        assert batches == []

    def test_default_batch_size_is_five(self):
        items = _make_items(12)
        batches = create_batch(items)
        assert len(batches) == 3
        assert len(batches[0].items) == 5
        assert len(batches[1].items) == 5
        assert len(batches[2].items) == 2

    def test_each_batch_has_unique_id(self):
        items = _make_items(10)
        batches = create_batch(items, batch_size=3)
        ids = [b.batch_id for b in batches]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# get_batch_progress
# ---------------------------------------------------------------------------


class TestGetBatchProgress:
    def test_all_pending(self):
        items = _make_items(5)
        batch = BatchAnnotation(items=items, batch_id="b", batch_size=5)
        progress = get_batch_progress(batch)
        assert progress == {"total": 5, "annotated": 0, "skipped": 0, "remaining": 5}

    def test_mixed_statuses(self):
        items = _make_items(4)
        items[0].status = "annotated"
        items[1].status = "skipped"
        batch = BatchAnnotation(items=items, batch_id="b", batch_size=4)
        progress = get_batch_progress(batch)
        assert progress["annotated"] == 1
        assert progress["skipped"] == 1
        assert progress["remaining"] == 2

    def test_all_annotated(self):
        items = _make_items(3)
        for item in items:
            item.status = "annotated"
        batch = BatchAnnotation(items=items, batch_id="b", batch_size=3)
        progress = get_batch_progress(batch)
        assert progress["remaining"] == 0
        assert progress["annotated"] == 3

    def test_empty_batch(self):
        batch = BatchAnnotation(items=[], batch_id="b", batch_size=0)
        progress = get_batch_progress(batch)
        assert progress == {"total": 0, "annotated": 0, "skipped": 0, "remaining": 0}


# ---------------------------------------------------------------------------
# format_annotation_prompt
# ---------------------------------------------------------------------------


class TestFormatAnnotationPrompt:
    def test_contains_item_id(self):
        item = _make_item()
        prompt = format_annotation_prompt(item)
        assert "item-001" in prompt

    def test_contains_input(self):
        item = _make_item()
        prompt = format_annotation_prompt(item)
        assert "What is Python?" in prompt

    def test_contains_output(self):
        item = _make_item()
        prompt = format_annotation_prompt(item)
        assert "Python is a programming language." in prompt

    def test_contains_scores(self):
        item = _make_item()
        prompt = format_annotation_prompt(item)
        assert "helpfulness" in prompt
        assert "0.90" in prompt

    def test_no_scores_section_when_empty(self):
        item = _make_item(metric_scores={})
        prompt = format_annotation_prompt(item)
        assert "Scores:" not in prompt


# ---------------------------------------------------------------------------
# format_keymap_help
# ---------------------------------------------------------------------------


class TestFormatKeymapHelp:
    def test_shows_all_keys(self):
        help_text = format_keymap_help(DEFAULT_KEYMAP)
        for key in DEFAULT_KEYMAP:
            assert f"[{key}]" in help_text

    def test_shows_actions(self):
        help_text = format_keymap_help(DEFAULT_KEYMAP)
        assert "pass" in help_text
        assert "fail" in help_text
        assert "skip" in help_text

    def test_shows_confidence_values(self):
        help_text = format_keymap_help(DEFAULT_KEYMAP)
        for i in range(1, 6):
            assert f"({i})" in help_text

    def test_empty_keymap(self):
        help_text = format_keymap_help({})
        assert help_text == ""
