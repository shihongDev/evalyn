from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from evalyn_sdk.trace.annotation import AnnotationStore, TraceAnnotation


# -- TraceAnnotation dataclass --


class TestTraceAnnotationCreate:
    def test_create_with_defaults(self):
        ann = TraceAnnotation(span_id="s1")
        assert ann.span_id == "s1"
        assert ann.notes == []
        assert ann.tags == []
        assert isinstance(ann.created_at, datetime)
        assert isinstance(ann.updated_at, datetime)

    def test_create_with_values(self):
        ann = TraceAnnotation(span_id="s2", notes=["a"], tags=["t1"])
        assert ann.notes == ["a"]
        assert ann.tags == ["t1"]


class TestTraceAnnotationAddNote:
    def test_add_note_appends(self):
        ann = TraceAnnotation(span_id="s1")
        ann.add_note("first")
        assert ann.notes == ["first"]

    def test_add_note_updates_timestamp(self):
        ann = TraceAnnotation(span_id="s1")
        before = ann.updated_at
        time.sleep(0.01)
        ann.add_note("note")
        assert ann.updated_at >= before

    def test_add_multiple_notes(self):
        ann = TraceAnnotation(span_id="s1")
        ann.add_note("a")
        ann.add_note("b")
        assert ann.notes == ["a", "b"]


class TestTraceAnnotationAddTag:
    def test_add_tag_appends(self):
        ann = TraceAnnotation(span_id="s1")
        ann.add_tag("important")
        assert ann.tags == ["important"]

    def test_add_tag_updates_timestamp(self):
        ann = TraceAnnotation(span_id="s1")
        before = ann.updated_at
        time.sleep(0.01)
        ann.add_tag("x")
        assert ann.updated_at >= before

    def test_duplicate_tag_not_added(self):
        ann = TraceAnnotation(span_id="s1")
        ann.add_tag("dup")
        ann.add_tag("dup")
        assert ann.tags == ["dup"]

    def test_duplicate_tag_does_not_update_timestamp(self):
        ann = TraceAnnotation(span_id="s1")
        ann.add_tag("dup")
        ts = ann.updated_at
        ann.add_tag("dup")
        assert ann.updated_at == ts


class TestTraceAnnotationRemoveTag:
    def test_remove_existing_tag(self):
        ann = TraceAnnotation(span_id="s1", tags=["a", "b"])
        assert ann.remove_tag("a") is True
        assert ann.tags == ["b"]

    def test_remove_missing_tag(self):
        ann = TraceAnnotation(span_id="s1", tags=["a"])
        assert ann.remove_tag("z") is False
        assert ann.tags == ["a"]

    def test_remove_tag_updates_timestamp(self):
        ann = TraceAnnotation(span_id="s1", tags=["x"])
        before = ann.updated_at
        time.sleep(0.01)
        ann.remove_tag("x")
        assert ann.updated_at >= before


class TestTraceAnnotationRoundtrip:
    def test_as_dict_fields(self):
        ann = TraceAnnotation(span_id="s1", notes=["n"], tags=["t"])
        d = ann.as_dict()
        assert d["span_id"] == "s1"
        assert d["notes"] == ["n"]
        assert d["tags"] == ["t"]
        assert "created_at" in d
        assert "updated_at" in d

    def test_from_dict_roundtrip(self):
        original = TraceAnnotation(span_id="s1", notes=["n1"], tags=["t1", "t2"])
        restored = TraceAnnotation.from_dict(original.as_dict())
        assert restored.span_id == original.span_id
        assert restored.notes == original.notes
        assert restored.tags == original.tags

    def test_from_dict_defaults(self):
        ann = TraceAnnotation.from_dict({"span_id": "s1"})
        assert ann.notes == []
        assert ann.tags == []


# -- AnnotationStore --


class TestAnnotationStoreAnnotate:
    def test_annotate_creates_new(self):
        store = AnnotationStore()
        ann = store.annotate("s1", note="hello")
        assert ann.span_id == "s1"
        assert ann.notes == ["hello"]

    def test_annotate_updates_existing(self):
        store = AnnotationStore()
        store.annotate("s1", note="first")
        store.annotate("s1", note="second")
        ann = store.get("s1")
        assert ann is not None
        assert ann.notes == ["first", "second"]

    def test_annotate_with_tag(self):
        store = AnnotationStore()
        store.annotate("s1", tag="important")
        ann = store.get("s1")
        assert ann is not None
        assert ann.tags == ["important"]

    def test_annotate_with_both(self):
        store = AnnotationStore()
        ann = store.annotate("s1", note="n", tag="t")
        assert ann.notes == ["n"]
        assert ann.tags == ["t"]


class TestAnnotationStoreGet:
    def test_get_existing(self):
        store = AnnotationStore()
        store.annotate("s1", note="x")
        assert store.get("s1") is not None

    def test_get_missing(self):
        store = AnnotationStore()
        assert store.get("missing") is None


class TestAnnotationStoreFindByTag:
    def test_find_by_tag(self):
        store = AnnotationStore()
        store.annotate("s1", tag="bug")
        store.annotate("s2", tag="bug")
        store.annotate("s3", tag="feature")
        results = store.find_by_tag("bug")
        assert len(results) == 2
        ids = {a.span_id for a in results}
        assert ids == {"s1", "s2"}

    def test_find_by_tag_empty(self):
        store = AnnotationStore()
        assert store.find_by_tag("nothing") == []


class TestAnnotationStoreFindByNoteKeyword:
    def test_find_by_note_keyword(self):
        store = AnnotationStore()
        store.annotate("s1", note="performance issue here")
        store.annotate("s2", note="all good")
        results = store.find_by_note_keyword("performance")
        assert len(results) == 1
        assert results[0].span_id == "s1"

    def test_find_by_note_keyword_no_match(self):
        store = AnnotationStore()
        store.annotate("s1", note="fine")
        assert store.find_by_note_keyword("missing") == []


class TestAnnotationStoreRemove:
    def test_remove_existing(self):
        store = AnnotationStore()
        store.annotate("s1", note="x")
        assert store.remove("s1") is True
        assert store.get("s1") is None

    def test_remove_missing(self):
        store = AnnotationStore()
        assert store.remove("nope") is False


class TestAnnotationStoreCount:
    def test_count_empty(self):
        store = AnnotationStore()
        assert store.count == 0

    def test_count_after_adds(self):
        store = AnnotationStore()
        store.annotate("s1", note="a")
        store.annotate("s2", note="b")
        assert store.count == 2

    def test_count_after_remove(self):
        store = AnnotationStore()
        store.annotate("s1", note="a")
        store.remove("s1")
        assert store.count == 0


class TestAnnotationStoreRoundtrip:
    def test_as_dict_from_dict(self):
        store = AnnotationStore()
        store.annotate("s1", note="n1", tag="t1")
        store.annotate("s2", note="n2")
        restored = AnnotationStore.from_dict(store.as_dict())
        assert restored.count == 2
        assert restored.get("s1") is not None
        assert restored.get("s1").notes == ["n1"]
        assert restored.get("s1").tags == ["t1"]

    def test_to_json_from_json(self):
        store = AnnotationStore()
        store.annotate("s1", note="n", tag="t")
        json_str = store.to_json()
        restored = AnnotationStore.from_json(json_str)
        assert restored.count == 1
        assert restored.get("s1").notes == ["n"]

    def test_to_json_is_valid_json(self):
        store = AnnotationStore()
        store.annotate("s1", note="test")
        parsed = json.loads(store.to_json())
        assert "annotations" in parsed

    def test_empty_store_roundtrip(self):
        store = AnnotationStore()
        restored = AnnotationStore.from_json(store.to_json())
        assert restored.count == 0
