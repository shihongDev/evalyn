from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from evalyn_sdk.trace.bookmarking import Bookmark, BookmarkCollection


# --- Bookmark ---


class TestBookmark:
    def test_defaults(self):
        bm = Bookmark(span_id="s1")
        assert bm.span_id == "s1"
        assert bm.label == ""
        assert bm.notes == ""
        assert bm.tags == []
        assert isinstance(bm.created_at, datetime)

    def test_as_dict(self):
        bm = Bookmark(span_id="s1", label="interesting", tags=["bug"])
        d = bm.as_dict()
        assert d["span_id"] == "s1"
        assert d["label"] == "interesting"
        assert d["tags"] == ["bug"]
        assert "created_at" in d

    def test_from_dict_roundtrip(self):
        bm = Bookmark(
            span_id="s1",
            label="test",
            notes="note",
            tags=["a", "b"],
        )
        d = bm.as_dict()
        restored = Bookmark.from_dict(d)
        assert restored.span_id == bm.span_id
        assert restored.label == bm.label
        assert restored.notes == bm.notes
        assert restored.tags == bm.tags

    def test_from_dict_defaults(self):
        bm = Bookmark.from_dict({"span_id": "x"})
        assert bm.label == ""
        assert bm.tags == []


# --- BookmarkCollection ---


class TestBookmarkCollection:
    def test_add_bookmark(self):
        col = BookmarkCollection()
        col.add("s1", label="first")
        assert col.count == 1
        assert col.get("s1") is not None
        assert col.get("s1").label == "first"

    def test_remove_bookmark(self):
        col = BookmarkCollection()
        col.add("s1")
        assert col.remove("s1") is True
        assert col.count == 0

    def test_remove_nonexistent(self):
        col = BookmarkCollection()
        assert col.remove("nope") is False

    def test_get_bookmark(self):
        col = BookmarkCollection()
        col.add("s1", label="found")
        bm = col.get("s1")
        assert bm is not None
        assert bm.label == "found"

    def test_get_nonexistent_returns_none(self):
        col = BookmarkCollection()
        assert col.get("missing") is None

    def test_list_by_tag(self):
        col = BookmarkCollection()
        col.add("s1", tags=["bug", "urgent"])
        col.add("s2", tags=["feature"])
        col.add("s3", tags=["bug"])
        result = col.list_by_tag("bug")
        ids = [b.span_id for b in result]
        assert "s1" in ids
        assert "s3" in ids
        assert "s2" not in ids

    def test_list_by_tag_no_match(self):
        col = BookmarkCollection()
        col.add("s1", tags=["a"])
        assert col.list_by_tag("z") == []

    def test_list_by_label(self):
        col = BookmarkCollection()
        col.add("s1", label="review")
        col.add("s2", label="review")
        col.add("s3", label="skip")
        result = col.list_by_label("review")
        assert len(result) == 2

    def test_list_by_label_no_match(self):
        col = BookmarkCollection()
        col.add("s1", label="x")
        assert col.list_by_label("y") == []

    def test_count_property(self):
        col = BookmarkCollection()
        assert col.count == 0
        col.add("s1")
        assert col.count == 1
        col.add("s2")
        assert col.count == 2

    def test_duplicate_add_replaces(self):
        col = BookmarkCollection()
        col.add("s1", label="old")
        col.add("s1", label="new")
        assert col.count == 1
        assert col.get("s1").label == "new"

    def test_as_dict_from_dict_roundtrip(self):
        col = BookmarkCollection()
        col.add("s1", label="a", tags=["x"])
        col.add("s2", label="b", notes="n")
        d = col.as_dict()
        restored = BookmarkCollection.from_dict(d)
        assert restored.count == 2
        assert restored.get("s1").label == "a"
        assert restored.get("s2").notes == "n"

    def test_to_json_from_json_roundtrip(self):
        col = BookmarkCollection()
        col.add("s1", label="test", tags=["t1"])
        col.add("s2", notes="hello")
        j = col.to_json()
        # Verify it is valid JSON
        parsed = json.loads(j)
        assert "bookmarks" in parsed
        restored = BookmarkCollection.from_json(j)
        assert restored.count == 2
        assert restored.get("s1").label == "test"
        assert restored.get("s2").notes == "hello"

    def test_empty_collection(self):
        col = BookmarkCollection()
        assert col.count == 0
        assert col.as_dict() == {"bookmarks": []}
        j = col.to_json()
        restored = BookmarkCollection.from_json(j)
        assert restored.count == 0

    def test_multiple_bookmarks(self):
        col = BookmarkCollection()
        for i in range(10):
            col.add(f"s{i}", label=f"label{i}")
        assert col.count == 10
        assert col.get("s5").label == "label5"

    def test_add_returns_bookmark(self):
        col = BookmarkCollection()
        bm = col.add("s1", label="ret")
        assert isinstance(bm, Bookmark)
        assert bm.span_id == "s1"
        assert bm.label == "ret"

    def test_remove_then_get(self):
        col = BookmarkCollection()
        col.add("s1")
        col.remove("s1")
        assert col.get("s1") is None

    def test_tags_preserved_on_add(self):
        col = BookmarkCollection()
        col.add("s1", tags=["a", "b", "c"])
        bm = col.get("s1")
        assert bm.tags == ["a", "b", "c"]

    def test_from_dict_empty(self):
        col = BookmarkCollection.from_dict({})
        assert col.count == 0

    def test_from_json_empty_list(self):
        col = BookmarkCollection.from_json('{"bookmarks": []}')
        assert col.count == 0
