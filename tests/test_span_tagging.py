from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.span_tagging import (
    STANDARD_TAGS,
    SpanTag,
    TagSet,
    apply_tags,
    create_tag_set,
    filter_spans_by_tag,
    get_tags,
    has_tag,
)


def _span(sid: str = "s1", **attrs) -> Span:
    return Span(
        id=sid,
        name="test",
        span_type="custom",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# -- SpanTag -------------------------------------------------------------------

class TestSpanTag:
    def test_basic(self):
        tag = SpanTag(key="env", value="prod")
        assert tag.key == "env"
        assert tag.value == "prod"

    def test_as_dict(self):
        tag = SpanTag(key="k", value=42)
        assert tag.as_dict() == {"key": "k", "value": 42}

    def test_complex_value(self):
        tag = SpanTag(key="data", value={"nested": True})
        d = tag.as_dict()
        assert d["value"] == {"nested": True}


# -- TagSet --------------------------------------------------------------------

class TestTagSet:
    def test_empty(self):
        ts = TagSet()
        assert ts.tags == {}

    def test_as_dict(self):
        ts = TagSet(tags={"a": 1, "b": 2})
        assert ts.as_dict() == {"tags": {"a": 1, "b": 2}}

    def test_from_dict(self):
        ts = TagSet.from_dict({"tags": {"x": "y"}})
        assert ts.tags == {"x": "y"}

    def test_from_dict_missing_tags(self):
        ts = TagSet.from_dict({})
        assert ts.tags == {}

    def test_roundtrip(self):
        original = TagSet(tags={"env": "staging", "version": "1.2"})
        rebuilt = TagSet.from_dict(original.as_dict())
        assert rebuilt.tags == original.tags

    def test_merge(self):
        a = TagSet(tags={"x": 1, "y": 2})
        b = TagSet(tags={"y": 99, "z": 3})
        merged = a.merge(b)
        assert merged.tags == {"x": 1, "y": 99, "z": 3}

    def test_merge_does_not_mutate(self):
        a = TagSet(tags={"x": 1})
        b = TagSet(tags={"x": 2})
        a.merge(b)
        assert a.tags == {"x": 1}

    def test_merge_empty(self):
        a = TagSet(tags={"k": "v"})
        merged = a.merge(TagSet())
        assert merged.tags == {"k": "v"}


# -- create_tag_set ------------------------------------------------------------

class TestCreateTagSet:
    def test_basic(self):
        ts = create_tag_set(env="prod", version="2")
        assert ts.tags["env"] == "prod"
        assert ts.tags["version"] == "2"

    def test_empty(self):
        ts = create_tag_set()
        assert ts.tags == {}


# -- apply_tags ----------------------------------------------------------------

class TestApplyTags:
    def test_creates_new_span(self):
        s = _span()
        ts = create_tag_set(env="prod")
        new_s = apply_tags(s, ts)
        assert new_s is not s

    def test_original_unchanged(self):
        s = _span()
        ts = create_tag_set(env="prod")
        apply_tags(s, ts)
        assert "tags" not in s.attributes

    def test_tags_applied(self):
        s = _span()
        ts = create_tag_set(env="prod", user_id="u1")
        new_s = apply_tags(s, ts)
        assert new_s.attributes["tags"]["env"] == "prod"
        assert new_s.attributes["tags"]["user_id"] == "u1"

    def test_merge_with_existing_tags(self):
        s = _span(tags={"old": "val"})
        ts = create_tag_set(new="val2")
        new_s = apply_tags(s, ts)
        assert new_s.attributes["tags"]["old"] == "val"
        assert new_s.attributes["tags"]["new"] == "val2"

    def test_preserves_other_attributes(self):
        s = _span(foo="bar")
        ts = create_tag_set(env="dev")
        new_s = apply_tags(s, ts)
        assert new_s.attributes["foo"] == "bar"


# -- get_tags ------------------------------------------------------------------

class TestGetTags:
    def test_from_tagged_span(self):
        s = _span(tags={"env": "prod", "v": "1"})
        ts = get_tags(s)
        assert ts.tags == {"env": "prod", "v": "1"}

    def test_from_untagged_span(self):
        s = _span()
        ts = get_tags(s)
        assert ts.tags == {}

    def test_does_not_share_reference(self):
        s = _span(tags={"a": "b"})
        ts = get_tags(s)
        ts.tags["a"] = "changed"
        assert s.attributes["tags"]["a"] == "b"


# -- has_tag -------------------------------------------------------------------

class TestHasTag:
    def test_true(self):
        s = _span(tags={"env": "prod"})
        assert has_tag(s, "env") is True

    def test_false(self):
        s = _span(tags={"env": "prod"})
        assert has_tag(s, "user_id") is False

    def test_no_tags_at_all(self):
        s = _span()
        assert has_tag(s, "env") is False


# -- filter_spans_by_tag -------------------------------------------------------

class TestFilterSpansByTag:
    def test_key_only(self):
        spans = [
            _span("a", tags={"env": "prod"}),
            _span("b", tags={"version": "1"}),
            _span("c", tags={"env": "staging"}),
        ]
        result = filter_spans_by_tag(spans, "env")
        assert [s.id for s in result] == ["a", "c"]

    def test_key_and_value(self):
        spans = [
            _span("a", tags={"env": "prod"}),
            _span("b", tags={"env": "staging"}),
        ]
        result = filter_spans_by_tag(spans, "env", "prod")
        assert [s.id for s in result] == ["a"]

    def test_no_match(self):
        spans = [_span("a", tags={"x": 1})]
        result = filter_spans_by_tag(spans, "y")
        assert result == []

    def test_empty_list(self):
        assert filter_spans_by_tag([], "env") == []

    def test_spans_without_tags(self):
        spans = [_span("a"), _span("b")]
        result = filter_spans_by_tag(spans, "env")
        assert result == []


# -- STANDARD_TAGS constant ----------------------------------------------------

class TestStandardTags:
    def test_is_set(self):
        assert isinstance(STANDARD_TAGS, set)

    def test_expected_members(self):
        expected = {"environment", "user_id", "experiment_id", "variant", "version"}
        assert STANDARD_TAGS == expected
