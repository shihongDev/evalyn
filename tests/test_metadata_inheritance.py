"""Tests for trace metadata inheritance."""

import pytest
from datetime import datetime, timezone
from evalyn_sdk.models import Span
from evalyn_sdk.trace.metadata_inheritance import (
    InheritanceConfig,
    InheritanceResult,
    get_inheritable_tags,
    apply_inheritance,
)


def _span(sid="s1", span_type="custom", parent_id=None, **attrs):
    return Span(
        id=sid, name="test", span_type=span_type,
        parent_id=parent_id, start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# =============================================================================
# InheritanceConfig
# =============================================================================

class TestInheritanceConfig:
    def test_defaults(self):
        cfg = InheritanceConfig()
        assert cfg.mode == "inherit_all"
        assert cfg.inherit_keys == []

    def test_as_dict(self):
        cfg = InheritanceConfig(mode="inherit_listed", inherit_keys=["env", "team"])
        d = cfg.as_dict()
        assert d["mode"] == "inherit_listed"
        assert d["inherit_keys"] == ["env", "team"]

    def test_from_dict(self):
        d = {"mode": "no_inherit", "inherit_keys": ["x"]}
        cfg = InheritanceConfig.from_dict(d)
        assert cfg.mode == "no_inherit"
        assert cfg.inherit_keys == ["x"]

    def test_roundtrip(self):
        original = InheritanceConfig(mode="inherit_listed", inherit_keys=["a", "b"])
        restored = InheritanceConfig.from_dict(original.as_dict())
        assert restored.mode == original.mode
        assert restored.inherit_keys == original.inherit_keys

    def test_from_dict_defaults(self):
        cfg = InheritanceConfig.from_dict({})
        assert cfg.mode == "inherit_all"
        assert cfg.inherit_keys == []


# =============================================================================
# InheritanceResult
# =============================================================================

class TestInheritanceResult:
    def test_as_dict(self):
        r = InheritanceResult(spans_updated=3, tags_inherited=7)
        d = r.as_dict()
        assert d["spans_updated"] == 3
        assert d["tags_inherited"] == 7


# =============================================================================
# get_inheritable_tags
# =============================================================================

class TestGetInheritableTags:
    def test_inherit_all(self):
        s = _span(tags={"env": "prod", "team": "ml"})
        cfg = InheritanceConfig(mode="inherit_all")
        tags = get_inheritable_tags(s, cfg)
        assert tags == {"env": "prod", "team": "ml"}

    def test_inherit_listed(self):
        s = _span(tags={"env": "prod", "team": "ml", "region": "us"})
        cfg = InheritanceConfig(mode="inherit_listed", inherit_keys=["env", "region"])
        tags = get_inheritable_tags(s, cfg)
        assert tags == {"env": "prod", "region": "us"}

    def test_no_inherit(self):
        s = _span(tags={"env": "prod"})
        cfg = InheritanceConfig(mode="no_inherit")
        tags = get_inheritable_tags(s, cfg)
        assert tags == {}

    def test_no_tags_attribute(self):
        s = _span()
        cfg = InheritanceConfig(mode="inherit_all")
        tags = get_inheritable_tags(s, cfg)
        assert tags == {}

    def test_non_dict_tags(self):
        s = _span(tags="not a dict")
        cfg = InheritanceConfig(mode="inherit_all")
        tags = get_inheritable_tags(s, cfg)
        assert tags == {}


# =============================================================================
# apply_inheritance - inherit_all mode
# =============================================================================

class TestApplyInheritanceAll:
    def test_parent_tags_copied_to_child(self):
        parent = _span(sid="p", tags={"env": "prod", "team": "ml"})
        child = _span(sid="c", parent_id="p")
        result_spans, result = apply_inheritance([parent, child])
        child_out = next(s for s in result_spans if s.id == "c")
        assert child_out.attributes["tags"]["env"] == "prod"
        assert child_out.attributes["tags"]["team"] == "ml"

    def test_child_tags_override_parent(self):
        parent = _span(sid="p", tags={"env": "prod", "ver": "1"})
        child = _span(sid="c", parent_id="p", tags={"env": "staging"})
        result_spans, _ = apply_inheritance([parent, child])
        child_out = next(s for s in result_spans if s.id == "c")
        assert child_out.attributes["tags"]["env"] == "staging"
        assert child_out.attributes["tags"]["ver"] == "1"

    def test_deep_hierarchy(self):
        gp = _span(sid="gp", tags={"level": "0", "shared": "gp"})
        p = _span(sid="p", parent_id="gp", tags={"level": "1"})
        c = _span(sid="c", parent_id="p", tags={"level": "2"})
        result_spans, result = apply_inheritance([gp, p, c])
        child = next(s for s in result_spans if s.id == "c")
        # Child should have grandparent's "shared" propagated through parent
        assert child.attributes["tags"]["shared"] == "gp"
        # Each level overrides "level"
        assert child.attributes["tags"]["level"] == "2"

    def test_result_counts(self):
        parent = _span(sid="p", tags={"a": 1, "b": 2})
        child = _span(sid="c", parent_id="p")
        _, result = apply_inheritance([parent, child])
        assert result.spans_updated == 1
        assert result.tags_inherited == 2

    def test_originals_unchanged(self):
        parent = _span(sid="p", tags={"env": "prod"})
        child = _span(sid="c", parent_id="p")
        original_child_attrs = dict(child.attributes)
        apply_inheritance([parent, child])
        assert child.attributes == original_child_attrs

    def test_spans_without_parents_unaffected(self):
        root1 = _span(sid="r1", tags={"a": 1})
        root2 = _span(sid="r2", tags={"b": 2})
        result_spans, result = apply_inheritance([root1, root2])
        r1_out = next(s for s in result_spans if s.id == "r1")
        r2_out = next(s for s in result_spans if s.id == "r2")
        assert r1_out.attributes["tags"] == {"a": 1}
        assert r2_out.attributes["tags"] == {"b": 2}
        assert result.spans_updated == 0


# =============================================================================
# apply_inheritance - inherit_listed mode
# =============================================================================

class TestApplyInheritanceListed:
    def test_only_listed_keys_inherited(self):
        parent = _span(sid="p", tags={"env": "prod", "team": "ml", "secret": "x"})
        child = _span(sid="c", parent_id="p")
        cfg = InheritanceConfig(mode="inherit_listed", inherit_keys=["env", "team"])
        result_spans, _ = apply_inheritance([parent, child], cfg)
        child_out = next(s for s in result_spans if s.id == "c")
        assert "env" in child_out.attributes["tags"]
        assert "team" in child_out.attributes["tags"]
        assert "secret" not in child_out.attributes["tags"]


# =============================================================================
# apply_inheritance - no_inherit mode
# =============================================================================

class TestApplyInheritanceNoInherit:
    def test_nothing_inherited(self):
        parent = _span(sid="p", tags={"env": "prod"})
        child = _span(sid="c", parent_id="p")
        cfg = InheritanceConfig(mode="no_inherit")
        result_spans, result = apply_inheritance([parent, child], cfg)
        child_out = next(s for s in result_spans if s.id == "c")
        assert child_out.attributes.get("tags", {}) == {}
        assert result.spans_updated == 0
        assert result.tags_inherited == 0


# =============================================================================
# apply_inheritance - edge cases
# =============================================================================

class TestApplyInheritanceEdgeCases:
    def test_empty_spans(self):
        result_spans, result = apply_inheritance([])
        assert result_spans == []
        assert result.spans_updated == 0
        assert result.tags_inherited == 0

    def test_single_span_no_parent(self):
        s = _span(sid="solo", tags={"x": 1})
        result_spans, result = apply_inheritance([s])
        assert len(result_spans) == 1
        assert result.spans_updated == 0

    def test_parent_not_in_list(self):
        # Child references a parent that is not in the span list
        child = _span(sid="c", parent_id="missing")
        result_spans, result = apply_inheritance([child])
        assert len(result_spans) == 1
        assert result.spans_updated == 0

    def test_default_config_used(self):
        parent = _span(sid="p", tags={"env": "prod"})
        child = _span(sid="c", parent_id="p")
        result_spans, result = apply_inheritance([parent, child])
        child_out = next(s for s in result_spans if s.id == "c")
        assert child_out.attributes["tags"]["env"] == "prod"

    def test_multiple_children(self):
        parent = _span(sid="p", tags={"env": "prod"})
        c1 = _span(sid="c1", parent_id="p")
        c2 = _span(sid="c2", parent_id="p", tags={"env": "dev"})
        result_spans, result = apply_inheritance([parent, c1, c2])
        c1_out = next(s for s in result_spans if s.id == "c1")
        c2_out = next(s for s in result_spans if s.id == "c2")
        assert c1_out.attributes["tags"]["env"] == "prod"
        assert c2_out.attributes["tags"]["env"] == "dev"
        assert result.spans_updated == 1  # c1 gets new tag, c2 already had it
