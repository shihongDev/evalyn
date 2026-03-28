from __future__ import annotations

import pytest

from evalyn_sdk.trace.span_types import (
    SpanTypeInfo,
    get_span_type,
    is_valid_span_type,
    list_span_types,
    register_span_type,
    reset_registry,
    unregister_span_type,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


# -- SpanTypeInfo dataclass --------------------------------------------------

class TestSpanTypeInfo:
    def test_defaults(self):
        info = SpanTypeInfo(name="foo")
        assert info.icon == ""
        assert info.color == ""
        assert info.description == ""

    def test_as_dict(self):
        info = SpanTypeInfo(name="x", icon="i", color="c", description="d")
        d = info.as_dict()
        assert d == {"name": "x", "icon": "i", "color": "c", "description": "d"}

    def test_from_dict_full(self):
        d = {"name": "x", "icon": "i", "color": "c", "description": "d"}
        info = SpanTypeInfo.from_dict(d)
        assert info.name == "x"
        assert info.icon == "i"
        assert info.color == "c"
        assert info.description == "d"

    def test_from_dict_minimal(self):
        info = SpanTypeInfo.from_dict({"name": "z"})
        assert info.name == "z"
        assert info.icon == ""

    def test_roundtrip(self):
        original = SpanTypeInfo(name="rt", icon="!", color="#fff", description="round")
        rebuilt = SpanTypeInfo.from_dict(original.as_dict())
        assert rebuilt == original


# -- register_span_type -------------------------------------------------------

class TestRegister:
    def test_register_custom_type(self):
        info = register_span_type("my_type", icon="M", description="mine")
        assert info.name == "my_type"
        assert get_span_type("my_type") is info

    def test_duplicate_raises(self):
        register_span_type("dup")
        with pytest.raises(ValueError, match="already registered"):
            register_span_type("dup")

    def test_duplicate_builtin_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_span_type("llm_call")

    def test_register_returns_span_type_info(self):
        result = register_span_type("new_one")
        assert isinstance(result, SpanTypeInfo)


# -- get_span_type -------------------------------------------------------------

class TestGetSpanType:
    def test_builtin(self):
        info = get_span_type("llm_call")
        assert info is not None
        assert info.name == "llm_call"

    def test_unknown_returns_none(self):
        assert get_span_type("nonexistent") is None

    def test_custom_after_register(self):
        register_span_type("custom2", color="red")
        info = get_span_type("custom2")
        assert info is not None
        assert info.color == "red"


# -- list_span_types -----------------------------------------------------------

class TestListSpanTypes:
    def test_includes_builtins(self):
        names = [t.name for t in list_span_types()]
        for builtin in ("llm_call", "tool_call", "retrieval", "node", "agent", "custom", "scorer"):
            assert builtin in names

    def test_sorted_by_name(self):
        names = [t.name for t in list_span_types()]
        assert names == sorted(names)

    def test_includes_custom(self):
        register_span_type("aaa_first")
        names = [t.name for t in list_span_types()]
        assert "aaa_first" in names


# -- is_valid_span_type --------------------------------------------------------

class TestIsValid:
    def test_builtin_valid(self):
        assert is_valid_span_type("llm_call") is True

    def test_unknown_invalid(self):
        assert is_valid_span_type("bogus") is False

    def test_custom_valid_after_register(self):
        register_span_type("valid_check")
        assert is_valid_span_type("valid_check") is True


# -- unregister_span_type ------------------------------------------------------

class TestUnregister:
    def test_unregister_custom(self):
        register_span_type("removable")
        assert unregister_span_type("removable") is True
        assert get_span_type("removable") is None

    def test_cannot_unregister_builtin(self):
        assert unregister_span_type("llm_call") is False
        assert get_span_type("llm_call") is not None

    def test_unregister_unknown_returns_false(self):
        assert unregister_span_type("nope") is False


# -- reset_registry ------------------------------------------------------------

class TestResetRegistry:
    def test_reset_removes_custom(self):
        register_span_type("tmp")
        reset_registry()
        assert get_span_type("tmp") is None

    def test_reset_preserves_builtins(self):
        reset_registry()
        assert get_span_type("llm_call") is not None
        assert get_span_type("scorer") is not None
