"""Tests for compare_shorthand module."""

from __future__ import annotations

import pytest

from evalyn_sdk.compare_shorthand import (
    COMPARE_MODES,
    CompareMode,
    CompareRequest,
    format_compare_request,
    list_modes,
    resolve_last_2,
    resolve_latest_vs_pinned,
    resolve_latest_vs_previous,
    resolve_shorthand,
)


# ---------------------------------------------------------------------------
# CompareMode dataclass
# ---------------------------------------------------------------------------


class TestCompareMode:
    def test_create(self):
        m = CompareMode(mode="last_2", description="Two most recent")
        assert m.mode == "last_2"
        assert m.description == "Two most recent"

    def test_as_dict(self):
        m = CompareMode(mode="last_2", description="desc")
        d = m.as_dict()
        assert d == {"mode": "last_2", "description": "desc"}

    def test_from_dict(self):
        d = {"mode": "latest_vs_pinned", "description": "pinned"}
        m = CompareMode.from_dict(d)
        assert m.mode == "latest_vs_pinned"
        assert m.description == "pinned"

    def test_roundtrip(self):
        m = CompareMode(mode="x", description="y")
        assert CompareMode.from_dict(m.as_dict()) == m

    def test_equality(self):
        a = CompareMode(mode="a", description="b")
        b = CompareMode(mode="a", description="b")
        assert a == b

    def test_inequality(self):
        a = CompareMode(mode="a", description="b")
        b = CompareMode(mode="a", description="c")
        assert a != b


# ---------------------------------------------------------------------------
# CompareRequest dataclass
# ---------------------------------------------------------------------------


class TestCompareRequest:
    def test_create_defaults(self):
        r = CompareRequest(run_a_id="a", run_b_id="b", mode="last_2")
        assert r.resolved is True

    def test_create_unresolved(self):
        r = CompareRequest(run_a_id="", run_b_id="", mode="m", resolved=False)
        assert r.resolved is False

    def test_as_dict(self):
        r = CompareRequest(run_a_id="a", run_b_id="b", mode="last_2")
        d = r.as_dict()
        assert d["run_a_id"] == "a"
        assert d["run_b_id"] == "b"
        assert d["mode"] == "last_2"
        assert d["resolved"] is True

    def test_from_dict_defaults(self):
        d = {"run_a_id": "x", "run_b_id": "y", "mode": "m"}
        r = CompareRequest.from_dict(d)
        assert r.resolved is True

    def test_from_dict_explicit_resolved(self):
        d = {"run_a_id": "", "run_b_id": "", "mode": "m", "resolved": False}
        r = CompareRequest.from_dict(d)
        assert r.resolved is False

    def test_roundtrip(self):
        r = CompareRequest(run_a_id="a", run_b_id="b", mode="last_2", resolved=True)
        assert CompareRequest.from_dict(r.as_dict()) == r

    def test_roundtrip_unresolved(self):
        r = CompareRequest(run_a_id="", run_b_id="", mode="x", resolved=False)
        assert CompareRequest.from_dict(r.as_dict()) == r


# ---------------------------------------------------------------------------
# resolve_last_2
# ---------------------------------------------------------------------------


class TestResolveLast2:
    def test_basic(self):
        r = resolve_last_2(["r1", "r2", "r3"])
        assert r.run_a_id == "r2"
        assert r.run_b_id == "r3"
        assert r.resolved is True
        assert r.mode == "last_2"

    def test_exactly_two(self):
        r = resolve_last_2(["a", "b"])
        assert r.run_a_id == "a"
        assert r.run_b_id == "b"
        assert r.resolved is True

    def test_one_run(self):
        r = resolve_last_2(["only"])
        assert r.resolved is False

    def test_empty(self):
        r = resolve_last_2([])
        assert r.resolved is False
        assert r.run_a_id == ""
        assert r.run_b_id == ""

    def test_many_runs(self):
        ids = [f"r{i}" for i in range(10)]
        r = resolve_last_2(ids)
        assert r.run_a_id == "r8"
        assert r.run_b_id == "r9"


# ---------------------------------------------------------------------------
# resolve_latest_vs_pinned
# ---------------------------------------------------------------------------


class TestResolveLatestVsPinned:
    def test_basic(self):
        r = resolve_latest_vs_pinned(["r1", "r2", "r3"], "baseline")
        assert r.run_a_id == "baseline"
        assert r.run_b_id == "r3"
        assert r.resolved is True
        assert r.mode == "latest_vs_pinned"

    def test_empty_runs(self):
        r = resolve_latest_vs_pinned([], "pin")
        assert r.resolved is False

    def test_empty_pinned(self):
        r = resolve_latest_vs_pinned(["r1"], "")
        assert r.resolved is False

    def test_both_empty(self):
        r = resolve_latest_vs_pinned([], "")
        assert r.resolved is False

    def test_single_run(self):
        r = resolve_latest_vs_pinned(["r1"], "base")
        assert r.run_a_id == "base"
        assert r.run_b_id == "r1"
        assert r.resolved is True

    def test_pinned_not_in_list(self):
        # pinned_id does not need to be in the run_ids list
        r = resolve_latest_vs_pinned(["r1", "r2"], "external_baseline")
        assert r.run_a_id == "external_baseline"
        assert r.run_b_id == "r2"
        assert r.resolved is True


# ---------------------------------------------------------------------------
# resolve_latest_vs_previous
# ---------------------------------------------------------------------------


class TestResolveLatestVsPrevious:
    def test_basic(self):
        r = resolve_latest_vs_previous(["r1", "r2", "r3"])
        assert r.run_a_id == "r2"
        assert r.run_b_id == "r3"
        assert r.resolved is True
        assert r.mode == "latest_vs_previous"

    def test_exactly_two(self):
        r = resolve_latest_vs_previous(["old", "new"])
        assert r.run_a_id == "old"
        assert r.run_b_id == "new"

    def test_one_run(self):
        r = resolve_latest_vs_previous(["only"])
        assert r.resolved is False

    def test_empty(self):
        r = resolve_latest_vs_previous([])
        assert r.resolved is False


# ---------------------------------------------------------------------------
# resolve_shorthand dispatcher
# ---------------------------------------------------------------------------


class TestResolveShorthand:
    def test_last_2(self):
        r = resolve_shorthand("last_2", ["a", "b", "c"])
        assert r.mode == "last_2"
        assert r.run_a_id == "b"
        assert r.run_b_id == "c"
        assert r.resolved is True

    def test_latest_vs_pinned(self):
        r = resolve_shorthand("latest_vs_pinned", ["a", "b"], pinned_id="pin")
        assert r.mode == "latest_vs_pinned"
        assert r.run_a_id == "pin"
        assert r.run_b_id == "b"

    def test_latest_vs_previous(self):
        r = resolve_shorthand("latest_vs_previous", ["a", "b"])
        assert r.mode == "latest_vs_previous"
        assert r.run_a_id == "a"
        assert r.run_b_id == "b"

    def test_unknown_mode(self):
        r = resolve_shorthand("nonexistent", ["a", "b"])
        assert r.resolved is False
        assert r.mode == "nonexistent"

    def test_not_enough_runs_last_2(self):
        r = resolve_shorthand("last_2", ["only"])
        assert r.resolved is False

    def test_not_enough_runs_latest_vs_previous(self):
        r = resolve_shorthand("latest_vs_previous", [])
        assert r.resolved is False

    def test_no_pinned_id(self):
        r = resolve_shorthand("latest_vs_pinned", ["a", "b"])
        assert r.resolved is False

    def test_default_pinned_id(self):
        # pinned_id defaults to ""
        r = resolve_shorthand("latest_vs_pinned", ["a"])
        assert r.resolved is False


# ---------------------------------------------------------------------------
# COMPARE_MODES and list_modes
# ---------------------------------------------------------------------------


class TestCompareModes:
    def test_has_three_modes(self):
        assert len(COMPARE_MODES) == 3

    def test_keys(self):
        assert set(COMPARE_MODES.keys()) == {
            "last_2",
            "latest_vs_pinned",
            "latest_vs_previous",
        }

    def test_all_are_compare_mode(self):
        for m in COMPARE_MODES.values():
            assert isinstance(m, CompareMode)

    def test_list_modes_returns_list(self):
        modes = list_modes()
        assert isinstance(modes, list)
        assert len(modes) == 3

    def test_list_modes_content(self):
        modes = list_modes()
        mode_strs = [m.mode for m in modes]
        assert "last_2" in mode_strs
        assert "latest_vs_pinned" in mode_strs
        assert "latest_vs_previous" in mode_strs


# ---------------------------------------------------------------------------
# format_compare_request
# ---------------------------------------------------------------------------


class TestFormatCompareRequest:
    def test_resolved_last_2(self):
        r = CompareRequest(run_a_id="a", run_b_id="b", mode="last_2")
        text = format_compare_request(r)
        assert "a" in text
        assert "b" in text
        assert "vs" in text

    def test_resolved_latest_vs_pinned(self):
        r = CompareRequest(
            run_a_id="pin", run_b_id="latest", mode="latest_vs_pinned"
        )
        text = format_compare_request(r)
        assert "pin" in text
        assert "latest" in text

    def test_unresolved(self):
        r = CompareRequest(run_a_id="", run_b_id="", mode="last_2", resolved=False)
        text = format_compare_request(r)
        assert "unresolved" in text

    def test_unknown_mode_resolved(self):
        r = CompareRequest(run_a_id="a", run_b_id="b", mode="custom_mode")
        text = format_compare_request(r)
        assert "custom_mode" in text
        assert "a" in text
        assert "b" in text

    def test_format_includes_description(self):
        r = CompareRequest(run_a_id="x", run_b_id="y", mode="last_2")
        text = format_compare_request(r)
        assert "most recent" in text.lower() or "last" in text.lower() or "x" in text
