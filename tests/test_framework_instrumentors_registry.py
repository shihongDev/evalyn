"""Tests for the framework detection and instrumentation registry.

Run with: uv run python -m pytest tests/test_framework_instrumentors_registry.py -v
"""
from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from evalyn_sdk.trace.framework_instrumentors import (
    BUILTIN_FRAMEWORKS,
    FrameworkDetectionResult,
    FrameworkRegistry,
    FrameworkSpec,
    create_default_registry,
    format_framework_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_find_spec(module_path: str, available: set[str]):
    """Return a callable that mimics importlib.util.find_spec for given modules."""
    def finder(name: str):
        if name in available:
            spec = MagicMock()
            spec.name = name
            return spec
        return None
    return finder


# ---------------------------------------------------------------------------
# FrameworkSpec - construction and fields
# ---------------------------------------------------------------------------


class TestFrameworkSpecConstruction:
    def test_basic_fields(self):
        spec = FrameworkSpec(
            framework_id="test",
            name="Test Framework",
            module_path="test_fw",
            detect_fn_hint="import test_fw",
            entry_points=["Main", "Runner"],
        )
        assert spec.framework_id == "test"
        assert spec.name == "Test Framework"
        assert spec.module_path == "test_fw"
        assert spec.detect_fn_hint == "import test_fw"
        assert spec.entry_points == ["Main", "Runner"]

    def test_empty_entry_points(self):
        spec = FrameworkSpec(
            framework_id="e",
            name="Empty",
            module_path="e",
            detect_fn_hint="",
            entry_points=[],
        )
        assert spec.entry_points == []

    def test_multiple_entry_points(self):
        spec = FrameworkSpec(
            framework_id="m",
            name="Multi",
            module_path="m",
            detect_fn_hint="",
            entry_points=["A", "B", "C", "D"],
        )
        assert len(spec.entry_points) == 4


# ---------------------------------------------------------------------------
# FrameworkSpec - as_dict / from_dict
# ---------------------------------------------------------------------------


class TestFrameworkSpecSerialization:
    def test_as_dict_keys(self):
        spec = FrameworkSpec("f", "F", "f_mod", "hint", ["X"])
        d = spec.as_dict()
        assert set(d.keys()) == {
            "framework_id",
            "name",
            "module_path",
            "detect_fn_hint",
            "entry_points",
        }

    def test_roundtrip(self):
        spec = FrameworkSpec(
            framework_id="rt",
            name="Roundtrip",
            module_path="rt_mod",
            detect_fn_hint="import rt_mod",
            entry_points=["A", "B"],
        )
        restored = FrameworkSpec.from_dict(spec.as_dict())
        assert restored.framework_id == spec.framework_id
        assert restored.name == spec.name
        assert restored.module_path == spec.module_path
        assert restored.detect_fn_hint == spec.detect_fn_hint
        assert restored.entry_points == spec.entry_points

    def test_from_dict_missing_entry_points(self):
        d = {
            "framework_id": "x",
            "name": "X",
            "module_path": "xm",
        }
        spec = FrameworkSpec.from_dict(d)
        assert spec.entry_points == []
        assert spec.detect_fn_hint == ""

    def test_as_dict_entry_points_is_copy(self):
        spec = FrameworkSpec("cp", "Copy", "cp_mod", "", ["A"])
        d = spec.as_dict()
        d["entry_points"].append("injected")
        assert "injected" not in spec.entry_points


# ---------------------------------------------------------------------------
# FrameworkDetectionResult - construction and fields
# ---------------------------------------------------------------------------


class TestFrameworkDetectionResultConstruction:
    def test_basic_fields(self):
        r = FrameworkDetectionResult(
            framework_id="test",
            detected=True,
        )
        assert r.framework_id == "test"
        assert r.detected is True
        assert r.version == ""
        assert r.confidence == 1.0

    def test_with_version(self):
        r = FrameworkDetectionResult(
            framework_id="test",
            detected=True,
            version="2.1.0",
        )
        assert r.version == "2.1.0"

    def test_with_low_confidence(self):
        r = FrameworkDetectionResult(
            framework_id="test",
            detected=False,
            confidence=0.0,
        )
        assert r.confidence == 0.0

    def test_defaults(self):
        r = FrameworkDetectionResult(framework_id="f", detected=False)
        assert r.version == ""
        assert r.confidence == 1.0


# ---------------------------------------------------------------------------
# FrameworkDetectionResult - as_dict / from_dict
# ---------------------------------------------------------------------------


class TestFrameworkDetectionResultSerialization:
    def test_as_dict_keys(self):
        r = FrameworkDetectionResult("f", True)
        d = r.as_dict()
        assert set(d.keys()) == {
            "framework_id",
            "detected",
            "version",
            "confidence",
        }

    def test_roundtrip(self):
        r = FrameworkDetectionResult(
            framework_id="rt",
            detected=True,
            version="3.0.0",
            confidence=0.95,
        )
        restored = FrameworkDetectionResult.from_dict(r.as_dict())
        assert restored.framework_id == r.framework_id
        assert restored.detected == r.detected
        assert restored.version == r.version
        assert restored.confidence == pytest.approx(r.confidence)

    def test_from_dict_defaults(self):
        d = {"framework_id": "x", "detected": False}
        r = FrameworkDetectionResult.from_dict(d)
        assert r.version == ""
        assert r.confidence == 1.0


# ---------------------------------------------------------------------------
# FrameworkRegistry - register / get / list
# ---------------------------------------------------------------------------


class TestFrameworkRegistryBasics:
    def test_empty_registry(self):
        reg = FrameworkRegistry()
        assert reg.list_frameworks() == []
        assert reg.get("nonexistent") is None

    def test_register_and_get(self):
        reg = FrameworkRegistry()
        spec = FrameworkSpec("fw", "FW", "fw_mod", "hint", ["A"])
        reg.register(spec)
        assert reg.get("fw") is spec

    def test_list_frameworks_order(self):
        reg = FrameworkRegistry()
        ids = ["alpha", "beta", "gamma"]
        for fid in ids:
            reg.register(FrameworkSpec(fid, fid.title(), fid, "", []))
        listed = reg.list_frameworks()
        assert [s.framework_id for s in listed] == ids

    def test_overwrite_existing(self):
        reg = FrameworkRegistry()
        v1 = FrameworkSpec("dup", "V1", "m1", "", [])
        v2 = FrameworkSpec("dup", "V2", "m2", "", [])
        reg.register(v1)
        reg.register(v2)
        assert reg.get("dup").name == "V2"
        assert len(reg.list_frameworks()) == 1

    def test_get_nonexistent_returns_none(self):
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("x", "X", "x", "", []))
        assert reg.get("missing") is None


# ---------------------------------------------------------------------------
# FrameworkRegistry - detect_installed
# ---------------------------------------------------------------------------


class TestFrameworkRegistryDetection:
    def test_detect_none_installed(self):
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("fake", "Fake", "fake_nonexistent_pkg_xyz", "", []))
        results = reg.detect_installed()
        assert len(results) == 1
        assert results[0].detected is False
        assert results[0].confidence == 0.0

    def test_detect_installed_module(self):
        """json is always available - use it as a stand-in."""
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("json_fw", "JSON", "json", "", ["loads"]))
        results = reg.detect_installed()
        assert len(results) == 1
        assert results[0].detected is True
        assert results[0].confidence == 1.0

    def test_detect_version_attribute(self):
        """pytest has __version__ - verify it is captured."""
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("pytest_fw", "Pytest", "pytest", "", []))
        results = reg.detect_installed()
        r = results[0]
        assert r.detected is True
        assert r.version != ""

    def test_detect_no_version_attribute(self):
        """os has no __version__ - version should be empty string."""
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("os_fw", "OS", "os", "", []))
        results = reg.detect_installed()
        assert results[0].version == ""

    def test_detect_multiple_mixed(self):
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("real", "Real", "json", "", []))
        reg.register(FrameworkSpec("fake", "Fake", "totally_fake_xyz_123", "", []))
        results = reg.detect_installed()
        by_id = {r.framework_id: r for r in results}
        assert by_id["real"].detected is True
        assert by_id["fake"].detected is False

    def test_detect_empty_registry(self):
        reg = FrameworkRegistry()
        assert reg.detect_installed() == []

    def test_detect_import_error_still_detected(self):
        """If find_spec succeeds but import fails, still mark as detected."""
        reg = FrameworkRegistry()
        reg.register(FrameworkSpec("bad", "Bad", "bad_import_mod", "", []))

        fake_spec = MagicMock()

        def fake_find_spec(name):
            if name == "bad_import_mod":
                return fake_spec
            return importlib.util.find_spec(name)

        def fake_import(name):
            if name == "bad_import_mod":
                raise ImportError("broken")
            return importlib.import_module(name)

        with patch("evalyn_sdk.trace.framework_instrumentors.importlib.util.find_spec", side_effect=fake_find_spec):
            with patch("evalyn_sdk.trace.framework_instrumentors.importlib.import_module", side_effect=fake_import):
                results = reg.detect_installed()
                assert results[0].detected is True
                assert results[0].version == ""


# ---------------------------------------------------------------------------
# BUILTIN_FRAMEWORKS
# ---------------------------------------------------------------------------


class TestBuiltinFrameworks:
    def test_count(self):
        assert len(BUILTIN_FRAMEWORKS) == 6

    def test_ids_unique(self):
        ids = [s.framework_id for s in BUILTIN_FRAMEWORKS]
        assert len(ids) == len(set(ids))

    def test_all_have_entry_points(self):
        for spec in BUILTIN_FRAMEWORKS:
            assert spec.entry_points, f"{spec.framework_id} has no entry_points"

    def test_all_have_module_path(self):
        for spec in BUILTIN_FRAMEWORKS:
            assert spec.module_path, f"{spec.framework_id} has no module_path"

    def test_crewai_spec(self):
        crewai = next(s for s in BUILTIN_FRAMEWORKS if s.framework_id == "crewai")
        assert crewai.module_path == "crewai"
        assert "Crew" in crewai.entry_points

    def test_autogen_spec(self):
        autogen = next(s for s in BUILTIN_FRAMEWORKS if s.framework_id == "autogen")
        assert autogen.module_path == "autogen"
        assert "ConversableAgent" in autogen.entry_points

    def test_dspy_spec(self):
        dspy = next(s for s in BUILTIN_FRAMEWORKS if s.framework_id == "dspy")
        assert dspy.module_path == "dspy"
        assert "Module" in dspy.entry_points

    def test_haystack_spec(self):
        haystack = next(s for s in BUILTIN_FRAMEWORKS if s.framework_id == "haystack")
        assert haystack.module_path == "haystack"

    def test_llamaindex_spec(self):
        llamaindex = next(s for s in BUILTIN_FRAMEWORKS if s.framework_id == "llamaindex")
        assert llamaindex.module_path == "llama_index"

    def test_semantic_kernel_spec(self):
        sk = next(s for s in BUILTIN_FRAMEWORKS if s.framework_id == "semantic_kernel")
        assert sk.module_path == "semantic_kernel"
        assert "Kernel" in sk.entry_points


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    def test_returns_registry(self):
        reg = create_default_registry()
        assert isinstance(reg, FrameworkRegistry)

    def test_contains_all_builtins(self):
        reg = create_default_registry()
        for spec in BUILTIN_FRAMEWORKS:
            assert reg.get(spec.framework_id) is not None

    def test_count_matches(self):
        reg = create_default_registry()
        assert len(reg.list_frameworks()) == len(BUILTIN_FRAMEWORKS)

    def test_independent_instances(self):
        r1 = create_default_registry()
        r2 = create_default_registry()
        r1.register(FrameworkSpec("extra", "Extra", "e", "", []))
        assert r2.get("extra") is None


# ---------------------------------------------------------------------------
# format_framework_status
# ---------------------------------------------------------------------------


class TestFormatFrameworkStatus:
    def test_empty_list(self):
        output = format_framework_status([])
        assert output == "No frameworks checked."

    def test_header_present(self):
        results = [FrameworkDetectionResult("f", True)]
        output = format_framework_status(results)
        assert "Framework" in output
        assert "Detected" in output
        assert "Version" in output
        assert "Confidence" in output

    def test_detected_yes(self):
        results = [FrameworkDetectionResult("myfw", True, version="1.0")]
        output = format_framework_status(results)
        assert "yes" in output
        assert "myfw" in output

    def test_detected_no(self):
        results = [FrameworkDetectionResult("gone", False, confidence=0.0)]
        output = format_framework_status(results)
        assert "no" in output

    def test_version_displayed(self):
        results = [FrameworkDetectionResult("fw", True, version="4.2.1")]
        output = format_framework_status(results)
        assert "4.2.1" in output

    def test_confidence_displayed(self):
        results = [FrameworkDetectionResult("fw", True, confidence=0.8)]
        output = format_framework_status(results)
        assert "0.8" in output

    def test_separator_line(self):
        results = [FrameworkDetectionResult("fw", True)]
        output = format_framework_status(results)
        lines = output.split("\n")
        assert len(lines) >= 3
        assert set(lines[1].strip().replace(" ", "")) == {"-"}

    def test_multiple_results(self):
        results = [
            FrameworkDetectionResult("alpha", True, version="1.0"),
            FrameworkDetectionResult("beta", False),
        ]
        output = format_framework_status(results)
        assert "alpha" in output
        assert "beta" in output
