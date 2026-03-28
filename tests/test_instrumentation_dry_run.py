"""Tests for trace.dry_run module."""

from __future__ import annotations

from unittest import mock

import pytest

from evalyn_sdk.trace.dry_run import (
    KNOWN_PATCH_TARGETS,
    DryRunReport,
    PatchTarget,
    detect_installed_providers,
    format_dry_run_table,
    get_patch_targets,
    run_dry_run,
)


# ---------------------------------------------------------------------------
# PatchTarget
# ---------------------------------------------------------------------------


class TestPatchTarget:
    def test_defaults(self):
        t = PatchTarget(module_name="mod")
        assert t.class_name == ""
        assert t.method_name == ""
        assert t.provider == ""
        assert t.strategy == "wrap"

    def test_full_path_all_parts(self):
        t = PatchTarget("openai.resources.chat", "Completions", "create")
        assert t.full_path == "openai.resources.chat.Completions.create"

    def test_full_path_module_only(self):
        t = PatchTarget(module_name="mymod")
        assert t.full_path == "mymod"

    def test_full_path_module_and_class(self):
        t = PatchTarget(module_name="mod", class_name="Cls")
        assert t.full_path == "mod.Cls"

    def test_as_dict(self):
        t = PatchTarget("mod", "Cls", "meth", "prov", "wrap")
        d = t.as_dict()
        assert d["module_name"] == "mod"
        assert d["class_name"] == "Cls"
        assert d["method_name"] == "meth"
        assert d["provider"] == "prov"
        assert d["strategy"] == "wrap"
        assert d["full_path"] == "mod.Cls.meth"

    def test_from_dict(self):
        d = {
            "module_name": "a",
            "class_name": "B",
            "method_name": "c",
            "provider": "p",
            "strategy": "hook",
        }
        t = PatchTarget.from_dict(d)
        assert t.module_name == "a"
        assert t.class_name == "B"
        assert t.method_name == "c"
        assert t.provider == "p"
        assert t.strategy == "hook"

    def test_from_dict_defaults(self):
        t = PatchTarget.from_dict({})
        assert t.module_name == ""
        assert t.strategy == "wrap"

    def test_as_dict_from_dict_roundtrip(self):
        original = PatchTarget("x.y", "Z", "do", "pr", "wrap")
        restored = PatchTarget.from_dict(original.as_dict())
        assert restored.module_name == original.module_name
        assert restored.class_name == original.class_name
        assert restored.method_name == original.method_name
        assert restored.provider == original.provider
        assert restored.strategy == original.strategy
        assert restored.full_path == original.full_path


# ---------------------------------------------------------------------------
# DryRunReport
# ---------------------------------------------------------------------------


class TestDryRunReport:
    def test_defaults(self):
        r = DryRunReport()
        assert r.targets == []
        assert r.total_patches == 0
        assert r.providers_detected == []
        assert r.sdk_versions == {}

    def test_as_dict(self):
        t = PatchTarget("mod", "Cls", "m", "prov")
        r = DryRunReport(
            targets=[t],
            total_patches=1,
            providers_detected=["prov"],
            sdk_versions={"prov": "1.0"},
        )
        d = r.as_dict()
        assert len(d["targets"]) == 1
        assert d["total_patches"] == 1
        assert d["providers_detected"] == ["prov"]
        assert d["sdk_versions"] == {"prov": "1.0"}

    def test_from_dict_roundtrip(self):
        original = DryRunReport(
            targets=[PatchTarget("a", "B", "c", "p", "wrap")],
            total_patches=1,
            providers_detected=["p"],
            sdk_versions={"p": "2.0"},
        )
        d = original.as_dict()
        restored = DryRunReport.from_dict(d)
        assert len(restored.targets) == 1
        assert restored.targets[0].module_name == "a"
        assert restored.total_patches == 1
        assert restored.providers_detected == ["p"]
        assert restored.sdk_versions == {"p": "2.0"}

    def test_from_dict_empty(self):
        r = DryRunReport.from_dict({})
        assert r.targets == []
        assert r.total_patches == 0

    def test_format_text_with_targets(self):
        t = PatchTarget("mod", "Cls", "method", "prov", "wrap")
        r = DryRunReport(
            targets=[t],
            total_patches=1,
            providers_detected=["prov"],
            sdk_versions={"prov": "3.0"},
        )
        text = r.format_text()
        assert "Instrumentation Dry Run" in text
        assert "Providers detected: 1" in text
        assert "prov" in text
        assert "3.0" in text
        assert "Total patches: 1" in text

    def test_format_text_no_targets(self):
        r = DryRunReport()
        text = r.format_text()
        assert "Total patches: 0" in text


# ---------------------------------------------------------------------------
# KNOWN_PATCH_TARGETS structure
# ---------------------------------------------------------------------------


class TestKnownPatchTargets:
    def test_has_google(self):
        assert "google-generativeai" in KNOWN_PATCH_TARGETS

    def test_has_openai(self):
        assert "openai" in KNOWN_PATCH_TARGETS

    def test_has_anthropic(self):
        assert "anthropic" in KNOWN_PATCH_TARGETS

    def test_google_has_two_targets(self):
        targets = KNOWN_PATCH_TARGETS["google-generativeai"]
        assert len(targets) == 2

    def test_openai_target_structure(self):
        targets = KNOWN_PATCH_TARGETS["openai"]
        assert len(targets) == 1
        t = targets[0]
        assert t.module_name == "openai.resources.chat"
        assert t.class_name == "Completions"
        assert t.method_name == "create"
        assert t.provider == "openai"

    def test_anthropic_target_structure(self):
        targets = KNOWN_PATCH_TARGETS["anthropic"]
        assert len(targets) == 1
        t = targets[0]
        assert t.module_name == "anthropic.resources"
        assert t.class_name == "Messages"

    def test_all_targets_have_strategy(self):
        for provider, targets in KNOWN_PATCH_TARGETS.items():
            for t in targets:
                assert t.strategy == "wrap", f"{provider}: missing strategy"


# ---------------------------------------------------------------------------
# detect_installed_providers
# ---------------------------------------------------------------------------


class TestDetectInstalledProviders:
    def test_returns_list(self):
        result = detect_installed_providers()
        assert isinstance(result, list)

    def test_with_no_providers(self):
        with mock.patch(
            "evalyn_sdk.trace.dry_run.importlib.metadata.version",
            side_effect=__import__("importlib.metadata", fromlist=["metadata"]).PackageNotFoundError,
        ):
            result = detect_installed_providers()
            assert result == []

    def test_with_one_provider(self):
        def fake_version(pkg):
            if pkg == "openai":
                return "1.0.0"
            raise __import__("importlib.metadata", fromlist=["metadata"]).PackageNotFoundError

        with mock.patch(
            "evalyn_sdk.trace.dry_run.importlib.metadata.version",
            side_effect=fake_version,
        ):
            result = detect_installed_providers()
            assert "openai" in result
            assert "anthropic" not in result


# ---------------------------------------------------------------------------
# get_patch_targets
# ---------------------------------------------------------------------------


class TestGetPatchTargets:
    def test_known_provider(self):
        targets = get_patch_targets("openai")
        assert len(targets) == 1
        assert targets[0].method_name == "create"

    def test_unknown_provider(self):
        targets = get_patch_targets("nonexistent-provider")
        assert targets == []

    def test_returns_copy(self):
        targets1 = get_patch_targets("openai")
        targets2 = get_patch_targets("openai")
        assert targets1 is not targets2


# ---------------------------------------------------------------------------
# run_dry_run
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_returns_report(self):
        report = run_dry_run()
        assert isinstance(report, DryRunReport)

    def test_total_matches_targets_len(self):
        report = run_dry_run()
        assert report.total_patches == len(report.targets)

    def test_with_mock_providers(self):
        def fake_version(pkg):
            versions = {"openai": "1.5.0", "anthropic": "0.8.0"}
            if pkg in versions:
                return versions[pkg]
            raise __import__("importlib.metadata", fromlist=["metadata"]).PackageNotFoundError

        with mock.patch(
            "evalyn_sdk.trace.dry_run.importlib.metadata.version",
            side_effect=fake_version,
        ):
            report = run_dry_run()
            assert "openai" in report.providers_detected
            assert "anthropic" in report.providers_detected
            assert report.sdk_versions["openai"] == "1.5.0"
            assert report.sdk_versions["anthropic"] == "0.8.0"
            assert report.total_patches == 2

    def test_with_no_providers(self):
        with mock.patch(
            "evalyn_sdk.trace.dry_run.importlib.metadata.version",
            side_effect=__import__("importlib.metadata", fromlist=["metadata"]).PackageNotFoundError,
        ):
            report = run_dry_run()
            assert report.providers_detected == []
            assert report.targets == []
            assert report.total_patches == 0


# ---------------------------------------------------------------------------
# format_dry_run_table
# ---------------------------------------------------------------------------


class TestFormatDryRunTable:
    def test_empty_report(self):
        r = DryRunReport()
        result = format_dry_run_table(r)
        assert "No patch targets" in result

    def test_table_has_headers(self):
        t = PatchTarget("mod", "Cls", "m", "prov", "wrap")
        r = DryRunReport(targets=[t], total_patches=1)
        result = format_dry_run_table(r)
        assert "Provider" in result
        assert "Module" in result
        assert "Class" in result
        assert "Method" in result
        assert "Strategy" in result

    def test_table_has_data(self):
        t = PatchTarget("openai.resources.chat", "Completions", "create", "openai", "wrap")
        r = DryRunReport(targets=[t], total_patches=1)
        result = format_dry_run_table(r)
        assert "openai" in result
        assert "Completions" in result
        assert "create" in result

    def test_table_has_borders(self):
        t = PatchTarget("mod", "Cls", "m", "prov", "wrap")
        r = DryRunReport(targets=[t], total_patches=1)
        result = format_dry_run_table(r)
        assert result.startswith("+")
        assert result.endswith("+")

    def test_multiple_rows(self):
        targets = [
            PatchTarget("mod1", "C1", "m1", "p1", "wrap"),
            PatchTarget("mod2", "C2", "m2", "p2", "wrap"),
        ]
        r = DryRunReport(targets=targets, total_patches=2)
        result = format_dry_run_table(r)
        assert "p1" in result
        assert "p2" in result
