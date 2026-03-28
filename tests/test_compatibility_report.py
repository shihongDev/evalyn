"""Tests for trace.compatibility_report module."""

from __future__ import annotations

import copy

import pytest

from evalyn_sdk.trace.compatibility_report import (
    KNOWN_COMPATIBLE_VERSIONS,
    CompatibilityEntry,
    CompatibilityReport,
    TestedVersion,
    check_compatibility,
    generate_compatibility_report,
    get_installed_version,
    is_version_tested,
    record_tested_version,
)


# -- TestedVersion ----------------------------------------------------------


class TestTestedVersion:
    def test_defaults(self):
        tv = TestedVersion(provider="openai", version="1.50.0")
        assert tv.tested_at == ""
        assert tv.status == "compatible"

    def test_as_dict(self):
        tv = TestedVersion(provider="openai", version="1.50.0", status="incompatible")
        d = tv.as_dict()
        assert d["provider"] == "openai"
        assert d["version"] == "1.50.0"
        assert d["status"] == "incompatible"

    def test_from_dict(self):
        d = {"provider": "anthropic", "version": "0.40.0", "tested_at": "2026-01-01"}
        tv = TestedVersion.from_dict(d)
        assert tv.provider == "anthropic"
        assert tv.version == "0.40.0"
        assert tv.tested_at == "2026-01-01"
        assert tv.status == "compatible"

    def test_roundtrip(self):
        tv = TestedVersion(
            provider="openai", version="1.55.0", tested_at="2026-03-01", status="untested"
        )
        restored = TestedVersion.from_dict(tv.as_dict())
        assert restored.provider == tv.provider
        assert restored.version == tv.version
        assert restored.tested_at == tv.tested_at
        assert restored.status == tv.status


# -- CompatibilityEntry -----------------------------------------------------


class TestCompatibilityEntry:
    def test_as_dict(self):
        entry = CompatibilityEntry(
            provider="openai",
            current_version="1.50.0",
            tested_versions=["1.50.0", "1.51.0"],
            is_tested=True,
        )
        d = entry.as_dict()
        assert d["provider"] == "openai"
        assert d["is_tested"] is True
        assert len(d["tested_versions"]) == 2

    def test_warning_field(self):
        entry = CompatibilityEntry(
            provider="openai",
            current_version="9.9.9",
            warning="openai==9.9.9 has not been tested",
        )
        assert "not been tested" in entry.warning


# -- CompatibilityReport ----------------------------------------------------


class TestCompatibilityReport:
    def test_empty_report(self):
        report = CompatibilityReport()
        assert report.all_compatible is True
        assert report.entries == []
        assert report.warnings == []

    def test_as_dict(self):
        entry = CompatibilityEntry(provider="openai", current_version="1.50.0", is_tested=True)
        report = CompatibilityReport(entries=[entry], all_compatible=True)
        d = report.as_dict()
        assert len(d["entries"]) == 1
        assert d["all_compatible"] is True

    def test_from_dict(self):
        d = {
            "entries": [
                {"provider": "openai", "current_version": "1.50.0", "is_tested": True}
            ],
            "all_compatible": True,
            "warnings": [],
        }
        report = CompatibilityReport.from_dict(d)
        assert len(report.entries) == 1
        assert report.entries[0].provider == "openai"

    def test_roundtrip(self):
        entry = CompatibilityEntry(
            provider="anthropic",
            current_version="0.40.0",
            tested_versions=["0.40.0"],
            is_tested=True,
        )
        original = CompatibilityReport(
            entries=[entry], all_compatible=True, warnings=[]
        )
        restored = CompatibilityReport.from_dict(original.as_dict())
        assert len(restored.entries) == 1
        assert restored.entries[0].provider == "anthropic"
        assert restored.all_compatible is True

    def test_format_text_installed(self):
        entry = CompatibilityEntry(
            provider="openai", current_version="1.50.0", is_tested=True
        )
        report = CompatibilityReport(entries=[entry], all_compatible=True)
        text = report.format_text()
        assert "openai" in text
        assert "1.50.0" in text
        assert "tested" in text.lower()

    def test_format_text_not_installed(self):
        entry = CompatibilityEntry(provider="openai", current_version="")
        report = CompatibilityReport(entries=[entry])
        text = report.format_text()
        assert "not installed" in text

    def test_format_text_with_warnings(self):
        entry = CompatibilityEntry(
            provider="openai",
            current_version="9.9.9",
            is_tested=False,
            warning="openai==9.9.9 has not been tested",
        )
        report = CompatibilityReport(
            entries=[entry],
            all_compatible=False,
            warnings=["openai==9.9.9 has not been tested"],
        )
        text = report.format_text()
        assert "Warnings:" in text
        assert "UNTESTED" in text


# -- get_installed_version ---------------------------------------------------


class TestGetInstalledVersion:
    def test_known_package(self):
        # pytest is always installed in the test environment
        version = get_installed_version("pytest")
        assert version != ""
        assert "." in version

    def test_unknown_package(self):
        version = get_installed_version("definitely-not-a-real-package-xyz")
        assert version == ""


# -- is_version_tested -------------------------------------------------------


class TestIsVersionTested:
    def test_true(self):
        assert is_version_tested("openai", "1.50.0") is True

    def test_false_wrong_version(self):
        assert is_version_tested("openai", "0.0.1") is False

    def test_false_unknown_provider(self):
        assert is_version_tested("unknown-provider", "1.0.0") is False


# -- check_compatibility -----------------------------------------------------


class TestCheckCompatibility:
    def test_not_installed(self):
        entry = check_compatibility("definitely-not-a-real-package-xyz")
        assert entry.current_version == ""
        assert entry.is_tested is False
        assert entry.warning == ""

    def test_installed_provider(self):
        # We cannot guarantee openai/anthropic are installed, so test the
        # logic path with a provider we can mock via record_tested_version
        original = copy.deepcopy(KNOWN_COMPATIBLE_VERSIONS)
        try:
            # pytest is installed; register it as a "provider"
            pytest_ver = get_installed_version("pytest")
            KNOWN_COMPATIBLE_VERSIONS["pytest"] = [pytest_ver]
            entry = check_compatibility("pytest")
            assert entry.current_version == pytest_ver
            assert entry.is_tested is True
            assert entry.warning == ""
        finally:
            # Restore original state
            KNOWN_COMPATIBLE_VERSIONS.clear()
            KNOWN_COMPATIBLE_VERSIONS.update(original)

    def test_installed_but_untested(self):
        original = copy.deepcopy(KNOWN_COMPATIBLE_VERSIONS)
        try:
            KNOWN_COMPATIBLE_VERSIONS["pytest"] = ["0.0.0"]
            entry = check_compatibility("pytest")
            assert entry.is_tested is False
            assert "not been tested" in entry.warning
        finally:
            KNOWN_COMPATIBLE_VERSIONS.clear()
            KNOWN_COMPATIBLE_VERSIONS.update(original)


# -- generate_compatibility_report -------------------------------------------


class TestGenerateCompatibilityReport:
    def test_returns_report(self):
        report = generate_compatibility_report()
        assert isinstance(report, CompatibilityReport)
        assert len(report.entries) == len(KNOWN_COMPATIBLE_VERSIONS)

    def test_entries_match_known_providers(self):
        report = generate_compatibility_report()
        providers = {e.provider for e in report.entries}
        for p in KNOWN_COMPATIBLE_VERSIONS:
            assert p in providers


# -- record_tested_version ---------------------------------------------------


class TestRecordTestedVersion:
    def test_add_new_version(self):
        original = copy.deepcopy(KNOWN_COMPATIBLE_VERSIONS)
        try:
            record_tested_version("openai", "2.0.0")
            assert "2.0.0" in KNOWN_COMPATIBLE_VERSIONS["openai"]
        finally:
            KNOWN_COMPATIBLE_VERSIONS.clear()
            KNOWN_COMPATIBLE_VERSIONS.update(original)

    def test_add_new_provider(self):
        original = copy.deepcopy(KNOWN_COMPATIBLE_VERSIONS)
        try:
            record_tested_version("new-provider", "1.0.0")
            assert "new-provider" in KNOWN_COMPATIBLE_VERSIONS
            assert "1.0.0" in KNOWN_COMPATIBLE_VERSIONS["new-provider"]
        finally:
            KNOWN_COMPATIBLE_VERSIONS.clear()
            KNOWN_COMPATIBLE_VERSIONS.update(original)

    def test_no_duplicate(self):
        original = copy.deepcopy(KNOWN_COMPATIBLE_VERSIONS)
        try:
            record_tested_version("openai", "1.50.0")
            count = KNOWN_COMPATIBLE_VERSIONS["openai"].count("1.50.0")
            assert count == 1
        finally:
            KNOWN_COMPATIBLE_VERSIONS.clear()
            KNOWN_COMPATIBLE_VERSIONS.update(original)


# -- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    def test_no_providers_installed(self):
        """Report should work even when no known providers are installed."""
        report = generate_compatibility_report()
        # The report should have entries for every known provider
        assert len(report.entries) > 0
