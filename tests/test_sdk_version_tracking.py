from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.sdk_version_tracking import (
    SDKVersionInfo,
    VersionReport,
    TRACKED_PACKAGES,
    get_package_version,
    detect_all_versions,
    check_version_mismatch,
    inject_version_info,
)


def _span(sid, **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- SDKVersionInfo ---


def test_sdk_version_info_as_dict():
    info = SDKVersionInfo(package_name="openai", version="1.2.3", installed=True)
    d = info.as_dict()
    assert d["package_name"] == "openai"
    assert d["version"] == "1.2.3"
    assert d["installed"] is True


def test_sdk_version_info_defaults():
    info = SDKVersionInfo(package_name="foo")
    assert info.version == "unknown"
    assert info.installed is False


# --- get_package_version ---


def test_get_package_version_installed():
    # pytest is always installed in test environments
    info = get_package_version("pytest")
    assert info.installed is True
    assert info.version != "unknown"
    assert info.package_name == "pytest"


def test_get_package_version_unknown():
    info = get_package_version("this-package-does-not-exist-xyz-999")
    assert info.installed is False
    assert info.version == "unknown"


# --- detect_all_versions ---


def test_detect_all_versions_returns_report():
    report = detect_all_versions()
    assert isinstance(report, VersionReport)
    assert len(report.versions) == len(TRACKED_PACKAGES)
    names = [v.package_name for v in report.versions]
    for pkg in TRACKED_PACKAGES:
        assert pkg in names


def test_detect_all_versions_no_mismatches():
    report = detect_all_versions()
    # detect_all_versions does not check spans, so mismatches should be empty
    assert report.mismatches == []


# --- check_version_mismatch ---


def test_check_version_mismatch_no_mismatch():
    spans = [
        _span("s1", **{"evalyn.provider_sdk_version": "openai==1.0.0"}),
        _span("s2", **{"evalyn.provider_sdk_version": "openai==1.0.0"}),
    ]
    result = check_version_mismatch(spans)
    assert result == []


def test_check_version_mismatch_detects_mismatch():
    spans = [
        _span("s1", **{"evalyn.provider_sdk_version": "openai==1.0.0"}),
        _span("s2", **{"evalyn.provider_sdk_version": "openai==1.1.0"}),
    ]
    result = check_version_mismatch(spans)
    assert len(result) == 1
    assert "openai" in result[0]
    assert "1.0.0" in result[0]
    assert "1.1.0" in result[0]


def test_check_version_mismatch_multiple_providers():
    spans = [
        _span("s1", **{"evalyn.provider_sdk_version": "openai==1.0.0"}),
        _span("s2", **{"evalyn.provider_sdk_version": "openai==1.1.0"}),
        _span("s3", **{"evalyn.provider_sdk_version": "anthropic==0.5.0"}),
        _span("s4", **{"evalyn.provider_sdk_version": "anthropic==0.6.0"}),
    ]
    result = check_version_mismatch(spans)
    assert len(result) == 2


def test_check_version_mismatch_empty_spans():
    result = check_version_mismatch([])
    assert result == []


def test_check_version_mismatch_no_version_attr():
    spans = [_span("s1"), _span("s2")]
    result = check_version_mismatch(spans)
    assert result == []


def test_check_version_mismatch_ignores_non_string():
    spans = [
        _span("s1", **{"evalyn.provider_sdk_version": 12345}),
    ]
    result = check_version_mismatch(spans)
    assert result == []


def test_check_version_mismatch_ignores_bad_format():
    spans = [
        _span("s1", **{"evalyn.provider_sdk_version": "no-equals-sign"}),
    ]
    result = check_version_mismatch(spans)
    assert result == []


# --- inject_version_info ---


def test_inject_version_info_adds_attribute():
    s = _span("s1")
    new_s = inject_version_info(s)
    assert "evalyn.sdk_versions" in new_s.attributes
    assert isinstance(new_s.attributes["evalyn.sdk_versions"], dict)


def test_inject_version_info_original_unchanged():
    s = _span("s1")
    original_attrs = dict(s.attributes)
    inject_version_info(s)
    assert s.attributes == original_attrs
    assert "evalyn.sdk_versions" not in s.attributes


def test_inject_version_info_preserves_existing_attrs():
    s = _span("s1", custom_key="custom_value")
    new_s = inject_version_info(s)
    assert new_s.attributes["custom_key"] == "custom_value"
    assert "evalyn.sdk_versions" in new_s.attributes


# --- VersionReport ---


def test_version_report_format_text():
    report = VersionReport(
        versions=[
            SDKVersionInfo("openai", "1.0.0", True),
            SDKVersionInfo("anthropic", "unknown", False),
        ],
        mismatches=["openai: multiple versions found (1.0.0, 1.1.0)"],
    )
    text = report.format_text()
    assert "openai: 1.0.0" in text
    assert "anthropic: not installed" in text
    assert "Mismatches:" in text
    assert "openai" in text


def test_version_report_format_text_no_mismatches():
    report = VersionReport(
        versions=[SDKVersionInfo("openai", "1.0.0", True)],
    )
    text = report.format_text()
    assert "Mismatches:" not in text


def test_version_report_as_dict():
    report = VersionReport(
        versions=[SDKVersionInfo("openai", "1.0.0", True)],
        mismatches=["test mismatch"],
    )
    d = report.as_dict()
    assert len(d["versions"]) == 1
    assert d["versions"][0]["package_name"] == "openai"
    assert d["mismatches"] == ["test mismatch"]


def test_version_report_from_dict():
    d = {
        "versions": [
            {"package_name": "openai", "version": "1.0.0", "installed": True},
            {"package_name": "anthropic", "version": "unknown", "installed": False},
        ],
        "mismatches": ["test"],
    }
    report = VersionReport.from_dict(d)
    assert len(report.versions) == 2
    assert report.versions[0].package_name == "openai"
    assert report.versions[0].installed is True
    assert report.mismatches == ["test"]


def test_version_report_roundtrip():
    original = VersionReport(
        versions=[
            SDKVersionInfo("openai", "1.0.0", True),
            SDKVersionInfo("cohere", "unknown", False),
        ],
        mismatches=["some mismatch"],
    )
    d = original.as_dict()
    restored = VersionReport.from_dict(d)
    assert len(restored.versions) == len(original.versions)
    assert restored.mismatches == original.mismatches
    for orig, rest in zip(original.versions, restored.versions):
        assert orig.package_name == rest.package_name
        assert orig.version == rest.version
        assert orig.installed == rest.installed


def test_version_report_from_dict_empty():
    report = VersionReport.from_dict({})
    assert report.versions == []
    assert report.mismatches == []
