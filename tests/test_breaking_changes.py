from __future__ import annotations

import hashlib
import json

import pytest

from evalyn_sdk.breaking_changes import (
    BreakingChange,
    CompatibilityReport,
    MetricFingerprint,
    RunManifest,
    compare_manifests,
    compute_fingerprint,
    create_manifest,
    format_compatibility_report,
    suggest_migration,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _params_hash(params: dict) -> str:
    return _sha(json.dumps(params, sort_keys=True, separators=(",", ":")))


def _fp(metric_id: str = "m1", vh: str = "vh", ph: str = "ph") -> MetricFingerprint:
    return MetricFingerprint(metric_id=metric_id, version_hash=vh, parameters_hash=ph)


def _manifest(
    run_id: str = "r1",
    version: str = "0.1.0",
    fingerprints: list | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        evalyn_version=version,
        fingerprints=fingerprints or [],
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# MetricFingerprint
# ---------------------------------------------------------------------------


class TestMetricFingerprint:
    def test_creation(self):
        fp = _fp()
        assert fp.metric_id == "m1"
        assert fp.version_hash == "vh"
        assert fp.parameters_hash == "ph"

    def test_as_dict(self):
        fp = _fp()
        d = fp.as_dict()
        assert d == {"metric_id": "m1", "version_hash": "vh", "parameters_hash": "ph"}

    def test_from_dict(self):
        d = {"metric_id": "x", "version_hash": "a", "parameters_hash": "b"}
        fp = MetricFingerprint.from_dict(d)
        assert fp.metric_id == "x"

    def test_roundtrip(self):
        fp = _fp(metric_id="test", vh="abc", ph="def")
        assert MetricFingerprint.from_dict(fp.as_dict()) == fp

    def test_equality(self):
        assert _fp() == _fp()

    def test_inequality(self):
        assert _fp(metric_id="a") != _fp(metric_id="b")


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------


class TestRunManifest:
    def test_creation(self):
        m = _manifest()
        assert m.run_id == "r1"
        assert m.fingerprints == []

    def test_as_dict(self):
        fp = _fp()
        m = _manifest(fingerprints=[fp])
        d = m.as_dict()
        assert d["run_id"] == "r1"
        assert len(d["fingerprints"]) == 1
        assert d["fingerprints"][0]["metric_id"] == "m1"

    def test_from_dict(self):
        d = {
            "run_id": "r2",
            "evalyn_version": "0.2.0",
            "fingerprints": [
                {"metric_id": "x", "version_hash": "a", "parameters_hash": "b"}
            ],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        m = RunManifest.from_dict(d)
        assert m.run_id == "r2"
        assert len(m.fingerprints) == 1

    def test_roundtrip(self):
        fp = _fp()
        m = _manifest(fingerprints=[fp])
        assert RunManifest.from_dict(m.as_dict()) == m

    def test_empty_fingerprints_from_dict(self):
        d = {
            "run_id": "r",
            "evalyn_version": "0.1.0",
            "created_at": "t",
        }
        m = RunManifest.from_dict(d)
        assert m.fingerprints == []


# ---------------------------------------------------------------------------
# BreakingChange
# ---------------------------------------------------------------------------


class TestBreakingChange:
    def test_creation(self):
        bc = BreakingChange(
            metric_id="m", old_hash="a", new_hash="b",
            change_type="modified", migration_hint="hint",
        )
        assert bc.change_type == "modified"

    def test_as_dict(self):
        bc = BreakingChange(
            metric_id="m", old_hash="a", new_hash="b",
            change_type="removed", migration_hint="hint",
        )
        d = bc.as_dict()
        assert d["change_type"] == "removed"

    def test_from_dict(self):
        d = {
            "metric_id": "m",
            "old_hash": "a",
            "new_hash": "b",
            "change_type": "parameters_changed",
            "migration_hint": "h",
        }
        bc = BreakingChange.from_dict(d)
        assert bc.change_type == "parameters_changed"

    def test_roundtrip(self):
        bc = BreakingChange(
            metric_id="m", old_hash="a", new_hash="b",
            change_type="modified", migration_hint="hint",
        )
        assert BreakingChange.from_dict(bc.as_dict()) == bc


# ---------------------------------------------------------------------------
# CompatibilityReport
# ---------------------------------------------------------------------------


class TestCompatibilityReport:
    def test_compatible(self):
        r = CompatibilityReport(
            compatible=True, changes=[], old_version="0.1", new_version="0.2",
        )
        assert r.compatible is True
        assert r.changes == []

    def test_incompatible(self):
        bc = BreakingChange(
            metric_id="m", old_hash="a", new_hash="b",
            change_type="modified", migration_hint="h",
        )
        r = CompatibilityReport(
            compatible=False, changes=[bc], old_version="0.1", new_version="0.2",
        )
        assert r.compatible is False
        assert len(r.changes) == 1

    def test_as_dict(self):
        r = CompatibilityReport(
            compatible=True, changes=[], old_version="0.1", new_version="0.2",
        )
        d = r.as_dict()
        assert d["compatible"] is True
        assert d["changes"] == []

    def test_from_dict(self):
        d = {
            "compatible": False,
            "changes": [{
                "metric_id": "m",
                "old_hash": "a",
                "new_hash": "b",
                "change_type": "removed",
                "migration_hint": "h",
            }],
            "old_version": "1.0",
            "new_version": "2.0",
        }
        r = CompatibilityReport.from_dict(d)
        assert r.compatible is False
        assert len(r.changes) == 1

    def test_roundtrip(self):
        bc = BreakingChange(
            metric_id="m", old_hash="a", new_hash="b",
            change_type="modified", migration_hint="h",
        )
        r = CompatibilityReport(
            compatible=False, changes=[bc], old_version="0.1", new_version="0.2",
        )
        assert CompatibilityReport.from_dict(r.as_dict()) == r

    def test_empty_changes_from_dict(self):
        d = {
            "compatible": True,
            "old_version": "0.1",
            "new_version": "0.2",
        }
        r = CompatibilityReport.from_dict(d)
        assert r.changes == []


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    def test_returns_fingerprint(self):
        fp = compute_fingerprint("m1", "def score(): pass", {"threshold": 0.5})
        assert fp.metric_id == "m1"
        assert isinstance(fp.version_hash, str)
        assert isinstance(fp.parameters_hash, str)
        assert len(fp.version_hash) == 64  # SHA-256 hex

    def test_deterministic(self):
        a = compute_fingerprint("m", "code", {"k": 1})
        b = compute_fingerprint("m", "code", {"k": 1})
        assert a == b

    def test_different_code_different_hash(self):
        a = compute_fingerprint("m", "code_v1", {})
        b = compute_fingerprint("m", "code_v2", {})
        assert a.version_hash != b.version_hash

    def test_different_params_different_hash(self):
        a = compute_fingerprint("m", "code", {"k": 1})
        b = compute_fingerprint("m", "code", {"k": 2})
        assert a.parameters_hash != b.parameters_hash

    def test_same_code_same_params_same_hash(self):
        a = compute_fingerprint("m", "code", {"a": 1, "b": 2})
        b = compute_fingerprint("m", "code", {"b": 2, "a": 1})
        # sort_keys ensures order independence
        assert a.parameters_hash == b.parameters_hash

    def test_version_hash_matches_sha256(self):
        code = "def score(): return 1"
        fp = compute_fingerprint("m", code, {})
        expected = _sha(code)
        assert fp.version_hash == expected

    def test_params_hash_matches_sha256(self):
        params = {"threshold": 0.5}
        fp = compute_fingerprint("m", "code", params)
        expected = _params_hash(params)
        assert fp.parameters_hash == expected

    def test_empty_code(self):
        fp = compute_fingerprint("m", "", {})
        assert fp.version_hash == _sha("")

    def test_empty_params(self):
        fp = compute_fingerprint("m", "code", {})
        assert fp.parameters_hash == _params_hash({})

    def test_nested_params(self):
        params = {"outer": {"inner": [1, 2, 3]}}
        fp = compute_fingerprint("m", "code", params)
        assert fp.parameters_hash == _params_hash(params)


# ---------------------------------------------------------------------------
# create_manifest
# ---------------------------------------------------------------------------


class TestCreateManifest:
    def test_basic(self):
        fps = [_fp(metric_id="m1"), _fp(metric_id="m2")]
        m = create_manifest("run-1", "0.5.0", fps)
        assert m.run_id == "run-1"
        assert m.evalyn_version == "0.5.0"
        assert len(m.fingerprints) == 2
        assert m.created_at  # non-empty timestamp

    def test_empty_fingerprints(self):
        m = create_manifest("r", "0.1.0", [])
        assert m.fingerprints == []

    def test_created_at_is_iso(self):
        m = create_manifest("r", "0.1.0", [])
        # Basic check that it looks like an ISO timestamp
        assert "T" in m.created_at


# ---------------------------------------------------------------------------
# compare_manifests
# ---------------------------------------------------------------------------


class TestCompareManifests:
    def test_identical_manifests(self):
        fps = [_fp(metric_id="m1", vh="h1", ph="p1")]
        old = _manifest(fingerprints=fps, version="0.1.0")
        new = _manifest(fingerprints=fps, version="0.2.0")
        report = compare_manifests(old, new)
        assert report.compatible is True
        assert report.changes == []

    def test_removed_metric(self):
        old = _manifest(fingerprints=[_fp(metric_id="m1")], version="0.1.0")
        new = _manifest(fingerprints=[], version="0.2.0")
        report = compare_manifests(old, new)
        assert report.compatible is False
        assert len(report.changes) == 1
        assert report.changes[0].change_type == "removed"
        assert report.changes[0].metric_id == "m1"

    def test_modified_metric(self):
        old = _manifest(
            fingerprints=[_fp(metric_id="m1", vh="old_code", ph="p")],
            version="0.1.0",
        )
        new = _manifest(
            fingerprints=[_fp(metric_id="m1", vh="new_code", ph="p")],
            version="0.2.0",
        )
        report = compare_manifests(old, new)
        assert report.compatible is False
        assert report.changes[0].change_type == "modified"

    def test_parameters_changed(self):
        old = _manifest(
            fingerprints=[_fp(metric_id="m1", vh="code", ph="old_p")],
            version="0.1.0",
        )
        new = _manifest(
            fingerprints=[_fp(metric_id="m1", vh="code", ph="new_p")],
            version="0.2.0",
        )
        report = compare_manifests(old, new)
        assert report.compatible is False
        assert report.changes[0].change_type == "parameters_changed"

    def test_new_metric_not_breaking(self):
        old = _manifest(fingerprints=[], version="0.1.0")
        new = _manifest(fingerprints=[_fp(metric_id="m1")], version="0.2.0")
        report = compare_manifests(old, new)
        assert report.compatible is True

    def test_multiple_changes(self):
        old = _manifest(
            fingerprints=[
                _fp(metric_id="m1", vh="h1", ph="p1"),
                _fp(metric_id="m2", vh="h2", ph="p2"),
                _fp(metric_id="m3", vh="h3", ph="p3"),
            ],
            version="0.1.0",
        )
        new = _manifest(
            fingerprints=[
                _fp(metric_id="m1", vh="h1_changed", ph="p1"),  # modified
                # m2 removed
                _fp(metric_id="m3", vh="h3", ph="p3_changed"),  # params changed
            ],
            version="0.2.0",
        )
        report = compare_manifests(old, new)
        assert report.compatible is False
        assert len(report.changes) == 3
        types = {c.change_type for c in report.changes}
        assert types == {"removed", "modified", "parameters_changed"}

    def test_versions_in_report(self):
        old = _manifest(version="0.1.0")
        new = _manifest(version="0.2.0")
        report = compare_manifests(old, new)
        assert report.old_version == "0.1.0"
        assert report.new_version == "0.2.0"

    def test_empty_manifests(self):
        old = _manifest()
        new = _manifest()
        report = compare_manifests(old, new)
        assert report.compatible is True

    def test_removed_metric_has_old_hash(self):
        old = _manifest(fingerprints=[_fp(metric_id="m1", vh="abc")])
        new = _manifest(fingerprints=[])
        report = compare_manifests(old, new)
        assert report.changes[0].old_hash == "abc"
        assert report.changes[0].new_hash == ""

    def test_modified_has_both_hashes(self):
        old = _manifest(fingerprints=[_fp(metric_id="m1", vh="old", ph="p")])
        new = _manifest(fingerprints=[_fp(metric_id="m1", vh="new", ph="p")])
        report = compare_manifests(old, new)
        c = report.changes[0]
        assert c.old_hash == "old"
        assert c.new_hash == "new"

    def test_implementation_change_trumps_params(self):
        # If both version_hash and parameters_hash change, report "modified"
        old = _manifest(fingerprints=[_fp(metric_id="m1", vh="old_v", ph="old_p")])
        new = _manifest(fingerprints=[_fp(metric_id="m1", vh="new_v", ph="new_p")])
        report = compare_manifests(old, new)
        assert len(report.changes) == 1
        assert report.changes[0].change_type == "modified"


# ---------------------------------------------------------------------------
# format_compatibility_report
# ---------------------------------------------------------------------------


class TestFormatCompatibilityReport:
    def test_compatible(self):
        r = CompatibilityReport(
            compatible=True, changes=[], old_version="0.1", new_version="0.2",
        )
        text = format_compatibility_report(r)
        assert "COMPATIBLE" in text
        assert "0.1" in text
        assert "0.2" in text

    def test_incompatible(self):
        bc = BreakingChange(
            metric_id="m1", old_hash="a", new_hash="b",
            change_type="modified", migration_hint="Re-run evaluation.",
        )
        r = CompatibilityReport(
            compatible=False, changes=[bc], old_version="0.1", new_version="0.2",
        )
        text = format_compatibility_report(r)
        assert "INCOMPATIBLE" in text
        assert "m1" in text
        assert "modified" in text
        assert "Re-run evaluation." in text

    def test_multiple_changes(self):
        changes = [
            BreakingChange("m1", "a", "b", "removed", "hint1"),
            BreakingChange("m2", "c", "d", "modified", "hint2"),
        ]
        r = CompatibilityReport(
            compatible=False, changes=changes, old_version="1.0", new_version="2.0",
        )
        text = format_compatibility_report(r)
        assert "2 breaking change" in text

    def test_shows_change_count(self):
        changes = [
            BreakingChange("m1", "", "", "removed", "h"),
        ]
        r = CompatibilityReport(False, changes, "0.1", "0.2")
        text = format_compatibility_report(r)
        assert "1 breaking change" in text


# ---------------------------------------------------------------------------
# suggest_migration
# ---------------------------------------------------------------------------


class TestSuggestMigration:
    def test_returns_hint(self):
        bc = BreakingChange("m", "a", "b", "modified", "Do this.")
        assert suggest_migration(bc) == "Do this."

    def test_empty_hint(self):
        bc = BreakingChange("m", "a", "b", "removed", "")
        assert suggest_migration(bc) == ""
