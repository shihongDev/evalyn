"""
Tests for evalyn_sdk.storage.retention.

Run with: uv run pytest tests/test_storage_retention.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.retention import (
    RetentionCandidate,
    RetentionPolicy,
    RetentionResult,
    apply_retention,
    compute_retention_savings,
    identify_candidates,
    suggest_retention_policy,
    validate_policy,
)


# ---------------------------------------------------------------------------
# RetentionPolicy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_defaults(self):
        p = RetentionPolicy()
        assert p.max_age_days == 90
        assert p.max_runs is None
        assert p.max_size_mb is None
        assert p.exclude_pinned is True
        assert p.dry_run is False

    def test_as_dict_from_dict_roundtrip(self):
        p = RetentionPolicy(
            max_age_days=30,
            max_runs=500,
            max_size_mb=100.0,
            exclude_pinned=False,
            dry_run=True,
        )
        d = p.as_dict()
        restored = RetentionPolicy.from_dict(d)
        assert restored.max_age_days == 30
        assert restored.max_runs == 500
        assert restored.max_size_mb == 100.0
        assert restored.exclude_pinned is False
        assert restored.dry_run is True

    def test_from_dict_defaults(self):
        p = RetentionPolicy.from_dict({})
        assert p.max_age_days == 90
        assert p.max_runs is None


# ---------------------------------------------------------------------------
# RetentionCandidate
# ---------------------------------------------------------------------------


class TestRetentionCandidate:
    def test_as_dict(self):
        c = RetentionCandidate(
            item_id="run-001",
            item_type="run",
            age_days=120,
            size_bytes=4096,
            pinned=False,
            reason="age 120d exceeds max 90d",
        )
        d = c.as_dict()
        assert d["item_id"] == "run-001"
        assert d["item_type"] == "run"
        assert d["age_days"] == 120
        assert d["size_bytes"] == 4096
        assert d["pinned"] is False
        assert "exceeds" in d["reason"]

    def test_defaults(self):
        c = RetentionCandidate(item_id="x", item_type="trace", age_days=10)
        assert c.size_bytes == 0
        assert c.pinned is False
        assert c.reason == ""


# ---------------------------------------------------------------------------
# RetentionResult
# ---------------------------------------------------------------------------


class TestRetentionResult:
    def test_as_dict_from_dict_roundtrip(self):
        c = RetentionCandidate("run-001", "run", 120, 4096, False, "old")
        result = RetentionResult(
            candidates=[c],
            deleted_count=1,
            freed_bytes=4096,
            skipped_pinned=0,
            dry_run=False,
        )
        d = result.as_dict()
        restored = RetentionResult.from_dict(d)
        assert len(restored.candidates) == 1
        assert restored.deleted_count == 1
        assert restored.freed_bytes == 4096
        assert restored.dry_run is False

    def test_from_dict_defaults(self):
        r = RetentionResult.from_dict({})
        assert r.candidates == []
        assert r.deleted_count == 0

    def test_format_text_applied(self):
        result = RetentionResult(
            candidates=[
                RetentionCandidate("r1", "run", 100, 1000),
            ],
            deleted_count=1,
            freed_bytes=1000,
            dry_run=False,
        )
        text = result.format_text()
        assert "APPLIED" in text
        assert "Deleted: 1" in text
        assert "1000 bytes" in text

    def test_format_text_dry_run(self):
        result = RetentionResult(
            candidates=[
                RetentionCandidate("r1", "run", 100, 1000),
            ],
            deleted_count=1,
            freed_bytes=1000,
            dry_run=True,
        )
        text = result.format_text()
        assert "DRY RUN" in text

    def test_format_text_skipped_pinned(self):
        result = RetentionResult(
            skipped_pinned=3,
            dry_run=False,
        )
        text = result.format_text()
        assert "Skipped (pinned): 3" in text


# ---------------------------------------------------------------------------
# identify_candidates
# ---------------------------------------------------------------------------


class TestIdentifyCandidates:
    def test_old_items(self):
        items = [
            {"id": "r1", "type": "run", "age_days": 120, "size_bytes": 100, "pinned": False},
            {"id": "r2", "type": "trace", "age_days": 200, "size_bytes": 200, "pinned": False},
        ]
        policy = RetentionPolicy(max_age_days=90)
        candidates = identify_candidates(items, policy)
        assert len(candidates) == 2
        assert candidates[0].item_id == "r1"
        assert "exceeds" in candidates[0].reason

    def test_within_policy(self):
        items = [
            {"id": "r1", "type": "run", "age_days": 30, "size_bytes": 100, "pinned": False},
            {"id": "r2", "type": "run", "age_days": 60, "size_bytes": 100, "pinned": False},
        ]
        policy = RetentionPolicy(max_age_days=90)
        candidates = identify_candidates(items, policy)
        assert len(candidates) == 0

    def test_mixed_ages(self):
        items = [
            {"id": "r1", "type": "run", "age_days": 30, "size_bytes": 100, "pinned": False},
            {"id": "r2", "type": "run", "age_days": 120, "size_bytes": 200, "pinned": False},
        ]
        policy = RetentionPolicy(max_age_days=90)
        candidates = identify_candidates(items, policy)
        assert len(candidates) == 1
        assert candidates[0].item_id == "r2"

    def test_empty_items(self):
        policy = RetentionPolicy(max_age_days=90)
        candidates = identify_candidates([], policy)
        assert candidates == []

    def test_pinned_items_still_identified(self):
        """Pinned items are identified as candidates; filtering is done at apply time."""
        items = [
            {"id": "r1", "type": "run", "age_days": 120, "size_bytes": 100, "pinned": True},
        ]
        policy = RetentionPolicy(max_age_days=90)
        candidates = identify_candidates(items, policy)
        assert len(candidates) == 1
        assert candidates[0].pinned is True

    def test_boundary_age(self):
        """Items at exactly max_age_days should NOT be candidates."""
        items = [
            {"id": "r1", "type": "run", "age_days": 90, "size_bytes": 100, "pinned": False},
        ]
        policy = RetentionPolicy(max_age_days=90)
        candidates = identify_candidates(items, policy)
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# apply_retention
# ---------------------------------------------------------------------------


class TestApplyRetention:
    def test_deletes_old(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 1000, False, "old"),
            RetentionCandidate("r2", "trace", 150, 2000, False, "old"),
        ]
        policy = RetentionPolicy(max_age_days=90, dry_run=False)
        result = apply_retention(candidates, policy)
        assert result.deleted_count == 2
        assert result.freed_bytes == 3000
        assert result.dry_run is False

    def test_dry_run_no_delete(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 1000, False, "old"),
        ]
        policy = RetentionPolicy(max_age_days=90, dry_run=True)
        result = apply_retention(candidates, policy)
        assert result.deleted_count == 1
        assert result.freed_bytes == 1000
        assert result.dry_run is True

    def test_skip_pinned(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 1000, True, "old"),
            RetentionCandidate("r2", "run", 150, 2000, False, "old"),
        ]
        policy = RetentionPolicy(max_age_days=90, exclude_pinned=True)
        result = apply_retention(candidates, policy)
        assert result.deleted_count == 1
        assert result.freed_bytes == 2000
        assert result.skipped_pinned == 1

    def test_include_pinned(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 1000, True, "old"),
            RetentionCandidate("r2", "run", 150, 2000, False, "old"),
        ]
        policy = RetentionPolicy(max_age_days=90, exclude_pinned=False)
        result = apply_retention(candidates, policy)
        assert result.deleted_count == 2
        assert result.freed_bytes == 3000
        assert result.skipped_pinned == 0

    def test_empty_candidates(self):
        policy = RetentionPolicy(max_age_days=90)
        result = apply_retention([], policy)
        assert result.deleted_count == 0
        assert result.freed_bytes == 0
        assert result.candidates == []

    def test_all_pinned(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 1000, True, "old"),
            RetentionCandidate("r2", "run", 150, 2000, True, "old"),
        ]
        policy = RetentionPolicy(max_age_days=90, exclude_pinned=True)
        result = apply_retention(candidates, policy)
        assert result.deleted_count == 0
        assert result.freed_bytes == 0
        assert result.skipped_pinned == 2


# ---------------------------------------------------------------------------
# compute_retention_savings
# ---------------------------------------------------------------------------


class TestComputeRetentionSavings:
    def test_basic(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 1000),
            RetentionCandidate("r2", "trace", 150, 2000),
        ]
        assert compute_retention_savings(candidates) == 3000

    def test_empty(self):
        assert compute_retention_savings([]) == 0

    def test_zero_size(self):
        candidates = [
            RetentionCandidate("r1", "run", 120, 0),
        ]
        assert compute_retention_savings(candidates) == 0


# ---------------------------------------------------------------------------
# validate_policy
# ---------------------------------------------------------------------------


class TestValidatePolicy:
    def test_valid_default(self):
        policy = RetentionPolicy()
        valid, errors = validate_policy(policy)
        assert valid is True
        assert errors == []

    def test_valid_custom(self):
        policy = RetentionPolicy(max_age_days=30, max_runs=100, max_size_mb=50.0)
        valid, errors = validate_policy(policy)
        assert valid is True

    def test_invalid_negative_age(self):
        policy = RetentionPolicy(max_age_days=-1)
        valid, errors = validate_policy(policy)
        assert valid is False
        assert any("max_age_days" in e for e in errors)

    def test_invalid_zero_age(self):
        policy = RetentionPolicy(max_age_days=0)
        valid, errors = validate_policy(policy)
        assert valid is False

    def test_invalid_zero_max_runs(self):
        policy = RetentionPolicy(max_runs=0)
        valid, errors = validate_policy(policy)
        assert valid is False
        assert any("max_runs" in e for e in errors)

    def test_invalid_negative_size(self):
        policy = RetentionPolicy(max_size_mb=-10.0)
        valid, errors = validate_policy(policy)
        assert valid is False
        assert any("max_size_mb" in e for e in errors)

    def test_multiple_errors(self):
        policy = RetentionPolicy(max_age_days=-1, max_runs=0, max_size_mb=-5.0)
        valid, errors = validate_policy(policy)
        assert valid is False
        assert len(errors) == 3


# ---------------------------------------------------------------------------
# suggest_retention_policy
# ---------------------------------------------------------------------------


class TestSuggestRetentionPolicy:
    def test_small_storage(self):
        policy = suggest_retention_policy(total_size_mb=50, total_items=100)
        assert policy.max_age_days == 180
        assert policy.max_runs is None
        assert policy.dry_run is True

    def test_medium_storage(self):
        policy = suggest_retention_policy(total_size_mb=200, total_items=500)
        assert policy.max_age_days == 90

    def test_large_storage(self):
        policy = suggest_retention_policy(total_size_mb=700, total_items=5000)
        assert policy.max_age_days == 60
        assert policy.max_runs == 1000
        assert policy.max_size_mb is not None

    def test_very_large_storage(self):
        policy = suggest_retention_policy(total_size_mb=2000, total_items=20000)
        assert policy.max_age_days == 30
        assert policy.max_runs == 5000
        assert policy.max_size_mb is not None

    def test_large_item_count(self):
        policy = suggest_retention_policy(total_size_mb=50, total_items=15000)
        assert policy.max_runs == 5000
