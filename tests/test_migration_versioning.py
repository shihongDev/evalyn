"""Tests for storage migration versioning module."""

from __future__ import annotations

from evalyn_sdk.storage.migration_versioning import (
    MigrationHistory,
    MigrationManager,
    MigrationStep,
    create_migration,
    validate_migration_order,
)


# ---------------------------------------------------------------------------
# MigrationStep
# ---------------------------------------------------------------------------


class TestMigrationStep:
    def test_as_dict(self):
        step = MigrationStep(
            version=1, name="init", up_description="create tables"
        )
        d = step.as_dict()
        assert d["version"] == 1
        assert d["name"] == "init"
        assert d["up_description"] == "create tables"

    def test_from_dict(self):
        d = {
            "version": 2,
            "name": "add_index",
            "up_description": "add index on name",
            "down_description": "drop index on name",
            "applied_at": "2026-01-01T00:00:00",
        }
        step = MigrationStep.from_dict(d)
        assert step.version == 2
        assert step.name == "add_index"
        assert step.applied_at == "2026-01-01T00:00:00"

    def test_from_dict_defaults(self):
        step = MigrationStep.from_dict({})
        assert step.version == 0
        assert step.name == ""
        assert step.up_description == ""
        assert step.down_description == ""
        assert step.applied_at == ""

    def test_roundtrip(self):
        step = MigrationStep(
            version=3,
            name="add_column",
            up_description="add col",
            down_description="drop col",
            applied_at="2026-03-01T12:00:00",
        )
        restored = MigrationStep.from_dict(step.as_dict())
        assert restored.version == step.version
        assert restored.name == step.name
        assert restored.up_description == step.up_description
        assert restored.down_description == step.down_description
        assert restored.applied_at == step.applied_at


# ---------------------------------------------------------------------------
# MigrationHistory
# ---------------------------------------------------------------------------


class TestMigrationHistory:
    def test_as_dict(self):
        history = MigrationHistory(
            steps=[MigrationStep(version=1, name="init")],
            current_version=1,
        )
        d = history.as_dict()
        assert d["current_version"] == 1
        assert len(d["steps"]) == 1

    def test_from_dict(self):
        d = {
            "steps": [{"version": 1, "name": "init"}],
            "current_version": 1,
        }
        history = MigrationHistory.from_dict(d)
        assert history.current_version == 1
        assert len(history.steps) == 1
        assert history.steps[0].name == "init"

    def test_from_dict_empty(self):
        history = MigrationHistory.from_dict({})
        assert history.current_version == 0
        assert history.steps == []

    def test_roundtrip(self):
        history = MigrationHistory(
            steps=[
                MigrationStep(version=1, name="init"),
                MigrationStep(version=2, name="add_index"),
            ],
            current_version=2,
        )
        restored = MigrationHistory.from_dict(history.as_dict())
        assert restored.current_version == history.current_version
        assert len(restored.steps) == len(history.steps)

    def test_format_text(self):
        history = MigrationHistory(
            steps=[
                MigrationStep(version=1, name="init", applied_at="2026-01-01"),
                MigrationStep(version=2, name="add_index"),
            ],
            current_version=2,
        )
        text = history.format_text()
        assert "Migration History" in text
        assert "Current version: 2" in text
        assert "Total steps: 2" in text
        assert "v1: init" in text
        assert "v2: add_index" in text
        assert "applied: 2026-01-01" in text

    def test_format_text_empty(self):
        history = MigrationHistory()
        text = history.format_text()
        assert "Current version: 0" in text
        assert "Total steps: 0" in text

    def test_is_up_to_date_true(self):
        history = MigrationHistory(current_version=5)
        assert history.is_up_to_date(5) is True
        assert history.is_up_to_date(3) is True

    def test_is_up_to_date_false(self):
        history = MigrationHistory(current_version=3)
        assert history.is_up_to_date(5) is False

    def test_pending_count(self):
        history = MigrationHistory(current_version=3)
        assert history.pending_count(5) == 2
        assert history.pending_count(3) == 0
        assert history.pending_count(1) == 0


# ---------------------------------------------------------------------------
# MigrationManager
# ---------------------------------------------------------------------------


class TestMigrationManager:
    def test_register(self):
        mgr = MigrationManager()
        mgr.register(1, "init", up_desc="create tables")
        assert len(mgr._registered) == 1
        assert mgr._registered[0].version == 1
        assert mgr._registered[0].name == "init"

    def test_register_sorts(self):
        mgr = MigrationManager()
        mgr.register(3, "third")
        mgr.register(1, "first")
        mgr.register(2, "second")
        versions = [s.version for s in mgr._registered]
        assert versions == [1, 2, 3]

    def test_apply(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        step = mgr.apply(1)
        assert step.version == 1
        assert step.applied_at != ""
        assert len(mgr._applied) == 1

    def test_apply_unregistered_raises(self):
        mgr = MigrationManager()
        try:
            mgr.apply(99)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "99" in str(e)

    def test_apply_idempotent(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        step1 = mgr.apply(1)
        step2 = mgr.apply(1)
        assert step1.applied_at == step2.applied_at
        assert len(mgr._applied) == 1

    def test_rollback(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        mgr.apply(1)
        rolled = mgr.rollback(1)
        assert rolled is not None
        assert rolled.version == 1
        assert len(mgr._applied) == 0

    def test_rollback_unapplied(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        result = mgr.rollback(1)
        assert result is None

    def test_get_current_version(self):
        mgr = MigrationManager()
        assert mgr.get_current_version() == 0
        mgr.register(1, "init")
        mgr.register(2, "add_index")
        mgr.apply(1)
        assert mgr.get_current_version() == 1
        mgr.apply(2)
        assert mgr.get_current_version() == 2

    def test_get_pending(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        mgr.register(2, "add_index")
        mgr.register(3, "add_column")
        mgr.apply(1)
        pending = mgr.get_pending()
        assert len(pending) == 2
        versions = [s.version for s in pending]
        assert 2 in versions
        assert 3 in versions

    def test_get_pending_all_applied(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        mgr.apply(1)
        assert mgr.get_pending() == []

    def test_get_history(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        mgr.register(2, "add_index")
        mgr.apply(1)
        mgr.apply(2)
        history = mgr.get_history()
        assert history.current_version == 2
        assert len(history.steps) == 2

    def test_as_dict(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        mgr.apply(1)
        d = mgr.as_dict()
        assert "registered" in d
        assert "applied" in d
        assert len(d["registered"]) == 1
        assert len(d["applied"]) == 1

    def test_from_dict(self):
        mgr = MigrationManager()
        mgr.register(1, "init")
        mgr.register(2, "add_index")
        mgr.apply(1)
        d = mgr.as_dict()
        restored = MigrationManager.from_dict(d)
        assert len(restored._registered) == 2
        assert len(restored._applied) == 1
        assert restored.get_current_version() == 1

    def test_from_dict_empty(self):
        mgr = MigrationManager.from_dict({})
        assert mgr.get_current_version() == 0
        assert mgr._registered == []
        assert mgr._applied == []

    def test_roundtrip(self):
        mgr = MigrationManager()
        mgr.register(1, "init", up_desc="create", down_desc="drop")
        mgr.register(2, "idx", up_desc="add idx", down_desc="drop idx")
        mgr.apply(1)
        mgr.apply(2)
        d = mgr.as_dict()
        restored = MigrationManager.from_dict(d)
        assert restored.get_current_version() == 2
        assert len(restored.get_pending()) == 0


# ---------------------------------------------------------------------------
# create_migration
# ---------------------------------------------------------------------------


class TestCreateMigration:
    def test_factory(self):
        step = create_migration(1, "init", up_desc="up", down_desc="down")
        assert step.version == 1
        assert step.name == "init"
        assert step.up_description == "up"
        assert step.down_description == "down"
        assert step.applied_at == ""

    def test_defaults(self):
        step = create_migration(1, "init")
        assert step.up_description == ""
        assert step.down_description == ""


# ---------------------------------------------------------------------------
# validate_migration_order
# ---------------------------------------------------------------------------


class TestValidateMigrationOrder:
    def test_valid_sequential(self):
        steps = [
            MigrationStep(version=1, name="init"),
            MigrationStep(version=2, name="add_index"),
            MigrationStep(version=3, name="add_column"),
        ]
        valid, errors = validate_migration_order(steps)
        assert valid is True
        assert errors == []

    def test_invalid_gap(self):
        steps = [
            MigrationStep(version=1, name="init"),
            MigrationStep(version=3, name="skip"),
        ]
        valid, errors = validate_migration_order(steps)
        assert valid is False
        assert len(errors) >= 1

    def test_invalid_not_starting_from_one(self):
        steps = [
            MigrationStep(version=2, name="wrong_start"),
        ]
        valid, errors = validate_migration_order(steps)
        assert valid is False

    def test_duplicate_versions(self):
        steps = [
            MigrationStep(version=1, name="first"),
            MigrationStep(version=1, name="duplicate"),
        ]
        valid, errors = validate_migration_order(steps)
        assert valid is False
        assert any("duplicate" in e.lower() or "Duplicate" in e for e in errors)

    def test_empty(self):
        valid, errors = validate_migration_order([])
        assert valid is True
        assert errors == []

    def test_single_valid(self):
        steps = [MigrationStep(version=1, name="init")]
        valid, errors = validate_migration_order(steps)
        assert valid is True

    def test_unordered_input_still_validates(self):
        steps = [
            MigrationStep(version=3, name="third"),
            MigrationStep(version=1, name="first"),
            MigrationStep(version=2, name="second"),
        ]
        valid, errors = validate_migration_order(steps)
        assert valid is True
        assert errors == []
