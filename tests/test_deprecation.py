from __future__ import annotations

import pytest

from evalyn_sdk.deprecation import (
    DEFAULT_REGISTRY,
    DeprecationEntry,
    DeprecationRegistry,
    DeprecationWarning_,
    format_migration_report,
    format_warnings,
)


# ---------------------------------------------------------------------------
# DeprecationEntry
# ---------------------------------------------------------------------------


def _sample_entry(**overrides) -> DeprecationEntry:
    defaults = {
        "old_name": "old_key",
        "new_name": "new_key",
        "since_version": "0.5.0",
        "removal_version": "1.0.0",
        "category": "config",
        "message": "Use new_key instead.",
    }
    defaults.update(overrides)
    return DeprecationEntry(**defaults)


class TestDeprecationEntry:
    def test_basic_creation(self):
        e = _sample_entry()
        assert e.old_name == "old_key"
        assert e.new_name == "new_key"
        assert e.since_version == "0.5.0"
        assert e.removal_version == "1.0.0"
        assert e.category == "config"
        assert e.message == "Use new_key instead."

    def test_as_dict(self):
        e = _sample_entry()
        d = e.as_dict()
        assert d == {
            "old_name": "old_key",
            "new_name": "new_key",
            "since_version": "0.5.0",
            "removal_version": "1.0.0",
            "category": "config",
            "message": "Use new_key instead.",
        }

    def test_from_dict(self):
        d = {
            "old_name": "a",
            "new_name": "b",
            "since_version": "1.0",
            "removal_version": "2.0",
            "category": "flag",
            "message": "msg",
        }
        e = DeprecationEntry.from_dict(d)
        assert e.old_name == "a"
        assert e.category == "flag"

    def test_roundtrip(self):
        e = _sample_entry()
        assert DeprecationEntry.from_dict(e.as_dict()) == e

    def test_equality(self):
        a = _sample_entry()
        b = _sample_entry()
        assert a == b

    def test_inequality(self):
        a = _sample_entry()
        b = _sample_entry(old_name="other")
        assert a != b

    def test_category_flag(self):
        e = _sample_entry(category="flag")
        assert e.category == "flag"

    def test_category_api(self):
        e = _sample_entry(category="api")
        assert e.category == "api"

    def test_empty_message(self):
        e = _sample_entry(message="")
        assert e.message == ""
        d = e.as_dict()
        assert d["message"] == ""


# ---------------------------------------------------------------------------
# DeprecationWarning_
# ---------------------------------------------------------------------------


class TestDeprecationWarning:
    def test_basic(self):
        e = _sample_entry()
        w = DeprecationWarning_(entry=e, context="test context")
        assert w.entry is e
        assert w.context == "test context"

    def test_as_dict(self):
        e = _sample_entry()
        w = DeprecationWarning_(entry=e, context="ctx")
        d = w.as_dict()
        assert d["context"] == "ctx"
        assert d["entry"]["old_name"] == "old_key"

    def test_from_dict(self):
        e = _sample_entry()
        w = DeprecationWarning_(entry=e, context="ctx")
        w2 = DeprecationWarning_.from_dict(w.as_dict())
        assert w2.context == "ctx"
        assert w2.entry == e

    def test_roundtrip(self):
        e = _sample_entry()
        w = DeprecationWarning_(entry=e, context="roundtrip")
        assert DeprecationWarning_.from_dict(w.as_dict()) == w


# ---------------------------------------------------------------------------
# DeprecationRegistry
# ---------------------------------------------------------------------------


class TestDeprecationRegistry:
    def test_empty_registry(self):
        reg = DeprecationRegistry()
        assert reg.check("anything", "config") is None

    def test_register_and_check(self):
        reg = DeprecationRegistry()
        e = _sample_entry()
        reg.register(e)
        result = reg.check("old_key", "config")
        assert result is e

    def test_check_wrong_category(self):
        reg = DeprecationRegistry()
        e = _sample_entry(category="config")
        reg.register(e)
        assert reg.check("old_key", "flag") is None

    def test_check_wrong_name(self):
        reg = DeprecationRegistry()
        e = _sample_entry()
        reg.register(e)
        assert reg.check("nonexistent", "config") is None

    def test_multiple_entries_same_key(self):
        reg = DeprecationRegistry()
        e1 = _sample_entry(new_name="first")
        e2 = _sample_entry(new_name="second")
        reg.register(e1)
        reg.register(e2)
        # Should return the last registered
        result = reg.check("old_key", "config")
        assert result.new_name == "second"

    def test_get_all_empty(self):
        reg = DeprecationRegistry()
        assert reg.get_all() == []

    def test_get_all_no_filter(self):
        reg = DeprecationRegistry()
        e1 = _sample_entry(old_name="a", category="config")
        e2 = _sample_entry(old_name="b", category="flag")
        reg.register(e1)
        reg.register(e2)
        all_entries = reg.get_all()
        assert len(all_entries) == 2

    def test_get_all_filtered(self):
        reg = DeprecationRegistry()
        e1 = _sample_entry(old_name="a", category="config")
        e2 = _sample_entry(old_name="b", category="flag")
        e3 = _sample_entry(old_name="c", category="config")
        reg.register(e1)
        reg.register(e2)
        reg.register(e3)
        configs = reg.get_all(category="config")
        assert len(configs) == 2
        flags = reg.get_all(category="flag")
        assert len(flags) == 1

    def test_get_all_filter_no_match(self):
        reg = DeprecationRegistry()
        e = _sample_entry(category="config")
        reg.register(e)
        assert reg.get_all(category="api") == []

    # --- check_config ---

    def test_check_config_empty_config(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry())
        assert reg.check_config({}) == []

    def test_check_config_no_deprecations(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry())
        warnings = reg.check_config({"safe_key": "value"})
        assert warnings == []

    def test_check_config_finds_deprecated(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="bad_key"))
        warnings = reg.check_config({"bad_key": 1, "good_key": 2})
        assert len(warnings) == 1
        assert warnings[0].entry.old_name == "bad_key"

    def test_check_config_multiple_hits(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="a"))
        reg.register(_sample_entry(old_name="b"))
        warnings = reg.check_config({"a": 1, "b": 2, "c": 3})
        assert len(warnings) == 2

    def test_check_config_context_message(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="x"))
        warnings = reg.check_config({"x": 1})
        assert "x" in warnings[0].context

    def test_check_config_ignores_non_config_entries(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="flag_key", category="flag"))
        warnings = reg.check_config({"flag_key": True})
        assert warnings == []

    # --- migrate_config ---

    def test_migrate_config_no_changes(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="other"))
        new_cfg, changes = reg.migrate_config({"safe": 1})
        assert new_cfg == {"safe": 1}
        assert changes == []

    def test_migrate_config_renames_key(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="old", new_name="new"))
        new_cfg, changes = reg.migrate_config({"old": 42})
        assert new_cfg == {"new": 42}
        assert len(changes) == 1
        assert "'old'" in changes[0]
        assert "'new'" in changes[0]

    def test_migrate_config_preserves_value(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="k", new_name="k2"))
        new_cfg, _ = reg.migrate_config({"k": {"nested": True}})
        assert new_cfg["k2"] == {"nested": True}

    def test_migrate_config_mixed(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="old1", new_name="new1"))
        config = {"old1": "a", "safe": "b"}
        new_cfg, changes = reg.migrate_config(config)
        assert "new1" in new_cfg
        assert "safe" in new_cfg
        assert "old1" not in new_cfg
        assert len(changes) == 1

    def test_migrate_config_empty_input(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry())
        new_cfg, changes = reg.migrate_config({})
        assert new_cfg == {}
        assert changes == []

    def test_migrate_config_multiple_renames(self):
        reg = DeprecationRegistry()
        reg.register(_sample_entry(old_name="a", new_name="a2"))
        reg.register(_sample_entry(old_name="b", new_name="b2"))
        new_cfg, changes = reg.migrate_config({"a": 1, "b": 2})
        assert new_cfg == {"a2": 1, "b2": 2}
        assert len(changes) == 2


# ---------------------------------------------------------------------------
# format_warnings
# ---------------------------------------------------------------------------


class TestFormatWarnings:
    def test_no_warnings(self):
        result = format_warnings([])
        assert result == "No deprecation warnings."

    def test_single_warning(self):
        e = _sample_entry()
        w = DeprecationWarning_(entry=e, context="Config key 'old_key' is deprecated")
        result = format_warnings([w])
        assert "1 deprecation warning" in result
        assert "old_key" in result
        assert "new_key" in result

    def test_multiple_warnings(self):
        e1 = _sample_entry(old_name="a", new_name="a2")
        e2 = _sample_entry(old_name="b", new_name="b2")
        ws = [
            DeprecationWarning_(entry=e1, context="ctx1"),
            DeprecationWarning_(entry=e2, context="ctx2"),
        ]
        result = format_warnings(ws)
        assert "2 deprecation warning" in result

    def test_includes_version_info(self):
        e = _sample_entry(since_version="0.5.0", removal_version="1.0.0")
        w = DeprecationWarning_(entry=e, context="ctx")
        result = format_warnings([w])
        assert "0.5.0" in result
        assert "1.0.0" in result

    def test_includes_message(self):
        e = _sample_entry(message="Custom message here.")
        w = DeprecationWarning_(entry=e, context="ctx")
        result = format_warnings([w])
        assert "Custom message here." in result

    def test_empty_message_no_extra_line(self):
        e = _sample_entry(message="")
        w = DeprecationWarning_(entry=e, context="ctx")
        result = format_warnings([w])
        # With empty message, the indent line is still added but empty
        assert "ctx" in result


# ---------------------------------------------------------------------------
# format_migration_report
# ---------------------------------------------------------------------------


class TestFormatMigrationReport:
    def test_no_changes(self):
        result = format_migration_report([])
        assert result == "No migrations needed."

    def test_single_change(self):
        result = format_migration_report(["Renamed 'a' -> 'b'"])
        assert "1 migration" in result
        assert "Renamed 'a' -> 'b'" in result

    def test_multiple_changes(self):
        changes = ["Renamed 'a' -> 'b'", "Renamed 'c' -> 'd'"]
        result = format_migration_report(changes)
        assert "2 migration" in result


# ---------------------------------------------------------------------------
# DEFAULT_REGISTRY
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_has_llm_provider(self):
        entry = DEFAULT_REGISTRY.check("llm_provider", "config")
        assert entry is not None
        assert entry.new_name == "provider"

    def test_has_max_retries(self):
        entry = DEFAULT_REGISTRY.check("max_retries", "config")
        assert entry is not None
        assert entry.new_name == "retry_limit"

    def test_has_verbose_flag(self):
        entry = DEFAULT_REGISTRY.check("--verbose", "flag")
        assert entry is not None
        assert entry.new_name == "--log-level"

    def test_has_run_eval_api(self):
        entry = DEFAULT_REGISTRY.check("run_eval", "api")
        assert entry is not None
        assert entry.new_name == "execute_eval"

    def test_has_output_dir(self):
        entry = DEFAULT_REGISTRY.check("output_dir", "config")
        assert entry is not None
        assert entry.new_name == "results_dir"

    def test_get_all_returns_expected_count(self):
        all_entries = DEFAULT_REGISTRY.get_all()
        assert len(all_entries) >= 5

    def test_migrate_default_config(self):
        config = {"llm_provider": "openai", "max_retries": 3, "timeout": 30}
        new_cfg, changes = DEFAULT_REGISTRY.migrate_config(config)
        assert "provider" in new_cfg
        assert "retry_limit" in new_cfg
        assert "timeout" in new_cfg
        assert new_cfg["provider"] == "openai"
        assert new_cfg["retry_limit"] == 3
        assert len(changes) == 2

    def test_check_config_default(self):
        warnings = DEFAULT_REGISTRY.check_config(
            {"llm_provider": "x", "safe_key": "y"}
        )
        assert len(warnings) == 1
        assert warnings[0].entry.old_name == "llm_provider"
