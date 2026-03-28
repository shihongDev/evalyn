"""Tests for environment validation and incremental dataset build."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from evalyn_sdk.cli.utils.env_check import (
    check_eval_environment,
    check_storage_environment,
    EnvCheckResult,
)
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.datasets_incremental import (
    incremental_build,
    load_build_state,
    save_build_state,
    IncrementalBuildResult,
)


# =============================================================================
# Environment validation tests
# =============================================================================

class TestCheckEvalEnvironment:
    def test_objective_only_passes(self):
        result = check_eval_environment(needs_subjective=False)
        assert result.passed is True

    def test_gemini_with_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSy_valid_key_here_1234"}):
            result = check_eval_environment(provider="gemini")
            assert result.passed is True
            assert "GEMINI_API_KEY" in result.found_keys

    def test_gemini_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure no keys in env
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                result = check_eval_environment(provider="gemini", config={})
                assert result.passed is False
                assert any("No API key" in e for e in result.errors)

    def test_openai_with_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-valid_key_here_12345678"}):
            result = check_eval_environment(provider="openai")
            assert result.passed is True

    def test_short_key_warning(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "short"}, clear=True):
            env = dict(os.environ)
            env.pop("GOOGLE_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                result = check_eval_environment(provider="gemini")
                assert any("too short" in w for w in result.warnings)

    def test_ollama_no_key_needed(self):
        result = check_eval_environment(provider="ollama")
        assert result.passed is True

    def test_config_key_lookup(self):
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                config = {"api_keys": {"gemini": "AIzaSy_from_config_1234567"}}
                result = check_eval_environment(provider="gemini", config=config)
                assert result.passed is True

    def test_format_text(self):
        result = EnvCheckResult(passed=True, found_keys={"KEY": "env"})
        text = result.format_text()
        assert "KEY" in text


class TestCheckStorageEnvironment:
    def test_default_storage(self):
        result = check_storage_environment()
        assert result.passed is True
        assert "storage" in result.found_keys or "EVALYN_DB" in result.found_keys

    def test_custom_db_path(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        with patch.dict(os.environ, {"EVALYN_DB": db_path}):
            result = check_storage_environment()
            assert result.passed is True
            assert "EVALYN_DB" in result.found_keys


# =============================================================================
# Incremental dataset build tests
# =============================================================================

def _item(item_id: str, input_text: str = "test") -> DatasetItem:
    return DatasetItem(id=item_id, input={"query": input_text}, output=f"answer-{item_id}")


class TestIncrementalBuild:
    def test_append_new_items(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1", "hello"), _item("2", "world")]
        new = [_item("3", "foo"), _item("4", "bar")]

        # Write existing dataset
        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, new, dataset_path)
        assert result.new_items_added == 2
        assert result.duplicates_skipped == 0
        assert result.total_items == 4

    def test_deduplicate(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1", "hello"), _item("2", "world")]
        new = [_item("3", "hello"), _item("4", "new")]  # "hello" is duplicate

        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, new, dataset_path)
        assert result.new_items_added == 1  # only "new" added
        assert result.duplicates_skipped == 1

    def test_all_duplicates(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1", "hello")]
        new = [_item("2", "hello")]  # same input

        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, new, dataset_path)
        assert result.new_items_added == 0
        assert result.duplicates_skipped == 1
        assert result.total_items == 1

    def test_empty_new_items(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        existing = [_item("1")]

        from evalyn_sdk.datasets import save_dataset
        save_dataset(existing, dataset_path)

        result = incremental_build(existing, [], dataset_path)
        assert result.new_items_added == 0
        assert result.total_items == 1

    def test_empty_existing(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        new = [_item("1", "hello"), _item("2", "world")]

        # Create empty file
        dataset_path.write_text("")

        result = incremental_build([], new, dataset_path)
        assert result.new_items_added == 2
        assert result.total_items == 2

    def test_build_state_saved(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        from evalyn_sdk.datasets import save_dataset
        save_dataset([_item("1")], dataset_path)

        incremental_build([_item("1")], [_item("2", "new")], dataset_path)
        state = load_build_state(tmp_path)
        assert state is not None

    def test_result_as_dict(self, tmp_path):
        dataset_path = tmp_path / "dataset.jsonl"
        from evalyn_sdk.datasets import save_dataset
        save_dataset([], dataset_path)

        result = incremental_build([], [_item("1")], dataset_path)
        d = result.as_dict()
        assert "new_items_added" in d
        assert "total_items" in d
        assert "last_build_timestamp" in d


class TestBuildState:
    def test_save_and_load(self, tmp_path):
        save_build_state(tmp_path, "2026-03-28T12:00:00Z")
        state = load_build_state(tmp_path)
        assert state == "2026-03-28T12:00:00Z"

    def test_load_missing(self, tmp_path):
        assert load_build_state(tmp_path) is None

    def test_load_corrupt(self, tmp_path):
        (tmp_path / "meta.json").write_text("not json{{{")
        assert load_build_state(tmp_path) is None

    def test_preserves_existing_meta(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"project": "test", "version": "v1"}))
        save_build_state(tmp_path, "2026-03-28T12:00:00Z")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["project"] == "test"
        assert meta["last_incremental_build"] == "2026-03-28T12:00:00Z"
