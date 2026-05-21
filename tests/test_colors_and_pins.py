"""Tests for color-coded output and dataset pinning."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from evalyn_sdk.cli.utils.colors import (
    green, red, yellow, blue, dim,
    delta_color,
    _colors_enabled,
)
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.datasets.pin import (
    create_pin, load_pin, verify_pin, remove_pin,
    compute_dataset_content_hash, DatasetPin, PIN_FILENAME,
)


# =============================================================================
# Color tests
# =============================================================================

class TestColorsEnabled:
    def test_no_color_env_disables(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert _colors_enabled() is False

    def test_evalyn_no_color_disables(self):
        with patch.dict(os.environ, {"EVALYN_NO_COLOR": "1"}):
            assert _colors_enabled() is False

    def test_non_tty_disables(self):
        # In test environment, stdout is usually not a TTY
        # so colors should be disabled
        if not (hasattr(__import__("sys").stdout, "isatty") and __import__("sys").stdout.isatty()):
            assert _colors_enabled() is False


class TestColorFunctions:
    """Test that color functions return strings (content preserved even when colors disabled)."""

    def test_green_contains_text(self):
        assert "hello" in green("hello")

    def test_red_contains_text(self):
        assert "error" in red("error")

    def test_yellow_contains_text(self):
        assert "warning" in yellow("warning")

    def test_blue_contains_text(self):
        assert "info" in blue("info")

    def test_dim_contains_text(self):
        assert "subtle" in dim("subtle")


class TestDeltaColor:
    def test_positive(self):
        result = delta_color(0.10)
        assert "+10.0%" in result

    def test_negative(self):
        result = delta_color(-0.15)
        assert "-15.0%" in result

    def test_zero(self):
        result = delta_color(0.0)
        assert "0.0%" in result


# =============================================================================
# Dataset pinning tests
# =============================================================================

def _item(item_id: str) -> DatasetItem:
    return DatasetItem(id=item_id, input=f"input-{item_id}", output=f"output-{item_id}")


class TestComputeHash:
    def test_deterministic(self):
        items = [_item("a"), _item("b")]
        h1 = compute_dataset_content_hash(items)
        h2 = compute_dataset_content_hash(items)
        assert h1 == h2

    def test_order_independent(self):
        h1 = compute_dataset_content_hash([_item("a"), _item("b")])
        h2 = compute_dataset_content_hash([_item("b"), _item("a")])
        assert h1 == h2

    def test_different_content(self):
        h1 = compute_dataset_content_hash([_item("a")])
        h2 = compute_dataset_content_hash([_item("b")])
        assert h1 != h2

    def test_hash_length(self):
        assert len(compute_dataset_content_hash([_item("x")])) == 32


class TestCreateAndLoadPin:
    def test_create_pin(self, tmp_path):
        items = [_item("1"), _item("2")]
        pin = create_pin(items, tmp_path, description="test pin")
        assert pin.item_count == 2
        assert pin.description == "test pin"
        assert pin.dataset_hash != ""
        assert (tmp_path / PIN_FILENAME).exists()

    def test_load_pin(self, tmp_path):
        items = [_item("1"), _item("2")]
        create_pin(items, tmp_path)
        loaded = load_pin(tmp_path)
        assert loaded is not None
        assert loaded.item_count == 2

    def test_load_missing_pin(self, tmp_path):
        assert load_pin(tmp_path) is None

    def test_load_corrupt_pin(self, tmp_path):
        (tmp_path / PIN_FILENAME).write_text("not json{{{")
        assert load_pin(tmp_path) is None


class TestVerifyPin:
    def test_matches(self, tmp_path):
        items = [_item("1"), _item("2")]
        pin = create_pin(items, tmp_path)
        assert verify_pin(items, pin) is True

    def test_changed_content(self, tmp_path):
        items_v1 = [_item("1"), _item("2")]
        pin = create_pin(items_v1, tmp_path)
        items_v2 = [_item("1"), _item("3")]  # changed
        assert verify_pin(items_v2, pin) is False

    def test_added_item(self, tmp_path):
        items_v1 = [_item("1")]
        pin = create_pin(items_v1, tmp_path)
        items_v2 = [_item("1"), _item("2")]
        assert verify_pin(items_v2, pin) is False

    def test_removed_item(self, tmp_path):
        items_v1 = [_item("1"), _item("2")]
        pin = create_pin(items_v1, tmp_path)
        items_v2 = [_item("1")]
        assert verify_pin(items_v2, pin) is False


class TestRemovePin:
    def test_remove_existing(self, tmp_path):
        create_pin([_item("1")], tmp_path)
        assert (tmp_path / PIN_FILENAME).exists()
        assert remove_pin(tmp_path) is True
        assert not (tmp_path / PIN_FILENAME).exists()

    def test_remove_nonexistent(self, tmp_path):
        assert remove_pin(tmp_path) is False


class TestDatasetPinSerialization:
    def test_roundtrip(self):
        pin = DatasetPin(
            dataset_hash="abc123", item_count=50,
            pinned_at="2026-03-28T00:00:00", description="test",
        )
        d = pin.as_dict()
        restored = DatasetPin.from_dict(d)
        assert restored.dataset_hash == "abc123"
        assert restored.item_count == 50
        assert restored.description == "test"
