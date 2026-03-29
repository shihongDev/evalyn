"""Tests for CLI output pagination."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.output_pagination import (
    PagerConfig,
    create_page_iterator,
    format_page,
    get_pager_command,
    get_terminal_height,
    paginate_builtin,
    paginate_text,
    should_paginate,
)


# ---------------------------------------------------------------------------
# PagerConfig
# ---------------------------------------------------------------------------


class TestPagerConfig:
    def test_defaults(self):
        cfg = PagerConfig()
        assert cfg.enabled is True
        assert cfg.page_size == 0
        assert cfg.pager_command == ""

    def test_custom_values(self):
        cfg = PagerConfig(enabled=False, page_size=50, pager_command="bat")
        assert cfg.enabled is False
        assert cfg.page_size == 50
        assert cfg.pager_command == "bat"

    def test_as_dict(self):
        cfg = PagerConfig(enabled=False, page_size=20, pager_command="less")
        d = cfg.as_dict()
        assert d == {"enabled": False, "page_size": 20, "pager_command": "less"}

    def test_from_dict(self):
        d = {"enabled": False, "page_size": 30, "pager_command": "more"}
        cfg = PagerConfig.from_dict(d)
        assert cfg.enabled is False
        assert cfg.page_size == 30
        assert cfg.pager_command == "more"

    def test_from_dict_defaults(self):
        cfg = PagerConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.page_size == 0
        assert cfg.pager_command == ""

    def test_roundtrip(self):
        cfg = PagerConfig(enabled=True, page_size=42, pager_command="bat -p")
        restored = PagerConfig.from_dict(cfg.as_dict())
        assert restored.enabled == cfg.enabled
        assert restored.page_size == cfg.page_size
        assert restored.pager_command == cfg.pager_command


# ---------------------------------------------------------------------------
# get_terminal_height
# ---------------------------------------------------------------------------


class TestGetTerminalHeight:
    def test_returns_int(self):
        result = get_terminal_height()
        assert isinstance(result, int)
        assert result > 0

    def test_fallback_when_no_terminal(self):
        with patch("os.get_terminal_size", side_effect=OSError("no tty")):
            assert get_terminal_height() == 24

    def test_fallback_on_value_error(self):
        with patch("os.get_terminal_size", side_effect=ValueError):
            assert get_terminal_height() == 24


# ---------------------------------------------------------------------------
# should_paginate
# ---------------------------------------------------------------------------


class TestShouldPaginate:
    def test_short_text_no_paginate(self):
        cfg = PagerConfig(page_size=10)
        text = "line1\nline2\nline3"
        assert should_paginate(text, cfg) is False

    def test_long_text_paginates(self):
        cfg = PagerConfig(page_size=5)
        text = "\n".join(f"line {i}" for i in range(10))
        assert should_paginate(text, cfg) is True

    def test_disabled_config(self):
        cfg = PagerConfig(enabled=False, page_size=2)
        text = "\n".join(f"line {i}" for i in range(100))
        assert should_paginate(text, cfg) is False

    def test_exact_page_size_no_paginate(self):
        cfg = PagerConfig(page_size=3)
        text = "a\nb\nc"
        assert should_paginate(text, cfg) is False

    def test_one_over_paginates(self):
        cfg = PagerConfig(page_size=3)
        text = "a\nb\nc\nd"
        assert should_paginate(text, cfg) is True

    def test_empty_text_no_paginate(self):
        cfg = PagerConfig(page_size=5)
        assert should_paginate("", cfg) is False

    def test_auto_page_size(self):
        """page_size=0 means auto-detect from terminal."""
        cfg = PagerConfig(page_size=0)
        with patch(
            "evalyn_sdk.output_pagination.get_terminal_height", return_value=10
        ):
            # 9 lines (terminal 10 - 1 for footer) should not paginate
            text = "\n".join(f"line {i}" for i in range(9))
            assert should_paginate(text, cfg) is False
            # 10 lines should paginate
            text = "\n".join(f"line {i}" for i in range(10))
            assert should_paginate(text, cfg) is True


# ---------------------------------------------------------------------------
# paginate_builtin
# ---------------------------------------------------------------------------


class TestPaginateBuiltin:
    def test_single_page(self):
        text = "a\nb\nc\n"
        pages = paginate_builtin(text, 10)
        assert len(pages) == 1
        assert pages[0] == "a\nb\nc\n"

    def test_two_pages(self):
        text = "a\nb\nc\nd\n"
        pages = paginate_builtin(text, 2)
        assert len(pages) == 2
        assert pages[0] == "a\nb\n"
        assert pages[1] == "c\nd\n"

    def test_uneven_split(self):
        text = "a\nb\nc\nd\ne\n"
        pages = paginate_builtin(text, 2)
        assert len(pages) == 3
        assert pages[2] == "e\n"

    def test_empty_text(self):
        pages = paginate_builtin("", 5)
        assert len(pages) == 1
        assert pages[0] == ""

    def test_page_size_one(self):
        text = "a\nb\nc"
        pages = paginate_builtin(text, 1)
        assert len(pages) == 3

    def test_page_size_zero_becomes_one(self):
        """Page size < 1 is clamped to 1."""
        text = "a\nb"
        pages = paginate_builtin(text, 0)
        assert len(pages) == 2

    def test_no_trailing_newline(self):
        text = "a\nb\nc"
        pages = paginate_builtin(text, 2)
        assert len(pages) == 2
        assert pages[0] == "a\nb\n"
        assert pages[1] == "c"

    def test_single_line(self):
        pages = paginate_builtin("hello", 5)
        assert len(pages) == 1
        assert pages[0] == "hello"


# ---------------------------------------------------------------------------
# format_page
# ---------------------------------------------------------------------------


class TestFormatPage:
    def test_basic_footer(self):
        result = format_page("content\n", 1, 3)
        assert result == "content\n-- Page 1/3 --"

    def test_footer_without_trailing_newline(self):
        result = format_page("content", 2, 5)
        assert result == "content\n-- Page 2/5 --"

    def test_last_page(self):
        result = format_page("end\n", 3, 3)
        assert "-- Page 3/3 --" in result

    def test_empty_page(self):
        result = format_page("", 1, 1)
        assert result == "-- Page 1/1 --"

    def test_page_numbers(self):
        result = format_page("x\n", 42, 100)
        assert "-- Page 42/100 --" in result


# ---------------------------------------------------------------------------
# get_pager_command
# ---------------------------------------------------------------------------


class TestGetPagerCommand:
    def test_config_takes_priority(self):
        cfg = PagerConfig(pager_command="bat")
        assert get_pager_command(cfg) == "bat"

    def test_env_pager_fallback(self):
        cfg = PagerConfig()
        with patch.dict(os.environ, {"PAGER": "most"}, clear=False):
            assert get_pager_command(cfg) == "most"

    def test_unix_default(self):
        cfg = PagerConfig()
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("PAGER", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("sys.platform", "linux"):
                    assert get_pager_command(cfg) == "less"

    def test_windows_default(self):
        cfg = PagerConfig()
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("PAGER", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("sys.platform", "win32"):
                    assert get_pager_command(cfg) == "more"

    def test_config_overrides_env(self):
        cfg = PagerConfig(pager_command="custom-pager")
        with patch.dict(os.environ, {"PAGER": "env-pager"}, clear=False):
            assert get_pager_command(cfg) == "custom-pager"


# ---------------------------------------------------------------------------
# paginate_text
# ---------------------------------------------------------------------------


class TestPaginateText:
    def test_short_text_unchanged(self):
        text = "hello\nworld"
        result = paginate_text(text, PagerConfig(page_size=10))
        assert result == text

    def test_long_text_first_page_with_footer(self):
        text = "\n".join(f"line {i}" for i in range(20))
        result = paginate_text(text, PagerConfig(page_size=5))
        assert "-- Page 1/" in result
        # Should only contain the first 5 lines
        lines = result.split("\n")
        # 5 content lines + footer
        assert len(lines) == 6

    def test_none_config_uses_defaults(self):
        text = "short"
        result = paginate_text(text, None)
        assert result == text

    def test_no_config_argument(self):
        text = "short"
        result = paginate_text(text)
        assert result == text

    def test_disabled_returns_unchanged(self):
        text = "\n".join(f"line {i}" for i in range(100))
        cfg = PagerConfig(enabled=False, page_size=5)
        result = paginate_text(text, cfg)
        assert result == text

    def test_page_footer_format(self):
        text = "\n".join(f"line {i}" for i in range(10))
        result = paginate_text(text, PagerConfig(page_size=3))
        assert result.endswith("--")
        assert "Page 1/" in result


# ---------------------------------------------------------------------------
# create_page_iterator
# ---------------------------------------------------------------------------


class TestCreatePageIterator:
    def test_basic(self):
        text = "a\nb\nc\nd"
        result = create_page_iterator(text, 2)
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 2

    def test_page_numbers_are_one_based(self):
        text = "a\nb\nc"
        result = create_page_iterator(text, 1)
        nums = [num for num, _ in result]
        assert nums == [1, 2, 3]

    def test_single_page(self):
        text = "hello"
        result = create_page_iterator(text, 10)
        assert len(result) == 1
        assert result[0] == (1, "hello")

    def test_empty_text(self):
        result = create_page_iterator("", 5)
        assert len(result) == 1
        assert result[0] == (1, "")

    def test_page_content(self):
        text = "a\nb\nc\nd\n"
        result = create_page_iterator(text, 2)
        assert result[0][1] == "a\nb\n"
        assert result[1][1] == "c\nd\n"

    def test_page_size_zero_becomes_one(self):
        text = "x\ny"
        result = create_page_iterator(text, 0)
        assert len(result) == 2
