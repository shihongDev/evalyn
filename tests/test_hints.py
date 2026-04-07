"""Tests for the hints system (HintCollector)."""

from __future__ import annotations

import os
from unittest.mock import patch

from evalyn_sdk.cli.utils.hints import HintCollector, _is_suppressed


# ---------------------------------------------------------------------------
# _is_suppressed
# ---------------------------------------------------------------------------


class TestIsSuppressed:
    def test_not_suppressed_by_default(self):
        assert _is_suppressed() is False

    def test_quiet_suppresses(self):
        assert _is_suppressed(quiet=True) is True

    def test_json_format_suppresses(self):
        assert _is_suppressed(format="json") is True

    def test_table_format_not_suppressed(self):
        assert _is_suppressed(format="table") is False

    def test_env_var_1_suppresses(self):
        with patch.dict(os.environ, {"EVALYN_NO_HINTS": "1"}):
            assert _is_suppressed() is True

    def test_env_var_true_suppresses(self):
        with patch.dict(os.environ, {"EVALYN_NO_HINTS": "true"}):
            assert _is_suppressed() is True

    def test_env_var_TRUE_suppresses(self):
        with patch.dict(os.environ, {"EVALYN_NO_HINTS": "TRUE"}):
            assert _is_suppressed() is True

    def test_env_var_false_does_not_suppress(self):
        with patch.dict(os.environ, {"EVALYN_NO_HINTS": "false"}):
            assert _is_suppressed() is False

    def test_env_var_empty_does_not_suppress(self):
        with patch.dict(os.environ, {"EVALYN_NO_HINTS": ""}):
            assert _is_suppressed() is False


# ---------------------------------------------------------------------------
# HintCollector
# ---------------------------------------------------------------------------


class TestHintCollector:
    def test_empty_collector_prints_nothing(self, capsys):
        hc = HintCollector()
        hc.render()
        assert capsys.readouterr().out == ""

    def test_single_hint(self, capsys):
        hc = HintCollector()
        hc.add("evalyn analyze --run abc", "Analyze results")
        hc.render()
        out = capsys.readouterr().out
        assert "Next steps:" in out
        assert "evalyn analyze --run abc" in out
        assert "Analyze results" in out

    def test_multiple_hints(self, capsys):
        hc = HintCollector()
        hc.add("evalyn cluster-failures --metric-id x", "Cluster failures")
        hc.add("evalyn analyze --run abc", "Analyze results")
        hc.render()
        out = capsys.readouterr().out
        assert "Next steps:" in out
        assert "evalyn cluster-failures --metric-id x" in out
        assert "evalyn analyze --run abc" in out

    def test_max_hints_caps_output(self, capsys):
        hc = HintCollector()
        hc.add("evalyn cmd1", "First")
        hc.add("evalyn cmd2", "Second")
        hc.add("evalyn cmd3", "Third")
        hc.add("evalyn cmd4", "Fourth - should not show")
        hc.render(max_hints=3)
        out = capsys.readouterr().out
        assert "evalyn cmd1" in out
        assert "evalyn cmd2" in out
        assert "evalyn cmd3" in out
        assert "evalyn cmd4" not in out

    def test_custom_max_hints(self, capsys):
        hc = HintCollector()
        hc.add("evalyn cmd1", "First")
        hc.add("evalyn cmd2", "Second")
        hc.render(max_hints=1)
        out = capsys.readouterr().out
        assert "evalyn cmd1" in out
        assert "evalyn cmd2" not in out

    def test_quiet_suppresses(self, capsys):
        hc = HintCollector(quiet=True)
        hc.add("evalyn analyze --run abc", "Analyze results")
        hc.render()
        assert capsys.readouterr().out == ""

    def test_json_format_suppresses(self, capsys):
        hc = HintCollector(format="json")
        hc.add("evalyn analyze --run abc", "Analyze results")
        hc.render()
        assert capsys.readouterr().out == ""

    def test_env_var_suppresses(self, capsys):
        with patch.dict(os.environ, {"EVALYN_NO_HINTS": "1"}):
            hc = HintCollector()
            hc.add("evalyn analyze --run abc", "Analyze results")
            hc.render()
        assert capsys.readouterr().out == ""

    def test_two_column_alignment(self, capsys):
        hc = HintCollector()
        hc.add("evalyn short", "Short command")
        hc.add("evalyn a-much-longer-command --with-flags", "Long command")
        hc.render()
        out = capsys.readouterr().out
        lines = [l for l in out.strip().split("\n") if l.startswith("  evalyn")]
        # Both description columns should start at the same position
        assert len(lines) == 2
        desc_pos_0 = lines[0].index("Short command")
        desc_pos_1 = lines[1].index("Long command")
        assert desc_pos_0 == desc_pos_1

    def test_header_is_next_steps(self, capsys):
        hc = HintCollector()
        hc.add("evalyn foo", "Do foo")
        hc.render()
        out = capsys.readouterr().out
        first_content_line = [l for l in out.split("\n") if l.strip()][0]
        assert first_content_line.strip() == "Next steps:"

    def test_hints_indented_with_two_spaces(self, capsys):
        hc = HintCollector()
        hc.add("evalyn foo", "Do foo")
        hc.render()
        out = capsys.readouterr().out
        hint_lines = [l for l in out.split("\n") if "evalyn" in l]
        for line in hint_lines:
            assert line.startswith("  ")
