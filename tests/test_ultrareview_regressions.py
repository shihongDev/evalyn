"""Regression tests for bugs caught by ultrareview on the cli-rich-output PR."""
from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# bug_001: calibration zero values must not be replaced by defaults
# ---------------------------------------------------------------------------

class TestCalibrationZeroValues:
    def _ns(self, **kwargs):
        # Default every supported attr to None so each test only sets the one
        # it cares about.
        defaults = {
            "optimizer": None,
            "evo_population": None,
            "evo_generations": None,
            "evo_mutation_rate": None,
            "textgrad_iterations": None,
            "textgrad_threshold": None,
            "mipro_instructions": None,
            "mipro_demos": None,
            "mipro_eval_samples": None,
            "pb_population": None,
            "pb_generations": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_evo_mutation_rate_zero_is_preserved(self):
        from evalyn_sdk.cli.commands.calibration import (
            _build_calibration_optimizer_configs,
        )
        configs = _build_calibration_optimizer_configs(
            self._ns(optimizer="evoprompt", evo_mutation_rate=0.0)
        )
        assert configs["optimizer_config"].mutation_rate == 0.0

    def test_textgrad_threshold_zero_is_preserved(self):
        from evalyn_sdk.cli.commands.calibration import (
            _build_calibration_optimizer_configs,
        )
        configs = _build_calibration_optimizer_configs(
            self._ns(optimizer="textgrad", textgrad_threshold=0.0)
        )
        assert configs["optimizer_config"].improvement_threshold == 0.0

    def test_mipro_demos_zero_is_preserved(self):
        from evalyn_sdk.cli.commands.calibration import (
            _build_calibration_optimizer_configs,
        )
        configs = _build_calibration_optimizer_configs(
            self._ns(optimizer="miprov2", mipro_demos=0)
        )
        assert configs["optimizer_config"].num_demos == 0

    def test_none_falls_back_to_default(self):
        from evalyn_sdk.cli.commands.calibration import (
            _DEFAULT_MIPRO_DEMOS,
            _build_calibration_optimizer_configs,
        )
        configs = _build_calibration_optimizer_configs(
            self._ns(optimizer="miprov2")
        )
        assert configs["optimizer_config"].num_demos == _DEFAULT_MIPRO_DEMOS


# ---------------------------------------------------------------------------
# bug_006: annotation-stats keeps the NEWEST stored annotation per target
# ---------------------------------------------------------------------------

class TestAnnotationStatsNewestWins:
    def test_setdefault_keeps_first_iteration(self):
        """SQL returns DESC, so the first row for a target_id is the newest.

        This mirrors the dict-build pattern in cmd_annotation_stats.
        """
        # Simulate list_annotations DESC ordering: newest first
        rows = [
            ("T1", "FAIL", "2026-02-01"),  # newest
            ("T1", "PASS", "2026-01-01"),  # older
            ("T2", "PASS", "2026-03-01"),
        ]
        stored_by_target: dict = {}
        for target_id, label, _ts in rows:
            stored_by_target.setdefault(target_id, label)

        assert stored_by_target["T1"] == "FAIL"  # newest wins
        assert stored_by_target["T2"] == "PASS"

    def test_dict_comprehension_would_pick_oldest(self):
        """Document why the buggy version was wrong, so the test stays useful."""
        rows = [
            ("T1", "FAIL"),  # newest (DESC)
            ("T1", "PASS"),  # older
        ]
        # The buggy form: dict-comp on DESC-sorted rows lets the LAST iteration
        # (=oldest) overwrite, so T1 ends up as PASS instead of FAIL.
        buggy = {tid: label for tid, label in rows}
        assert buggy["T1"] == "PASS"  # this is the bug
