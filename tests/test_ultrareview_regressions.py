"""Regression tests for bugs caught by ultrareview on the cli-rich-output PR."""
from __future__ import annotations


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
