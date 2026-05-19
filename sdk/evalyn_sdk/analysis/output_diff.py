"""Evaluation output diff: show expected vs actual text per item.

Compares agent outputs against expected references and produces
human-readable diffs highlighting added, removed, and changed text.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ItemDiff:
    """Text diff for a single item."""

    item_id: str
    expected: str
    actual: str
    similarity: float  # 0.0-1.0 (SequenceMatcher ratio)
    diff_lines: list[str] = field(default_factory=list)

    @property
    def is_match(self) -> bool:
        return self.expected.strip() == self.actual.strip()

    @property
    def is_close(self) -> bool:
        return self.similarity >= 0.8

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "similarity": round(self.similarity, 4),
            "is_match": self.is_match,
            "expected_length": len(self.expected),
            "actual_length": len(self.actual),
            "diff_line_count": len(self.diff_lines),
        }

    def format_text(self, max_lines: int = 20) -> str:
        lines = [f"Item {self.item_id} (similarity: {self.similarity:.1%}):"]
        if self.is_match:
            lines.append("  [MATCH] Outputs are identical")
        else:
            for line in self.diff_lines[:max_lines]:
                lines.append(f"  {line}")
            if len(self.diff_lines) > max_lines:
                lines.append(f"  ... ({len(self.diff_lines) - max_lines} more lines)")
        return "\n".join(lines)


@dataclass
class OutputDiffReport:
    """Report of output diffs across all items."""

    diffs: list[ItemDiff] = field(default_factory=list)
    total_items: int = 0
    exact_matches: int = 0
    close_matches: int = 0
    significant_diffs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "exact_matches": self.exact_matches,
            "close_matches": self.close_matches,
            "significant_diffs": self.significant_diffs,
            "avg_similarity": round(
                sum(d.similarity for d in self.diffs) / len(self.diffs), 4
            ) if self.diffs else 0.0,
        }

    def format_text(self) -> str:
        lines = [
            "Output Diff Report",
            f"  Total items:      {self.total_items}",
            f"  Exact matches:    {self.exact_matches}",
            f"  Close matches:    {self.close_matches} (>80% similar)",
            f"  Significant diffs: {self.significant_diffs}",
        ]
        if self.diffs:
            avg_sim = sum(d.similarity for d in self.diffs) / len(self.diffs)
            lines.append(f"  Avg similarity:   {avg_sim:.1%}")
        return "\n".join(lines)


def compute_output_diff(
    item_id: str,
    expected: str,
    actual: str,
) -> ItemDiff:
    """Compute a diff between expected and actual output.

    Args:
        item_id: Item identifier.
        expected: Expected/reference output text.
        actual: Actual agent output text.

    Returns:
        ItemDiff with similarity score and diff lines.
    """
    expected = str(expected or "")
    actual = str(actual or "")

    # Compute similarity
    matcher = difflib.SequenceMatcher(None, expected, actual)
    similarity = matcher.ratio()

    # Generate unified diff
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        expected_lines, actual_lines,
        fromfile="expected", tofile="actual",
        lineterm="",
    ))

    return ItemDiff(
        item_id=item_id,
        expected=expected,
        actual=actual,
        similarity=similarity,
        diff_lines=diff_lines,
    )


def diff_run_outputs(
    items: list,
    run_results: dict[str, str] | None = None,
) -> OutputDiffReport:
    """Compute output diffs for all items with expected references.

    Args:
        items: List of DatasetItem objects (must have human_label.reference
            or output as expected value).
        run_results: Optional dict of item_id -> actual output. If None,
            uses item.output as actual.

    Returns:
        OutputDiffReport with per-item diffs.
    """
    diffs = []
    exact = 0
    close = 0
    significant = 0

    for item in items:
        # Get expected value
        expected = None
        if item.human_label and isinstance(item.human_label, dict):
            expected = item.human_label.get("reference")
        if expected is None:
            continue  # skip items without expected reference

        # Get actual value
        if run_results:
            actual = run_results.get(item.id, str(item.output or ""))
        else:
            actual = str(item.output or "")

        diff = compute_output_diff(item.id, str(expected), actual)
        diffs.append(diff)

        if diff.is_match:
            exact += 1
        elif diff.is_close:
            close += 1
        else:
            significant += 1

    return OutputDiffReport(
        diffs=diffs,
        total_items=len(diffs),
        exact_matches=exact,
        close_matches=close,
        significant_diffs=significant,
    )
