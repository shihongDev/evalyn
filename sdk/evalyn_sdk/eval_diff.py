"""Diff two evaluation runs showing changed scores per item.

Pure Python, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ItemDiff:
    """Score difference for a single item on a single metric."""

    item_id: str
    metric_id: str
    score_a: float
    score_b: float
    delta: float
    direction: str  # "improved", "regressed", "unchanged"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "delta": self.delta,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ItemDiff:
        return cls(
            item_id=data["item_id"],
            metric_id=data["metric_id"],
            score_a=data["score_a"],
            score_b=data["score_b"],
            delta=data["delta"],
            direction=data["direction"],
        )


@dataclass
class RunDiff:
    """Diff between two evaluation runs."""

    run_a_id: str
    run_b_id: str
    item_diffs: List[ItemDiff]
    improved_count: int
    regressed_count: int
    unchanged_count: int
    total_items: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "item_diffs": [d.as_dict() for d in self.item_diffs],
            "improved_count": self.improved_count,
            "regressed_count": self.regressed_count,
            "unchanged_count": self.unchanged_count,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunDiff:
        return cls(
            run_a_id=data["run_a_id"],
            run_b_id=data["run_b_id"],
            item_diffs=[ItemDiff.from_dict(d) for d in data.get("item_diffs", [])],
            improved_count=data.get("improved_count", 0),
            regressed_count=data.get("regressed_count", 0),
            unchanged_count=data.get("unchanged_count", 0),
            total_items=data.get("total_items", 0),
        )


@dataclass
class TextDiff:
    """Word-level text diff for a single item."""

    item_id: str
    field: str
    text_a: str
    text_b: str
    added_words: List[str]
    removed_words: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "field": self.field,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "added_words": list(self.added_words),
            "removed_words": list(self.removed_words),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TextDiff:
        return cls(
            item_id=data["item_id"],
            field=data.get("field", ""),
            text_a=data.get("text_a", ""),
            text_b=data.get("text_b", ""),
            added_words=data.get("added_words", []),
            removed_words=data.get("removed_words", []),
        )


def compute_score_diff(
    scores_a: Dict[str, Dict[str, float]],
    scores_b: Dict[str, Dict[str, float]],
    threshold: float = 0.01,
    run_a_id: str = "run_a",
    run_b_id: str = "run_b",
) -> RunDiff:
    """Compute score diffs between two runs.

    scores_a/scores_b: {item_id: {metric_id: score}}
    """
    all_items = sorted(set(scores_a.keys()) | set(scores_b.keys()))
    diffs: List[ItemDiff] = []
    improved = 0
    regressed = 0
    unchanged = 0

    for item_id in all_items:
        metrics_a = scores_a.get(item_id, {})
        metrics_b = scores_b.get(item_id, {})
        all_metrics = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

        for metric_id in all_metrics:
            sa = metrics_a.get(metric_id, 0.0)
            sb = metrics_b.get(metric_id, 0.0)
            delta = sb - sa

            if delta > threshold:
                direction = "improved"
                improved += 1
            elif delta < -threshold:
                direction = "regressed"
                regressed += 1
            else:
                direction = "unchanged"
                unchanged += 1

            diffs.append(ItemDiff(
                item_id=item_id,
                metric_id=metric_id,
                score_a=sa,
                score_b=sb,
                delta=delta,
                direction=direction,
            ))

    return RunDiff(
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        item_diffs=diffs,
        improved_count=improved,
        regressed_count=regressed,
        unchanged_count=unchanged,
        total_items=len(all_items),
    )


def compute_text_diff(
    texts_a: Dict[str, str], texts_b: Dict[str, str], field_name: str = "output"
) -> List[TextDiff]:
    """Word-level diff for items present in both sets."""
    common = sorted(set(texts_a.keys()) & set(texts_b.keys()))
    diffs: List[TextDiff] = []
    for item_id in common:
        words_a = set(texts_a[item_id].lower().split())
        words_b = set(texts_b[item_id].lower().split())
        added = sorted(words_b - words_a)
        removed = sorted(words_a - words_b)
        if added or removed:
            diffs.append(TextDiff(
                item_id=item_id,
                field=field_name,
                text_a=texts_a[item_id],
                text_b=texts_b[item_id],
                added_words=added,
                removed_words=removed,
            ))
    return diffs


def filter_diffs(
    run_diff: RunDiff,
    direction: Optional[str] = None,
    metric_id: Optional[str] = None,
    min_delta: float = 0.0,
) -> List[ItemDiff]:
    """Filter item diffs by direction, metric, or minimum delta magnitude."""
    result: List[ItemDiff] = []
    for d in run_diff.item_diffs:
        if direction and d.direction != direction:
            continue
        if metric_id and d.metric_id != metric_id:
            continue
        if abs(d.delta) < min_delta:
            continue
        result.append(d)
    return result


def format_score_diff(run_diff: RunDiff) -> str:
    """Human-readable score diff table."""
    lines: List[str] = []
    lines.append(f"Diff: {run_diff.run_a_id} -> {run_diff.run_b_id}")
    lines.append(f"Items: {run_diff.total_items}")
    lines.append(
        f"Improved: {run_diff.improved_count}, "
        f"Regressed: {run_diff.regressed_count}, "
        f"Unchanged: {run_diff.unchanged_count}"
    )
    lines.append("")
    for d in run_diff.item_diffs:
        sign = "+" if d.delta > 0 else ""
        marker = "+" if d.direction == "improved" else "-" if d.direction == "regressed" else " "
        lines.append(
            f"  [{marker}] {d.item_id}/{d.metric_id}: "
            f"{d.score_a:.4f} -> {d.score_b:.4f} ({sign}{d.delta:.4f})"
        )
    return "\n".join(lines)


def format_text_diff(diffs: List[TextDiff]) -> str:
    """Human-readable text diff."""
    lines: List[str] = []
    for d in diffs:
        lines.append(f"Item: {d.item_id} ({d.field})")
        if d.removed_words:
            lines.append(f"  - removed: {', '.join(d.removed_words)}")
        if d.added_words:
            lines.append(f"  + added: {', '.join(d.added_words)}")
        lines.append("")
    return "\n".join(lines)


def summarize_diff(run_diff: RunDiff) -> Dict[str, Any]:
    """Summary stats for a run diff."""
    deltas = [d.delta for d in run_diff.item_diffs]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    most_improved = max(run_diff.item_diffs, key=lambda d: d.delta) if run_diff.item_diffs else None
    most_regressed = min(run_diff.item_diffs, key=lambda d: d.delta) if run_diff.item_diffs else None

    return {
        "improved_count": run_diff.improved_count,
        "regressed_count": run_diff.regressed_count,
        "unchanged_count": run_diff.unchanged_count,
        "mean_delta": mean_delta,
        "most_improved": most_improved.as_dict() if most_improved else None,
        "most_regressed": most_regressed.as_dict() if most_regressed else None,
    }
