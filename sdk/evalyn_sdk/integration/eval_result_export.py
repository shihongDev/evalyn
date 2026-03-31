"""
Export evaluation scores to observability platforms.

Formats eval results for Phoenix, Langfuse, and LangSmith so scores
can be pushed back to trace viewers. Simulated export - no actual HTTP calls.
Does NOT hardcode API keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EvalScore:
    """A single evaluation score tied to a span."""

    span_id: str
    metric_id: str
    score: float
    label: str = ""
    explanation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "metric_id": self.metric_id,
            "score": self.score,
            "label": self.label,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalScore:
        return cls(
            span_id=data.get("span_id", ""),
            metric_id=data.get("metric_id", ""),
            score=data.get("score", 0.0),
            label=data.get("label", ""),
            explanation=data.get("explanation", ""),
        )


@dataclass
class EvalExportConfig:
    """Configuration for exporting eval scores to a platform."""

    platform: str = "phoenix"  # "phoenix" / "langfuse" / "langsmith"
    project_name: str = "evalyn"
    batch_size: int = 100

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "project_name": self.project_name,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalExportConfig:
        return cls(
            platform=data.get("platform", "phoenix"),
            project_name=data.get("project_name", "evalyn"),
            batch_size=data.get("batch_size", 100),
        )


@dataclass
class EvalExportResult:
    """Result of exporting eval scores."""

    total_scores: int = 0
    exported: int = 0
    failed: int = 0
    platform: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_scores": self.total_scores,
            "exported": self.exported,
            "failed": self.failed,
            "platform": self.platform,
        }

    def format_text(self) -> str:
        lines = [
            f"Platform: {self.platform}",
            f"Total scores: {self.total_scores}",
            f"Exported: {self.exported}",
            f"Failed: {self.failed}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Platform Formatters
# ---------------------------------------------------------------------------


def format_eval_for_phoenix(scores: List[EvalScore]) -> List[Dict[str, Any]]:
    """Format evaluation scores in Phoenix evaluation format."""
    result: List[Dict[str, Any]] = []
    for s in scores:
        entry: Dict[str, Any] = {
            "span_id": s.span_id,
            "name": s.metric_id,
            "score": s.score,
        }
        if s.label:
            entry["label"] = s.label
        if s.explanation:
            entry["explanation"] = s.explanation
        result.append(entry)
    return result


def format_eval_for_langfuse(scores: List[EvalScore]) -> List[Dict[str, Any]]:
    """Format evaluation scores in Langfuse score format."""
    result: List[Dict[str, Any]] = []
    for s in scores:
        entry: Dict[str, Any] = {
            "observationId": s.span_id,
            "name": s.metric_id,
            "value": s.score,
        }
        comment_parts = []
        if s.label:
            comment_parts.append(s.label)
        if s.explanation:
            comment_parts.append(s.explanation)
        if comment_parts:
            entry["comment"] = " - ".join(comment_parts)
        result.append(entry)
    return result


def format_eval_for_langsmith(scores: List[EvalScore]) -> List[Dict[str, Any]]:
    """Format evaluation scores in LangSmith feedback format."""
    result: List[Dict[str, Any]] = []
    for s in scores:
        entry: Dict[str, Any] = {
            "run_id": s.span_id,
            "key": s.metric_id,
            "score": s.score,
        }
        if s.label:
            entry["value"] = s.label
        if s.explanation:
            entry["comment"] = s.explanation
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def export_eval_results(
    scores: List[EvalScore], config: EvalExportConfig
) -> EvalExportResult:
    """Export evaluation scores in the specified platform format.

    This is a simulated export - no actual HTTP calls are made.
    The function validates, batches, and formats the scores.
    """
    valid, issues = validate_scores(scores)
    if not valid:
        return EvalExportResult(
            total_scores=len(scores),
            exported=0,
            failed=len(scores),
            platform=config.platform,
        )

    batches = batch_scores(scores, config.batch_size)

    exported = 0
    failed = 0
    for batch in batches:
        if config.platform == "phoenix":
            format_eval_for_phoenix(batch)
        elif config.platform == "langfuse":
            format_eval_for_langfuse(batch)
        elif config.platform == "langsmith":
            format_eval_for_langsmith(batch)
        else:
            failed += len(batch)
            continue
        exported += len(batch)

    return EvalExportResult(
        total_scores=len(scores),
        exported=exported,
        failed=failed,
        platform=config.platform,
    )


def batch_scores(
    scores: List[EvalScore], batch_size: int = 100
) -> List[List[EvalScore]]:
    """Split scores into batches of the given size."""
    if batch_size <= 0:
        batch_size = len(scores) or 1
    batches: List[List[EvalScore]] = []
    for i in range(0, len(scores), batch_size):
        batches.append(scores[i : i + batch_size])
    return batches


def validate_scores(scores: List[EvalScore]) -> Tuple[bool, List[str]]:
    """Validate that all scores have required fields.

    Returns (is_valid, list_of_issues).
    """
    issues: List[str] = []
    for idx, s in enumerate(scores):
        if not s.span_id:
            issues.append(f"score[{idx}]: missing span_id")
        if not s.metric_id:
            issues.append(f"score[{idx}]: missing metric_id")
    return (len(issues) == 0, issues)
