"""OpenAI Evals format export for compatibility with OpenAI's evaluation framework.

Pure Python, no external dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class OpenAIEvalSample:
    """A single evaluation sample in OpenAI Evals format."""

    input_messages: list[dict[str, str]]
    ideal: str = ""
    output: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": [dict(m) for m in self.input_messages],
            "ideal": self.ideal,
            "output": self.output,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenAIEvalSample:
        return cls(
            input_messages=data.get("input", []),
            ideal=data.get("ideal", ""),
            output=data.get("output", ""),
            metrics=data.get("metrics", {}),
        )


@dataclass
class OpenAIEvalSet:
    """A collection of evaluation samples."""

    eval_name: str
    samples: list[OpenAIEvalSample]
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "samples": [s.as_dict() for s in self.samples],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenAIEvalSet:
        return cls(
            eval_name=data.get("eval_name", ""),
            samples=[OpenAIEvalSample.from_dict(s) for s in data.get("samples", [])],
            created_at=data.get("created_at", ""),
        )


def convert_evalyn_to_openai(
    items: list[dict[str, Any]],
    results: list[dict[str, Any]] | None = None,
    input_field: str = "input",
    output_field: str = "output",
    expected_field: str = "expected_output",
) -> OpenAIEvalSet:
    """Convert evalyn dataset items to OpenAI Evals format."""
    results_by_id: dict[str, dict[str, float]] = {}
    if results:
        for r in results:
            item_id = r.get("item_id", "")
            metric_id = r.get("metric_id", "")
            score = r.get("score", 0.0)
            if item_id not in results_by_id:
                results_by_id[item_id] = {}
            results_by_id[item_id][metric_id] = score

    samples: list[OpenAIEvalSample] = []
    for item in items:
        input_text = str(item.get(input_field, ""))
        output_text = str(item.get(output_field, ""))
        expected = str(item.get(expected_field, ""))
        item_id = item.get("id", item.get("item_id", ""))
        metrics = results_by_id.get(item_id, {})

        messages = [{"role": "user", "content": input_text}]
        samples.append(OpenAIEvalSample(
            input_messages=messages,
            ideal=expected,
            output=output_text,
            metrics=metrics,
        ))

    return OpenAIEvalSet(
        eval_name="evalyn_export",
        samples=samples,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def export_as_jsonl(eval_set: OpenAIEvalSet) -> str:
    """Export as JSONL - one JSON object per line."""
    lines: list[str] = []
    for sample in eval_set.samples:
        lines.append(json.dumps(sample.as_dict(), separators=(",", ":")))
    return "\n".join(lines)


def export_as_json(eval_set: OpenAIEvalSet) -> str:
    """Export as pretty-printed JSON."""
    return json.dumps(eval_set.as_dict(), indent=2)


def import_from_openai_jsonl(content: str) -> OpenAIEvalSet:
    """Parse OpenAI JSONL back into an OpenAIEvalSet."""
    samples: list[OpenAIEvalSample] = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        samples.append(OpenAIEvalSample.from_dict(data))
    return OpenAIEvalSet(
        eval_name="imported",
        samples=samples,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def format_export_summary(eval_set: OpenAIEvalSet) -> str:
    """Human-readable summary of the export."""
    lines: list[str] = []
    lines.append(f"OpenAI Evals Export: {eval_set.eval_name}")
    lines.append(f"Samples: {len(eval_set.samples)}")
    lines.append(f"Created: {eval_set.created_at}")
    with_ideal = sum(1 for s in eval_set.samples if s.ideal)
    with_output = sum(1 for s in eval_set.samples if s.output)
    with_metrics = sum(1 for s in eval_set.samples if s.metrics)
    lines.append(f"With ideal: {with_ideal}")
    lines.append(f"With output: {with_output}")
    lines.append(f"With metrics: {with_metrics}")
    return "\n".join(lines)
