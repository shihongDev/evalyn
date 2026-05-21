"""Community rubric library for importing and exporting portable rubrics.

Pure Python, no external dependencies. Supports JSON serialization and a
custom YAML-like text format for human-readable rubric exchange.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricMetadata:
    """Metadata about a portable rubric's origin and quality."""

    author: str
    version: str
    tested_on: str
    accuracy: float
    created_at: str
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "version": self.version,
            "tested_on": self.tested_on,
            "accuracy": self.accuracy,
            "created_at": self.created_at,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricMetadata:
        return cls(
            author=data.get("author", ""),
            version=data.get("version", "0.0.0"),
            tested_on=data.get("tested_on", ""),
            accuracy=data.get("accuracy", 0.0),
            created_at=data.get("created_at", ""),
            tags=data.get("tags", []),
        )


@dataclass
class PortableRubric:
    """A self-contained rubric that can be shared across projects."""

    metric_id: str
    metric_type: str
    description: str
    category: str
    scope: str
    prompt: str
    rubric: dict[str, str]
    metadata: RubricMetadata

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type,
            "description": self.description,
            "category": self.category,
            "scope": self.scope,
            "prompt": self.prompt,
            "rubric": dict(self.rubric),
            "metadata": self.metadata.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortableRubric:
        return cls(
            metric_id=data.get("metric_id", ""),
            metric_type=data.get("metric_type", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            scope=data.get("scope", ""),
            prompt=data.get("prompt", ""),
            rubric=data.get("rubric", {}),
            metadata=RubricMetadata.from_dict(data.get("metadata", {})),
        )


def export_rubric(rubric_dict: dict, metadata: RubricMetadata) -> PortableRubric:
    """Convert a registry-style rubric dict into a portable rubric."""
    return PortableRubric(
        metric_id=rubric_dict.get("metric_id", rubric_dict.get("id", "")),
        metric_type=rubric_dict.get("metric_type", rubric_dict.get("type", "")),
        description=rubric_dict.get("description", ""),
        category=rubric_dict.get("category", ""),
        scope=rubric_dict.get("scope", ""),
        prompt=rubric_dict.get("prompt", ""),
        rubric=rubric_dict.get("rubric", {}),
        metadata=metadata,
    )


def export_to_yaml_string(rubric: PortableRubric) -> str:
    """Serialize a rubric to a custom YAML-like text format.

    Uses simple key: value pairs with --- separators between sections.
    Indented blocks for nested data.
    """
    lines: list[str] = []
    lines.append("---")
    lines.append(f"metric_id: {rubric.metric_id}")
    lines.append(f"metric_type: {rubric.metric_type}")
    lines.append(f"description: {rubric.description}")
    lines.append(f"category: {rubric.category}")
    lines.append(f"scope: {rubric.scope}")
    lines.append("---")
    lines.append("prompt: |")
    for prompt_line in rubric.prompt.split("\n"):
        lines.append(f"  {prompt_line}")
    lines.append("---")
    lines.append("rubric:")
    for level, text in sorted(rubric.rubric.items()):
        lines.append(f"  {level}: {text}")
    lines.append("---")
    lines.append("metadata:")
    lines.append(f"  author: {rubric.metadata.author}")
    lines.append(f"  version: {rubric.metadata.version}")
    lines.append(f"  tested_on: {rubric.metadata.tested_on}")
    lines.append(f"  accuracy: {rubric.metadata.accuracy}")
    lines.append(f"  created_at: {rubric.metadata.created_at}")
    lines.append(f"  tags: {', '.join(rubric.metadata.tags)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _parse_section(lines: list[str]) -> dict[str, str]:
    """Parse simple key: value lines into a dict."""
    result: dict[str, str] = {}
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key.strip()] = value.strip()
    return result


def _parse_indented_block(lines: list[str]) -> dict[str, str]:
    """Parse indented key: value lines into a dict."""
    result: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and ": " in stripped:
            key, value = stripped.split(": ", 1)
            result[key] = value
    return result


def import_from_yaml_string(content: str) -> PortableRubric:
    """Parse a rubric from the custom YAML-like text format."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in content.split("\n"):
        if line.strip() == "---":
            if current:
                sections.append(current)
            current = []
        else:
            current.append(line)
    if current:
        sections.append(current)

    # Section 0: header fields
    header = _parse_section(sections[0]) if sections else {}

    # Section 1: prompt (multiline block)
    prompt_lines: list[str] = []
    if len(sections) > 1:
        for line in sections[1]:
            stripped = line.strip()
            if stripped.startswith("prompt:"):
                continue
            # Remove the 2-space indent from prompt block
            if line.startswith("  "):
                prompt_lines.append(line[2:])
            else:
                prompt_lines.append(line)
    prompt = "\n".join(prompt_lines)

    # Section 2: rubric levels
    rubric = _parse_indented_block(sections[2]) if len(sections) > 2 else {}
    # Remove "rubric" key header if present
    rubric.pop("rubric", None)

    # Section 3: metadata
    meta_raw = _parse_indented_block(sections[3]) if len(sections) > 3 else {}
    meta_raw.pop("metadata", None)

    tags_str = meta_raw.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

    accuracy_str = meta_raw.get("accuracy", "0.0")
    try:
        accuracy = float(accuracy_str)
    except ValueError:
        accuracy = 0.0

    metadata = RubricMetadata(
        author=meta_raw.get("author", ""),
        version=meta_raw.get("version", "0.0.0"),
        tested_on=meta_raw.get("tested_on", ""),
        accuracy=accuracy,
        created_at=meta_raw.get("created_at", ""),
        tags=tags,
    )

    return PortableRubric(
        metric_id=header.get("metric_id", ""),
        metric_type=header.get("metric_type", ""),
        description=header.get("description", ""),
        category=header.get("category", ""),
        scope=header.get("scope", ""),
        prompt=prompt,
        rubric=rubric,
        metadata=metadata,
    )


def export_to_json_string(rubric: PortableRubric) -> str:
    """Serialize a rubric to JSON."""
    return json.dumps(rubric.as_dict(), indent=2)


def import_from_json_string(content: str) -> PortableRubric:
    """Parse a rubric from JSON."""
    data = json.loads(content)
    return PortableRubric.from_dict(data)


def validate_rubric(rubric: PortableRubric) -> list[str]:
    """Return a list of validation errors for a rubric.

    Returns an empty list if the rubric is valid.
    """
    errors: list[str] = []
    if not rubric.metric_id:
        errors.append("metric_id is empty")
    if not rubric.prompt:
        errors.append("prompt is missing")
    if not rubric.rubric:
        errors.append("no rubric levels defined")
    if not rubric.metric_type:
        errors.append("metric_type is empty")
    if rubric.metadata.accuracy < 0.0 or rubric.metadata.accuracy > 1.0:
        errors.append("accuracy must be between 0.0 and 1.0")
    if not rubric.metadata.author:
        errors.append("metadata author is missing")
    return errors


def merge_rubrics(
    existing: list[PortableRubric],
    incoming: list[PortableRubric],
    strategy: str = "skip",
) -> tuple[list[PortableRubric], list[str]]:
    """Merge incoming rubrics into an existing list.

    Strategies:
        skip - keep existing rubric on conflict
        overwrite - replace existing with incoming on conflict
        rename - add incoming with a renamed metric_id on conflict

    Returns (merged list, list of actions taken).
    """
    existing_ids = {r.metric_id for r in existing}
    merged = list(existing)
    actions: list[str] = []

    for rubric in incoming:
        if rubric.metric_id not in existing_ids:
            merged.append(rubric)
            actions.append(f"added {rubric.metric_id}")
            existing_ids.add(rubric.metric_id)
        elif strategy == "skip":
            actions.append(f"skipped {rubric.metric_id} (already exists)")
        elif strategy == "overwrite":
            merged = [r for r in merged if r.metric_id != rubric.metric_id]
            merged.append(rubric)
            actions.append(f"overwrote {rubric.metric_id}")
        elif strategy == "rename":
            suffix = 1
            new_id = f"{rubric.metric_id}_v{suffix}"
            while new_id in existing_ids:
                suffix += 1
                new_id = f"{rubric.metric_id}_v{suffix}"
            renamed = PortableRubric(
                metric_id=new_id,
                metric_type=rubric.metric_type,
                description=rubric.description,
                category=rubric.category,
                scope=rubric.scope,
                prompt=rubric.prompt,
                rubric=dict(rubric.rubric),
                metadata=rubric.metadata,
            )
            merged.append(renamed)
            existing_ids.add(new_id)
            actions.append(f"renamed {rubric.metric_id} -> {new_id}")
        else:
            actions.append(f"unknown strategy {strategy} for {rubric.metric_id}")

    return merged, actions


def format_rubric_summary(rubric: PortableRubric) -> str:
    """Return a one-line summary of a rubric."""
    levels = len(rubric.rubric)
    acc = f"{rubric.metadata.accuracy:.0%}"
    return (
        f"[{rubric.metric_id}] {rubric.metric_type} - "
        f"{rubric.description} ({levels} levels, {acc} accuracy)"
    )


def format_rubric_detail(rubric: PortableRubric) -> str:
    """Return a full detail view of a rubric."""
    lines: list[str] = []
    lines.append(f"Metric ID:   {rubric.metric_id}")
    lines.append(f"Type:        {rubric.metric_type}")
    lines.append(f"Description: {rubric.description}")
    lines.append(f"Category:    {rubric.category}")
    lines.append(f"Scope:       {rubric.scope}")
    lines.append("")
    lines.append("Prompt:")
    for prompt_line in rubric.prompt.split("\n"):
        lines.append(f"  {prompt_line}")
    lines.append("")
    lines.append("Rubric Levels:")
    for level, text in sorted(rubric.rubric.items()):
        lines.append(f"  {level}: {text}")
    lines.append("")
    lines.append("Metadata:")
    lines.append(f"  Author:     {rubric.metadata.author}")
    lines.append(f"  Version:    {rubric.metadata.version}")
    lines.append(f"  Tested on:  {rubric.metadata.tested_on}")
    lines.append(f"  Accuracy:   {rubric.metadata.accuracy:.0%}")
    lines.append(f"  Created at: {rubric.metadata.created_at}")
    lines.append(f"  Tags:       {', '.join(rubric.metadata.tags)}")
    return "\n".join(lines)
