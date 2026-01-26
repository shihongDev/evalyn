"""Shared utilities for prompt optimizers (APE, OPRO, GEPA).

Contains common functions that were duplicated across optimizer implementations:
- Dataset building from annotations
- Response parsing (candidates and judge verdicts)
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from ..models import Annotation, DatasetItem, MetricResult


def build_dataset_from_annotations(
    metric_results: List[MetricResult],
    annotations: List[Annotation],
    dataset_items: Optional[List[DatasetItem]] = None,
    train_split: float = 0.7,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert calibration data to train/val sets.

    Each example contains: input, output, expected (human label), call_id.

    Args:
        metric_results: Evaluation results from the judge
        annotations: Human annotations
        dataset_items: Optional dataset items for context
        train_split: Ratio for train/val split (default: 0.7)

    Returns:
        Tuple of (trainset, valset)
    """
    ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
    items_by_call: Dict[str, DatasetItem] = {}
    if dataset_items:
        for item in dataset_items:
            call_id = item.metadata.get("call_id", item.id)
            items_by_call[call_id] = item

    examples = []
    for res in metric_results:
        ann = ann_by_call.get(res.call_id)
        if not ann:
            continue

        item = items_by_call.get(res.call_id)
        input_text = ""
        output_text = ""
        if item:
            input_text = json.dumps(item.input, default=str) if item.input else ""
            output_text = str(item.output) if item.output else ""

        examples.append(
            {
                "id": res.call_id,
                "input": input_text,
                "output": output_text,
                "expected": "PASS" if ann.label else "FAIL",
                "call_id": res.call_id,
            }
        )

    # Shuffle and split into train/val
    random.shuffle(examples)
    split_idx = int(len(examples) * train_split)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    return trainset, valset


def parse_candidates_response(response: str) -> List[str]:
    """
    Parse the LLM's response to extract candidate preambles.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Raw JSON arrays
    - Nested arrays

    Args:
        response: Raw LLM response text

    Returns:
        List of candidate preamble strings
    """
    text = response.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    # Find JSON array
    try:
        start = text.find("[")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        json_str = text[start : i + 1]
                        candidates = json.loads(json_str)
                        # Ensure all items are strings
                        return [str(c) for c in candidates if c]
    except (json.JSONDecodeError, ValueError):
        pass

    return []


def parse_judge_response(response: str) -> bool:
    """
    Parse judge response to extract pass/fail verdict.

    Handles:
    - JSON with "passed" field
    - Plain text with "true"/"false" keywords

    Args:
        response: Raw judge response text

    Returns:
        True if passed, False otherwise
    """
    text = response.strip().lower()

    # Try to find JSON
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            return bool(data.get("passed", False))
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: look for keywords
    if "true" in text or '"passed": true' in text:
        return True
    if "false" in text or '"passed": false' in text:
        return False

    # Default to fail if unclear
    return False


__all__ = [
    "build_dataset_from_annotations",
    "parse_candidates_response",
    "parse_judge_response",
]
