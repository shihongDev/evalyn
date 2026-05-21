"""Shared utilities for calibration module.

Contains:
- Prompt building utilities
- Dataset building from annotations
- Response parsing (candidates and judge verdicts)
- Save/load calibration records
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import Annotation, DatasetItem, MetricResult

if TYPE_CHECKING:
    from ..models import CalibrationRecord


def build_full_prompt(preamble: str, rubric: list[str]) -> str:
    """
    Combine preamble with rubric to create a complete judge prompt.

    This is the standard prompt structure used by all optimizers.
    The preamble is the part that gets optimized; the rubric stays fixed.

    Args:
        preamble: The system prompt/instructions (optimized by APE/OPRO/etc)
        rubric: Fixed evaluation criteria defined by humans

    Returns:
        Complete prompt ready for use by LLM judge
    """
    rubric_text = ""
    if rubric:
        rubric_lines = "\n".join([f"- {r}" for r in rubric])
        rubric_text = (
            f"\n\nEvaluate using this rubric (PASS only if all criteria met):\n{rubric_lines}"
        )

    output_format = """

After your analysis, provide your verdict as a JSON object:
{"passed": true/false, "reason": "brief explanation", "score": 0.0-1.0}"""

    return preamble + rubric_text + output_format


def build_dataset_from_annotations(
    metric_results: list[MetricResult],
    annotations: list[Annotation],
    dataset_items: list[DatasetItem] | None = None,
    train_split: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    ann_by_call: dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
    items_by_call: dict[str, DatasetItem] = {}
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


def parse_candidates_response(response: str) -> list[str]:
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

    # Fallback: check for standalone "true"/"false" keywords (not substrings)
    if re.search(r"\btrue\b", text):
        return True
    if re.search(r"\bfalse\b", text):
        return False

    # Default to fail if unparseable
    return False


def save_calibration(
    record: CalibrationRecord,
    dataset_path: str,
    metric_id: str,
) -> dict[str, str]:
    """
    Save calibration record and optimized prompts to the dataset's calibrations folder.

    Directory structure:
        <dataset>/
          calibrations/
            <metric_id>/
              <timestamp>_<optimizer>.json     # Full calibration record
              prompts/
                <timestamp>_preamble.txt       # Optimized preamble only
                <timestamp>_full.txt           # Full prompt (ready to use)

    Args:
        record: CalibrationRecord to save
        dataset_path: Path to the dataset folder
        metric_id: Metric ID being calibrated

    Returns:
        Dict with paths to saved files:
        - "calibration": Path to the calibration JSON
        - "preamble": Path to preamble file (if available)
        - "full_prompt": Path to full prompt file (if available)
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Create calibrations directory structure
    calibrations_dir = dataset_dir / "calibrations" / metric_id
    prompts_dir = calibrations_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_type = record.adjustments.get("optimizer_type", "basic")

    saved_paths: dict[str, str] = {}

    # Save the calibration record JSON
    calibration_file = calibrations_dir / f"{timestamp}_{optimizer_type}.json"
    calibration_file.write_text(
        json.dumps(record.as_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    saved_paths["calibration"] = str(calibration_file)

    # Extract and save optimized prompts if available
    optimization = record.adjustments.get("prompt_optimization", {})
    if optimization:
        # Save optimized preamble
        optimized_preamble = optimization.get("optimized_preamble", "")
        if optimized_preamble:
            preamble_file = prompts_dir / f"{timestamp}_preamble.txt"
            preamble_file.write_text(optimized_preamble, encoding="utf-8")
            saved_paths["preamble"] = str(preamble_file)

        # Save full prompt (ready to use)
        full_prompt = optimization.get("full_prompt", "")
        if full_prompt:
            full_file = prompts_dir / f"{timestamp}_full.txt"
            full_file.write_text(full_prompt, encoding="utf-8")
            saved_paths["full_prompt"] = str(full_file)

        # If no full_prompt but we have improved_rubric, build it
        if not full_prompt and optimization.get("improved_rubric"):
            preamble = optimization.get("optimized_preamble", "")
            rubric = optimization.get("improved_rubric", [])
            if preamble or rubric:
                full_built = build_full_prompt(preamble or "", rubric)
                if full_built.strip():
                    full_file = prompts_dir / f"{timestamp}_full.txt"
                    full_file.write_text(full_built, encoding="utf-8")
                    saved_paths["full_prompt"] = str(full_file)

    return saved_paths


def load_optimized_prompt(
    dataset_path: str,
    metric_id: str,
    version: str | None = None,
) -> str | None:
    """
    Load an optimized prompt from the calibrations folder.

    Args:
        dataset_path: Path to the dataset folder
        metric_id: Metric ID to load prompt for
        version: Specific version timestamp (e.g., "20250101_120000").
                 If None, loads the most recent.

    Returns:
        The full optimized prompt text, or None if not found.
    """
    prompts_dir = Path(dataset_path) / "calibrations" / metric_id / "prompts"
    if not prompts_dir.exists():
        return None

    # Find full prompt files
    full_files = sorted(prompts_dir.glob("*_full.txt"), reverse=True)
    if not full_files:
        return None

    if version:
        # Find specific version
        for f in full_files:
            if f.name.startswith(version):
                return f.read_text(encoding="utf-8")
        return None
    else:
        # Return most recent
        return full_files[0].read_text(encoding="utf-8")


__all__ = [
    "build_full_prompt",
    "build_dataset_from_annotations",
    "parse_candidates_response",
    "parse_judge_response",
    "save_calibration",
    "load_optimized_prompt",
]
