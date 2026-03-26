"""Shared parsing helpers for LLM output: JSON extraction and judge verdicts."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .models import FunctionCall


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text, handling markdown code blocks."""
    text = (text or "").strip()
    if not text:
        return None

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Find JSON object using bracket-finding (handles braces inside strings correctly
    # by delegating to json.loads rather than manual brace-depth counting)
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_json_list(text: str) -> List[dict]:
    """Extract a JSON list from LLM text that may contain markdown or extra prose.

    Tries, in order: direct parse, markdown code block extraction, regex for
    bare JSON array, bracket-finding fallback.
    """
    if not text:
        return []

    # Direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass

    # Markdown code block extraction
    for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                continue

    # Bracket-finding fallback
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass

    return []


def _parse_passed(value: Any) -> Optional[bool]:
    """Parse various representations of pass/fail to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"pass", "passed", "true", "yes", "1"}:
            return True
        if low in {"fail", "failed", "false", "no", "0"}:
            return False
    return None


def _safe_trace_excerpt(
    call: FunctionCall, max_events: int = 20, max_chars: int = 2000
) -> str:
    """Create a safe string excerpt of trace events for the judge prompt."""
    events = call.trace or []
    lines = []
    for ev in events[:max_events]:
        detail = ev.detail or {}
        try:
            detail_str = json.dumps(detail, default=str)
        except Exception:
            detail_str = str(detail)
        lines.append(f"- {ev.kind}: {detail_str[:300]}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "..."
