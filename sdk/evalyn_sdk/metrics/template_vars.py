"""Metric template variables: custom variables in judge prompts.

Supports {{variable}} placeholders in judge prompt templates that are
resolved at evaluation time from evalyn.yaml or item metadata.

Built-in variables: {{input}}, {{output}}, {{expected}}, {{metadata.*}}
Custom variables: defined in evalyn.yaml template_vars section.
"""

from __future__ import annotations

import re
from typing import Any

# Built-in variable names
BUILTIN_VARS = {"input", "output", "expected", "item_id"}


def resolve_template(
    template: str,
    item: Any | None = None,
    custom_vars: dict[str, str] | None = None,
) -> str:
    """Resolve {{variable}} placeholders in a template string.

    Resolution order:
    1. Built-in vars from the DatasetItem (input, output, expected, item_id)
    2. metadata.* vars from item.metadata
    3. Custom vars from config/arguments

    Args:
        template: Template string with {{variable}} placeholders.
        item: DatasetItem for built-in variable resolution.
        custom_vars: Dict of custom variable name -> value.

    Returns:
        Template with all resolvable variables replaced.
    """
    if "{{" not in template:
        return template

    resolved = template
    variables = extract_variables(template)

    for var_name in variables:
        value = _resolve_variable(var_name, item, custom_vars)
        if value is not None:
            resolved = resolved.replace("{{" + var_name + "}}", str(value))

    return resolved


def extract_variables(template: str) -> list[str]:
    """Extract all {{variable}} names from a template.

    Args:
        template: Template string.

    Returns:
        List of unique variable names found.
    """
    pattern = r"\{\{(\w+(?:\.\w+)*)\}\}"
    return list(dict.fromkeys(re.findall(pattern, template)))


def validate_template(
    template: str,
    available_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a template and check for unresolvable variables.

    Args:
        template: Template string.
        available_vars: Custom variables that will be available at runtime.

    Returns:
        Dict with "valid" (bool), "variables" (list), "unresolved" (list).
    """
    variables = extract_variables(template)
    available = set(BUILTIN_VARS)
    available.add("metadata")  # metadata.* is always available

    if available_vars:
        available.update(available_vars.keys())

    unresolved = []
    for var in variables:
        root = var.split(".")[0]
        if root not in available:
            unresolved.append(var)

    return {
        "valid": len(unresolved) == 0,
        "variables": variables,
        "unresolved": unresolved,
        "builtin": [v for v in variables if v.split(".")[0] in BUILTIN_VARS],
        "custom": [
            v for v in variables if v.split(".")[0] not in BUILTIN_VARS and v not in unresolved
        ],
    }


def _resolve_variable(
    var_name: str,
    item: Any | None,
    custom_vars: dict[str, str] | None,
) -> str | None:
    """Resolve a single variable name to its value."""
    # Built-in variables
    if item is not None:
        if var_name == "input":
            return _to_str(item.input)
        if var_name == "output":
            return _to_str(item.output)
        if var_name == "expected":
            return _to_str(getattr(item, "expected", None))
        if var_name == "item_id":
            return item.id

        # metadata.* variables
        if var_name.startswith("metadata."):
            key = var_name[len("metadata.") :]
            meta = getattr(item, "metadata", {}) or {}
            return _to_str(meta.get(key))

    # Custom variables
    if custom_vars and var_name in custom_vars:
        return str(custom_vars[var_name])

    return None


def _to_str(value: Any) -> str:
    """Convert a value to string for template substitution."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)
