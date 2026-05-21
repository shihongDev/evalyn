"""Dataset filtering DSL for query-based item selection.

Supports filtering dataset items by field values, metadata, and text content.

Syntax examples:
    "output_length > 500"
    "metadata.tag = production"
    "input contains refund"
    "metadata.source != test and output_length < 1000"
"""

from __future__ import annotations

import json
import re
from typing import Any


class FilterExpression:
    """A parsed filter expression that can evaluate against dataset items."""

    def __init__(self, field: str, operator: str, value: str):
        self.field = field
        self.operator = operator
        self.value = value

    def evaluate(self, item) -> bool:
        """Evaluate this expression against a DatasetItem."""
        actual = self._resolve_field(item)
        return self._compare(actual, self.operator, self.value)

    def _resolve_field(self, item) -> Any:
        """Resolve a field path to a value from the item."""
        if self.field == "input_length":
            return len(_to_text(item.input))
        if self.field == "output_length":
            return len(_to_text(item.output))
        if self.field == "id":
            return item.id
        if self.field == "input":
            return _to_text(item.input)
        if self.field == "output":
            return _to_text(item.output)
        if self.field == "has_reference":
            return item.human_label is not None

        # metadata.* fields
        if self.field.startswith("metadata."):
            key = self.field[len("metadata.") :]
            return (item.metadata or {}).get(key)

        return None

    @staticmethod
    def _compare(actual: Any, operator: str, value: str) -> bool:
        """Compare a field value against the filter value."""
        if actual is None:
            return operator == "is_null"

        if operator == "=":
            return str(actual) == value
        if operator == "!=":
            return str(actual) != value
        if operator == "contains":
            return value.lower() in str(actual).lower()
        if operator == "not_contains":
            return value.lower() not in str(actual).lower()
        if operator == "matches":
            try:
                return bool(re.search(value, str(actual)))
            except re.error:
                return False
        if operator == "is_null":
            return actual is None
        if operator == "is_not_null":
            return actual is not None

        # Numeric comparisons
        try:
            num_actual = float(actual) if not isinstance(actual, (int, float)) else actual
            num_value = float(value)
        except (ValueError, TypeError):
            return False

        if operator == ">":
            return num_actual > num_value
        if operator == ">=":
            return num_actual >= num_value
        if operator == "<":
            return num_actual < num_value
        if operator == "<=":
            return num_actual <= num_value

        return False

    def __repr__(self) -> str:
        return f"FilterExpression({self.field} {self.operator} {self.value})"


class CompoundFilter:
    """A compound filter combining multiple expressions with AND/OR."""

    def __init__(self, expressions: list[FilterExpression], logic: str = "and"):
        self.expressions = expressions
        self.logic = logic.lower()

    def evaluate(self, item) -> bool:
        if not self.expressions:
            return True
        if self.logic == "and":
            return all(expr.evaluate(item) for expr in self.expressions)
        elif self.logic == "or":
            return any(expr.evaluate(item) for expr in self.expressions)
        return True

    def __repr__(self) -> str:
        return f"CompoundFilter({self.logic}: {self.expressions})"


def parse_filter(query: str) -> CompoundFilter:
    """Parse a filter query string into a CompoundFilter.

    Supports:
        "field op value"
        "field op value and field op value"
        "field op value or field op value"

    Args:
        query: Filter query string.

    Returns:
        CompoundFilter ready to evaluate against items.

    Raises:
        ValueError: If query cannot be parsed.
    """
    query = query.strip()
    if not query:
        return CompoundFilter([])

    # Detect logic operator (AND takes precedence - split on OR first)
    logic = "and"
    parts = []

    # Split on " or " (case insensitive)
    or_parts = re.split(r"\s+or\s+", query, flags=re.IGNORECASE)
    if len(or_parts) > 1:
        logic = "or"
        parts = or_parts
    else:
        # Split on " and " (case insensitive)
        and_parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)
        parts = and_parts

    expressions = []
    for part in parts:
        expr = _parse_single_expression(part.strip())
        if expr:
            expressions.append(expr)

    if not expressions:
        raise ValueError(f"Could not parse filter: {query!r}")

    return CompoundFilter(expressions, logic)


def _parse_single_expression(expr_str: str) -> FilterExpression | None:
    """Parse a single 'field op value' expression."""
    # Try operators from longest to shortest to avoid partial matches
    operators = [
        ("is not null", "is_not_null"),
        ("is null", "is_null"),
        ("not_contains", "not_contains"),
        ("not contains", "not_contains"),
        ("contains", "contains"),
        ("matches", "matches"),
        (">=", ">="),
        ("<=", "<="),
        ("!=", "!="),
        (">", ">"),
        ("<", "<"),
        ("=", "="),
    ]

    for op_str, op_name in operators:
        # Special case for "is null" / "is not null" (no value)
        if op_name in ("is_null", "is_not_null"):
            pattern = rf"^(\S+)\s+{re.escape(op_str)}$"
            match = re.match(pattern, expr_str, re.IGNORECASE)
            if match:
                return FilterExpression(match.group(1), op_name, "")
            continue

        # Standard "field op value" pattern
        idx = expr_str.lower().find(f" {op_str} ") if len(op_str) > 1 else -1
        if idx == -1 and len(op_str) <= 2:
            # For short operators (=, !=, >, <, >=, <=), find them more carefully
            # Split on the operator, respecting spaces
            pattern = rf"^(\S+)\s*{re.escape(op_str)}\s*(.+)$"
            match = re.match(pattern, expr_str)
            if match:
                return FilterExpression(match.group(1).strip(), op_name, match.group(2).strip())
        elif idx >= 0:
            field = expr_str[:idx].strip()
            value = expr_str[idx + len(op_str) + 2 :].strip()
            return FilterExpression(field, op_name, value)

    return None


def filter_items(items: list, query: str) -> list:
    """Filter dataset items using a query string.

    Args:
        items: List of DatasetItem objects.
        query: Filter query string.

    Returns:
        Filtered list of items matching the query.
    """
    filt = parse_filter(query)
    return [item for item in items if filt.evaluate(item)]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)
