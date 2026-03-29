"""
Constraint-guided simulation: generate inputs satisfying specific constraints.

Parse constraint expressions, check items against constraints, filter datasets,
and generate text that satisfies constraint requirements.
Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """A single field-level constraint."""

    field: str
    operator: str  # eq/ne/gt/lt/gte/lte/contains/not_contains
    value: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Constraint:
        return cls(
            field=data["field"],
            operator=data["operator"],
            value=data["value"],
        )


@dataclass
class ConstraintSet:
    """A collection of constraints joined by AND or OR logic."""

    constraints: List[Constraint] = field(default_factory=list)
    logic: str = "AND"  # "AND" or "OR"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "constraints": [c.as_dict() for c in self.constraints],
            "logic": self.logic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstraintSet:
        return cls(
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            logic=data.get("logic", "AND"),
        )


@dataclass
class ConstraintCheckResult:
    """Result of checking an item against a constraint set."""

    item_id: str
    passed: bool
    failed_constraints: List[str] = field(default_factory=list)
    all_constraints: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "passed": self.passed,
            "failed_constraints": list(self.failed_constraints),
            "all_constraints": self.all_constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstraintCheckResult:
        return cls(
            item_id=data["item_id"],
            passed=data["passed"],
            failed_constraints=data.get("failed_constraints", []),
            all_constraints=data.get("all_constraints", 0),
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Operator mapping from expression syntax to internal names.
# Order matters: longer operators must come before shorter ones to avoid
# partial matches (e.g. ">=" before ">").
_OPERATOR_MAP = [
    (">=", "gte"),
    ("<=", "lte"),
    ("!=", "ne"),
    ("!~", "not_contains"),
    ("~=", "contains"),
    (">", "gt"),
    ("<", "lt"),
    ("=", "eq"),
]


def parse_constraint(expr: str) -> Constraint:
    """Parse a single constraint expression like 'field>=value'.

    Supported operators: =, !=, >, <, >=, <=, ~= (contains), !~ (not_contains).
    """
    expr = expr.strip()
    for symbol, op_name in _OPERATOR_MAP:
        idx = expr.find(symbol)
        if idx > 0:
            field_name = expr[:idx].strip()
            raw_value = expr[idx + len(symbol) :].strip()
            # Try numeric conversion
            value: Any = raw_value
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    pass
            return Constraint(field=field_name, operator=op_name, value=value)
    raise ValueError(f"Cannot parse constraint expression: {expr!r}")


def parse_constraint_string(expr: str) -> ConstraintSet:
    """Parse a compound constraint string like 'field1=val1 AND field2>val2'.

    Splits on AND/OR keywords. The logic of the resulting ConstraintSet is
    determined by the first connector found (all connectors should be the same;
    mixing AND/OR produces whichever appears first).
    """
    expr = expr.strip()
    if not expr:
        return ConstraintSet()

    # Split on AND / OR while capturing the keyword
    parts = re.split(r"\s+(AND|OR)\s+", expr)
    # parts looks like: [expr1, 'AND', expr2, 'AND', expr3, ...]

    logic = "AND"
    constraint_exprs: List[str] = []
    for i, part in enumerate(parts):
        upper = part.strip().upper()
        if upper in ("AND", "OR"):
            logic = upper
        else:
            constraint_exprs.append(part.strip())

    constraints = [parse_constraint(e) for e in constraint_exprs if e]
    return ConstraintSet(constraints=constraints, logic=logic)


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def check_constraint(item: dict, constraint: Constraint) -> bool:
    """Check whether a single constraint holds for the given item dict."""
    raw = item.get(constraint.field)
    target = constraint.value
    op = constraint.operator

    # Missing field always fails
    if raw is None:
        return False

    # Numeric comparison path
    if op in ("gt", "lt", "gte", "lte"):
        try:
            num_raw = float(raw)
            num_target = float(target)
        except (ValueError, TypeError):
            return False
        if op == "gt":
            return num_raw > num_target
        if op == "lt":
            return num_raw < num_target
        if op == "gte":
            return num_raw >= num_target
        if op == "lte":
            return num_raw <= num_target

    # Equality / inequality
    if op == "eq":
        # Try numeric equality first
        try:
            return float(raw) == float(target)
        except (ValueError, TypeError):
            return str(raw) == str(target)

    if op == "ne":
        try:
            return float(raw) != float(target)
        except (ValueError, TypeError):
            return str(raw) != str(target)

    # String matching
    if op == "contains":
        return str(target).lower() in str(raw).lower()

    if op == "not_contains":
        return str(target).lower() not in str(raw).lower()

    return False


def check_constraints(item: dict, constraint_set: ConstraintSet) -> ConstraintCheckResult:
    """Check all constraints in a set against an item.

    Uses AND/OR logic as specified. Returns a ConstraintCheckResult with
    item_id taken from item['id'] or 'unknown'.
    """
    item_id = str(item.get("id", "unknown"))
    total = len(constraint_set.constraints)
    failed: List[str] = []

    for c in constraint_set.constraints:
        if not check_constraint(item, c):
            failed.append(f"{c.field} {c.operator} {c.value}")

    if constraint_set.logic == "OR":
        passed = len(failed) < total  # at least one passed
        if total == 0:
            passed = True
    else:
        passed = len(failed) == 0

    return ConstraintCheckResult(
        item_id=item_id,
        passed=passed,
        failed_constraints=failed,
        all_constraints=total,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_by_constraints(items: List[dict], constraint_set: ConstraintSet) -> List[dict]:
    """Return items that pass the constraint set."""
    return [item for item in items if check_constraints(item, constraint_set).passed]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_constrained_text(base_text: str, constraints: ConstraintSet) -> str:
    """Modify base_text to satisfy constraints where possible.

    Applies simple heuristics:
    - length constraints: pad or truncate
    - contains constraints: prepend keyword if missing
    - not_contains constraints: remove keyword if present
    """
    text = base_text
    for c in constraints.constraints:
        if c.field == "length" or c.field == "min_length":
            try:
                target_len = int(c.value)
            except (ValueError, TypeError):
                continue
            if c.operator in ("gte", "gt", "eq"):
                needed = target_len if c.operator != "gt" else target_len + 1
                while len(text) < needed:
                    text = text + " " + text
                if c.operator == "eq":
                    text = text[:target_len]
            elif c.operator in ("lte", "lt"):
                max_len = target_len if c.operator != "lt" else target_len - 1
                if len(text) > max_len:
                    text = text[:max_len]

        elif c.field == "max_length":
            try:
                max_len = int(c.value)
            except (ValueError, TypeError):
                continue
            if c.operator in ("lte", "lt", "eq"):
                limit = max_len if c.operator != "lt" else max_len - 1
                if len(text) > limit:
                    text = text[:limit]

        elif c.field == "topic" and c.operator == "contains":
            keyword = str(c.value)
            if keyword.lower() not in text.lower():
                text = keyword + " " + text

        elif c.field == "topic" and c.operator == "not_contains":
            keyword = str(c.value)
            # Simple removal - case insensitive
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub("", text).strip()

        elif c.operator == "contains":
            keyword = str(c.value)
            if keyword.lower() not in text.lower():
                text = keyword + " " + text

        elif c.operator == "not_contains":
            keyword = str(c.value)
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub("", text).strip()

    return text


# ---------------------------------------------------------------------------
# Formatting / Utilities
# ---------------------------------------------------------------------------


def format_constraint_summary(constraint_set: ConstraintSet) -> str:
    """Return a human-readable summary of the constraint set."""
    if not constraint_set.constraints:
        return "No constraints defined."

    lines: List[str] = []
    lines.append(f"Constraint set ({constraint_set.logic} logic):")
    for i, c in enumerate(constraint_set.constraints, 1):
        lines.append(f"  {i}. {c.field} {c.operator} {c.value}")
    lines.append(f"Total: {len(constraint_set.constraints)} constraint(s)")
    return "\n".join(lines)


def count_passing(items: List[dict], constraint_set: ConstraintSet) -> int:
    """Count items that pass the constraint set."""
    return len(filter_by_constraints(items, constraint_set))
