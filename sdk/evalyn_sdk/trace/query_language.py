from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..models import Span


@dataclass
class QueryPredicate:
    """Single predicate: field operator value."""

    field: str
    operator: str  # =, !=, >, <, >=, <=, ~ (contains)
    value: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class Query:
    """Collection of predicates with a combine mode."""

    predicates: List[QueryPredicate] = field(default_factory=list)
    combine: str = "and"  # "and" or "or"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "predicates": [p.as_dict() for p in self.predicates],
            "combine": self.combine,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Query":
        return cls(
            predicates=[
                QueryPredicate(
                    field=p["field"],
                    operator=p["operator"],
                    value=p["value"],
                )
                for p in data.get("predicates", [])
            ],
            combine=data.get("combine", "and"),
        )


# Regex for a single predicate token: field operator value
_PRED_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.]*)"  # field name (allows dots for nested)
    r"(!=|>=|<=|>|<|=|~)"  # operator
    r"(.+)"  # value (rest of token)
)


def parse_query(query_str: str) -> Query:
    """Parse a query string into a Query.

    Multiple predicates separated by whitespace are ANDed.
    Use the literal word ``or`` between predicates for OR logic.

    Examples:
        span_type=llm_call duration_ms>1000
        status=error or status=running
    """
    query_str = query_str.strip()
    if not query_str:
        return Query(predicates=[], combine="and")

    # Split on whitespace
    tokens = query_str.split()

    # Detect "or" keyword to decide combine mode
    has_or = False
    predicate_tokens: List[str] = []
    for token in tokens:
        if token.lower() == "or":
            has_or = True
        else:
            predicate_tokens.append(token)

    combine = "or" if has_or else "and"

    predicates: List[QueryPredicate] = []
    for token in predicate_tokens:
        m = _PRED_RE.match(token)
        if m:
            predicates.append(
                QueryPredicate(
                    field=m.group(1),
                    operator=m.group(2),
                    value=m.group(3),
                )
            )
        # Skip tokens that don't match - graceful handling

    return Query(predicates=predicates, combine=combine)


def _get_span_value(span: Span, field_name: str) -> Any:
    """Get a value from a span by field name.

    Supports: span_type, name, status, duration_ms, id, parent_id,
    and attributes.<key> for nested access.
    """
    if field_name.startswith("attributes."):
        attr_key = field_name[len("attributes."):]
        return span.attributes.get(attr_key)

    direct = {
        "span_type": span.span_type,
        "name": span.name,
        "status": span.status,
        "duration_ms": span.duration_ms,
        "id": span.id,
        "parent_id": span.parent_id,
    }
    return direct.get(field_name)


def _compare(actual: Any, operator: str, expected: str) -> bool:
    """Compare actual value against expected using the given operator.

    Tries float coercion for numeric operators (>, <, >=, <=).
    """
    if actual is None:
        return False

    if operator == "~":
        # Contains / substring match
        return expected.lower() in str(actual).lower()

    if operator in (">", "<", ">=", "<="):
        try:
            actual_f = float(actual)
            expected_f = float(expected)
        except (ValueError, TypeError):
            return False
        if operator == ">":
            return actual_f > expected_f
        if operator == "<":
            return actual_f < expected_f
        if operator == ">=":
            return actual_f >= expected_f
        if operator == "<=":
            return actual_f <= expected_f

    if operator == "=":
        # Try numeric equality first, fall back to string
        try:
            return float(actual) == float(expected)
        except (ValueError, TypeError):
            return str(actual) == expected

    if operator == "!=":
        try:
            return float(actual) != float(expected)
        except (ValueError, TypeError):
            return str(actual) != expected

    return False


def match_span(span: Span, query: Query) -> bool:
    """Test if a span matches the query."""
    if not query.predicates:
        return True

    results = []
    for pred in query.predicates:
        actual = _get_span_value(span, pred.field)
        results.append(_compare(actual, pred.operator, pred.value))

    if query.combine == "or":
        return any(results)
    return all(results)


def filter_spans(spans: List[Span], query: Query) -> List[Span]:
    """Return spans that match the query."""
    return [s for s in spans if match_span(s, query)]


def filter_spans_by_string(spans: List[Span], query_str: str) -> List[Span]:
    """Convenience: parse query string then filter."""
    return filter_spans(spans, parse_query(query_str))
