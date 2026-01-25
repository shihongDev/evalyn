"""
Evaluation units for span-level evaluation.

This package provides abstractions for discovering evaluatable units from
trace structure and projecting them into normalized views for metric evaluation.
"""

from .builders import (
    EvalUnitBuilder,
    OutcomeBuilder,
    SingleTurnBuilder,
    ToolUseBuilder,
    MultiTurnBuilder,
    CustomBuilder,
    get_default_builders,
    get_builder_for_type,
    get_builders_for_types,
)
from .views import project_unit

__all__ = [
    "EvalUnitBuilder",
    "OutcomeBuilder",
    "SingleTurnBuilder",
    "ToolUseBuilder",
    "MultiTurnBuilder",
    "CustomBuilder",
    "get_default_builders",
    "get_builder_for_type",
    "get_builders_for_types",
    "project_unit",
]
