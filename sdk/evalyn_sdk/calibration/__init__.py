"""Calibration module for Evalyn SDK.

This module provides tools for calibrating LLM judges against human annotations:
- AlignmentMetrics: Compute precision, recall, F1, Cohen's kappa
- DisagreementAnalysis: Analyze false positives/negatives
- Prompt optimizers: Basic, GEPA, GEPA-Native, OPRO, APE
- CalibrationEngine: Full calibration pipeline

IMPORTANT: All optimizers only modify the preamble (system prompt/instructions).
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from .ape import APEConfig, APEOptimizer
from .basic import BasicOptimizer
from .engine import CalibrationConfig, CalibrationEngine
from .gepa import GEPA_AVAILABLE, GEPAConfig, GEPAOptimizer
from .gepa_native import GEPANativeConfig, GEPANativeOptimizer
from .models import (
    AlignmentMetrics,
    DisagreementAnalysis,
    DisagreementCase,
    PromptOptimizationResult,
    TokenAccumulator,
    ValidationResult,
)
from .opro import OPROConfig, OPROOptimizer, TrajectoryEntry
from .utils import (
    build_dataset_from_annotations,
    build_full_prompt,
    load_optimized_prompt,
    parse_candidates_response,
    parse_judge_response,
    save_calibration,
)

__all__ = [
    # Engine
    "CalibrationConfig",
    "CalibrationEngine",
    # Models
    "AlignmentMetrics",
    "DisagreementAnalysis",
    "DisagreementCase",
    "PromptOptimizationResult",
    "TokenAccumulator",
    "ValidationResult",
    # Optimizers
    "BasicOptimizer",
    "GEPAConfig",
    "GEPAOptimizer",
    "GEPA_AVAILABLE",
    "GEPANativeConfig",
    "GEPANativeOptimizer",
    "OPROConfig",
    "OPROOptimizer",
    "TrajectoryEntry",
    "APEConfig",
    "APEOptimizer",
    # Utils
    "build_full_prompt",
    "build_dataset_from_annotations",
    "parse_candidates_response",
    "parse_judge_response",
    "save_calibration",
    "load_optimized_prompt",
]
