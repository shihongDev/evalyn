"""Calibration module for Evalyn SDK.

This module provides tools for calibrating LLM judges against human annotations:
- AlignmentMetrics: Compute precision, recall, F1, Cohen's kappa
- DisagreementAnalysis: Analyze false positives/negatives
- Prompt optimizers: Basic, GEPA, GEPA-Native, OPRO, APE, EvoPrompt
- CalibrationEngine: Full calibration pipeline

IMPORTANT: All optimizers only modify the preamble (system prompt/instructions).
The rubric (evaluation criteria) is kept FIXED as defined by humans.

All symbols are loaded lazily so that importing the calibration package
(e.g., to access models or utils) does not pull in every optimizer.
"""

from __future__ import annotations

import importlib as _importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Models (lightweight dataclasses)
    "AlignmentMetrics": (".models", "AlignmentMetrics"),
    "DisagreementAnalysis": (".models", "DisagreementAnalysis"),
    "DisagreementCase": (".models", "DisagreementCase"),
    "PromptOptimizationResult": (".models", "PromptOptimizationResult"),
    "TokenAccumulator": (".models", "TokenAccumulator"),
    "ValidationResult": (".models", "ValidationResult"),
    # Engine
    "CalibrationConfig": (".engine", "CalibrationConfig"),
    "CalibrationEngine": (".engine", "CalibrationEngine"),
    # Factory
    "BaseOptimizer": (".base_optimizer", "BaseOptimizer"),
    "OPTIMIZER_REGISTRY": (".factory", "OPTIMIZER_REGISTRY"),
    "call_optimizer": (".factory", "call_optimizer"),
    "create_optimizer": (".factory", "create_optimizer"),
    # Utils
    "build_dataset_from_annotations": (".utils", "build_dataset_from_annotations"),
    "build_full_prompt": (".utils", "build_full_prompt"),
    "load_optimized_prompt": (".utils", "load_optimized_prompt"),
    "parse_candidates_response": (".utils", "parse_candidates_response"),
    "parse_judge_response": (".utils", "parse_judge_response"),
    "save_calibration": (".utils", "save_calibration"),
    # Optimizers
    "BasicOptimizer": (".optimizers.basic", "BasicOptimizer"),
    "GEPAConfig": (".optimizers.gepa", "GEPAConfig"),
    "GEPAOptimizer": (".optimizers.gepa", "GEPAOptimizer"),
    "GEPA_AVAILABLE": (".optimizers.gepa", "GEPA_AVAILABLE"),
    "GEPANativeConfig": (".optimizers.gepa_native", "GEPANativeConfig"),
    "GEPANativeOptimizer": (".optimizers.gepa_native", "GEPANativeOptimizer"),
    "OPROConfig": (".optimizers.opro", "OPROConfig"),
    "OPROOptimizer": (".optimizers.opro", "OPROOptimizer"),
    "TrajectoryEntry": (".optimizers.opro", "TrajectoryEntry"),
    "APEConfig": (".optimizers.ape", "APEConfig"),
    "APEOptimizer": (".optimizers.ape", "APEOptimizer"),
    "EvoPromptConfig": (".optimizers.evoprompt", "EvoPromptConfig"),
    "EvoPromptOptimizer": (".optimizers.evoprompt", "EvoPromptOptimizer"),
    "TextGradConfig": (".optimizers.textgrad", "TextGradConfig"),
    "TextGradOptimizer": (".optimizers.textgrad", "TextGradOptimizer"),
    "MIPROv2Config": (".optimizers.miprov2", "MIPROv2Config"),
    "MIPROv2Optimizer": (".optimizers.miprov2", "MIPROv2Optimizer"),
    "BreederUnit": (".optimizers.promptbreeder", "BreederUnit"),
    "PromptBreederConfig": (".optimizers.promptbreeder", "PromptBreederConfig"),
    "PromptBreederOptimizer": (".optimizers.promptbreeder", "PromptBreederOptimizer"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
