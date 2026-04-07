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
    "BasicOptimizer": (".basic", "BasicOptimizer"),
    "GEPAConfig": (".gepa", "GEPAConfig"),
    "GEPAOptimizer": (".gepa", "GEPAOptimizer"),
    "GEPA_AVAILABLE": (".gepa", "GEPA_AVAILABLE"),
    "GEPANativeConfig": (".gepa_native", "GEPANativeConfig"),
    "GEPANativeOptimizer": (".gepa_native", "GEPANativeOptimizer"),
    "OPROConfig": (".opro", "OPROConfig"),
    "OPROOptimizer": (".opro", "OPROOptimizer"),
    "TrajectoryEntry": (".opro", "TrajectoryEntry"),
    "APEConfig": (".ape", "APEConfig"),
    "APEOptimizer": (".ape", "APEOptimizer"),
    "EvoPromptConfig": (".evoprompt", "EvoPromptConfig"),
    "EvoPromptOptimizer": (".evoprompt", "EvoPromptOptimizer"),
    "TextGradConfig": (".textgrad", "TextGradConfig"),
    "TextGradOptimizer": (".textgrad", "TextGradOptimizer"),
    "MIPROv2Config": (".miprov2", "MIPROv2Config"),
    "MIPROv2Optimizer": (".miprov2", "MIPROv2Optimizer"),
    "BreederUnit": (".promptbreeder", "BreederUnit"),
    "PromptBreederConfig": (".promptbreeder", "PromptBreederConfig"),
    "PromptBreederOptimizer": (".promptbreeder", "PromptBreederOptimizer"),
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
