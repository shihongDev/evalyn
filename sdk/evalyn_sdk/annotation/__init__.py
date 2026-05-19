"""
Annotation module for human labeling.

Provides tools for:
- Importing/exporting annotations
- Span-level annotation schemas

Note: Calibration and prompt optimization have moved to evalyn_sdk.calibration.
Re-exports are provided here for backwards compatibility via lazy loading.
"""

from .annotations import (
    export_annotation_items,
    export_annotations,
    import_annotation_items,
    import_annotations,
    merge_annotations_into_dataset,
)
from .span_annotation import (
    ANNOTATION_SCHEMAS,
    AnnotationSchema,
    AnnotationSpanType,
    LLMCallAnnotation,
    OverallAnnotation,
    ReasoningAnnotation,
    RetrievalAnnotation,
    SpanAnnotation,
    ToolCallAnnotation,
    extract_spans_from_trace,
    get_annotation_prompts,
)

# Calibration re-exports are loaded lazily to avoid importing
# all optimizer modules when only annotation functions are needed.
_CALIBRATION_REEXPORTS = {
    "AlignmentMetrics",
    "DisagreementCase",
    "DisagreementAnalysis",
    "PromptOptimizationResult",
    "ValidationResult",
    "BasicOptimizer",
    "GEPAConfig",
    "GEPAOptimizer",
    "GEPA_AVAILABLE",
    "GEPANativeConfig",
    "GEPANativeOptimizer",
    "OPROConfig",
    "OPROOptimizer",
    "APEConfig",
    "APEOptimizer",
    "CalibrationConfig",
    "CalibrationEngine",
    "save_calibration",
    "load_optimized_prompt",
    "TokenAccumulator",
    "build_full_prompt",
}


def __getattr__(name: str):
    if name in _CALIBRATION_REEXPORTS:
        from .. import calibration as _cal

        val = getattr(_cal, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Annotations
    "export_annotations",
    "export_annotation_items",
    "import_annotations",
    "import_annotation_items",
    "merge_annotations_into_dataset",
    # Span annotations
    "SpanAnnotation",
    "AnnotationSpanType",
    "LLMCallAnnotation",
    "ToolCallAnnotation",
    "ReasoningAnnotation",
    "RetrievalAnnotation",
    "OverallAnnotation",
    "AnnotationSchema",
    "ANNOTATION_SCHEMAS",
    "extract_spans_from_trace",
    "get_annotation_prompts",
    # Calibration (lazy re-exports)
    *_CALIBRATION_REEXPORTS,
]
