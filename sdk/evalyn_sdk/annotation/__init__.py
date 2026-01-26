"""
Annotation module for human labeling and calibration.

Provides tools for:
- Importing/exporting annotations
- Span-level annotation schemas
- LLM judge calibration
- Prompt optimization (Basic, APE, OPRO, GEPA)
"""

from .annotations import (
    export_annotations,
    export_annotation_items,
    import_annotations,
    import_annotation_items,
    merge_annotations_into_dataset,
)
from .span_annotation import (
    SpanAnnotation,
    AnnotationSpanType,
    LLMCallAnnotation,
    ToolCallAnnotation,
    ReasoningAnnotation,
    RetrievalAnnotation,
    OverallAnnotation,
    AnnotationSchema,
    ANNOTATION_SCHEMAS,
    extract_spans_from_trace,
    get_annotation_prompts,
)
from .calibration import (
    AlignmentMetrics,
    DisagreementCase,
    DisagreementAnalysis,
    PromptOptimizationResult,
    ValidationResult,
    BasicOptimizer,
    GEPAConfig,
    GEPAOptimizer,
    GEPA_AVAILABLE,
    CalibrationConfig,
    CalibrationEngine,
    save_calibration,
    load_optimized_prompt,
    TokenAccumulator,
    build_full_prompt,
)
from .opro import (
    OPROConfig,
    OPROOptimizer,
)
from .ape import (
    APEConfig,
    APEOptimizer,
)
from .gepa_native import (
    GEPANativeConfig,
    GEPANativeOptimizer,
)

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
    # Calibration
    "AlignmentMetrics",
    "DisagreementCase",
    "DisagreementAnalysis",
    "PromptOptimizationResult",
    "ValidationResult",
    "BasicOptimizer",
    "GEPAConfig",
    "GEPAOptimizer",
    "GEPA_AVAILABLE",
    "CalibrationConfig",
    "CalibrationEngine",
    "save_calibration",
    "load_optimized_prompt",
    "TokenAccumulator",
    "build_full_prompt",
    # OPRO
    "OPROConfig",
    "OPROOptimizer",
    # APE
    "APEConfig",
    "APEOptimizer",
    # GEPA Native
    "GEPANativeConfig",
    "GEPANativeOptimizer",
]
