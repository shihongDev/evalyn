"""
Span-level annotation for different types of traced events.

Supports annotating:
- LLM calls (quality, accuracy, hallucination)
- Tool calls (correctness, appropriateness)
- Reasoning steps (logic, coherence)
- Retrieval (relevance, coverage)
- Overall result (pass/fail)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from ..models import now_utc


# Annotation span types (distinct from models.SpanType which covers execution spans)
AnnotationSpanType = Literal["llm_call", "tool_call", "reasoning", "retrieval", "overall"]


@dataclass
class LLMCallAnnotation:
    """Annotation schema for LLM API calls."""

    quality: Optional[int] = None  # 1-5 scale
    accurate: Optional[bool] = None  # Is the response factually correct?
    relevant: Optional[bool] = None  # Does it address the query?
    hallucinated: Optional[bool] = None  # Contains made-up information?
    helpful: Optional[bool] = None  # Is it useful to the user?
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMCallAnnotation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolCallAnnotation:
    """Annotation schema for tool/function calls."""

    correct_tool: Optional[bool] = None  # Was the right tool chosen?
    correct_args: Optional[bool] = None  # Were the arguments correct?
    successful: Optional[bool] = None  # Did the tool succeed?
    necessary: Optional[bool] = None  # Was the tool call needed?
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallAnnotation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReasoningAnnotation:
    """Annotation schema for reasoning/thinking steps."""

    logical: Optional[bool] = None  # Is the reasoning logical?
    coherent: Optional[bool] = None  # Does it flow well?
    complete: Optional[bool] = None  # Are all steps present?
    correct: Optional[bool] = None  # Is the conclusion right?
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningAnnotation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RetrievalAnnotation:
    """Annotation schema for RAG retrieval steps."""

    relevant: Optional[bool] = None  # Are retrieved docs relevant?
    sufficient: Optional[bool] = None  # Enough context retrieved?
    accurate: Optional[bool] = None  # Is the retrieved info correct?
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalAnnotation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OverallAnnotation:
    """Annotation schema for overall result (backwards compatible)."""

    passed: Optional[bool] = None
    confidence: Optional[int] = None  # 1-5
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverallAnnotation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Type alias for any annotation schema
AnnotationSchema = (
    LLMCallAnnotation
    | ToolCallAnnotation
    | ReasoningAnnotation
    | RetrievalAnnotation
    | OverallAnnotation
)


ANNOTATION_SCHEMAS = {
    "llm_call": LLMCallAnnotation,
    "tool_call": ToolCallAnnotation,
    "reasoning": ReasoningAnnotation,
    "retrieval": RetrievalAnnotation,
    "overall": OverallAnnotation,
}


@dataclass
class SpanAnnotation:
    """
    Annotation for a specific span within a trace.

    A span can be:
    - An LLM call (gemini.completion, openai.completion, etc.)
    - A tool call (tool.call)
    - A reasoning step (trace.*.start/end)
    - A retrieval step
    - The overall result
    """

    id: str
    call_id: str  # Parent FunctionCall ID
    span_id: str  # ID of the specific span/event
    span_type: AnnotationSpanType
    annotation: AnnotationSchema
    annotator: str = "human"
    created_at: datetime = field(default_factory=now_utc)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "call_id": self.call_id,
            "span_id": self.span_id,
            "span_type": self.span_type,
            "annotation": self.annotation.as_dict(),
            "annotator": self.annotator,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanAnnotation":
        span_type = data.get("span_type", "overall")
        schema_cls = ANNOTATION_SCHEMAS.get(span_type, OverallAnnotation)
        annotation = schema_cls.from_dict(data.get("annotation", {}))

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return cls(
            id=data["id"],
            call_id=data["call_id"],
            span_id=data["span_id"],
            span_type=span_type,
            annotation=annotation,
            annotator=data.get("annotator", "human"),
            created_at=created_at or now_utc(),
        )


def extract_spans_from_trace(call) -> List[Dict[str, Any]]:
    """
    Extract annotatable spans from a FunctionCall's trace events.

    Returns a list of spans with:
    - span_id: Unique identifier
    - span_type: Type of span (llm_call, tool_call, etc.)
    - event: The original trace event
    - summary: Human-readable summary
    """
    spans = []

    # Add overall as the first span
    spans.append(
        {
            "span_id": f"{call.id}:overall",
            "span_type": "overall",
            "event": None,
            "summary": f"Overall result of {call.function_name}",
            "input": call.inputs,
            "output": call.output,
        }
    )

    # Process trace events
    for i, event in enumerate(call.trace):
        span_id = f"{call.id}:event:{i}"
        kind = event.kind
        detail = event.detail or {}

        # Classify span type
        if (
            "completion" in kind
            or "generate" in kind
            or "gemini" in kind.lower()
            or "openai" in kind.lower()
            or "anthropic" in kind.lower()
        ):
            span_type = "llm_call"
            summary = f"LLM: {detail.get('model', 'unknown')} - {detail.get('input_tokens', '?')} in / {detail.get('output_tokens', '?')} out"
        elif "tool" in kind:
            span_type = "tool_call"
            tool_name = detail.get("tool_name", "unknown")
            summary = f"Tool: {tool_name}"
        elif "trace." in kind:
            span_type = "reasoning"
            func_name = (
                kind.replace("trace.", "").replace(".start", "").replace(".end", "")
            )
            summary = f"Step: {func_name}"
        elif "retriev" in kind.lower() or "search" in kind.lower():
            span_type = "retrieval"
            summary = f"Retrieval: {kind}"
        else:
            # Skip unknown event types
            continue

        spans.append(
            {
                "span_id": span_id,
                "span_type": span_type,
                "event": event,
                "summary": summary,
                "detail": detail,
            }
        )

    return spans


def get_annotation_prompts(span_type: AnnotationSpanType) -> List[Dict[str, Any]]:
    """
    Get the annotation prompts for a given span type.

    Returns a list of questions to ask the annotator.
    """
    prompts = {
        "llm_call": [
            {
                "field": "quality",
                "question": "Quality (1-5)?",
                "type": "int",
                "range": (1, 5),
            },
            {"field": "accurate", "question": "Accurate?", "type": "bool"},
            {"field": "relevant", "question": "Relevant to query?", "type": "bool"},
            {
                "field": "hallucinated",
                "question": "Contains hallucinations?",
                "type": "bool",
            },
            {"field": "helpful", "question": "Helpful?", "type": "bool"},
            {"field": "notes", "question": "Notes:", "type": "str"},
        ],
        "tool_call": [
            {
                "field": "correct_tool",
                "question": "Correct tool chosen?",
                "type": "bool",
            },
            {"field": "correct_args", "question": "Correct arguments?", "type": "bool"},
            {"field": "successful", "question": "Succeeded?", "type": "bool"},
            {"field": "necessary", "question": "Was it necessary?", "type": "bool"},
            {"field": "notes", "question": "Notes:", "type": "str"},
        ],
        "reasoning": [
            {"field": "logical", "question": "Logical?", "type": "bool"},
            {"field": "coherent", "question": "Coherent?", "type": "bool"},
            {"field": "complete", "question": "Complete?", "type": "bool"},
            {"field": "correct", "question": "Correct conclusion?", "type": "bool"},
            {"field": "notes", "question": "Notes:", "type": "str"},
        ],
        "retrieval": [
            {"field": "relevant", "question": "Relevant docs?", "type": "bool"},
            {"field": "sufficient", "question": "Sufficient context?", "type": "bool"},
            {"field": "accurate", "question": "Accurate info?", "type": "bool"},
            {"field": "notes", "question": "Notes:", "type": "str"},
        ],
        "overall": [
            {"field": "passed", "question": "Pass?", "type": "bool"},
            {
                "field": "confidence",
                "question": "Confidence (1-5)?",
                "type": "int",
                "range": (1, 5),
            },
            {"field": "notes", "question": "Notes:", "type": "str"},
        ],
    }
    return prompts.get(span_type, prompts["overall"])
