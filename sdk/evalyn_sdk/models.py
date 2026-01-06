from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
import uuid


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_datetime(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


def _default_id() -> str:
    return str(uuid.uuid4())


def _safe_details(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return data or {}


@dataclass
class TraceEvent:
    kind: str
    timestamp: datetime
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "timestamp": _iso(self.timestamp),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        return cls(
            kind=data["kind"],
            timestamp=_parse_datetime(data["timestamp"]) or now_utc(),
            detail=_safe_details(data.get("detail")),
        )


@dataclass
class FunctionCall:
    id: str
    function_name: str
    inputs: Dict[str, Any]
    output: Any
    error: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    duration_ms: Optional[float]
    session_id: Optional[str]
    trace: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "function_name": self.function_name,
            "inputs": self.inputs,
            "output": self.output,
            "error": self.error,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "duration_ms": self.duration_ms,
            "session_id": self.session_id,
            "trace": [t.as_dict() for t in self.trace],
            "metadata": self.metadata,
        }

    @classmethod
    def new(
        cls,
        function_name: str,
        inputs: Dict[str, Any],
        session_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FunctionCall":
        return cls(
            id=_default_id(),
            function_name=function_name,
            inputs=inputs,
            output=None,
            error=None,
            started_at=now_utc(),
            ended_at=None,
            duration_ms=None,
            session_id=session_id,
            trace=[],
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionCall":
        return cls(
            id=data["id"],
            function_name=data["function_name"],
            inputs=data.get("inputs", {}),
            output=data.get("output"),
            error=data.get("error"),
            started_at=_parse_datetime(data.get("started_at")) or now_utc(),
            ended_at=_parse_datetime(data.get("ended_at")),
            duration_ms=data.get("duration_ms"),
            session_id=data.get("session_id"),
            trace=[TraceEvent.from_dict(t) for t in data.get("trace", [])],
            metadata=data.get("metadata", {}),
        )


MetricType = Literal["objective", "subjective"]


@dataclass
class MetricSpec:
    id: str
    name: str
    type: MetricType
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    why: str = ""


@dataclass
class MetricResult:
    metric_id: str
    item_id: str
    call_id: str
    score: Optional[float]
    passed: Optional[bool]
    details: Dict[str, Any] = field(default_factory=dict)
    raw_judge: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricResult":
        return cls(
            metric_id=data["metric_id"],
            item_id=data["item_id"],
            call_id=data["call_id"],
            score=data.get("score"),
            passed=data.get("passed"),
            details=data.get("details", {}),
            raw_judge=data.get("raw_judge"),
        )


@dataclass
class DatasetItem:
    """
    Represents a single evaluation item with 4 core columns:

    - input: User/system input to the agent
    - output: Agent/LLM response (captured from trace)
    - human_label: Human judgement/annotation (optional, for calibration)
    - metadata: Additional info (call_id, trace data, etc.)

    The old 'inputs' and 'expected' fields are kept for backwards compatibility.
    """

    id: str
    input: Dict[str, Any] = field(default_factory=dict)  # User input
    output: Optional[Any] = None  # Agent output
    human_label: Optional[Dict[str, Any]] = None  # Human judgement
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backwards compatibility
    inputs: Dict[str, Any] = field(default_factory=dict)  # Alias for input
    expected: Optional[Any] = None  # Deprecated

    def __post_init__(self):
        # Merge inputs into input for backwards compat
        if self.inputs and not self.input:
            self.input = self.inputs
        elif self.input and not self.inputs:
            self.inputs = self.input

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "human_label": self.human_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DatasetItem":
        # Support both old format (inputs/expected) and new format (input/output)
        input_data = payload.get("input") or payload.get("inputs", {})
        output_data = payload.get("output") or payload.get("expected")

        return cls(
            id=payload.get("id", _default_id()),
            input=input_data,
            output=output_data,
            human_label=payload.get("human_label"),
            metadata=payload.get("metadata", {}),
            inputs=input_data,  # Backwards compat
            expected=output_data,  # Backwards compat
        )

    @classmethod
    def from_call(cls, call: "FunctionCall") -> "DatasetItem":
        """Create a DatasetItem from a traced FunctionCall."""
        return cls(
            id=_default_id(),
            input=call.inputs,
            output=call.output,
            human_label=None,
            metadata={
                "call_id": call.id,
                "function": call.function_name,
                "duration_ms": call.duration_ms,
                "error": call.error,
                "session_id": call.session_id,
            },
            inputs=call.inputs,
            expected=None,
        )


@dataclass
class JudgeConfig:
    id: str
    model: str
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "v0"


@dataclass
class EvalRun:
    id: str
    dataset_name: str
    created_at: datetime
    metric_results: List[MetricResult]
    metrics: List[MetricSpec] = field(default_factory=list)
    judge_configs: List[JudgeConfig] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "created_at": _iso(self.created_at),
            "metric_results": [r.as_dict() for r in self.metric_results],
            "metrics": [asdict(m) for m in self.metrics],
            "judge_configs": [asdict(j) for j in self.judge_configs],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalRun":
        return cls(
            id=data["id"],
            dataset_name=data["dataset_name"],
            created_at=_parse_datetime(data.get("created_at")) or now_utc(),
            metric_results=[
                MetricResult.from_dict(r) for r in data.get("metric_results", [])
            ],
            metrics=[MetricSpec(**m) for m in data.get("metrics", [])],
            judge_configs=[JudgeConfig(**j) for j in data.get("judge_configs", [])],
            summary=data.get("summary", {}),
        )


@dataclass
class HumanLabel:
    """
    Human annotation for calibration.

    Schema for human judgement on (input, output, eval_result) tuples:
    - passed: Overall pass/fail judgement
    - scores: Per-metric human scores (0-1)
    - notes: Free-form notes from annotator
    - annotator: Annotator identifier
    - timestamp: When annotation was made
    """

    passed: bool
    scores: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    annotator: str = ""
    timestamp: datetime = field(default_factory=now_utc)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "scores": self.scores,
            "notes": self.notes,
            "annotator": self.annotator,
            "timestamp": _iso(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanLabel":
        return cls(
            passed=data.get("passed", True),
            scores=data.get("scores", {}),
            notes=data.get("notes", ""),
            annotator=data.get("annotator", ""),
            timestamp=_parse_datetime(data.get("timestamp")) or now_utc(),
        )


@dataclass
class AnnotationItem:
    """
    Export format for human annotation workflow.

    Contains the (input, output, eval_results) tuple for annotators to review.
    """

    id: str
    input: Dict[str, Any]
    output: Any
    eval_results: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # metric_id -> result
    human_label: Optional[HumanLabel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "eval_results": self.eval_results,
            "human_label": self.human_label.as_dict() if self.human_label else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationItem":
        human_label_data = data.get("human_label")
        return cls(
            id=data.get("id", _default_id()),
            input=data.get("input", {}),
            output=data.get("output"),
            eval_results=data.get("eval_results", {}),
            human_label=HumanLabel.from_dict(human_label_data)
            if human_label_data
            else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class MetricLabel:
    """Human label for a specific metric."""

    metric_id: str
    agree_with_llm: bool  # Does human agree with LLM judge?
    human_label: bool  # Human's own judgement (pass/fail)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "agree_with_llm": self.agree_with_llm,
            "human_label": self.human_label,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricLabel":
        return cls(
            metric_id=data.get("metric_id", ""),
            agree_with_llm=data.get("agree_with_llm", True),
            human_label=data.get("human_label", True),
            notes=data.get("notes", ""),
        )


@dataclass
class Annotation:
    """
    Human annotation for an eval item.

    Supports two modes:
    1. Simple mode: Just overall label (bool) for backwards compatibility
    2. Per-metric mode: metric_labels dict with per-metric human judgements

    confidence: 1-5 scale (1=very uncertain, 5=very confident)
    """

    id: str
    target_id: str
    label: Any  # Overall pass/fail (bool) - for backwards compat
    rationale: Optional[str]
    annotator: str
    source: str = "human"
    confidence: Optional[int] = None  # 1-5 scale
    metric_labels: Dict[str, MetricLabel] = field(
        default_factory=dict
    )  # metric_id -> MetricLabel
    created_at: datetime = field(default_factory=now_utc)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target_id": self.target_id,
            "label": self.label,
            "rationale": self.rationale,
            "annotator": self.annotator,
            "source": self.source,
            "confidence": self.confidence,
            "metric_labels": {k: v.as_dict() for k, v in self.metric_labels.items()},
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        metric_labels_raw = data.get("metric_labels", {})
        metric_labels = {}
        for k, v in metric_labels_raw.items():
            if isinstance(v, dict):
                metric_labels[k] = MetricLabel.from_dict(v)

        return cls(
            id=data["id"],
            target_id=data["target_id"],
            label=data.get("label"),
            rationale=data.get("rationale"),
            annotator=data.get("annotator", "unknown"),
            source=data.get("source", "human"),
            confidence=data.get("confidence"),
            metric_labels=metric_labels,
            created_at=_parse_datetime(data.get("created_at")) or now_utc(),
        )


@dataclass
class CalibrationRecord:
    id: str
    judge_config_id: str
    gold_items: List[str]
    adjustments: Dict[str, Any]
    created_at: datetime = field(default_factory=now_utc)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "judge_config_id": self.judge_config_id,
            "gold_items": self.gold_items,
            "adjustments": self.adjustments,
            "created_at": _iso(self.created_at),
        }
