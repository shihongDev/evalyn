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
    id: str
    inputs: Dict[str, Any]
    expected: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DatasetItem":
        return cls(
            id=payload.get("id", _default_id()),
            inputs=payload.get("inputs", {}),
            expected=payload.get("expected"),
            metadata=payload.get("metadata", {}),
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
            metric_results=[MetricResult.from_dict(r) for r in data.get("metric_results", [])],
            metrics=[MetricSpec(**m) for m in data.get("metrics", [])],
            judge_configs=[JudgeConfig(**j) for j in data.get("judge_configs", [])],
            summary=data.get("summary", {}),
        )


@dataclass
class Annotation:
    id: str
    target_id: str
    label: Any
    rationale: Optional[str]
    annotator: str
    source: str = "human"
    confidence: Optional[float] = None
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
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        return cls(
            id=data["id"],
            target_id=data["target_id"],
            label=data.get("label"),
            rationale=data.get("rationale"),
            annotator=data.get("annotator", "unknown"),
            source=data.get("source", "human"),
            confidence=data.get("confidence"),
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
