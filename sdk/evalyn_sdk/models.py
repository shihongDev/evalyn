from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from collections.abc import Callable
import builtins


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_datetime(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


def _default_id() -> str:
    return str(uuid.uuid4())


def _safe_details(data: dict[str, Any] | None) -> dict[str, Any]:
    return data or {}


# Span types for hierarchical tracing
SpanType = Literal[
    "session",  # Root session span
    "graph",  # LangGraph execution
    "node",  # LangGraph node
    "llm_call",  # LLM API call
    "tool_call",  # Tool/function call
    "retrieval",  # RAG retrieval
    "scorer",  # Metric evaluation
    "agent",  # Agent execution (Google ADK, Anthropic Agents, etc.)
    "custom",  # User-defined span
    # Semantic span kinds for fine-grained evaluation
    "input_message",  # User/system message input
    "output_message",  # Assistant message output
    "tool_use",  # Tool invocation request
    "tool_result",  # Tool execution result
]

SpanStatus = Literal["ok", "error", "running"]

# Evaluation unit types for span-level evaluation
EvalUnitType = Literal[
    "outcome",  # Full trace outcome (default, backward-compatible)
    "single_turn",  # Single LLM call: input -> output
    "tool_use",  # Tool invocation: request -> result
    "multi_turn",  # Consecutive exchanges in a conversation
    "custom",  # User-defined evaluation boundary
]


@dataclass
class Span:
    """
    A hierarchical span with parent-child relationships.

    Enables Phoenix/LangSmith-style trace visualization:

    Trace (run_agent)
     └── graph.execution
         ├── node (generate_query)
         │    └── llm_call (gemini-2.5-flash)
         └── node (finalize_answer)
              └── llm_call (gemini-2.5-flash)
     ├── scorer (helpfulness) ✓
     └── scorer (hallucination) ✗
    """

    id: str
    name: str
    span_type: str  # SpanType
    parent_id: str | None  # Parent span ID (None for root)
    start_time: datetime
    end_time: datetime | None = None
    status: str = "ok"  # SpanStatus
    attributes: dict[str, Any] = field(default_factory=dict)

    # Computed properties
    @property
    def duration_ms(self) -> float | None:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "span_type": self.span_type,
            "parent_id": self.parent_id,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "status": self.status,
            "attributes": self.attributes,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        return cls(
            id=data["id"],
            name=data["name"],
            span_type=data.get("span_type", "custom"),
            parent_id=data.get("parent_id"),
            start_time=_parse_datetime(data.get("start_time")) or now_utc(),
            end_time=_parse_datetime(data.get("end_time")),
            status=data.get("status", "ok"),
            attributes=data.get("attributes", {}),
        )

    @classmethod
    def new(
        cls,
        name: str,
        span_type: str,
        parent_id: str | None = None,
        **attributes: Any,
    ) -> Span:
        """Create a new span with auto-generated ID."""
        return cls(
            id=_default_id(),
            name=name,
            span_type=span_type,
            parent_id=parent_id,
            start_time=now_utc(),
            status="running",
            attributes=attributes,
        )

    def finish(self, status: str = "ok", **extra_attributes: Any) -> Span:
        """Mark span as finished."""
        self.end_time = now_utc()
        self.status = status
        self.attributes.update(extra_attributes)
        return self


@dataclass
class TraceEvent:
    kind: str
    timestamp: datetime
    detail: dict[str, Any] = field(default_factory=dict)
    span_id: str | None = None  # Link to associated Span
    parent_span_id: str | None = None  # Parent span for hierarchy

    def as_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "timestamp": _iso(self.timestamp),
            "detail": self.detail,
        }
        if self.span_id:
            result["span_id"] = self.span_id
        if self.parent_span_id:
            result["parent_span_id"] = self.parent_span_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        return cls(
            kind=data["kind"],
            timestamp=_parse_datetime(data["timestamp"]) or now_utc(),
            detail=_safe_details(data.get("detail")),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
        )


@dataclass
class FunctionCall:
    id: str
    function_name: str
    inputs: dict[str, Any]
    output: Any
    error: str | None
    started_at: datetime
    ended_at: datetime | None
    duration_ms: float | None
    session_id: str | None
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Hierarchical span support
    parent_call_id: str | None = None  # Parent @eval call (for nested calls)
    spans: list[Span] = field(default_factory=list)  # Hierarchical span tree

    def as_dict(self) -> dict[str, Any]:
        result = {
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
        if self.parent_call_id:
            result["parent_call_id"] = self.parent_call_id
        if self.spans:
            result["spans"] = [s.as_dict() for s in self.spans]
        return result

    @classmethod
    def new(
        cls,
        function_name: str,
        inputs: dict[str, Any],
        session_id: str | None,
        metadata: dict[str, Any] | None = None,
        parent_call_id: str | None = None,
    ) -> FunctionCall:
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
            parent_call_id=parent_call_id,
            spans=[],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunctionCall:
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
            parent_call_id=data.get("parent_call_id"),
            spans=[Span.from_dict(s) for s in data.get("spans", [])],
        )

    def add_span(self, span: Span) -> None:
        """Add a span to this call's span tree."""
        self.spans.append(span)

    def get_span_tree(self) -> dict[str, Any]:
        """Build hierarchical span tree for visualization."""
        # Build lookup by id
        by_id = {s.id: {"span": s, "children": []} for s in self.spans}
        roots = []

        for s in self.spans:
            node = by_id[s.id]
            if s.parent_id and s.parent_id in by_id:
                by_id[s.parent_id]["children"].append(node)
            else:
                roots.append(node)

        return {"roots": roots, "by_id": by_id}


@dataclass
class EvalUnit:
    """
    An evaluatable unit discovered from trace structure.

    Units can represent different evaluation granularities:
    - outcome: Full trace (backward-compatible default)
    - single_turn: Single LLM call with input/output
    - tool_use: Tool invocation with request/result
    - multi_turn: Sequence of exchanges
    - custom: User-defined boundary
    """

    id: str
    unit_type: str  # EvalUnitType
    call_id: str  # Parent FunctionCall ID
    span_ids: list[str]  # Spans comprising this unit
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "unit_type": self.unit_type,
            "call_id": self.call_id,
            "span_ids": list(self.span_ids),
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalUnit:
        return cls(
            id=data["id"],
            unit_type=data["unit_type"],
            call_id=data["call_id"],
            span_ids=data.get("span_ids", []),
            context=data.get("context", {}),
        )


@dataclass
class EvalView:
    """
    Projected view of an EvalUnit for metric evaluation.

    Provides a normalized interface regardless of unit type,
    allowing metrics to evaluate different granularities uniformly.
    """

    unit_id: str
    unit_type: str
    input: Any  # Projected input (varies by unit type)
    output: Any  # Projected output (varies by unit type)
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "input": self.input,
            "output": self.output,
            "context": dict(self.context),
        }


MetricType = Literal["objective", "subjective"]


@dataclass
class MetricSpec:
    id: str
    name: str
    type: MetricType
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    why: str = ""
    # Unit types this metric can evaluate (default: ["outcome"] for backward compat)
    unit_types: list[str] = field(default_factory=lambda: ["outcome"])

    @property
    def version_hash(self) -> str:
        """Deterministic hash of the metric's identity-defining fields.

        Changes when the metric's prompt, rubric, threshold, or type changes.
        Used for detecting metric version drift between evaluation runs.
        Cached after first computation.
        """
        cached = getattr(self, "_version_hash_cache", None)
        if cached is not None:
            return cached
        import hashlib
        import json
        identity = {
            "id": self.id,
            "type": self.type,
            "config": self.config or {},
        }
        content = json.dumps(identity, sort_keys=True, ensure_ascii=True)
        result = hashlib.sha256(content.encode()).hexdigest()[:16]
        object.__setattr__(self, "_version_hash_cache", result)
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "config": dict(self.config or {}),
            "why": self.why,
            "unit_types": list(self.unit_types or ["outcome"]),
            "version_hash": self.version_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricSpec:
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            type=data.get("type", "objective"),
            description=data.get("description", ""),
            config=data.get("config", {}),
            why=data.get("why", ""),
            unit_types=data.get("unit_types", ["outcome"]),
        )


@dataclass
class MetricResult:
    metric_id: str
    item_id: str
    call_id: str
    score: float | None
    passed: bool | None
    details: dict[str, Any] = field(default_factory=dict)
    raw_judge: dict[str, Any] | None = None
    # Token usage for subjective metrics (LLM judge)
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    # Unit-level evaluation fields (optional, for span-level evals)
    unit_id: str | None = None
    unit_type: str | None = None  # EvalUnitType
    span_ids: list[str] | None = None
    # Metric version tracking
    config_hash: str | None = None  # MetricSpec.version_hash at evaluation time

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "item_id": self.item_id,
            "call_id": self.call_id,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "raw_judge": self.raw_judge,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "span_ids": self.span_ids,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        return cls(
            metric_id=data["metric_id"],
            item_id=data["item_id"],
            call_id=data["call_id"],
            score=data.get("score"),
            passed=data.get("passed"),
            details=data.get("details", {}),
            raw_judge=data.get("raw_judge"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            model=data.get("model"),
            unit_id=data.get("unit_id"),
            unit_type=data.get("unit_type"),
            span_ids=data.get("span_ids"),
            config_hash=data.get("config_hash"),
        )


@dataclass
class SpanMetricLink:
    """Links a metric result to a specific span with relevance scoring."""

    id: str
    metric_result_id: str  # composite: metric_id:item_id:call_id
    span_id: str
    relevance: float  # 0.0-1.0
    reason: str
    run_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "metric_result_id": self.metric_result_id,
            "span_id": self.span_id,
            "relevance": self.relevance,
            "reason": self.reason,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanMetricLink:
        return cls(
            id=data["id"],
            metric_result_id=data["metric_result_id"],
            span_id=data["span_id"],
            relevance=data.get("relevance", 0.0),
            reason=data.get("reason", ""),
            run_id=data["run_id"],
        )


class Metric:
    """Runtime metric that binds a spec to an evaluation function."""

    def __init__(
        self,
        spec: MetricSpec,
        handler: Callable[[FunctionCall, DatasetItem], MetricResult],
        unit_handler: Callable[[EvalView, DatasetItem], MetricResult] | None = None,
    ):
        self.spec = spec
        self.handler = handler
        self.unit_handler = unit_handler

    def evaluate(self, call: FunctionCall, item: DatasetItem) -> MetricResult:
        return self.handler(call, item)

    def evaluate_unit(self, view: EvalView, item: DatasetItem) -> MetricResult:
        """Evaluate a unit view. Falls back to handler with synthetic call if no unit_handler."""
        if self.unit_handler:
            return self.unit_handler(view, item)
        # Fallback: create synthetic call from view for backward compat
        synthetic_call = FunctionCall(
            id=view.unit_id,
            function_name="unit_eval",
            inputs={"input": view.input},
            output=view.output,
            error=None,
            started_at=now_utc(),
            ended_at=now_utc(),
            duration_ms=0,
            session_id=None,
        )
        return self.handler(synthetic_call, item)

    def supports_unit_type(self, unit_type: str) -> bool:
        """Check if this metric supports a given unit type."""
        return unit_type in self.spec.unit_types


class CompositeMetric(Metric):
    """A metric that combines multiple child metrics into a weighted score.

    Aggregation strategies:
    - weighted_average: weighted mean of child scores
    - min: minimum child score (strictest)
    - max: maximum child score (most lenient)
    - all_pass: 1.0 if all children pass, 0.0 otherwise

    Example:
        composite = CompositeMetric(
            metric_id="quality_score",
            children=[(helpfulness_metric, 0.4), (safety_metric, 0.3), (accuracy_metric, 0.3)],
            aggregation="weighted_average",
            threshold=0.7,
        )
    """

    def __init__(
        self,
        metric_id: str,
        children: list[tuple],  # [(Metric, weight), ...]
        aggregation: str = "weighted_average",
        threshold: float = 0.5,
        description: str = "",
    ):
        valid_aggregations = {"weighted_average", "min", "max", "all_pass"}
        if aggregation not in valid_aggregations:
            raise ValueError(f"aggregation must be one of {valid_aggregations}, got {aggregation!r}")

        self.children = children
        self.aggregation = aggregation
        self.threshold = threshold

        spec = MetricSpec(
            id=metric_id,
            name=f"Composite: {metric_id}",
            type="objective",
            description=description or f"Composite metric ({aggregation}) of {len(children)} children",
        )

        def composite_handler(call: FunctionCall, item: DatasetItem) -> MetricResult:
            child_results = []
            for child_metric, weight in self.children:
                result = child_metric.evaluate(call, item)
                child_results.append((result, weight))

            score = self._aggregate(child_results)
            passed = score >= self.threshold

            return MetricResult(
                metric_id=spec.id,
                item_id=item.id,
                call_id=call.id,
                score=score,
                passed=passed,
                details={
                    "aggregation": self.aggregation,
                    "threshold": self.threshold,
                    "children": [
                        {
                            "metric_id": r.metric_id,
                            "score": r.score,
                            "passed": r.passed,
                            "weight": w,
                        }
                        for r, w in child_results
                    ],
                },
            )

        super().__init__(spec, composite_handler)

    def _aggregate(self, child_results: list[tuple]) -> float:
        """Compute aggregate score from child results."""
        scores = [(r.score or 0.0, w) for r, w in child_results]

        if self.aggregation == "weighted_average":
            total_weight = sum(w for _, w in scores)
            if total_weight == 0:
                return 0.0
            return sum(s * w for s, w in scores) / total_weight

        elif self.aggregation == "min":
            return min(s for s, _ in scores) if scores else 0.0

        elif self.aggregation == "max":
            return max(s for s, _ in scores) if scores else 0.0

        elif self.aggregation == "all_pass":
            all_passed = all(r.passed for r, _ in child_results)
            return 1.0 if all_passed else 0.0

        return 0.0


class MetricRegistry:
    """Registry for managing multiple metrics."""

    def __init__(self):
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        self._metrics[metric.spec.id] = metric

    def get(self, metric_id: str) -> Metric | None:
        return self._metrics.get(metric_id)

    def list(self) -> builtins.list[Metric]:
        return list(self._metrics.values())

    def apply_all(
        self, call: FunctionCall, item: DatasetItem
    ) -> builtins.list[MetricResult]:
        return [metric.evaluate(call, item) for metric in self._metrics.values()]


@dataclass
class DatasetItem:
    """
    Represents a single evaluation item with 4 core columns:

    - input: User/system input to the agent
    - output: Agent/LLM response (captured from trace)
    - human_label: Human judgement/annotation (optional, for calibration)
    - metadata: Additional info (call_id, trace data, etc.)

    The 'inputs' and 'expected' properties are kept for backwards compatibility
    and always delegate to the canonical 'input' and 'output' fields.
    """

    id: str
    input: dict[str, Any] = field(default_factory=dict)  # User input
    output: Any | None = None  # Agent output
    human_label: dict[str, Any] | None = None  # Human judgement
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> dict[str, Any]:
        """Backward-compat alias for input."""
        return self.input

    @inputs.setter
    def inputs(self, value: dict[str, Any]) -> None:
        self.input = value

    @property
    def expected(self) -> Any | None:
        """Backward-compat alias for output."""
        return self.output

    @expected.setter
    def expected(self, value: Any) -> None:
        self.output = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "human_label": self.human_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DatasetItem:
        # Support both old format (inputs/expected) and new format (input/output)
        input_data = payload.get("input") or payload.get("inputs", {})
        output_data = payload.get("output") or payload.get("expected")

        return cls(
            id=payload.get("id", _default_id()),
            input=input_data,
            output=output_data,
            human_label=payload.get("human_label"),
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_call(cls, call: FunctionCall) -> DatasetItem:
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
        )


@dataclass
class JudgeConfig:
    id: str
    model: str
    prompt: str
    parameters: dict[str, Any] = field(default_factory=dict)
    version: str = "v0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "prompt": self.prompt,
            "parameters": dict(self.parameters or {}),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeConfig:
        return cls(
            id=data["id"],
            model=data["model"],
            prompt=data.get("prompt", ""),
            parameters=data.get("parameters", {}),
            version=data.get("version", "v0"),
        )


@dataclass
class EvalRun:
    id: str
    dataset_name: str
    created_at: datetime
    metric_results: list[MetricResult]
    metrics: list[MetricSpec] = field(default_factory=list)
    judge_configs: list[JudgeConfig] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    # Token usage summary for LLM judge evaluations
    usage_summary: dict[str, Any] = field(default_factory=dict)
    # Run management
    name: str | None = None
    pinned: bool = False
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "created_at": _iso(self.created_at),
            "metric_results": [r.as_dict() for r in self.metric_results],
            "metrics": [m.as_dict() for m in self.metrics],
            "judge_configs": [j.as_dict() for j in self.judge_configs],
            "summary": self.summary,
            "usage_summary": self.usage_summary,
        }
        if self.name:
            d["name"] = self.name
        if self.pinned:
            d["pinned"] = self.pinned
        if self.tags:
            d["tags"] = list(self.tags)
        return d

    def has_tag(self, tag: str) -> bool:
        """Check if this run has a specific tag."""
        return tag in self.tags

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRun:
        return cls(
            id=data["id"],
            dataset_name=data["dataset_name"],
            created_at=_parse_datetime(data.get("created_at")) or now_utc(),
            metric_results=[
                MetricResult.from_dict(r) for r in data.get("metric_results", [])
            ],
            metrics=[MetricSpec.from_dict(m) for m in data.get("metrics", [])],
            judge_configs=[JudgeConfig.from_dict(j) for j in data.get("judge_configs", [])],
            summary=data.get("summary", {}),
            usage_summary=data.get("usage_summary", {}),
            name=data.get("name"),
            pinned=data.get("pinned", False),
            tags=data.get("tags", []),
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
    scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    annotator: str = ""
    timestamp: datetime = field(default_factory=now_utc)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scores": self.scores,
            "notes": self.notes,
            "annotator": self.annotator,
            "timestamp": _iso(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanLabel:
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
    input: dict[str, Any]
    output: Any
    eval_results: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # metric_id -> result
    human_label: HumanLabel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "eval_results": self.eval_results,
            "human_label": self.human_label.as_dict() if self.human_label else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationItem:
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "agree_with_llm": self.agree_with_llm,
            "human_label": self.human_label,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricLabel:
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
    rationale: str | None
    annotator: str
    source: str = "human"
    confidence: int | None = None  # 1-5 scale
    metric_labels: dict[str, MetricLabel] = field(
        default_factory=dict
    )  # metric_id -> MetricLabel
    created_at: datetime = field(default_factory=now_utc)

    def as_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        metric_labels_raw = data.get("metric_labels", {})
        metric_labels = {
            k: MetricLabel.from_dict(v)
            for k, v in metric_labels_raw.items()
            if isinstance(v, dict)
        }
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
    gold_items: list[str]
    adjustments: dict[str, Any]
    created_at: datetime = field(default_factory=now_utc)
    # Token usage summary for calibration (LLM optimizer calls)
    usage_summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "judge_config_id": self.judge_config_id,
            "gold_items": self.gold_items,
            "adjustments": self.adjustments,
            "created_at": _iso(self.created_at),
            "usage_summary": self.usage_summary,
        }
