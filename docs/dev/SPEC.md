# Span-Level Evaluation: EvalUnit Architecture

Trace-shape-driven evaluation. Discover evaluatable units from trace structure, then apply rubrics.

## Project Status

- 559 completed roadmap items
- 1500+ Python modules across the SDK
- 460+ test files
- CLI: 35+ commands covering tracing, datasets, metrics, evaluation, annotation, calibration, simulation, and infrastructure
- Optimizer algorithms: LLM, GEPA-Native, OPRO, APE, CAPO
- Standalone modules for doctor, playground, dashboard export, quickstart templates, and more

## Core Concept

```
FunctionCall.spans
       |
EvalUnitBuilder.build()  -> List[EvalUnit]
       |
Rubric.evaluate(unit)    -> EvalResult
```

Key insight: System discovers "what CAN be evaluated" rather than "run metric X on span Y".

---

## 1. Extended Semantic Span Kinds

**models.py** - Extend SpanType:

```python
SpanType = Literal[
    # Existing
    "session", "graph", "node", "llm_call", "tool_call",
    "retrieval", "scorer", "agent", "custom",
    # NEW
    "message",        # Conversation turn
    "state_snapshot", # Agent state checkpoint
    "outcome",        # Final result/decision
    "decision",       # Agent decision point
]
```

---

## 2. EvalUnit Data Model

**models.py**:

```python
EvalUnitType = Literal[
    "single_turn",  # Single LLM input->output
    "multi_turn",   # Conversation sequence
    "trajectory",   # Action sequence (tool calls)
    "outcome",      # Final result evaluation
    "subgraph",     # Nested agent/graph
]

@dataclass
class EvalUnit:
    id: str
    unit_type: EvalUnitType
    call_id: str
    span_ids: List[str]
    input_view: Any
    output_view: Any
    context: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    parent_unit_id: Optional[str] = None

    @classmethod
    def from_span(cls, span: Span, call_id: str) -> "EvalUnit": ...
```

---

## 3. EvalUnitBuilder Protocol

**NEW: sdk/evalyn_sdk/eval_units/builders.py**

```python
class EvalUnitBuilder(Protocol):
    unit_type: str
    def build(self, call: FunctionCall) -> List[EvalUnit]: ...

class SingleTurnBuilder:    # EvalUnit per llm_call span
class MultiTurnBuilder:     # Group message spans into conversation
class TrajectoryBuilder:    # Sequence of tool calls
class OutcomeBuilder:       # Final call result (DEFAULT)
class SubgraphBuilder:      # Nested agent/graph executions
```

---

## 4. EvalView Projection

**NEW: sdk/evalyn_sdk/eval_units/views.py**

```python
@dataclass
class EvalView:
    unit_id: str
    unit_type: str
    input_text: str         # Flattened for LLM judge
    output_text: str
    input_structured: Any   # For objective metrics
    output_structured: Any
    conversation_history: List[Dict] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_unit(cls, unit: EvalUnit, call: FunctionCall) -> "EvalView": ...
```

---

## 5. Extended Metric & MetricResult

**models.py**:

```python
class Metric:
    def __init__(self, spec, handler, unit_types: List[str] = None):
        self.unit_types = unit_types or ["outcome"]  # Default: call-level

    def evaluate(self, call, item) -> MetricResult:  # KEEP existing
        ...

    def evaluate_unit(self, unit: EvalUnit, view: EvalView) -> MetricResult:  # NEW
        ...

@dataclass
class MetricResult:
    # ... existing fields unchanged ...
    unit_id: Optional[str] = None      # NEW
    unit_type: Optional[str] = None    # NEW
    span_ids: List[str] = field(default_factory=list)  # NEW
```

---

## 6. Runner Integration

**runner.py**:

```python
class EvalRunner:
    def __init__(self, ..., unit_builders: List[EvalUnitBuilder] = None):
        self.unit_builders = unit_builders or [OutcomeBuilder()]

    def _discover_units(self, call: FunctionCall) -> List[EvalUnit]:
        units = []
        for builder in self.unit_builders:
            units.extend(builder.build(call))
        return units

    def run_dataset(self, dataset, ...) -> EvalRun:
        for item, call in prepared:
            units = self._discover_units(call)
            for unit in units:
                view = EvalView.from_unit(unit, call)
                for metric in self.metrics:
                    if unit.unit_type in (metric.unit_types or ["outcome"]):
                        result = metric.evaluate_unit(unit, view)
                        results.append(result)
```

---

## 7. CLI Flags

**evaluation.py**:

```
--unit-types TYPE...     # single_turn multi_turn trajectory outcome
--builders BUILDER...    # SingleTurnBuilder TrajectoryBuilder etc
--span-types TYPE...     # For SingleTurnBuilder: which spans to evaluate
```

Examples:
```bash
evalyn run-eval data.jsonl -m metrics.json --unit-types single_turn --span-types llm_call
evalyn run-eval data.jsonl -m metrics.json --unit-types trajectory
evalyn run-eval data.jsonl -m metrics.json --unit-types outcome single_turn
```

---

## 8. Metrics JSON Extension

```json
[
  {"id": "helpfulness", "type": "subjective", "unit_types": ["outcome"], ...},
  {"id": "llm_quality", "type": "subjective", "unit_types": ["single_turn"], ...},
  {"id": "trajectory_efficiency", "type": "objective", "unit_types": ["trajectory"], ...}
]
```

---

## 9. Files to Create/Modify

| File | Change |
|------|--------|
| sdk/evalyn_sdk/models.py | Extend SpanType, add EvalUnitType, EvalUnit, EvalView |
| sdk/evalyn_sdk/models.py | Extend Metric (unit_types), MetricResult (+3 Optional) |
| sdk/evalyn_sdk/eval_units/__init__.py | NEW: Package |
| sdk/evalyn_sdk/eval_units/builders.py | NEW: EvalUnitBuilder + impls |
| sdk/evalyn_sdk/eval_units/views.py | NEW: EvalView projection |
| sdk/evalyn_sdk/runner.py | unit_builders, _discover_units(), updated loop |
| sdk/evalyn_sdk/metrics/factory.py | Handle unit_types |
| sdk/evalyn_sdk/cli/commands/evaluation.py | New CLI flags |

---

## 10. Backwards Compatibility (GUARANTEED)

- Default `unit_builders = [OutcomeBuilder()]` = existing behavior
- Metrics without `unit_types` default to `["outcome"]`
- New MetricResult fields are Optional with defaults
- Existing handler signature unchanged
- Existing metrics.json files work without modification

---

## 11. Implementation Order

1. Extend SpanType with new semantic kinds
2. Add EvalUnit, EvalUnitType, EvalView to models.py
3. Create eval_units/ package with builders
4. Extend Metric and MetricResult
5. Update EvalRunner with unit discovery
6. Add CLI flags
7. Write tests
8. Update docs

---

## 12. Verification

1. Unit tests for each EvalUnitBuilder
2. Test EvalView projection from various unit types
3. Integration: trace with 3 LLM calls -> 3 single_turn units -> 3 results
4. CLI test: `--unit-types single_turn --span-types llm_call`
5. Backwards compat: existing eval runs produce identical results
