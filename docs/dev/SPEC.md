# Span-Level Evaluation: EvalUnit Architecture

Trace-shape-driven evaluation. Discover evaluatable units from trace structure, then apply rubrics.

## Project Status

- 559 completed roadmap items
- 1500+ Python modules across the SDK
- 460+ test files
- CLI: 34 commands covering tracing, datasets, metrics, evaluation, annotation, calibration, simulation, and infrastructure
- Optimizer algorithms: Basic, GEPA, GEPA-Native, OPRO, APE, EvoPrompt, TextGrad, MIPROv2, PromptBreeder
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
    "session",         # Root session span
    "graph",           # LangGraph execution
    "node",            # LangGraph node
    "llm_call",        # LLM API call
    "tool_call",       # Tool/function call
    "retrieval",       # RAG retrieval
    "scorer",          # Metric evaluation
    "agent",           # Agent execution (Google ADK, Anthropic Agents, etc.)
    "custom",          # User-defined span
    "input_message",   # User/system message input
    "output_message",  # Assistant message output
    "tool_use",        # Tool invocation request
    "tool_result",     # Tool execution result
]
```

---

## 2. EvalUnit Data Model

**models.py**:

```python
EvalUnitType = Literal[
    "outcome",      # Full trace outcome (default, backward-compatible)
    "single_turn",  # Single LLM call: input -> output
    "tool_use",     # Tool invocation: request -> result
    "multi_turn",   # Consecutive exchanges in a conversation
    "custom",       # User-defined evaluation boundary
]

@dataclass
class EvalUnit:
    id: str
    unit_type: str  # EvalUnitType
    call_id: str    # Parent FunctionCall ID
    span_ids: List[str]  # Spans comprising this unit
    context: Dict[str, Any] = field(default_factory=dict)
```

---

## 3. EvalUnitBuilder ABC

**sdk/evalyn_sdk/evaluation/units/builders.py**

```python
class EvalUnitBuilder(ABC):
    @property
    @abstractmethod
    def unit_type(self) -> str: ...

    @abstractmethod
    def discover(self, call: FunctionCall) -> List[EvalUnit]: ...

class OutcomeBuilder:       # Full trace outcome (DEFAULT)
class SingleTurnBuilder:    # EvalUnit per llm_call span
class ToolUseBuilder:       # Tool invocation request/result
class MultiTurnBuilder:     # Group llm_call spans by parent into conversation
class CustomBuilder:        # User-defined eval boundaries via span attributes
```

---

## 4. EvalView Projection

**sdk/evalyn_sdk/evaluation/units/views.py**

```python
@dataclass
class EvalView:
    unit_id: str
    unit_type: str
    input: Any              # Projected input (varies by unit type)
    output: Any             # Projected output (varies by unit type)
    context: Dict[str, Any] = field(default_factory=dict)

# Standalone projection function (not a classmethod):
def project_unit(unit: EvalUnit, call: FunctionCall) -> EvalView: ...
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

    def evaluate_unit(self, view: EvalView, item: DatasetItem) -> MetricResult:  # NEW
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
--unit-types TYPE...     # outcome, single_turn, tool_use, multi_turn, custom (comma-separated)
```

Examples:
```bash
evalyn run-eval data.jsonl -m metrics.json --unit-types single_turn
evalyn run-eval data.jsonl -m metrics.json --unit-types tool_use
evalyn run-eval data.jsonl -m metrics.json --unit-types outcome,single_turn
```

---

## 8. Metrics JSON Extension

```json
[
  {"id": "helpfulness", "type": "subjective", "unit_types": ["outcome"], ...},
  {"id": "llm_quality", "type": "subjective", "unit_types": ["single_turn"], ...},
  {"id": "tool_use_quality", "type": "objective", "unit_types": ["tool_use"], ...}
]
```

---

## 9. Files to Create/Modify

| File | Change |
|------|--------|
| sdk/evalyn_sdk/models.py | Extend SpanType, add EvalUnitType, EvalUnit, EvalView |
| sdk/evalyn_sdk/models.py | Extend Metric (unit_types), MetricResult (+3 Optional) |
| sdk/evalyn_sdk/evaluation/units/__init__.py | Package |
| sdk/evalyn_sdk/evaluation/units/builders.py | EvalUnitBuilder ABC + impls |
| sdk/evalyn_sdk/evaluation/units/views.py | EvalView projection (project_unit) |
| sdk/evalyn_sdk/evaluation/runner.py | unit_builders, _discover_units(), updated loop |
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
4. CLI test: `--unit-types single_turn`
5. Backwards compat: existing eval runs produce identical results
