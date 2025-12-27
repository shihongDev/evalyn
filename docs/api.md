# API Reference

## Decorator

### @eval

```python
from evalyn_sdk import eval

@eval(
    project="myapp",           # Project name
    version="v1",              # Version tag
    is_simulation=False,       # True for test runs
    name=None,                 # Custom function name
)
def my_agent(query: str) -> str:
    return process(query)
```

Supports sync and async functions.

### eval_session

```python
from evalyn_sdk import eval_session

with eval_session(session_id="user-123"):
    my_agent("query 1")
    my_agent("query 2")
```

Groups related calls with a session ID.

## Tracer

### EvalTracer

```python
from evalyn_sdk import EvalTracer, configure_tracer

tracer = EvalTracer(storage=my_storage)
configure_tracer(tracer)
```

### get_default_tracer

```python
from evalyn_sdk import get_default_tracer

tracer = get_default_tracer()
```

## Datasets

### load_dataset / save_dataset

```python
from evalyn_sdk import load_dataset, save_dataset

items = load_dataset("data/myapp/dataset.jsonl")
save_dataset(items, "output.jsonl")
```

### build_dataset_from_storage

```python
from evalyn_sdk import build_dataset_from_storage
from evalyn_sdk.storage import get_default_storage

storage = get_default_storage()
items = build_dataset_from_storage(
    storage,
    project_name="myapp",
    version="v1",
    production_only=True,
    limit=500
)
```

## Metrics

### MetricRegistry

```python
from evalyn_sdk import MetricRegistry

registry = MetricRegistry()
registry.add_from_file("metrics.json")

for spec in registry.list_metrics():
    print(spec.id, spec.type)
```

### Built-in Metrics

```python
from evalyn_sdk import (
    exact_match_metric,
    latency_metric,
    json_valid_metric,
    subjective_metric,
)
```

### Build from Templates

```python
from evalyn_sdk import build_objective_metric, build_subjective_metric

latency = build_objective_metric("latency_ms")
helpful = build_subjective_metric("helpfulness_accuracy")
```

## Evaluation

### EvalRunner

```python
from evalyn_sdk import EvalRunner, MetricRegistry, load_dataset

registry = MetricRegistry()
registry.add_from_file("metrics.json")

items = load_dataset("dataset.jsonl")
runner = EvalRunner(registry=registry)
eval_run = runner.run(items, target_fn=my_agent)

print(eval_run.summary)
for result in eval_run.results:
    print(result.metric_id, result.score, result.passed)
```

## Calibration

### CalibrationEngine

```python
from evalyn_sdk import CalibrationEngine, load_dataset
from evalyn_sdk.storage import get_default_storage

storage = get_default_storage()
dataset = load_dataset("dataset.jsonl")

engine = CalibrationEngine(storage=storage, dataset_items=dataset)
record = engine.calibrate(
    metric_id="helpfulness_accuracy",
    annotations=annotations,
    use_optimizer=True,
    optimizer="gepa"
)

# Check validation
validation = record.adjustments.get("validation", {})
if validation.get("is_better"):
    print(f"Improved by {validation['improvement_delta']:.1%}")
```

### load_optimized_prompt

```python
from evalyn_sdk import load_optimized_prompt

prompt = load_optimized_prompt("data/myapp", "helpfulness_accuracy")
if prompt:
    print("Using optimized prompt")
```

## Simulation

### UserSimulator

```python
from evalyn_sdk.simulator import UserSimulator

simulator = UserSimulator(model="gemini-2.5-flash-lite")
similar = simulator.generate_similar(seeds, num_per_seed=3)
outliers = simulator.generate_outliers(seeds, num_per_seed=2)
```

### AgentSimulator

```python
from evalyn_sdk.simulator import AgentSimulator, SimulationConfig

config = SimulationConfig(
    num_similar_per_seed=3,
    num_outlier_per_seed=2,
    max_seeds=20
)

agent_sim = AgentSimulator(
    target_fn=my_agent,
    simulator=simulator,
    config=config
)
results = agent_sim.run(seeds)
```

## Models

### DatasetItem

```python
from evalyn_sdk import DatasetItem

item = DatasetItem(
    id="abc123",
    input={"query": "What is AI?"},
    output="AI is...",
    human_label=None,
    metadata={"project": "myapp"}
)
```

### MetricSpec

```python
from evalyn_sdk import MetricSpec, MetricType

spec = MetricSpec(
    id="my_metric",
    name="My Metric",
    type=MetricType.OBJECTIVE,
    description="Description",
    config={}
)
```

### MetricResult

```python
from evalyn_sdk import MetricResult

result = MetricResult(
    metric_id="my_metric",
    score=0.85,
    passed=True,
    details={"reason": "Good"}
)
```

## Judges

### GeminiJudge

```python
from evalyn_sdk import GeminiJudge

judge = GeminiJudge(
    name="helpfulness",
    prompt="Evaluate if the response is helpful...",
    model="gemini-2.5-flash-lite"
)

result = judge.evaluate(input_text, output_text)
```

### OpenAIJudge

```python
from evalyn_sdk import OpenAIJudge

judge = OpenAIJudge(
    name="toxicity",
    prompt="Check for toxic content...",
    model="gpt-4"
)
```
