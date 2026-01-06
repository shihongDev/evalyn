# Evalyn

**Local-first evaluation and calibration framework for LLM agents**

Evalyn focuses on making GenAI App evaluation practical and easy. It provides lightweight tracing, human-in-the-loop annotation, metric suggestion, and calibration workflows to help developers/non-tech folks understand what to evaluate, align metrics with real usage, and continuously improve your GenAI App behavior over time.

## Why Evalyn?

| | |
|---|---|
| **Fully Local** | All data stays on your machine. SQLite storage, no cloud dependencies. |
| **Easy Onboarding** | Just `import evalyn_sdk` — LLM calls auto-captured with tokens & cost. |
| **Metric Bank** | 50+ metrics including both code-based and LLM-based judges templates for quality assessment. |
| **Auto Calibration** | Align LLM judges with human feedback through automatic prompt optimization such as GEPA. |
| **One Command** | Run the entire pipeline with `evalyn one-click`. |

## Install

```bash
pip install -e ".[dev,llm]"
```

## 30-Second Start

```python
from evalyn_sdk import eval

@eval(project="myapp", version="v1")
def my_agent(query: str) -> str:
    return call_llm(query)  # LLM calls auto-captured
```

```bash
export GEMINI_API_KEY="your-key"
python my_agent.py                           # Run agent, traces captured
evalyn one-click --project myapp             # Full evaluation pipeline
```


## The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  1. COLLECT                                                      │
│                                                                  │
│     @eval decorator  →  TRACE (SQLite)  →  DATASET (JSONL)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. EVALUATE & CALIBRATE (iterate until aligned)                 │
│                                                                  │
│          ┌────────────────────────────────────┐                  │
│          │                                    │                  │
│          ▼                                    │                  │
│     EVALUATE  ───→  ANNOTATE  ───→  CALIBRATE │                  │
│    (LLM Judge)      (Human)       (Optimize)  │                  │
│          │                                    │                  │
│          └─────── RE-EVALUATE ◄───────────────┘                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. EXPAND                                                       │
│                                                                  │
│     SIMULATE  →  Generate synthetic queries  →  Back to Step 2   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```



## Sample Workflow

### Step 1: Instrument & Collect
```bash
# Add @eval decorator to your agent, then run it
python my_agent.py "your query"
evalyn list-calls --limit 5
```
```
id       | function | project | status | duration_ms
---------|----------|---------|--------|------------
fde2d07e | my_agent | myapp   | OK     | 1234.56
47fe2576 | my_agent | myapp   | OK     | 2345.67
```

### Step 2: Build Dataset
```bash
evalyn build-dataset --project myapp
```
```
Wrote 10 items to data/myapp-v1-20250115-120000/dataset.jsonl
```

### Step 3: Select Metrics
```bash
evalyn suggest-metrics --project myapp --dataset data/myapp-v1-20250115-120000 --mode basic
```
```
- latency_ms [objective] :: Measure execution time
- output_nonempty [objective] :: Check output is not empty
- helpfulness_accuracy [subjective] :: LLM judge scoring
Saved metrics to data/myapp-v1-20250115-120000/metrics/basic-20250115.json
```

### Step 4: Run Evaluation
```bash
evalyn run-eval --dataset data/myapp-v1-20250115-120000
```
```
Loaded 3 metrics (2 objective, 1 subjective)
Dataset: 10 items

Eval run abc12345-...
Run folder: data/myapp-v1-20250115-120000/eval_runs/20250115-120500_abc12345
  results.json - evaluation data
  report.html  - analysis report   ← Open in browser

Results:
Metric                 Type    Pass Rate
latency_ms             [obj]   N/A
output_nonempty        [obj]   100.0%
helpfulness_accuracy   [llm]   85.0%
```

### Step 5: Annotate & Calibrate (Optional)
```bash
evalyn annotate --dataset data/myapp-v1-20250115-120000
evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl
evalyn run-eval --dataset data/myapp-v1-20250115-120000 --use-calibrated
```

### Step 6: Expand with Simulation (Optional)
```bash
evalyn simulate --dataset data/myapp-v1-20250115-120000 --modes similar,outlier
evalyn run-eval --dataset data/myapp-v1-20250115-120000/simulations/sim-similar-...
```

## Key Commands

| Command | What it does |
|---------|--------------|
| `evalyn one-click --project X` | Run full pipeline |
| `evalyn list-calls` | View captured traces |
| `evalyn build-dataset --project X` | Create dataset from traces |
| `evalyn suggest-metrics --project X --dataset D` | Get metric recommendations |
| `evalyn run-eval --dataset D` | Run evaluation + generate HTML report |
| `evalyn annotate --dataset D` | Human annotation (interactive) |
| `evalyn calibrate --metric-id X` | Calibrate LLM judge |
| `evalyn simulate --dataset D` | Generate synthetic test data |


## Documentation

### CLI Reference

| Command | Description |
|---------|-------------|
| [one-click](docs/clis/one-click.md) | Run full evaluation pipeline |
| [init](docs/clis/init.md) | Initialize config file |
| **Tracing** | |
| [list-calls](docs/clis/list-calls.md) | List captured traces |
| [show-call](docs/clis/show-call.md) | View trace details |
| [show-projects](docs/clis/show-projects.md) | View project summaries |
| **Dataset** | |
| [build-dataset](docs/clis/build-dataset.md) | Create dataset from traces |
| **Metrics** | |
| [suggest-metrics](docs/clis/suggest-metrics.md) | Get metric recommendations |
| [list-metrics](docs/clis/list-metrics.md) | List available metric templates |
| **Evaluation** | |
| [run-eval](docs/clis/run-eval.md) | Run evaluation + generate report |
| [list-runs](docs/clis/list-runs.md) | List past eval runs |
| [show-run](docs/clis/show-run.md) | View eval run details |
| **Calibration** | |
| [annotate](docs/clis/annotate.md) | Human annotation (interactive) |
| [calibrate](docs/clis/calibrate.md) | Calibrate LLM judges |
| **Simulation** | |
| [simulate](docs/clis/simulate.md) | Generate synthetic test data |

### Guides

| Guide | Description |
|-------|-------------|
| [technical-manual.md](docs/technical-manual.md) | Architecture & internals |

## Example

See [`example_agent/`](example_agent/) for a LangGraph integration. 

## License

MIT

## Contact
Submit issues on GitHub or email lsh98dev@gmail.com