# Evalyn

**Trace, evaluate, and calibrate your LLM agents — fully local, fully yours.**

## Why Evalyn?

| | |
|---|---|
| **Fully Local** | All data stays on your machine. SQLite storage, no cloud dependencies. |
| **Zero Config** | Just `import evalyn_sdk` — LLM calls auto-captured with tokens & cost. |
| **LLM Judges** | 50+ metrics including LLM-based judges for quality assessment. |
| **Human Calibration** | Align LLM judges with human feedback through annotation workflow. |
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
# Add @eval decorator to your agent
python my_agent.py                        # Traces auto-captured 
evalyn list-calls                         # View traces
```

### Step 2: Build Dataset
```bash
evalyn build-dataset --project myapp      # Creates data/myapp-v1-<timestamp>/
```

### Step 3: Select Metrics
```bash
evalyn suggest-metrics --latest --mode basic    # Fast, no LLM
# or
evalyn suggest-metrics --latest --mode llm-registry  # LLM picks from 50+ templates
```

### Step 4: Run Evaluation
```bash
evalyn run-eval --latest                  # Runs eval + generates HTML report
# Output: eval_runs/<timestamp>/
#   ├── results.json
#   └── report.html    ← Open in browser
```

### Step 5: Annotate & Calibrate (Optional)
```bash
evalyn annotate --latest                  # Interactive: label pass/fail
evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl
evalyn run-eval --latest --use-calibrated # Re-run with calibrated prompts
```

### Step 6: Expand with Simulation (Optional)
```bash
evalyn simulate --latest --modes similar,outlier  # Generate synthetic queries
evalyn run-eval --dataset data/simulations/...    # Evaluate on synthetic data
```

## Key Commands

| Command | What it does |
|---------|--------------|
| `evalyn one-click --project X` | Run full pipeline |
| `evalyn list-calls` | View captured traces |
| `evalyn build-dataset --project X` | Create dataset from traces |
| `evalyn suggest-metrics --latest` | Get metric recommendations |
| `evalyn run-eval --latest` | Run evaluation + generate HTML report |
| `evalyn annotate --latest` | Human annotation (interactive) |
| `evalyn calibrate --metric-id X` | Calibrate LLM judge |
| `evalyn simulate --latest` | Generate synthetic test data |


## Documentation

| Guide | Description |
|-------|-------------|
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/commands/](docs/commands/) | Detailed command docs |
| [docs/metrics.md](docs/metrics.md) | Metrics guide |
| [docs/calibration.md](docs/calibration.md) | Calibration workflow |

## Example

See [`example_agent/`](example_agent/) for a LangGraph integration. 

## License

MIT
