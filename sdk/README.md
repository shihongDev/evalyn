# Evalyn SDK

A Python SDK for tracing, evaluating, and calibrating LLM agents.

## Quick Start

```bash
pip install -e ".[dev,llm]"
```

```python
from evalyn_sdk import eval

@eval(project="myapp", version="v1")
def my_agent(query: str) -> str:
    return call_llm(query)
```

```bash
# Set API key
export GEMINI_API_KEY="your-key"

# Run your agent (traces captured automatically)
python my_agent.py

# Build dataset → suggest metrics → evaluate
evalyn one-click --project myapp --target my_agent.py:my_agent
```

## Features

| Feature | Description |
|---------|-------------|
| **Tracing** | Automatic capture of inputs, outputs, errors, latency via `@eval` decorator |
| **Datasets** | Build evaluation datasets from production traces |
| **Metrics** | 50+ objective & subjective (LLM-judge) metrics |
| **Evaluation** | Run metrics against datasets with detailed results |
| **Annotation** | Interactive CLI for human labeling |
| **Calibration** | Align LLM judges with human feedback, optimize prompts |
| **Simulation** | Generate synthetic test queries (similar & edge cases) |
| **One-Click** | Full pipeline in a single command |

## CLI Commands

| Command | Description |
|---------|-------------|
| `evalyn init` | Create config file |
| `evalyn list-calls` | View traced calls |
| `evalyn build-dataset` | Build dataset from traces |
| `evalyn suggest-metrics` | Suggest metrics for your agent |
| `evalyn run-eval` | Run evaluation on dataset |
| `evalyn annotate` | Interactive human annotation |
| `evalyn calibrate` | Calibrate LLM judges with annotations |
| `evalyn simulate` | Generate synthetic test data |
| `evalyn one-click` | Run full pipeline |
| `evalyn status` | View dataset status |

Run `evalyn <command> --help` for options.

## Configuration

```bash
evalyn init                        # Creates evalyn.yaml
export GEMINI_API_KEY="your-key"   # Or OPENAI_API_KEY
```

**evalyn.yaml:**
```yaml
api_keys:
  gemini: "${GEMINI_API_KEY}"

model: "gemini-2.5-flash-lite"

defaults:
  project: "myapp"
```

## Workflow

```
1. Instrument    @eval decorator captures traces
       ↓
2. Build         evalyn build-dataset --project myapp
       ↓
3. Metrics       evalyn suggest-metrics --target agent.py:func
       ↓
4. Evaluate      evalyn run-eval --latest
       ↓
5. Annotate      evalyn annotate --latest
       ↓
6. Calibrate     evalyn calibrate --metric-id helpfulness --annotations ann.jsonl
       ↓
7. Re-evaluate   evalyn run-eval --latest --use-calibrated
```

Or run everything at once:
```bash
evalyn one-click --project myapp --target agent.py:my_agent
```

## Decorator Options

```python
@eval(
    project="myapp",           # Project name for grouping
    version="v1",              # Version tag
    is_simulation=False,       # True for test runs (vs production)
)
def my_agent(query: str) -> str:
    return process(query)
```

## Metric Modes

```bash
# Fast heuristic (no LLM needed)
evalyn suggest-metrics --target agent.py:func --mode basic

# LLM selects from 50+ templates
evalyn suggest-metrics --target agent.py:func --mode llm-registry

# Pre-configured bundles
evalyn suggest-metrics --target agent.py:func --mode bundle --bundle research-agent
```

## Key Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--latest` | run-eval, annotate, calibrate | Use most recent dataset |
| `--use-calibrated` | run-eval | Apply calibrated prompts |
| `--production` | list-calls, build-dataset | Filter to production traces |
| `--simulation` | list-calls, build-dataset | Filter to simulation traces |
| `--per-metric` | annotate | Annotate each metric separately |
| `--skip-annotation` | one-click | Skip annotation step |

## Documentation

- [CLI Reference](docs/cli.md)
- [Metrics Guide](docs/metrics.md)
- [Calibration Guide](docs/calibration.md)
- [API Reference](docs/api.md)

## Example

See [`example_agent/`](example_agent/) for a complete LangGraph integration example.

## License

MIT
