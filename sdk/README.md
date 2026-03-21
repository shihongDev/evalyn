# Evalyn SDK

**Local-first evaluation framework for LLM agents**

Evalyn makes GenAI app evaluation practical and easy. Trace LLM calls, evaluate with 130+ metrics, calibrate judges with human feedback — all running locally on your machine.

## Why Evalyn?

| | |
|---|---|
| **Fully Local** | All data stays on your machine. SQLite storage, no cloud dependencies. |
| **Easy Onboarding** | Just `import evalyn_sdk` — LLM calls auto-captured with tokens & cost. |
| **Metric Bank** | 130+ built-in metrics (73 objective, 60 LLM judges). |
| **Auto Calibration** | 5 optimizers (GEPA, EvoPrompt, TextGrad, MIPROv2, PromptBreeder) align LLM judges with human feedback. |
| **One Command** | Run the entire pipeline with `evalyn one-click`. |

## Install

```bash
pip install evalyn-sdk

# With LLM judge support (OpenAI + Gemini):
pip install "evalyn-sdk[llm]"
```

## Quick Start

### 1. Instrument Your Agent

```python
import evalyn_sdk  # Auto-patches OpenAI, Anthropic, Gemini clients

from evalyn_sdk import eval

@eval(project="myapp", version="v1")
def my_agent(query: str) -> str:
    return call_llm(query)  # LLM calls auto-captured
```

> **Important:** `import evalyn_sdk` must come before importing your LLM client library.

### 2. Run Your Agent

```bash
python my_agent.py "What is quantum computing?"

# Initialize config
evalyn init
```

### 3. Evaluate

**One command:**
```bash
evalyn one-click --project myapp
```

**Or step by step:**
```bash
evalyn build-dataset --project myapp
evalyn suggest-metrics --dataset data/prod/datasets/myapp-*/dataset.jsonl --mode llm-registry
evalyn run-eval --dataset data/prod/datasets/myapp-*/dataset.jsonl
evalyn insights --latest --format html
```

### 4. Calibrate (Optional)

Align LLM judges with your human judgment:
```bash
evalyn annotate --latest
evalyn calibrate --metric-id factual_accuracy
evalyn run-eval --dataset <path> --use-calibrated
```

## The Pipeline

```
COLLECT  →  EVALUATE  →  CALIBRATE  →  EXPAND
@eval       suggest-     annotate      simulate
trace       metrics      calibrate     synthetic
build-      run-eval     cluster-      queries
dataset     insights     failures
```

## Key Commands

| Command | What it does |
|---------|--------------|
| `evalyn one-click --project X` | Run full pipeline |
| `evalyn list-calls` | View captured traces |
| `evalyn show-trace --last -v` | View span tree with details |
| `evalyn build-dataset --project X` | Create dataset from traces |
| `evalyn suggest-metrics --dataset D` | Get metric recommendations |
| `evalyn run-eval --dataset D` | Run evaluation |
| `evalyn insights --latest` | Analysis with recommendations |
| `evalyn insights --latest --deep` | Add LLM expert panel |
| `evalyn insights --latest --format html` | Interactive HTML dashboard |
| `evalyn annotate --latest` | Human annotation |
| `evalyn calibrate --metric-id X` | Calibrate LLM judges |

## Supported Frameworks

Auto-instrumentation for: **OpenAI**, **Anthropic**, **Google Gemini**, **LangChain**, **CrewAI**, **AutoGen**, **DSPy**, **Haystack**, **LlamaIndex**, **Semantic Kernel**

## Optional Extras

```bash
pip install "evalyn-sdk[llm]"              # OpenAI + Gemini for LLM judges
pip install "evalyn-sdk[agent]"            # LangChain + LangGraph agent support
pip install "evalyn-sdk[anthropic-agents]" # Anthropic Agent SDK support
pip install "evalyn-sdk[clustering]"       # Failure clustering with embeddings
pip install "evalyn-sdk[dev]"              # pytest + coverage for development
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Gemini API key for LLM judges (default provider) |
| `OPENAI_API_KEY` | OpenAI API key when using `--provider openai` |
| `EVALYN_NO_HINTS` | Set to `1` to suppress hint messages |
| `EVALYN_AUTO_INSTRUMENT` | Set to `off` to disable auto-patching |

## Claude Code Skills

Evalyn includes [Claude Code skills](https://github.com/shihongDev/evalyn) that let Claude guide you through the evaluation pipeline conversationally:

```bash
# Install skills
cp -r skills/evalyn-* ~/.claude/skills/
```

Then in Claude Code: *"Help me evaluate my agent"* — Claude handles the rest.

## License

MIT

## Links

- [GitHub](https://github.com/shihongDev/evalyn)
- [Changelog](https://github.com/shihongDev/evalyn/blob/main/sdk/CHANGELOG.md)
- [Issues](https://github.com/shihongDev/evalyn/issues)
