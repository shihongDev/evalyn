<p align="center">
  <img src="docs/evalyn-banner.png" alt="Evalyn - Evaluation as a service. Let your agents evolve — privately." width="600">
</p>

<p align="center">
  <strong>Local-first evaluation framework for LLM agents</strong>
</p>

---

Evalyn focuses on making GenAI App evaluation practical and easy. It provides lightweight tracing, human-in-the-loop annotation, metric suggestion, and calibration workflows to help developers/non-tech folks understand what to evaluate, align metrics with real usage, and continuously improve your GenAI App behavior over time.

## Why Evalyn?

| | |
|---|---|
| **Fully Local** | All data stays on your machine. SQLite storage, no cloud dependencies. |
| **Easy Onboarding** | Just `import evalyn_sdk` — LLM calls auto-captured with tokens & cost. |
| **Metric Bank** | 130+ built-in metrics (73 objective, 60 LLM judges). Community contributions welcome. |
| **Auto Calibration** | Align LLM judges with human feedback through automatic prompt optimization such as GEPA. |
| **One Command** | Run the entire pipeline with `evalyn one-click`. |

## The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  1. COLLECT                                                      │
│                                                                  │
│     @eval  ->  TRACE (SQLite)  ->  DATASET (JSONL)               │
│               show-trace -v        build-dataset                 │
│               show-span                                          │
└──────────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────────┐
│  2. EVALUATE                                                     │
│                                                                  │
│     suggest-metrics  ->  run-eval  ->  ANALYZE                   │
│                                        trend / compare / analyze │
└──────────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────────┐
│  3. CALIBRATE                                                    │
│                                                                  │
│     annotate  ->  calibrate  ->  run-eval --use-calibrated       │
│      (Human)      (Auto-iterates until aligned)                  │
│                                                                  │
│     cluster-failures / cluster-misalignments for insights        │
└──────────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────────┐
│  4. EXPAND                                                       │
│                                                                  │
│     simulate  ->  synthetic queries  ->  Back to Step 2          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Install

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment (Python 3.10+)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the SDK
uv pip install -e "./sdk[llm]"

# To run the example agent, also install agent dependencies:
uv pip install -e "./sdk[agent]"
```

## Quick Start (Example Gemini Deep Research Agent)

### 1. Instrument Your Agent

```python
from evalyn_sdk import eval

@eval(project="myapp", version="v1")
def my_agent(query: str) -> str:
    return call_llm(query)  # LLM calls auto-captured
```

### 2. Run the Agent 
```bash
export GEMINI_API_KEY="your_api_key"  # This is for the example agent itself
python example_agents/langchain_deep_research_agent/agent.py "What is the current status of AI Agents?"  
evalyn init  # Create evalyn.yaml config and edit evalyn.yaml to set your GEMINI_API_KEY. This is for llm judge setup
```

### 3. Choose Your Workflow

**Option A: One-Click (Automated)** — Run the full pipeline in one command
```bash
evalyn one-click --project gemini-deep-research-agent    # Dataset -> Metrics -> Eval -> Report
```

**Option B: Step-by-Step (Manual)** — See [Sample Workflow](#sample-workflow) for granular control over each step.




## Sample Workflow

> For users who want manual control over each step. For automated pipeline, use `evalyn one-click` instead.

### Step 1: Instrument & Collect
```bash
# Add @eval decorator to the agent, then run it (already added for example agents)
evalyn list-calls
```
```
id       | function       | project                     | status | duration_ms
---------|----------------|-----------------------------|---------|-----------
960912fa | research_agent | gemini-deep-research-agent  | OK     | 21215.61
75489541 | research_agent | gemini-deep-research-agent  | OK     | 22097.82
2567ea62 | chat           | anthropic-research-agent    | OK     | 1461418.96
```

> **Tip:** Use `evalyn show-call --id 960912fa` to view trace details with short ID prefix.

### Step 2: Build Dataset
```bash
evalyn build-dataset --project gemini-deep-research-agent
```
```
Wrote 137 items to data/prod/datasets/gemini-deep-research-agent-v0-20260127-054032/dataset.jsonl

Hint: To suggest metrics, run: evalyn suggest-metrics --dataset <path> --mode basic
```

### Step 3: Select Metrics

Use `--mode llm-registry` to pick from 130+ built-in metrics via LLM selection:
```bash
evalyn suggest-metrics --dataset <dataset-path> --mode llm-registry
```
```
- helpfulness_accuracy [subjective] :: PASS if the output answers the user question accurately
- factual_accuracy [subjective] :: PASS if factual claims are accurate and verifiable
- output_nonempty [objective] :: PASS if output is not empty/None
- latency_ms [objective] :: Measure execution latency in milliseconds
- url_count [objective] :: Counts URLs in the output (proxy for citations)
Saved metrics to <dataset-path>/metrics/metrics.json
```

Use `--mode llm-brainstorm --append` to generate custom metrics:
```bash
evalyn suggest-metrics --dataset <dataset-path> --mode llm-brainstorm --append
```
```
- completeness_of_explanation [subjective] :: Evaluates if the explanation covers core aspects
- clarity_and_readability [subjective] :: Assesses how easy the explanation is to understand
- relevance_to_question [subjective] :: Determines if output directly answers the question
- source_credibility_and_integration [subjective] :: Evaluates quality of cited sources
- conciseness_and_efficiency [subjective] :: Measures whether explanation avoids verbosity
Appending 5 new metric(s) to 5 existing.
```

### Step 4: Run Evaluation
```bash
evalyn run-eval --dataset <dataset-path>
```
```
Loaded 10 metrics (3 objective, 7 subjective)
Dataset: 137 items

Eval run 220e8590-2ccd-4cb5-a585-a0110285e786
Run folder: <dataset-path>/eval_runs/20260127-055659_220e8590
  results.json - evaluation data
  report.html  - analysis report

Results:
--------------------------------------------------------------------------------
Metric                         Type    Count  Avg Score    Pass Rate  Errors
--------------------------------------------------------------------------------
output_nonempty                [obj]   137    1.0000       100.0%     -
latency_ms                     [obj]   137    23986.3494   N/A        -
url_count                      [obj]   137    55.1825      100.0%     -
factual_accuracy               [llm]   137    0.8015       79.6%      1
clarity_and_readability        [llm]   137    1.0000       97.1%      4
helpfulness_accuracy           [llm]   137    0.9850       95.6%      4
--------------------------------------------------------------------------------

Token usage: 10,048,041 input + 70,929 output = 10,118,970 total ($1.03)
```

### Step 5: Analyze Results
```bash
evalyn analyze --run 220e8590
```
```
======================================================================
  METRIC SUMMARY
======================================================================
  [FAIL] factual_accuracy               109/137 passed (80%)  avg=0.80
  [FAIL] conciseness_and_efficiency     124/137 passed (91%)  avg=0.91
  [PASS] output_nonempty                137/137 passed (100%) avg=1.00
  [PASS] url_count                      137/137 passed (100%) avg=55.18

======================================================================
  INSIGHTS
======================================================================
  'latency_ms' has the highest failure rate. Consider reviewing the rubric.
  2 metric(s) have 100% pass rate: output_nonempty, url_count
  Overall health is MODERATE (85% pass rate)

======================================================================
  RECOMMENDATIONS
======================================================================
  1. Run 'evalyn annotate' to provide human labels for failed items
  2. Run 'evalyn calibrate' to improve metric alignment
```

### Step 6: Annotate & Calibrate (Optional)
```bash
evalyn annotate --latest
```
```
======================================================================
INTERACTIVE ANNOTATION
======================================================================
Dataset: data/prod/datasets/gemini-deep-research-agent-v0-.../dataset.jsonl
Items to annotate: 137
Using eval run: 220e8590...

Commands: [y]es/pass  [n]o/fail  [s]kip  [v]iew full  [q]uit
======================================================================

Item 1/137 [960912fa-637...]
INPUT:  {"question": "What is antimatter?", ...}
OUTPUT: Antimatter is a form of matter that is the counterpart to...

LLM JUDGE RESULTS:
   factual_accuracy: PASS
   clarity_and_readability: PASS
   helpfulness_accuracy: PASS

Pass? [y/n/s/v/q]:
```

After annotation, calibrate LLM judges to align with human feedback:
```bash
evalyn calibrate --metric-id factual_accuracy --annotations annotations.jsonl
evalyn run-eval --dataset <dataset-path> --use-calibrated
```

### Step 7: Expand with Simulation (Optional)
```bash
evalyn simulate --dataset <dataset-path> --modes similar,outlier
evalyn run-eval --dataset <dataset-path>/simulations/sim-similar-...
```

## Key Commands

| Command | What it does |
|---------|--------------|
| `evalyn one-click --project X` | Run full pipeline |
| `evalyn list-calls` | View captured traces |
| `evalyn show-call --id abc123` | View trace details (supports short IDs) |
| `evalyn show-call --last` | View most recent trace |
| `evalyn show-trace --last -v` | View span tree with details |
| `evalyn show-span --call-id X --span Y` | Inspect single span |
| `evalyn delete-traces -n 5 --db test` | Delete traces from storage |
| `evalyn build-dataset --project X` | Create dataset from traces |
| `evalyn suggest-metrics --project X --dataset D` | Get metric recommendations |
| `evalyn run-eval --dataset D` | Run evaluation + generate HTML report (use `--provider` for OpenAI/Ollama) |
| `evalyn show-run --last` | View most recent eval run |
| `evalyn trend --project X` | View metric trends across eval runs |
| `evalyn annotate --dataset D` | Human annotation (interactive) |
| `evalyn calibrate --metric-id X` | Calibrate LLM judge |
| `evalyn simulate --dataset D` | Generate synthetic test data |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Gemini API key for LLM judges (default provider) |
| `OPENAI_API_KEY` | OpenAI API key for judges when using `--provider openai` |
| `EVALYN_NO_HINTS` | Set to `1` to suppress hint messages |
| `EVALYN_AUTO_INSTRUMENT` | Set to `off` to disable auto-patching |


## Documentation

### CLI Reference

| Command | Description |
|---------|-------------|
| [one-click](docs/clis/one-click.md) | Run full evaluation pipeline |
| [init](docs/clis/init.md) | Initialize config file |
| [status](docs/clis/status.md) | Show dataset status overview |
| **Tracing** | |
| [list-calls](docs/clis/list-calls.md) | List captured traces |
| [show-call](docs/clis/show-call.md) | View trace details |
| [show-trace](docs/clis/show-trace.md) | View span tree (with -v for details) |
| [show-span](docs/clis/show-span.md) | View single span details |
| [show-projects](docs/clis/show-projects.md) | View project summaries |
| [delete-traces](docs/clis/delete-traces.md) | Delete traces from storage |
| **Dataset** | |
| [build-dataset](docs/clis/build-dataset.md) | Create dataset from traces |
| [validate](docs/clis/validate.md) | Validate dataset format |
| **Metrics** | |
| [suggest-metrics](docs/clis/suggest-metrics.md) | Get metric recommendations |
| [select-metrics](docs/clis/select-metrics.md) | LLM-guided metric selection |
| [list-metrics](docs/clis/list-metrics.md) | List available metric templates |
| **Evaluation** | |
| [run-eval](docs/clis/run-eval.md) | Run evaluation + generate report |
| [list-runs](docs/clis/list-runs.md) | List past eval runs |
| [show-run](docs/clis/show-run.md) | View eval run details |
| [analyze](docs/clis/analyze.md) | Analyze eval run insights |
| [compare](docs/clis/compare.md) | Compare two eval runs |
| [trend](docs/clis/trend.md) | View metric trends across runs |
| [export](docs/clis/export.md) | Export results (JSON/CSV/MD/HTML) |
| **Annotation & Calibration** | |
| [annotate](docs/clis/annotate.md) | Human annotation (interactive) |
| [annotation-stats](docs/clis/annotation-stats.md) | Show annotation statistics |
| [export-for-annotation](docs/clis/export-for-annotation.md) | Export for external annotation |
| [import-annotations](docs/clis/import-annotations.md) | Import annotations from file |
| [calibrate](docs/clis/calibrate.md) | Calibrate LLM judges |
| [list-calibrations](docs/clis/list-calibrations.md) | List calibration records |
| [cluster-failures](docs/clis/cluster-failures.md) | Cluster and analyze failures |
| [cluster-misalignments](docs/clis/cluster-misalignments.md) | Cluster human/LLM disagreements |
| **Simulation** | |
| [simulate](docs/clis/simulate.md) | Generate synthetic test data |

### Guides

| Guide | Description |
|-------|-------------|
| [technical-manual.md](docs/technical-manual.md) | Architecture & internals |

## Example

See [`example_agents/`](example_agents/) for SDK integrations:
- **LangChain**: Auto-instrumentation via callback handler
- **Claude Agent SDK**: Hook-based integration (see [technical manual](docs/technical-manual.md#claude-agent-sdk-integration))
- **Google ADK**: Automatic callback injection (see [technical manual](docs/technical-manual.md#google-adk-integration))

## Contribution

We welcome community-contributed metrics. Upload your own evaluation metrics to help grow the open-source metric bank.

### Step 1: Choose Metric Type

Decide whether your metric is:
- **Objective**: Code-based evaluation (regex, JSON parsing, numeric checks, etc.)
- **Subjective**: LLM judge-based evaluation (requires rubric and prompt)

### Step 2: Define Your Metric

**For objective metrics**, add to `sdk/evalyn_sdk/metrics/objective.py`:

```python
{
    "id": "your_metric_name",           # Unique snake_case identifier
    "type": "objective",
    "description": "What this metric measures.",
    "config": {},                        # Any configurable parameters
    "category": "structure",             # One of: efficiency, structure, robustness, correctness, grounding, style, diversity
    "scope": "overall",                  # One of: overall, llm_call, tool_call, trace
    "requires_reference": False,         # True if needs human_label.reference
    "author": "Your Name <email>",       # Optional: credit for your contribution
}
```

**For subjective metrics**, add to `sdk/evalyn_sdk/metrics/subjective.py`:

```python
{
    "id": "your_metric_name",
    "type": "subjective",
    "description": "What this metric evaluates.",
    "category": "correctness",           # One of: safety, correctness, style, instruction, grounding, completeness, agent, ethics, domain, ux, conversation, creativity, explanation, persona, summarization, argumentation, accessibility
    "scope": "overall",
    "prompt": "You are a judge for X. Evaluate whether...",
    "config": {
        "rubric": [
            "Criterion 1: what to check.",
            "Criterion 2: what to check.",
            "If ANY issue is found, FAIL and explain.",
        ],
        "threshold": 0.5,
    },
    "requires_reference": False,
    "author": "Your Name <email>",       # Optional: credit for your contribution
}
```

### Step 3: Implement the Evaluation Logic (Objective Only)

For objective metrics, add a handler function in `sdk/evalyn_sdk/metrics/objective.py`:

```python
def compute_your_metric_name(output: str, config: dict, **kwargs) -> dict:
    """Evaluate your metric."""
    # Your logic here
    score = ...
    passed = score > config.get("threshold", 0.5)
    return {"score": score, "passed": passed, "reason": "Explanation"}
```

Then register it in the `METRIC_HANDLERS` dict.

### Step 4: Add Tests

Add tests in `tests/test_metrics.py` to verify your metric works correctly:

```python
def test_your_metric_name():
    result = compute_your_metric_name("test output", {})
    assert "score" in result
    assert "passed" in result
```

### Step 5: Submit a Pull Request

1. Fork the repository
2. Create a branch: `git checkout -b add-metric-your_metric_name`
3. Make your changes
4. Run tests: `uv run pytest tests/`
5. Submit a PR with:
   - Description of what the metric measures
   - Example use cases
   - Test results

### Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique snake_case identifier |
| `type` | Yes | `objective` or `subjective` |
| `description` | Yes | Brief explanation of what it measures |
| `category` | Yes | Metric category for grouping |
| `scope` | Yes | What the metric evaluates: `overall`, `llm_call`, `tool_call`, `trace` |
| `config` | Yes | Configuration options (can be empty `{}`) |
| `requires_reference` | Yes | Whether human_label.reference is needed |
| `prompt` | Subjective only | System prompt for the LLM judge |
| `author` | No | Your name/email for attribution |

## License

MIT

## Contact
Submit issues on GitHub or email lsh98dev@gmail.com
