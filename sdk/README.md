# Evalyn SDK

A Python SDK for instrumenting, tracing, and evaluating LLM agents with objective and subjective (LLM-judge) metrics.

## Quick Start

```bash
# Install
pip install -e ".[dev,llm,otel,agent]"

# Instrument your agent
from evalyn_sdk import eval

@eval(project="myproj", version="v1")
def my_agent(query: str) -> str:
    return process(query)

# Run your agent (traces captured automatically)
python my_agent.py

# View traces
evalyn list-calls --limit 20

# Build dataset from traces
evalyn build-dataset --project myproj --version v1

# Suggest metrics
evalyn suggest-metrics --target my_agent.py:my_agent --mode llm-registry --llm-mode api

# Run evaluation
evalyn run-eval --latest

# Annotate results
evalyn annotate --dataset data/myproj-v1-20250101

# Calibrate LLM judges
evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl --dataset data/myproj-v1
```

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. INSTRUMENT & COLLECT                                            │
│     @eval decorator → traces to SQLite                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  2. BUILD DATASET                                                   │
│     evalyn build-dataset → dataset.jsonl + meta.json                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  3. SELECT METRICS                                                  │
│     evalyn suggest-metrics → metrics/*.json                         │
│     (basic/llm-registry/llm-brainstorm/bundle modes)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  4. RUN EVALUATION                                                  │
│     evalyn run-eval → eval_runs/*.json                              │
│     (objective metrics + LLM judge subjective metrics)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  5. HUMAN ANNOTATION                                                │
│     evalyn annotate --dataset ... (interactive CLI)                 │
│     Shows: input, output, LLM judge results                         │
│     Collects: pass/fail, confidence (1-5), notes                    │
│     Saves: annotations.jsonl                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  6. CALIBRATE                                                       │
│     evalyn calibrate --metric-id ... --annotations ...              │
│     Validates: Is optimized prompt better? (F1 comparison)          │
│     Outputs: alignment metrics, validation results, optimized       │
│              prompts (only if validated as better)                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  7. RE-EVALUATE WITH CALIBRATED PROMPTS                             │
│     evalyn run-eval --latest --use-calibrated                       │
│     Loads optimized prompts from calibrations/ folder               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
┌─────────────────────────────────────────────────────────────────────┐
│  8. SIMULATE (expand coverage)                                      │
│     evalyn simulate --dataset ... --modes similar,outlier           │
│     Similar: variations of seed queries (test robustness)           │
│     Outlier: edge cases, adversarial inputs                         │
│     --target: run agent on generated queries                        │
│     Saves: simulations/sim-{mode}-{timestamp}/                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    v
                        ┌───────────────────────┐
                        │ Iterate: Evaluate     │
                        │ synthetic data        │
                        └───────────────────────┘
```

---

## 1. Instrumentation & Tracing

### Flow

```
┌──────────────────┐
│  Your Function   │
│  (decorated)     │
└────────┬─────────┘
         │
         v
┌──────────────────┐      ┌──────────────────┐
│  @eval decorator │ ───> │  EvalTracer      │
│  - project       │      │  - capture input │
│  - version       │      │  - capture output│
│  - is_simulation │      │  - duration      │
└──────────────────┘      │  - trace events  │
                          │  - metadata      │
                          └────────┬─────────┘
                                   │
                                   v
                          ┌──────────────────┐
                          │  SQLite Storage  │
                          │  evalyn.sqlite   │
                          │  - FunctionCall  │
                          │  - Metadata      │
                          └──────────────────┘
```

### Usage

```python
from evalyn_sdk import eval, eval_session

# Basic instrumentation
@eval(project="myproj", version="v1")
def my_agent(query: str) -> str:
    return process(query)

# Simulation mode (auto-tagged)
@eval(project="myproj", version="v1", is_simulation=True)
def test_agent(query: str) -> str:
    return process(query)

# Async functions supported
@eval(project="myproj", version="v1")
async def async_agent(query: str) -> str:
    return await async_process(query)

# Group calls into sessions
with eval_session(session_id="user-123-session"):
    result1 = my_agent("query 1")
    result2 = my_agent("query 2")
```

### CLI Commands

```bash
# List all traced calls
evalyn list-calls --limit 20

# Filter by project
evalyn list-calls --project myproj --version v1

# Filter by simulation/production
evalyn list-calls --simulation      # Only simulation traces
evalyn list-calls --production      # Only production traces

# View specific call
evalyn show-call --id abc123

# View projects
evalyn show-projects
```

### What Gets Captured

- **Inputs**: Function arguments (serialized as JSON)
- **Output**: Function return value
- **Error**: Exception details (if function fails)
- **Duration**: Execution time in milliseconds
- **Trace Events**: Timestamped events logged during execution
- **Function Metadata**: Signature, docstring, source code, hash
- **Custom Metadata**: Project, version, is_simulation flag
- **Session ID**: Groups related calls together

---

## 2. Dataset Management

### Flow

```
┌──────────────────┐
│  SQLite Storage  │
│  (FunctionCalls) │
└────────┬─────────┘
         │
         v
┌────────────────────────────────┐
│  build-dataset command         │
│  Filters:                      │
│  - project/version             │
│  - simulation_only=True/False  │
│  - production_only=True/False  │
│  - since/until (time range)    │
│  - success_only=True           │
│  - limit                       │
└────────┬───────────────────────┘
         │
         v
┌────────────────────────────────┐
│  Dataset Folder                │
│  data/<project>-<version>-ts/  │
│  ├── dataset.jsonl             │
│  ├── meta.json                 │
│  ├── metrics/                  │
│  │   ├── basic-*.json          │
│  │   └── llm-registry-*.json   │
│  ├── eval_runs/                │
│  │   └── <run_id>.json         │
│  ├── calibrations/             │
│  │   └── <metric_id>/          │
│  │       ├── calibration.json  │
│  │       └── prompts/           │
│  │           └── optimized.txt │
│  └── simulations/              │
│      └── sim-*/                │
└────────────────────────────────┘
```

### Dataset Structure

Each dataset item has 4 columns:

```json
{
  "id": "abc123",
  "input": {"query": "What is AI?"},
  "output": "AI is artificial intelligence...",
  "human_label": null,
  "metadata": {
    "function": "my_agent",
    "call_id": "abc123",
    "started_at": "2025-01-26T10:00:00Z",
    "duration_ms": 1234,
    "project_id": "myproj",
    "version": "v1",
    "is_simulation": false
  }
}
```

### CLI Commands

```bash
# Build dataset from all traces
evalyn build-dataset --project myproj --version v1

# Production traces only
evalyn build-dataset --project myproj --production

# Simulation traces only
evalyn build-dataset --project myproj --simulation

# Time-based filtering
evalyn build-dataset --project myproj --since "2025-01-01" --until "2025-01-31"

# Limit number of items
evalyn build-dataset --project myproj --limit 100
```

### Python API

```python
from evalyn_sdk import build_dataset_from_storage, save_dataset, load_dataset
from evalyn_sdk.storage import get_default_storage

# Build dataset
storage = get_default_storage()
items = build_dataset_from_storage(
    storage,
    project_name="myproj",
    version="v1",
    production_only=True,
    limit=500
)

# Save dataset
save_dataset(items, "data/myproj/dataset.jsonl")

# Load dataset
items = load_dataset("data/myproj/dataset.jsonl")
```

---

## 3. Metric Selection

### Flow

```
                     ┌──────────────────┐
                     │  User chooses    │
                     │  selection mode  │
                     └────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          v                   v                   v
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Basic Mode     │  │ LLM-Registry    │  │ LLM-Brainstorm  │
│  (heuristic)    │  │ (select from    │  │ (generate       │
│  - Fast         │  │  50+ templates) │  │  custom specs)  │
│  - No API key   │  │  - Needs API    │  │  - Needs API    │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         v                    v                    v
┌──────────────────────────────────────────────────────────┐
│  Metric Specifications (JSON)                            │
│  - Metric ID                                             │
│  - Type (objective/subjective)                           │
│  - Config (thresholds, prompts, etc.)                    │
└────────────────────┬─────────────────────────────────────┘
                     │
                     v
┌──────────────────────────────────────────────────────────┐
│  Saved to: data/<dataset>/metrics/<name>.json            │
│  Updated: meta.json (active_metric_set)                  │
└──────────────────────────────────────────────────────────┘
```

### Selection Modes

**1. Basic Mode (Heuristic)**
- Fast, offline, no API key needed
- Based on function signature and sample traces
- Suggests: latency, output_nonempty, json_valid, helpfulness

```bash
evalyn suggest-metrics --target my_agent.py:my_agent --mode basic
```

**2. LLM-Registry Mode**
- LLM selects from 50+ pre-defined metric templates
- Requires API key (Gemini or OpenAI)
- Analyzes function code + sample traces

```bash
# With Gemini (default)
evalyn suggest-metrics --target my_agent.py:my_agent \
  --mode llm-registry --llm-mode api --model gemini-2.5-flash-lite

# With OpenAI
evalyn suggest-metrics --target my_agent.py:my_agent \
  --mode llm-registry --llm-mode api --model gpt-4

# With local Ollama
evalyn suggest-metrics --target my_agent.py:my_agent \
  --mode llm-registry --llm-mode local --model llama3.1
```

**3. LLM-Brainstorm Mode**
- LLM generates custom metric specifications
- Not constrained to registry
- Most flexible but requires review

```bash
evalyn suggest-metrics --target my_agent.py:my_agent \
  --mode llm-brainstorm --llm-mode api --model gpt-4
```

**4. Bundle Mode**
- Pre-configured metric sets
- Available bundles: `summarization`, `orchestrator`, `research-agent`

```bash
evalyn suggest-metrics --target my_agent.py:my_agent \
  --mode bundle --bundle research-agent
```

### Save to Dataset

```bash
# Save with custom name
evalyn suggest-metrics --dataset data/myproj-v1 \
  --target my_agent.py:my_agent \
  --mode llm-registry --llm-mode api \
  --metrics-name llm-selected

# Saved to: data/myproj-v1/metrics/llm-selected.json
# Updated: data/myproj-v1/meta.json (active_metric_set)
```

### Available Metrics

```bash
# List all metric templates
evalyn list-metrics

# Output shows:
# - Objective metrics: latency, BLEU, ROUGE, JSON validation, token counts
# - Subjective metrics: helpfulness, toxicity, hallucination, clarity
```

**Objective Metrics** (deterministic, no LLM):
- `latency_ms`, `cost_usd`, `token_length`
- `exact_match`, `bleu`, `rouge_l`, `token_overlap_f1`
- `json_valid`, `regex_match`, `url_count`
- `tool_call_count`, `llm_call_count`

**Subjective Metrics** (LLM-judge):
- `helpfulness_accuracy`, `instruction_following`
- `toxicity_safety`, `hallucination_risk`
- `clarity_coherence`, `completeness`
- `tone_appropriateness`

---

## 4. Evaluation

### Flow

```
┌──────────────────┐     ┌──────────────────┐
│  Dataset         │     │  Metrics         │
│  dataset.jsonl   │     │  metrics/*.json  │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────┬───────────────┘
                  │
                  v
┌─────────────────────────────────────────────┐
│  evalyn run-eval --dataset ... --metrics ...│
│  For each dataset item:                     │
│    1. Load/run target function              │
│    2. Apply all metrics                     │
│    3. Compute scores                        │
└────────┬────────────────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  Objective Metrics               │
│  - Deterministic computation     │
│  - latency, BLEU, JSON valid     │
└────────┬─────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  Subjective Metrics              │
│  - LLM judge (Gemini/OpenAI)     │
│  - helpfulness, toxicity, etc.   │
│  - Returns: passed + reason      │
└────────┬─────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  EvalRun Results                 │
│  - Per-item metric results       │
│  - Summary statistics            │
│  - Saved to eval_runs/*.json     │
└──────────────────────────────────┘
```

### CLI Commands

```bash
# Run with auto-detected metrics (from meta.json)
evalyn run-eval --dataset data/myproj-v1

# Run with specific metrics file
evalyn run-eval --dataset data/myproj-v1 --metrics metrics/llm-selected.json

# Run with multiple metrics files
evalyn run-eval --dataset data/myproj-v1 --metrics "metrics/basic.json,metrics/llm-registry.json"

# Run with ALL metrics in folder
evalyn run-eval --dataset data/myproj-v1 --metrics-all

# Use calibrated prompts (load optimized prompts)
evalyn run-eval --dataset data/myproj-v1 --use-calibrated

# Run on latest dataset
evalyn run-eval --latest --use-calibrated
```

### Python API

```python
from evalyn_sdk import EvalRunner, load_dataset, MetricRegistry

# Load dataset
items = load_dataset("data/myproj/dataset.jsonl")

# Create metric registry
registry = MetricRegistry()
registry.add_from_file("metrics/llm-selected.json")

# Run evaluation
runner = EvalRunner(registry=registry)
eval_run = runner.run(items, target_fn=my_agent)

# Access results
print(f"Summary: {eval_run.summary}")
for result in eval_run.results:
    print(f"{result.metric_id}: {result.score} ({result.passed})")
```

### Output Example

```
Loaded 5 metrics (2 objective, 3 subjective)
Dataset: 50 items
Running evaluation...
[============================] 50/50

RESULTS:
  latency_ms:         avg=1234.5ms (min=500ms, max=3000ms)
  helpfulness:        pass_rate=0.92 (46/50)
  toxicity:           pass_rate=1.00 (50/50)
  json_valid:         pass_rate=0.98 (49/50)
  hallucination_risk: pass_rate=0.88 (44/50)

Saved to: data/myproj-v1/eval_runs/20250126_123456_run123.json
```

### Calibrated Prompts

When using `--use-calibrated`:
1. Loads optimized prompts from `calibrations/<metric_id>/prompts/`
2. Replaces default prompts in subjective metrics
3. Shows count: `Loaded 5 metrics (2 objective, 3 subjective, 2 calibrated)`

```bash
evalyn run-eval --latest --use-calibrated
```

---

## 5. Human Annotation

### Flow

```
┌──────────────────┐
│  Dataset         │
│  + Eval Results  │
└────────┬─────────┘
         │
         v
┌─────────────────────────────────────────────┐
│  evalyn annotate --dataset ...              │
│  For each item (interactive):               │
│    1. Show input                            │
│    2. Show output                           │
│    3. Show LLM judge results (if available) │
│    4. Collect human annotation              │
└────────┬────────────────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  Annotation Mode                 │
│  --per-metric: Annotate each     │
│    metric separately (agree/     │
│    disagree with LLM judge)      │
│  Default: Overall pass/fail      │
└────────┬─────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  Collect for each item:          │
│  - Pass/Fail (or agree/disagree) │
│  - Confidence (1-5)              │
│  - Notes (optional)              │
└────────┬─────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  annotations.jsonl               │
│  - item_id                       │
│  - human_label (pass/fail)       │
│  - confidence (1-5)              │
│  - metric_labels (per-metric)    │
│  - notes                         │
└──────────────────────────────────┘
```

### CLI Commands

```bash
# Interactive annotation (shows LLM judge results)
evalyn annotate --dataset data/myproj-v1

# Per-metric annotation (agree/disagree with each LLM judge)
evalyn annotate --dataset data/myproj-v1 --per-metric

# Resume annotation session
evalyn annotate --dataset data/myproj-v1 --resume

# View annotation stats
evalyn annotation-stats --dataset data/myproj-v1/annotations.jsonl
```

### Interactive Workflow

```
Item 1/50
─────────────────────────────────────────
Input:
  query: "What are the latest developments in AI?"

Output:
  "Recent AI developments include GPT-4, multimodal models..."

LLM Judge Results:
  ✓ helpfulness_accuracy: PASS (score: 0.92)
    Reason: Response is accurate and helpful
  ✓ toxicity_safety: PASS (score: 1.00)
    Reason: No toxic content
  ✗ completeness: FAIL (score: 0.65)
    Reason: Missing recent developments from 2025

Your annotation:
  Pass/Fail [p/f]: p
  Confidence (1-5): 4
  Notes (optional): Good overview but could be more recent

Saved annotation 1/50
```

### Per-Metric Mode

```bash
evalyn annotate --dataset data/myproj-v1 --per-metric
```

```
Item 1/50 - Metric: helpfulness_accuracy
─────────────────────────────────────────
Input: ...
Output: ...

LLM Judge: PASS (score: 0.92)
Reason: Response is accurate and helpful

Do you agree? [y/n]: y
Confidence (1-5): 5
Notes: Excellent response

─────────────────────────────────────────
Item 1/50 - Metric: completeness
─────────────────────────────────────────
LLM Judge: FAIL (score: 0.65)
Reason: Missing recent developments

Do you agree? [y/n]: n
Your label [pass/fail]: pass
Confidence (1-5): 3
Notes: Coverage is sufficient for general query
```

### Annotation Format

```json
{
  "item_id": "abc123",
  "human_label": "pass",
  "confidence": 4,
  "notes": "Good overview but could be more recent",
  "metric_labels": {
    "helpfulness_accuracy": {
      "agree_with_llm": true,
      "human_label": "pass",
      "confidence": 5,
      "notes": "Excellent response"
    },
    "completeness": {
      "agree_with_llm": false,
      "human_label": "pass",
      "confidence": 3,
      "notes": "Coverage is sufficient"
    }
  },
  "annotated_at": "2025-01-26T10:30:00Z",
  "annotator": "user123"
}
```

---

## 6. Calibration & Validation

### Flow

```
┌──────────────────┐     ┌──────────────────┐
│  Annotations     │     │  Eval Results    │
│  annotations.json│     │  eval_runs/*.json│
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────┬───────────────┘
                  │
                  v
┌─────────────────────────────────────────────┐
│  evalyn calibrate --metric-id ... --annot...│
│  1. Compute alignment metrics               │
│  2. Split data: 70% train, 30% validation   │
│  3. Run prompt optimization (GEPA/LLM)      │
│  4. VALIDATE: Re-run both prompts on val set│
│  5. Compare F1 scores                       │
└────────┬────────────────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  Validation Results              │
│  - Original F1                   │
│  - Optimized F1                  │
│  - Improvement delta             │
│  - is_better flag                │
│  - confidence (high/med/low)     │
│  - recommendation                │
└────────┬─────────────────────────┘
         │
         ├─────> ✅ Better (>2% improvement)
         │       → Save optimized prompt
         │       → Recommend: USE OPTIMIZED
         │
         └─────> ❌ Worse (<-2% degradation)
                 → Save for reference only
                 → Recommend: KEEP ORIGINAL
```

### Alignment Metrics

Compares human annotations vs LLM judge predictions:
- **Accuracy**: Overall agreement rate
- **Precision**: Of LLM's "pass" predictions, how many are correct?
- **Recall**: Of true "pass" cases, how many did LLM catch?
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: Of true "fail" cases, how many did LLM catch?
- **Cohen's Kappa**: Agreement adjusted for chance

### Prompt Optimization

**Method 1: LLM-based (default)**
- Analyzes disagreement cases
- Generates improved rubric suggestions
- Fast, deterministic

**Method 2: GEPA (Generative Evolutionary Prompt Adaptation)**
- Evolutionary search with LLM reflection
- Optimizes preamble while keeping rubric fixed
- More thorough but slower

```bash
# LLM-based optimization
evalyn calibrate --metric-id helpfulness_accuracy \
  --annotations annotations.jsonl \
  --dataset data/myproj-v1

# GEPA optimization
evalyn calibrate --metric-id helpfulness_accuracy \
  --annotations annotations.jsonl \
  --dataset data/myproj-v1 \
  --optimizer gepa

# No optimization (alignment metrics only)
evalyn calibrate --metric-id helpfulness_accuracy \
  --annotations annotations.jsonl \
  --no-optimize
```

### Validation

**Automatic validation** ensures optimized prompts are actually better:

1. **Split**: 70% training, 30% validation
2. **Re-run**: Both prompts on validation set
3. **Compare**: F1 scores
4. **Decide**:
   - Improvement >2% → Recommend optimized
   - Degradation <-2% → Keep original
   - Otherwise → Uncertain

**Thresholds**:
- High confidence: |delta| > 5%
- Medium confidence: 2% < |delta| < 5%
- Low confidence: |delta| < 2%

### CLI Output Example

```bash
evalyn calibrate --metric-id helpfulness_accuracy \
  --annotations annotations.jsonl \
  --dataset data/myproj-v1
```

**Success case:**
```
CALIBRATION REPORT: helpfulness_accuracy
========================================
Alignment Metrics:
  Accuracy:    0.85
  Precision:   0.88
  Recall:      0.82
  F1 Score:    0.85
  Specificity: 0.89
  Kappa:       0.70

Disagreement Analysis:
  Total disagreements: 8/50 (16%)

  False Positives (LLM said pass, human said fail): 3
  - Input: "What is AI?"
    LLM: "Response is helpful"
    Human: "Too vague, not specific enough"

  False Negatives (LLM said fail, human said pass): 5
  - Input: "Explain quantum computing"
    LLM: "Too technical"
    Human: "Appropriate level of detail"

Prompt Optimization (GEPA):
  Iterations: 10
  Best F1 on training set: 0.89

--- VALIDATION RESULTS ---
✅ SUCCESS - Optimized prompt is BETTER

  Original F1:     0.850
  Optimized F1:    0.920
  Improvement:     +0.070 (+7.0%)
  Confidence:      high
  Val samples:     15

💡 RECOMMENDATION: USE OPTIMIZED PROMPT

Saved to: data/myproj-v1/calibrations/helpfulness_accuracy/
  - calibration.json (full results)
  - prompts/20250126_123456_optimized.txt
  - prompts/20250126_123456_original.txt

Next: evalyn run-eval --latest --use-calibrated
```

**Failure case:**
```
--- VALIDATION RESULTS ---
❌ DEGRADED - Optimized prompt is WORSE

  Original F1:     0.850
  Optimized F1:    0.720
  Degradation:     -0.130 (-13.0%)
  Confidence:      high
  Val samples:     15

⚠️  RECOMMENDATION: KEEP ORIGINAL PROMPT

Note: Optimized prompt saved for reference but NOT recommended.
```

### Using Calibrated Prompts

```bash
# Re-run evaluation with calibrated prompts
evalyn run-eval --latest --use-calibrated

# Output shows which metrics use calibrated prompts:
# Loaded 5 metrics (2 objective, 3 subjective, 2 calibrated)
```

---

## 7. User Simulation

### Flow

```
┌──────────────────┐
│  Seed Dataset    │
│  (existing data) │
└────────┬─────────┘
         │
         v
┌─────────────────────────────────────────────┐
│  evalyn simulate --dataset ... --modes ...  │
│  Generation modes:                          │
│  - similar: Variations of seed queries      │
│  - outlier: Edge cases, adversarial inputs  │
└────────┬────────────────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  LLM-Generated Queries           │
│  (synthetic test cases)          │
└────────┬─────────────────────────┘
         │
         ├─────> Review only (--no-target)
         │       → Save to simulations/
         │
         └─────> Run through agent (--target)
                 → Execute queries
                 → Save results
                 → Auto-tag: is_simulation=True
```

### Generation Modes

**Similar Mode**: Generate variations to test robustness
```
Seed: "What is machine learning?"
Generated:
  - "Can you explain machine learning?"
  - "What's the concept of ML?"
  - "Define machine learning in simple terms"
```

**Outlier Mode**: Generate edge cases
```
Seed: "What is machine learning?"
Generated:
  - "What is machine learning using only emojis?"
  - "Explain ML in Shakespearean English"
  - "What if machine learning was invented in 1800?"
  - "ML but make it rhyme"
```

### CLI Commands

```bash
# Generate only (review before running agent)
evalyn simulate --dataset data/myproj-v1 --modes similar,outlier

# Generate and run through agent
evalyn simulate --dataset data/myproj-v1 \
  --target my_agent.py:my_agent \
  --modes similar,outlier

# Control generation parameters
evalyn simulate --dataset data/myproj-v1 \
  --target my_agent.py:my_agent \
  --num-similar 5 \
  --num-outlier 2 \
  --max-seeds 20 \
  --model gemini-2.5-flash-lite

# Use only similar mode
evalyn simulate --dataset data/myproj-v1 \
  --target my_agent.py:my_agent \
  --modes similar
```

### Output Structure

```
data/myproj-v1/simulations/
  sim-similar-20250126_123456/
    dataset.jsonl        # Generated queries + agent outputs
    meta.json            # Generation config

  sim-outlier-20250126_130000/
    dataset.jsonl
    meta.json
```

### Python API

```python
from evalyn_sdk.simulator import UserSimulator, AgentSimulator, SimulationConfig
from evalyn_sdk import load_dataset, eval

# Load seed dataset
seeds = load_dataset("data/myproj-v1/dataset.jsonl")

# Generate queries only
simulator = UserSimulator()
similar_queries = simulator.generate_similar(seeds, num_per_seed=3)
outlier_queries = simulator.generate_outliers(seeds, num_per_seed=2)

# Generate and run through agent
@eval(project="myproj", version="v1", is_simulation=True)
def my_agent(query: str) -> str:
    return process(query)

agent_sim = AgentSimulator(
    target_fn=my_agent,
    simulator=simulator,
    config=SimulationConfig(
        num_similar_per_seed=3,
        num_outlier_per_seed=2,
        max_seeds=20
    )
)

results = agent_sim.run(seeds)
# Results are automatically saved and tagged with is_simulation=True
```

### Use Cases

1. **Robustness Testing**: Similar queries test if agent handles variations consistently
2. **Edge Case Discovery**: Outlier queries find unexpected failure modes
3. **Regression Testing**: Re-run simulations after changes to detect degradation
4. **Data Augmentation**: Expand training/eval datasets with synthetic examples

---

## 8. Production vs Simulation Tracking

### Flow

```
┌──────────────────────────────────────┐
│  Production Traces                   │
│  @eval(is_simulation=False)          │
│  - Real user interactions            │
│  - Live system traces                │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  SQLite Storage                      │
│  metadata.is_simulation = False      │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  Filter in CLI                       │
│  --production flag                   │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  Simulation Traces                   │
│  @eval(is_simulation=True)           │
│  - Test/dev runs                     │
│  - Synthetic queries from simulator  │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  SQLite Storage                      │
│  metadata.is_simulation = True       │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  Filter in CLI                       │
│  --simulation flag                   │
└──────────────────────────────────────┘
```

### Usage

**In Code:**
```python
from evalyn_sdk import eval

# Production traces (default)
@eval(project="myproj", version="v1")
def production_agent(query: str) -> str:
    return process(query)

# Simulation traces (explicit)
@eval(project="myproj", version="v1", is_simulation=True)
def test_agent(query: str) -> str:
    return process(query)
```

**Auto-tagging in simulator:**
```python
# Automatically tagged with is_simulation=True
evalyn simulate --dataset data/myproj-v1 --target my_agent.py:my_agent
```

**CLI Filtering:**
```bash
# List only production traces
evalyn list-calls --production

# List only simulation traces
evalyn list-calls --simulation

# Build dataset from production only
evalyn build-dataset --project myproj --production

# Build dataset from simulations only
evalyn build-dataset --project myproj --simulation
```

**Output:**
```
id    | function   | project | version | sim? | status | duration
abc123| my_agent   | myproj  | v1      | Y    | OK     | 1234ms
def456| my_agent   | myproj  | v1      |      | OK     | 567ms
```

---

## Dataset Folder Reference

Complete structure of a dataset folder:

```
data/<project>-<version>-<timestamp>/
├── dataset.jsonl                    # Dataset items (input/output/label/metadata)
├── meta.json                        # Dataset metadata
│   ├── project_name
│   ├── version
│   ├── created_at
│   ├── item_count
│   ├── filters (what filters were used)
│   ├── active_metric_set (pointer to metrics file)
│   └── metric_sets (list of all metric files)
│
├── metrics/                         # Metric specifications
│   ├── basic-20250126_120000.json
│   ├── llm-registry-20250126_123000.json
│   └── bundle-research-agent.json
│
├── eval_runs/                       # Evaluation results
│   ├── 20250126_123456_run1.json
│   └── 20250126_140000_run2.json
│
├── calibrations/                    # Calibration results per metric
│   ├── helpfulness_accuracy/
│   │   ├── calibration.json         # Full calibration record
│   │   └── prompts/
│   │       ├── 20250126_123456_original.txt
│   │       └── 20250126_123456_optimized.txt
│   │
│   └── toxicity_safety/
│       ├── calibration.json
│       └── prompts/
│           └── 20250126_130000_optimized.txt
│
├── simulations/                     # Synthetic data generations
│   ├── sim-similar-20250126_140000/
│   │   ├── dataset.jsonl
│   │   └── meta.json
│   │
│   └── sim-outlier-20250126_150000/
│       ├── dataset.jsonl
│       └── meta.json
│
└── annotations.jsonl                # Human annotations
```

---

## CLI Command Reference

### Tracing & Collection

```bash
# List traced calls
evalyn list-calls [--limit N] [--project X] [--version Y] [--simulation] [--production]

# Show specific call details
evalyn show-call --id <call_id>

# View project summaries
evalyn show-projects
```

### Dataset Management

```bash
# Build dataset from traces
evalyn build-dataset --project <name> [--version V] [--limit N] [--production] [--simulation]

# Build from time range
evalyn build-dataset --project <name> --since "2025-01-01" --until "2025-01-31"
```

### Metric Selection

```bash
# Basic heuristic (no API)
evalyn suggest-metrics --target <file.py:func> --mode basic

# LLM-based selection from registry
evalyn suggest-metrics --target <file.py:func> --mode llm-registry --llm-mode api [--model MODEL]

# LLM brainstorm (custom metrics)
evalyn suggest-metrics --target <file.py:func> --mode llm-brainstorm --llm-mode api [--model MODEL]

# Pre-configured bundle
evalyn suggest-metrics --target <file.py:func> --mode bundle --bundle <name>

# Save to dataset
evalyn suggest-metrics --dataset <path> --target <file.py:func> --mode llm-registry --metrics-name <name>

# List available templates
evalyn list-metrics
```

### Evaluation

```bash
# Run with auto-detected metrics
evalyn run-eval --dataset <path>

# Run with specific metrics
evalyn run-eval --dataset <path> --metrics <metrics.json>

# Run with multiple metrics
evalyn run-eval --dataset <path> --metrics "file1.json,file2.json"

# Run with ALL metrics
evalyn run-eval --dataset <path> --metrics-all

# Use calibrated prompts
evalyn run-eval --dataset <path> --use-calibrated

# Run on latest dataset
evalyn run-eval --latest [--use-calibrated]

# View runs
evalyn list-runs
evalyn show-run --id <run_id>
```

### Annotation

```bash
# Interactive annotation
evalyn annotate --dataset <path>

# Per-metric annotation
evalyn annotate --dataset <path> --per-metric

# Resume session
evalyn annotate --dataset <path> --resume

# View stats
evalyn annotation-stats --dataset <annotations.jsonl>

# Export for external annotation
evalyn export-for-annotation --dataset <path> --output <file.jsonl>

# Import completed annotations
evalyn import-annotations --path <completed.jsonl>
```

### Calibration

```bash
# Full calibration (alignment + optimization + validation)
evalyn calibrate --metric-id <id> --annotations <file> --dataset <path>

# With GEPA optimizer
evalyn calibrate --metric-id <id> --annotations <file> --dataset <path> --optimizer gepa

# Alignment metrics only (no optimization)
evalyn calibrate --metric-id <id> --annotations <file> --no-optimize

# Show example disagreements
evalyn calibrate --metric-id <id> --annotations <file> --dataset <path> --show-examples

# Save calibration record
evalyn calibrate --metric-id <id> --annotations <file> --output <calibration.json>
```

### Simulation

```bash
# Generate only (review first)
evalyn simulate --dataset <path> --modes similar,outlier

# Generate and run through agent
evalyn simulate --dataset <path> --target <file.py:func> --modes similar,outlier

# Control parameters
evalyn simulate --dataset <path> --target <file.py:func> \
  --num-similar 5 --num-outlier 2 --max-seeds 20 --model gemini-2.5-flash-lite
```

---

## Python API Examples

### Basic Instrumentation

```python
from evalyn_sdk import eval, eval_session

@eval(project="myproj", version="v1")
def my_agent(query: str) -> str:
    # Your agent logic
    return result

# Group calls
with eval_session(session_id="user-123"):
    my_agent("query 1")
    my_agent("query 2")
```

### Custom Tracer

```python
from evalyn_sdk import EvalTracer, configure_tracer

tracer = EvalTracer(storage=my_storage)
configure_tracer(tracer)

@eval
def my_func():
    pass
```

### Dataset Operations

```python
from evalyn_sdk import load_dataset, save_dataset, build_dataset_from_storage
from evalyn_sdk.storage import get_default_storage

# Build from storage
storage = get_default_storage()
items = build_dataset_from_storage(
    storage,
    project_name="myproj",
    version="v1",
    production_only=True,
    limit=500
)

# Save/load
save_dataset(items, "dataset.jsonl")
items = load_dataset("dataset.jsonl")
```

### Metrics

```python
from evalyn_sdk import MetricRegistry, build_objective_metric, build_subjective_metric

registry = MetricRegistry()

# Add from template
registry.add_metric(build_objective_metric("latency_ms"))
registry.add_metric(build_subjective_metric("helpfulness_accuracy"))

# Add from file
registry.add_from_file("metrics/llm-selected.json")

# List available
from evalyn_sdk import list_template_ids
print(list_template_ids())
```

### Evaluation

```python
from evalyn_sdk import EvalRunner, MetricRegistry, load_dataset

# Setup
registry = MetricRegistry()
registry.add_from_file("metrics.json")

items = load_dataset("dataset.jsonl")

# Run
runner = EvalRunner(registry=registry)
eval_run = runner.run(items, target_fn=my_agent)

# Results
print(eval_run.summary)
for result in eval_run.results:
    print(f"{result.metric_id}: {result.score}")
```

### Calibration

```python
from evalyn_sdk import CalibrationEngine, Annotation, load_dataset
from evalyn_sdk.storage import get_default_storage

# Load data
storage = get_default_storage()
annotations = [Annotation.from_payload(a) for a in load_annotations()]
dataset = load_dataset("dataset.jsonl")

# Calibrate
engine = CalibrationEngine(storage=storage, dataset_items=dataset)
record = engine.calibrate(
    metric_id="helpfulness_accuracy",
    annotations=annotations,
    use_optimizer=True,
    optimizer="gepa"
)

# Check validation
if record.adjustments.get("validation", {}).get("is_better"):
    print("Optimized prompt is better!")
    print(f"Improvement: {record.adjustments['validation']['improvement_delta']:.1%}")
```

### Simulation

```python
from evalyn_sdk.simulator import UserSimulator, AgentSimulator, SimulationConfig
from evalyn_sdk import load_dataset, eval

seeds = load_dataset("dataset.jsonl")

# Generate queries only
simulator = UserSimulator()
similar = simulator.generate_similar(seeds, num_per_seed=3)
outliers = simulator.generate_outliers(seeds, num_per_seed=2)

# Run through agent
@eval(project="myproj", version="v1", is_simulation=True)
def my_agent(query: str) -> str:
    return process(query)

agent_sim = AgentSimulator(
    target_fn=my_agent,
    simulator=simulator,
    config=SimulationConfig(num_similar_per_seed=3)
)
results = agent_sim.run(seeds)
```

---

## Configuration

### Environment Variables

```bash
# API Keys
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# OpenTelemetry (optional)
export EVALYN_OTEL=off                    # Disable OTel
export EVALYN_OTEL_SERVICE=myservice      # Service name
export EVALYN_OTEL_EXPORTER=console       # console|sqlite
export EVALYN_OTEL_ENDPOINT=http://...    # OTLP endpoint

# Storage
export EVALYN_STORAGE_PATH=/custom/path/evalyn.sqlite
```

### Function-Level Configuration

```python
@eval(
    project="myproj",
    version="v1",
    is_simulation=False,
    metric_mode="llm-registry",      # auto-suggest mode
    metric_bundle="research-agent"   # pre-configured bundle
)
def my_agent(query: str) -> str:
    return process(query)
```

---

## Best Practices

### 1. Start Small, Iterate

```bash
# Day 1: Instrument and collect
@eval(project="myproj", version="v1")
def my_agent(query: str) -> str:
    return process(query)

# Day 2: Build dataset, suggest metrics
evalyn build-dataset --project myproj --limit 50
evalyn suggest-metrics --target agent.py:my_agent --mode basic

# Day 3: Run evaluation
evalyn run-eval --latest

# Day 4: Annotate subset
evalyn annotate --dataset data/myproj-v1 --limit 20

# Day 5: Calibrate
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl
```

### 2. Separate Production and Simulation

```python
# Production
@eval(project="myproj", version="v1", is_simulation=False)
def production_agent(query: str) -> str:
    return process(query)

# Testing/dev
@eval(project="myproj", version="v1-dev", is_simulation=True)
def test_agent(query: str) -> str:
    return process(query)
```

### 3. Use Calibrated Prompts

```bash
# Calibrate once
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl --dataset data/myproj-v1

# Always use calibrated prompts in evaluations
evalyn run-eval --latest --use-calibrated
```

### 4. Expand Coverage with Simulation

```bash
# Generate edge cases
evalyn simulate --dataset data/myproj-v1 --modes outlier --num-outlier 3

# Run agent on synthetic queries
evalyn simulate --dataset data/myproj-v1 --target agent.py:my_agent --modes similar,outlier

# Evaluate synthetic data
evalyn run-eval --dataset data/myproj-v1/simulations/sim-outlier-*
```

### 5. Version Your Metrics

```bash
# Save metrics with descriptive names
evalyn suggest-metrics --dataset data/myproj-v1 --target agent.py:my_agent \
  --mode llm-registry --metrics-name v1-initial

# Iterate
evalyn suggest-metrics --dataset data/myproj-v2 --target agent.py:my_agent \
  --mode llm-registry --metrics-name v2-improved
```

### 6. Monitor Calibration Quality

Always check validation results before using optimized prompts:

```bash
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl --dataset data/myproj-v1

# Look for:
# ✅ SUCCESS - Optimized prompt is BETTER
# 💡 RECOMMENDATION: USE OPTIMIZED PROMPT
```

If you see `❌ DEGRADED`, stick with the original prompt.

---

## Troubleshooting

### No traces captured

```bash
# Check if decorator is applied
@eval(project="myproj")
def my_func():
    pass

# Check storage
evalyn list-calls --limit 5

# Check if function is actually called
# (decorator only captures when function runs)
```

### LLM judge not working

```bash
# Check API key
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# Test with basic metrics first
evalyn suggest-metrics --target agent.py:func --mode basic

# Check metric configuration
cat data/myproj-v1/metrics/llm-selected.json
```

### Calibration fails

```bash
# Ensure you have enough annotations (minimum 10)
evalyn annotation-stats --dataset annotations.jsonl

# Check annotation format
cat annotations.jsonl | head -1 | jq

# Use --no-optimize to skip optimization and just see alignment
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl --no-optimize
```

### Simulation generates poor queries

```bash
# Reduce quantity, increase quality
evalyn simulate --dataset data/myproj-v1 --modes similar \
  --num-similar 2 --max-seeds 10

# Use better model
evalyn simulate --dataset data/myproj-v1 --modes outlier \
  --model gemini-2.5-flash-lite
```

---

## 9. One-Click Pipeline

Run the complete evaluation workflow with a single command - from dataset building to calibrated evaluation.

### Flow

```
┌──────────────────┐
│  evalyn one-click│
│  --project X     │
│  --target func   │
└────────┬─────────┘
         │
         v
┌─────────────────────────────────────────────┐
│  [1/7] Build Dataset                        │
│    → Filter traces by project/version       │
│    → Save to 1_dataset/                     │
└────────┬────────────────────────────────────┘
         v
┌─────────────────────────────────────────────┐
│  [2/7] Suggest Metrics                      │
│    → Mode: basic/llm-registry/bundle        │
│    → Save to 2_metrics/                     │
└────────┬────────────────────────────────────┘
         v
┌─────────────────────────────────────────────┐
│  [3/7] Run Initial Evaluation               │
│    → Apply all metrics                      │
│    → Save to 3_initial_eval/                │
└────────┬────────────────────────────────────┘
         v
┌─────────────────────────────────────────────┐
│  [4/7] Human Annotation (optional)          │
│    → Interactive CLI                        │
│    → Save to 4_annotations/                 │
│    → Can skip with Ctrl+C                   │
└────────┬────────────────────────────────────┘
         v
┌─────────────────────────────────────────────┐
│  [5/7] Calibrate LLM Judges (optional)      │
│    → For each subjective metric             │
│    → Validate optimized prompts             │
│    → Save to 5_calibrations/                │
└────────┬────────────────────────────────────┘
         v
┌─────────────────────────────────────────────┐
│  [6/7] Re-evaluate with Calibrated Prompts  │
│    → Load optimized prompts                 │
│    → Re-run evaluation                      │
│    → Save to 6_calibrated_eval/             │
└────────┬────────────────────────────────────┘
         v
┌─────────────────────────────────────────────┐
│  [7/7] Generate Simulations (optional)      │
│    → Generate synthetic queries             │
│    → Save to 7_simulations/                 │
└─────────────────────────────────────────────┘
         │
         v
┌──────────────────────────────────┐
│  pipeline_summary.json           │
│  (Complete execution report)     │
└──────────────────────────────────┘
```

### Quick Start

```bash
# Minimal pipeline (fast)
evalyn one-click --project myproj --target agent.py:my_agent

# Standard pipeline (with LLM metrics)
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry

# Full pipeline (with calibration and simulation)
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --annotation-limit 30 \
  --enable-simulation
```

### Command Options

**Required:**
- `--project <name>`: Project name to filter traces
- `--target <path>`: Target function (file.py:func or module:func)

**Dataset Options:**
- `--version <v>`: Version filter
- `--production-only`: Use only production traces
- `--simulation-only`: Use only simulation traces
- `--dataset-limit <N>`: Max items (default: 100)
- `--since <date>`: Filter traces since date (ISO format)
- `--until <date>`: Filter traces until date (ISO format)

**Metrics Options:**
- `--metric-mode <mode>`: basic|llm-registry|llm-brainstorm|bundle (default: basic)
- `--llm-mode <mode>`: api|local (default: api, for LLM modes)
- `--model <name>`: LLM model (default: gemini-2.5-flash-lite)
- `--bundle <name>`: Bundle name (if mode=bundle)

**Annotation Options:**
- `--skip-annotation`: Skip annotation step
- `--annotation-limit <N>`: Max items to annotate (default: 20)
- `--per-metric`: Use per-metric annotation mode

**Calibration Options:**
- `--skip-calibration`: Skip calibration step
- `--optimizer <type>`: llm|gepa (default: llm)
- `--calibrate-all-metrics`: Calibrate all subjective metrics

**Simulation Options:**
- `--enable-simulation`: Enable simulation step (off by default)
- `--simulation-modes <modes>`: similar,outlier (default: similar)
- `--num-similar <N>`: Similar queries per seed (default: 3)
- `--num-outlier <N>`: Outlier queries per seed (default: 2)
- `--max-sim-seeds <N>`: Max seeds for simulation (default: 10)

**Behavior:**
- `--output-dir <path>`: Custom output directory
- `--auto-yes`: Skip confirmation prompts
- `--verbose`: Show detailed logs
- `--dry-run`: Show what would be done without executing

### Output Structure

```
data/<project>-<version>-<timestamp>-oneclick/
├── 1_dataset/
│   ├── dataset.jsonl
│   └── meta.json
│
├── 2_metrics/
│   └── suggested.json
│
├── 3_initial_eval/
│   └── run_<timestamp>_<id>.json
│
├── 4_annotations/
│   └── annotations.jsonl
│
├── 5_calibrations/
│   ├── <metric_id_1>/
│   │   ├── calibration.json
│   │   └── prompts/
│   └── <metric_id_2>/
│       └── ...
│
├── 6_calibrated_eval/
│   └── run_<timestamp>_<id>.json
│
├── 7_simulations/
│   ├── sim-similar-<timestamp>/
│   └── sim-outlier-<timestamp>/
│
└── pipeline_summary.json  # Complete execution summary
```

### Usage Examples

**Minimal Pipeline (Basic Mode)**
```bash
# Fast, no LLM calls, no annotation
evalyn one-click --project myproj --target agent.py:my_agent \
  --skip-annotation

# Output:
# [1/7] Build dataset ✓
# [2/7] Suggest metrics (basic) ✓
# [3/7] Run evaluation ✓
# [4/7] Annotation ⏭️  SKIPPED
# [5/7] Calibration ⏭️  SKIPPED
# [6/7] Re-eval ⏭️  SKIPPED
# [7/7] Simulation ⏭️  SKIPPED
#
# Completes in ~30 seconds
```

**Standard Pipeline (LLM Metrics + Annotation)**
```bash
# Recommended workflow
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --llm-mode api \
  --annotation-limit 20

# Interactive steps:
# - Suggests metrics using LLM
# - Runs initial evaluation
# - Prompts for 20 annotations (interactive)
# - Calibrates metrics based on annotations
# - Re-runs with calibrated prompts
#
# Completes in ~5-10 minutes (with human input)
```

**Full Pipeline (With Simulation)**
```bash
# Complete workflow
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --annotation-limit 30 \
  --optimizer gepa \
  --enable-simulation \
  --simulation-modes similar,outlier

# All steps enabled:
# - LLM-powered metric selection
# - 30 human annotations
# - GEPA prompt optimization
# - Validated calibration
# - Synthetic data generation
#
# Completes in ~15-20 minutes
```

**Production-Only Dataset**
```bash
# Only use real production traces
evalyn one-click --project myproj --target agent.py:my_agent \
  --production-only \
  --dataset-limit 200 \
  --metric-mode llm-registry
```

**Custom Date Range**
```bash
# Specific time period
evalyn one-click --project myproj --target agent.py:my_agent \
  --since "2025-01-01" \
  --until "2025-01-31" \
  --metric-mode basic
```

**Dry Run (Preview)**
```bash
# See what would be done without executing
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --dry-run

# Output shows planned steps without execution
```

### Console Output Example

```
======================================================================
               EVALYN ONE-CLICK EVALUATION PIPELINE
======================================================================

Project:  myproj
Target:   agent.py:my_agent
Mode:     llm-registry (gemini-2.5-flash-lite)
Output:   data/myproj-20250126_143022-oneclick/

----------------------------------------------------------------------

[1/7] Building Dataset
  ✓ Found 100 items
  ✓ Saved to: data/myproj-20250126_143022-oneclick/1_dataset/dataset.jsonl

[2/7] Suggesting Metrics
  ✓ Selected 5 metrics (2 objective, 3 subjective)
    - latency_ms (objective)
    - json_valid (objective)
    - helpfulness_accuracy (subjective)
    - toxicity_safety (subjective)
    - completeness (subjective)
  ✓ Saved to: data/myproj-20250126_143022-oneclick/2_metrics/suggested.json

[3/7] Running Initial Evaluation
  ✓ Evaluated 100 items
  RESULTS:
    latency_ms: avg=1234.0
    json_valid: pass_rate=0.98
    helpfulness_accuracy: pass_rate=0.92
    toxicity_safety: pass_rate=1.00
    completeness: pass_rate=0.88
  ✓ Saved to: data/myproj-20250126_143022-oneclick/3_initial_eval/run_...json

[4/7] Human Annotation
  → Annotating 20 items...
  → Interactive annotation mode
  → Press Ctrl+C to skip this step

  [Interactive UI appears...]

  ✓ Completed 20 annotations
  ✓ Saved to: data/myproj-20250126_143022-oneclick/4_annotations/annotations.jsonl

[5/7] Calibrating LLM Judges
  → Calibrating 3 subjective metrics...

  [helpfulness_accuracy]
  CALIBRATION REPORT: helpfulness_accuracy
  ========================================
  Alignment Metrics:
    F1 Score: 0.85

  --- VALIDATION RESULTS ---
  ✅ SUCCESS - Optimized prompt is BETTER
  Original F1:     0.850
  Optimized F1:    0.920
  Improvement:     +0.070 (+7.0%)
  💡 RECOMMENDATION: USE OPTIMIZED PROMPT

  [toxicity_safety]
  F1 Score: 1.00 (perfect agreement!)
  ⏭️  SKIPPED optimization

  [completeness]
  ❌ DEGRADED - Optimized prompt is WORSE
  ⚠️  RECOMMENDATION: KEEP ORIGINAL PROMPT

[6/7] Re-evaluating with Calibrated Prompts
  ✓ Used 1 calibrated prompts
  ✓ Evaluated 100 items
  RESULTS:
    helpfulness_accuracy: pass_rate=0.96
  ✓ Saved to: data/myproj-20250126_143022-oneclick/6_calibrated_eval/run_...json

[7/7] Generating Simulations
  ⏭️  SKIPPED (use --enable-simulation to enable)

======================================================================
                    PIPELINE COMPLETE
======================================================================

Output directory: data/myproj-20250126_143022-oneclick/

Summary:
  ✓ 1_dataset: success
  ✓ 2_metrics: success
  ✓ 3_initial_eval: success
  ✓ 4_annotation: success
  ✓ 5_calibration: success
  ✓ 6_calibrated_eval: success
  ⏭️ 7_simulation: skipped

Next steps:
  1. Review results: cat data/myproj-20250126_143022-oneclick/6_calibrated_eval/run_...json
  2. View full summary: cat data/myproj-20250126_143022-oneclick/pipeline_summary.json
```

### Pipeline Summary JSON

After completion, `pipeline_summary.json` contains:

```json
{
  "started_at": "2025-01-26T14:30:22Z",
  "completed_at": "2025-01-26T14:38:56Z",
  "output_dir": "data/myproj-20250126_143022-oneclick",
  "config": {
    "project": "myproj",
    "target": "agent.py:my_agent",
    "metric_mode": "llm-registry",
    "dataset_limit": 100,
    "annotation_limit": 20
  },
  "steps": {
    "1_dataset": {
      "status": "success",
      "output": "...",
      "item_count": 100
    },
    "2_metrics": {
      "status": "success",
      "total": 5,
      "objective": 2,
      "subjective": 3
    },
    "3_initial_eval": {
      "status": "success",
      "run_id": "abc123..."
    },
    "4_annotation": {
      "status": "success",
      "count": 20
    },
    "5_calibration": {
      "status": "success",
      "metrics_calibrated": 3
    },
    "6_calibrated_eval": {
      "status": "success",
      "calibrated_count": 1
    },
    "7_simulation": {
      "status": "skipped"
    }
  }
}
```

### Skipping Steps

Steps can be skipped or interrupted:

**Skip Annotation:**
```bash
evalyn one-click --project myproj --target agent.py:my_agent --skip-annotation
```

**Skip Calibration:**
```bash
evalyn one-click --project myproj --target agent.py:my_agent --skip-calibration
```

**Skip Both:**
```bash
evalyn one-click --project myproj --target agent.py:my_agent \
  --skip-annotation --skip-calibration

# Runs: dataset → metrics → eval only
```

**Interrupt Annotation:**
During annotation, press `Ctrl+C` to skip remaining annotations and continue pipeline.

### Error Handling

**Pipeline Interruption:**
```
[3/7] Running Initial Evaluation
  [Processing...]
  ^C

⚠️  Pipeline interrupted by user
Partial results saved to: data/myproj-20250126_143022-oneclick/
Resume or inspect: cd data/myproj-20250126_143022-oneclick/
```

**Step Failure:**
```
[5/7] Calibrating LLM Judges
  [helpfulness_accuracy]
    ✗ Calibration failed: API rate limit exceeded

  [toxicity_safety]
    ✓ Calibration complete

# Pipeline continues with next step
```

All partial results are saved, allowing you to inspect intermediate outputs even if pipeline doesn't complete.

### Best Practices

**1. Start with Basic Mode**
```bash
# Test quickly before committing to LLM calls
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode basic \
  --skip-annotation
```

**2. Use Production-Only for Real Metrics**
```bash
# Exclude test/simulation traces
evalyn one-click --project myproj --target agent.py:my_agent \
  --production-only
```

**3. Preview with Dry Run**
```bash
# Check what would happen
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --dry-run
```

**4. Iterate on Small Samples**
```bash
# Use small limits while refining workflow
evalyn one-click --project myproj --target agent.py:my_agent \
  --dataset-limit 20 \
  --annotation-limit 10
```

**5. Enable Verbose for Debugging**
```bash
# See full error traces
evalyn one-click --project myproj --target agent.py:my_agent --verbose
```

### When to Use One-Click

**✓ Good for:**
- Initial evaluation of a new agent
- Regular automated evaluation runs
- Complete regression testing
- Onboarding team members (simple single command)
- CI/CD integration

**✗ Not ideal for:**
- Experimenting with different metric combinations (use `suggest-metrics` + `run-eval`)
- Custom annotation workflows (use `annotate` directly)
- When you need fine control over each step

For advanced workflows, use individual commands for more control.

---

## Advanced Topics

### Custom Metrics

```python
from evalyn_sdk import Metric, MetricSpec, MetricType, MetricResult

def my_custom_metric(output: str, **kwargs) -> MetricResult:
    score = len(output) / 100  # Example logic
    passed = score > 0.5
    return MetricResult(
        metric_id="custom_length",
        score=score,
        passed=passed,
        details={"length": len(output)}
    )

spec = MetricSpec(
    id="custom_length",
    name="Custom Length Check",
    type=MetricType.OBJECTIVE,
    description="Checks output length",
    config={}
)

metric = Metric(spec=spec, handler=my_custom_metric)
registry.add_metric(metric)
```

### Custom Storage Backend

```python
from evalyn_sdk.storage import StorageBackend
from evalyn_sdk import configure_tracer, EvalTracer

class MyStorage(StorageBackend):
    def save_call(self, call):
        # Your logic
        pass

    def list_calls(self, limit=100):
        # Your logic
        pass

storage = MyStorage()
tracer = EvalTracer(storage=storage)
configure_tracer(tracer)
```

### OpenTelemetry Integration

```python
from evalyn_sdk import configure_default_otel, eval

# Enable OTel with default config
configure_default_otel(service_name="myservice", exporter_type="console")

@eval(project="myproj")
def my_func():
    pass  # Spans automatically created
```

---

## Links

- **GitHub**: https://github.com/evalyn-ai/evalyn-sdk
- **Documentation**: https://evalyn.ai/docs
- **Examples**: https://github.com/evalyn-ai/evalyn-sdk/tree/main/example_agent

---

## License

MIT License - see LICENSE file for details.
