# Evalyn SDK Quick Start

This repo focuses on the **Python SDK** (`sdk/`). The former frontend demo has been removed.

## What's implemented now
- `@eval` decorator (sync/async) with trace capture: inputs, outputs/errors, duration, session id, trace events.
- Automatic capture of function metadata: signature, docstring, source (when available), hash, module/file path.
- Pluggable storage with SQLite backend (calls, runs, annotations).
- Metric system: registry with 50 metric templates (objective + subjective), covering latency/cost, text overlap, structure, tool usage, safety, and quality judging.
- Subjective metrics are rubric-based PASS/FAIL by default (LLM judge returns `passed` + `reason`).
- Use `evalyn list-metrics` to see each metric's required inputs and a short meaning description.
- LLM judges: GeminiJudge (default), OpenAIJudge (optional), and EchoJudge (debug).
- Dataset runner with optional caching (hash by inputs) and summary stats.
- OpenTelemetry spans: enabled by default if the otel dependency is installed; exporter defaults to `sqlite` (local). Disable with `EVALYN_OTEL=off` or configure exporter via env.
- Metric suggestion/selection:
  - Heuristic suggester.
  - LLM suggester for new specs.
  - LLM registry selector that picks from available metrics using code + traces.
  - Suggestions can be saved into a dataset folder (metrics sidecar + meta.json update).
- Human loop: import/export annotations; calibration engine to suggest judge thresholds.
- CLI commands for dataset runs, listing calls/runs, metric suggestion/selection, annotation import, and calibration.

## Pipeline (end-to-end)
1) **Instrument**: add `@eval` to your LLM-facing function. Calls are traced to SQLite (`evalyn.sqlite`) by default, with code metadata captured for metric selection.
2) **Collect**: run your agent; traces are stored (`evalyn list-calls`).
3) **Build dataset from traces**: use `evalyn build-dataset` (CLI) or `build_dataset_from_storage` (SDK) to export JSONL from stored calls.
4) **Suggest/Select Metrics**:
   - Heuristic suggestions: `evalyn suggest-metrics --target module:function --mode basic`
   - LLM registry selection: `evalyn suggest-metrics --target module:function --mode llm-registry`
   - Preset bundles: `evalyn suggest-metrics --target module:function --mode bundle --bundle research-agent`
   - Save suggestions into a dataset folder: `evalyn suggest-metrics --dataset data/<dataset_dir> --target module:function`
5) **Run Eval**: `evalyn run-eval --dataset data/myproj` (auto-detects metrics from meta.json) or specify metrics explicitly.
6) **Review Results**: Eval runs saved as JSON in `<dataset>/eval_runs/` with full scores and LLM judge reasoning.
7) **Annotate**: import human labels `evalyn import-annotations --path annotations.jsonl` (target_id should match call_id).
8) **Calibrate**: `evalyn calibrate --metric-id <judge-metric> --annotations annotations.jsonl --run-id <eval-run>` to get suggested threshold adjustments.

### Visual flow
```
@eval-wrapped function
      |
      v
 Traces (inputs/outputs/errors/code metadata, events)
      |
      +--> Build dataset ───────────────────────────────────────┐
      |     - evalyn build-dataset                              |
      |                                                         |
      +--> Metric selection                                     |
      |     - evalyn suggest-metrics                            |
      |     - basic/llm-registry/bundle modes                   |
      |                                                         v
      +--> Eval setup & run ──────────────────────────> dataset.jsonl
      |     - evalyn run-eval                            metrics/*.json
      |     - LLM judges score each item                 eval_runs/*.json
      |                                                         |
      v                                                         v
 Review results <───────────────────────────────────────────────┘
      |
      +--> Human annotation loop
      |     - evalyn annotate --dataset ... [--per-metric]
      |     - Interactive: y/n/skip with LLM judge context
      |     - Confidence scores (1-5)
      |     - Saves to annotations.jsonl
      |                    |
      v                    v
 Calibrate judges <────────┘
      |
      +--> Alignment metrics (precision, recall, F1, kappa)
      +--> Disagreement analysis (false positives/negatives)
      +--> LLM-optimized rubric suggestions
      |
      v
 Improved rubric ──> Re-run eval with better prompts (iterate)
      |
      v
 Simulation (expand coverage) ────────────────────────────────────┐
      |                                                           |
      +--> evalyn simulate --dataset ... --modes similar,outlier  |
      |     - Similar: Variations of seed queries (robustness)    |
      |     - Outlier: Edge cases, adversarial inputs             |
      |                                                           v
      +--> With --target: run agent on generated queries   simulations/
      |     and create new datasets for eval               ├─ sim-similar/
      |                                                    └─ sim-outlier/
      v
 Expanded dataset ──> Re-run eval on synthetic data (iterate)
```

## Minimal code example
```python
import json
from pathlib import Path
from evalyn_sdk import eval, EvalRunner, dataset_from_calls, get_default_tracer, build_objective_metric

@eval
def handle(user_input: str) -> str:
    return f"echo:{user_input}"

# 1) Run the function a few times (traces are captured automatically)
handle("hi")
handle("hello")

# 2) Build a dataset from successful past calls (no pre-created dataset needed)
tr = get_default_tracer()
dataset = dataset_from_calls(tr.storage.list_calls(), use_only_success=True)
# Optionally save to JSONL for reuse
Path("trace_dataset.jsonl").write_text("\n".join(json.dumps(item.__dict__) for item in dataset))

metrics = [
    build_objective_metric("latency_ms"),
    build_objective_metric("token_overlap_f1"),
]
runner = EvalRunner(target_fn=handle, metrics=metrics, instrument=False)
run = runner.run_dataset(dataset)
print(run.summary)
```

## CLI targets: paths first
All CLI commands that take `--target`/`--llm-caller` accept **file paths** without PYTHONPATH tweaks:
- Bash/WSL: `evalyn suggest-metrics --target example_agent/agent.py:run_agent --num-traces 5`
- PowerShell: `evalyn suggest-metrics --target example_agent\\agent.py:run_agent --num-traces 5`
If you prefer modules, dotted imports still work: `example_agent.agent:run_agent`.

### LLM-powered metric suggestion
`evalyn suggest-metrics` supports a built-in LLM mode:
- Registry-constrained LLM (default): `evalyn suggest-metrics --mode llm-registry --target ...`
- Free-form LLM brainstorm: `evalyn suggest-metrics --mode llm-brainstorm --target ...`
- Local Ollama: add `--llm-mode local --model llama3.1`
- API (OpenAI/Gemini): `--llm-mode api --model gpt-4` (or `gemini-2.5-flash-lite` with `GEMINI_API_KEY`)
- Preset bundles (no LLM): `evalyn suggest-metrics --mode bundle --bundle summarization` (bundles: summarization, orchestrator, research-agent)
- Default model: `gemini-2.5-flash-lite` (set `GEMINI_API_KEY` env var)
You can still supply your own callable with `--llm-caller` (takes a prompt string, returns metric dicts).

**Note on reference-based metrics**: Metrics like ROUGE, BLEU, and token_overlap require a golden standard. When using `suggest-metrics` with a dataset that has no reference values, these are automatically excluded.

## Subjective (LLM-judge) metric example
Subjective templates are **rubric-based PASS/FAIL**. Override the rubric (or policy/tone) via config.

```python
from evalyn_sdk import build_subjective_metric

# Requires GEMINI_API_KEY (default judge provider is Gemini)
metric = build_subjective_metric(
    "toxicity_safety",
    config={"rubric": ["No harassment or hate.", "No self-harm instructions."]},
)
```

## CLI cheatsheet (pipeline order)
- Instrument & run agent: `python -m example_agent.agent "your question"` (requires `GEMINI_API_KEY`)
- Inspect traces: `evalyn list-calls --limit 20`, then `evalyn show-call --id <call_id>`
- Project summary: `evalyn show-projects`
- Build dataset from stored traces: `evalyn build-dataset --project myproj --version v1 --limit 200`
- Suggest metrics (heuristic): `evalyn suggest-metrics --target example_agent/agent.py:run_agent --mode basic`
- Suggest metrics (LLM): `evalyn suggest-metrics --target example_agent/agent.py:run_agent --mode llm-registry --dataset data/myproj`
- Run eval (auto-detect metrics): `evalyn run-eval --dataset data/myproj`
- Run eval (specific metrics): `evalyn run-eval --dataset data/myproj --metrics metrics/llm-registry.json`
- Run eval (all metrics): `evalyn run-eval --dataset data/myproj --metrics-all`
- Run eval (multiple files): `evalyn run-eval --dataset data/myproj --metrics "metrics/basic.json,metrics/bundle.json"`
- List metric templates: `evalyn list-metrics` (shows category + required inputs)
- View eval runs/results: `evalyn list-runs`, `evalyn show-run --id <run_id>`
- Interactive annotation: `evalyn annotate --dataset data/myproj` (shows items one-by-one with LLM judge results)
- Import annotations: `evalyn import-annotations --path annotations.jsonl`
- Calibrate judges (basic): `evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl --run-id <run>`
- Calibrate with optimization: `evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl --dataset data/myproj --show-examples`
- Save calibration: `evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl --output calibration.json`
- Per-metric annotation: `evalyn annotate --dataset data/myproj --per-metric`
- Simulate users (generate queries): `evalyn simulate --dataset data/myproj --modes similar,outlier`
- Simulate users (with agent): `evalyn simulate --dataset data/myproj --target example_agent/agent.py:run_agent`

### Calibration workflow
The calibration system computes alignment metrics between LLM judges and human annotations, then uses LLM-based prompt optimization to suggest improved rubrics:

1. **Run eval first** (get LLM judge results): `evalyn run-eval --dataset data/myproj`
2. **Annotate interactively**: `evalyn annotate --dataset data/myproj`
3. **Calibrate**: `evalyn calibrate --metric-id <metric> --annotations data/myproj/annotations.jsonl --dataset data/myproj`

The interactive annotator shows each item with input, output, and LLM judge results side-by-side:
```
Item 1/5 [0063c461...]
📥 INPUT: {"question": "whats the current state of AI?"}
📤 OUTPUT: The current state of AI shows rapid advancement...
🤖 LLM JUDGE: helpfulness_accuracy: ✅ PASS

Pass? [y/n/s/v/q]: y
Notes (optional): Good comprehensive response
```

The calibration report includes:
- **Alignment metrics**: accuracy, precision, recall, F1, Cohen's kappa
- **Confusion matrix**: TP, TN, FP, FN counts
- **Disagreement analysis**: false positive/negative examples with reasons
- **Prompt optimization**: LLM-suggested rubric improvements

Use `--no-optimize` to skip LLM optimization, or `--model <model>` to change the optimizer model.

### Per-metric annotation
For fine-grained calibration, use `--per-metric` mode to annotate each metric separately:
```bash
evalyn annotate --dataset data/myproj --per-metric
```

This asks for each metric: "Do you agree with the LLM judge?" and "What's your own judgement?":
```
Item 1/5 [0063c461...]
📥 INPUT: {"question": "whats the current state of AI?"}
📤 OUTPUT: The current state of AI shows rapid advancement...
🤖 LLM JUDGE: helpfulness_accuracy: ✅ PASS

--- METRIC: helpfulness_accuracy ---
LLM said: PASS | Agree? [y/n/s]: n
Your judgement (pass/fail) [y/n]: n
Notes (optional): Response lacks specific examples

Confidence (1-5, 5=very confident): 4
```

### Simulation workflow
Generate synthetic test data by simulating users. Two modes:
- **similar**: Variations similar to seed inputs (test robustness)
- **outlier**: Edge cases, adversarial inputs, unusual requests

```bash
# Generate synthetic queries only (review before running)
evalyn simulate --dataset data/myproj --modes similar,outlier

# Generate and run through your agent
evalyn simulate --dataset data/myproj --target example_agent/agent.py:run_agent

# Control generation parameters
evalyn simulate --dataset data/myproj \
  --num-similar 5 \
  --num-outlier 2 \
  --max-seeds 20 \
  --model gemini-2.5-flash-lite
```

Output structure:
```
data/myproj/simulations/
  sim-similar-20250101-120000/
    dataset.jsonl
    meta.json
  sim-outlier-20250101-120000/
    dataset.jsonl
    meta.json
```

Run eval on simulated data:
```bash
evalyn run-eval --dataset data/myproj/simulations/sim-similar-20250101-120000
```

### Decorator hints for metric suggestion
You can annotate your function with a preferred suggestion mode:
```python
@eval(metric_mode="llm-registry", project="myproj", version="v1")
def my_agent(...):
    ...
```
- `metric_mode` options (validated): `llm-registry`, `llm-brainstorm`, `bundle`.
- For bundles, you can pair with `metric_bundle="summarization"` (or `orchestrator`, `research-agent`). Suggestion CLI with `--mode auto` will pick these defaults.

## Dataset layout (default)
`evalyn build-dataset` writes a folder per dataset:
```
data/<project>-<version>-<timestamp>/
  dataset.jsonl
  meta.json
  annotations.jsonl              # Human annotations (from evalyn annotate)
  metrics/
    llm-registry-<timestamp>.json
    bundle-summarization.json
    basic-<timestamp>.json
  eval_runs/
    <timestamp>_<run_id>.json
  simulations/                   # Synthetic data (from evalyn simulate)
    sim-similar-<timestamp>/
      dataset.jsonl
      meta.json
    sim-outlier-<timestamp>/
      dataset.jsonl
      meta.json
```
- `meta.json` includes filters, schema, counts, and `metric_sets` (list of saved metric selections).
- `active_metric_set` points to the most recently saved selection.
- `eval_runs/` contains JSON files for each evaluation with full results, scores, and LLM judge reasoning.
- `simulations/` contains synthetic datasets generated by the user simulator.

## Notes on cleanliness
- `.gitignore` covers node_modules/dist/__pycache__/venv/etc.
- Default SQLite file (`evalyn.sqlite`) is created lazily; delete it if you want a fresh slate.
