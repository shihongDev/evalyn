# Contributing to Evalyn

## Development Setup

```bash
# Clone the repository
git clone https://github.com/anthropics/evalyn.git
cd evalyn

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e "./sdk[dev,llm]"

# Set up API key for testing
export GEMINI_API_KEY="your-key"
```

## Running Tests

```bash
# Run all tests (excluding slow/integration)
uv run pytest tests/ -m "not slow"

# Run a specific test file
uv run pytest tests/test_analysis.py -v

# Run with coverage report
uv run pytest tests/ --cov=evalyn_sdk --cov-report=term-missing

# Run core unit tests only (no subprocess CLI tests)
uv run pytest tests/test_analysis.py tests/test_models_extended.py tests/test_storage.py tests/test_cli_utils.py tests/test_metrics_extended.py tests/test_cli_export.py tests/test_insights.py -v
```

Note: the `tests/` directory is gitignored. Use `git add -f tests/<file>` to stage test files.

### Test file overview

The test suite contains 17,300+ tests across 460 files. Key test files by area:

| Area | Key files | What they cover |
|------|-----------|----------------|
| **Core** | `test_analysis.py`, `test_models_extended.py`, `test_storage.py` | RunAnalysis, MetricStats, model roundtrips, SQLiteStorage CRUD |
| **CLI** | `test_cli.py`, `test_cli_commands.py`, `test_cli_integration.py`, `test_cli_utils.py`, `test_cli_export.py` | Argument parsing, command execution, end-to-end workflows, formatters, export formats |
| **Metrics** | `test_metrics.py`, `test_metrics_extended.py` | Metric registry, objective handlers, suggester, template validation |
| **Insights** | `test_insights.py` | Correlations, regressions, recommendations |
| **Tracing** | `test_tracing.py`, `test_*_instrumentors.py` | @eval decorator, spans, provider-specific instrumentation (mocked) |
| **Calibration** | `test_ape.py`, `test_calibration_*.py` | Optimizer configs, convergence, active learning, ensemble fusion |
| **Sampling** | `test_sampling.py`, `test_*_sampling.py` | Dataset sampling modes, stratified, diversity, importance, coreset |
| **Evaluation** | `test_agentic_benchmarks.py`, `test_async_strategy.py`, `test_consistency_testing.py` | Eval runner, batch processing, judge consistency, pairwise eval |
| **Analysis** | `test_analysis_snapshots.py`, `test_cohort_analysis.py`, `test_confusion_matrix.py`, ... | 75+ analysis module tests |
| **Storage** | `test_storage.py`, `test_compaction.py`, `test_encryption.py`, ... | SQLite, migrations, encryption, connection pooling, backup |
| **Integration** | `test_api_server.py`, `test_cicd.py`, `test_github_action.py` | API server, CI/CD, webhooks, Phoenix export |
| **Simulation** | `test_adversarial_simulation.py`, `test_multiturn_simulation.py`, ... | Synthetic data generation, persona simulation, constraint simulation |

Most modules have a corresponding `test_<module>.py` file. When adding a new module, add its test file following the same pattern.

### Writing tests

- Use realistic data, not trivial single-field examples
- Pure function tests are preferred over subprocess CLI tests where possible
- Import from `conftest.py` for shared fixtures (`temp_dir`, `temp_db`, `sample_dataset`, `realistic_eval_run_data`)
- Shared timestamp constants `T0`-`T3` are available from conftest for building test data

## Code Style

- **Formatter**: We use `ruff` for formatting
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for public functions

```bash
# Format code
ruff format .

# Check linting
ruff check .
```

## Pull Request Process

1. **Fork & branch**: Create a feature branch from `main`
2. **Make changes**: Keep PRs focused on a single feature/fix
3. **Test**: Ensure all tests pass
4. **Document**: Update docs if adding new features
5. **PR description**: Explain what and why

### Commit Messages

```
<type>: <short description>

<optional body>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Example:
```
feat: add rouge-l metric template

Adds ROUGE-L score calculation for summarization evaluation.
```

## Project Structure

The SDK contains 579 Python modules across 21 packages. Here is the high-level layout:

```
evalyn/
├── sdk/
│   ├── evalyn_sdk/                 # 579 modules total
│   │   ├── cli/                    # CLI module (37 files)
│   │   │   ├── main.py             # Entry point & command registration
│   │   │   ├── commands/           # 14 command modules
│   │   │   │   ├── analysis.py     # analyze, compare, trend, status, validate
│   │   │   │   ├── traces.py       # list-calls, show-call, show-trace
│   │   │   │   ├── runs.py         # list-runs, show-run
│   │   │   │   ├── export.py       # export (json/csv/md/html), export-for-annotation
│   │   │   │   ├── annotation.py   # annotate, import-annotations, annotation-stats
│   │   │   │   ├── calibration.py  # calibrate, list-calibrations
│   │   │   │   ├── evaluation.py   # run-eval, build-dataset
│   │   │   │   ├── simulate.py     # simulate
│   │   │   │   ├── dashboard.py    # dashboard
│   │   │   │   ├── dataset.py      # dataset management commands
│   │   │   │   ├── insights.py     # insights
│   │   │   │   ├── infrastructure.py # doctor, config, init
│   │   │   │   ├── quickstart.py   # quickstart, workflow
│   │   │   │   └── clustering.py   # cluster-failures, cluster-misalignments
│   │   │   └── utils/              # 17 CLI utility modules
│   │   │       ├── formatters.py   # print_table, format_cost, output_json
│   │   │       ├── config.py       # Config file loading, dataset path resolution
│   │   │       ├── validation.py   # API key checks, extract_project_id
│   │   │       ├── command_common.py # resolve_call_id, resolve_dataset_dir_and_file
│   │   │       ├── ui.py           # Spinner, ProgressBar
│   │   │       ├── loaders.py      # Module/callable loading
│   │   │       ├── llm_callers.py  # LLM API callers
│   │   │       ├── pipeline.py     # One-click pipeline orchestration
│   │   │       ├── colors.py       # Terminal color support
│   │   │       ├── hints.py        # Contextual hint system
│   │   │       └── ...             # dataset_resolver, env_check, errors, compact, etc.
│   │   ├── analysis/               # Report & analysis (78 modules)
│   │   │   ├── core.py             # RunAnalysis, MetricStats, ItemStats, analyze_run()
│   │   │   ├── reports.py          # Text/ASCII reports, ascii_bar, score distributions
│   │   │   ├── insights.py         # Correlations, regressions, recommendations
│   │   │   ├── clustering.py       # Failure clustering (LLM-based)
│   │   │   ├── html_report.py      # HTML dashboard generation with Chart.js
│   │   │   ├── trends.py           # Trend analysis over time
│   │   │   ├── cohort_analysis.py  # Cohort-based analysis
│   │   │   ├── confusion_matrix.py # Confusion matrix generation
│   │   │   ├── cost_dashboard.py   # Cost tracking dashboards
│   │   │   ├── forecast.py         # Metric forecasting
│   │   │   ├── root_cause.py       # Root cause analysis
│   │   │   ├── what_if.py          # What-if scenario simulation
│   │   │   ├── significance_testing.py # Statistical significance
│   │   │   └── ...                 # 65+ more: sensitivity, normalization, regression_bisection, etc.
│   │   ├── trace/                  # Tracing & instrumentation (47 modules)
│   │   │   ├── tracer.py           # Core tracing logic
│   │   │   ├── auto_instrument.py  # Auto-patching for LLM libraries
│   │   │   ├── instrumentation/    # Provider-specific instrumentors
│   │   │   │   └── providers/      # 14 providers + shared utilities
│   │   │   ├── pii_redaction.py    # PII redaction in traces
│   │   │   ├── otel_export.py      # OpenTelemetry export
│   │   │   ├── flame_graph.py      # Flame graph visualization
│   │   │   └── ...                 # compression, session_replay, query_language, etc.
│   │   ├── calibration/            # Calibration engine (47 modules)
│   │   │   ├── engine.py           # Core calibration logic
│   │   │   ├── gepa.py             # GEPA optimizer
│   │   │   ├── ape.py              # APE optimizer
│   │   │   ├── opro.py             # OPRO optimizer
│   │   │   ├── evoprompt.py        # EvoPrompt optimizer
│   │   │   ├── textgrad.py         # TextGrad optimizer
│   │   │   ├── miprov2.py          # MIPROv2 optimizer
│   │   │   ├── promptbreeder.py    # PromptBreeder optimizer
│   │   │   ├── active_learning.py  # Active learning strategies
│   │   │   ├── ensemble_fusion.py  # Ensemble fusion
│   │   │   └── ...                 # 37+ more: curriculum, sensitivity, cost_tracking, etc.
│   │   ├── evaluation/             # Evaluation engine (72 modules)
│   │   │   ├── runner.py           # EvalRunner with checkpointing and caching
│   │   │   ├── batch/              # Batch processing
│   │   │   ├── units/              # Span-level evaluation units
│   │   │   ├── pairwise.py         # Pairwise evaluation
│   │   │   ├── multi_turn.py       # Multi-turn evaluation
│   │   │   ├── cross_validation.py # Cross-validation
│   │   │   ├── ab_testing.py       # A/B testing
│   │   │   └── ...                 # 65+ more: streaming, distributed, circuit_breaker, etc.
│   │   ├── storage/                # Persistence backends (30 modules)
│   │   │   ├── base.py             # StorageBackend protocol
│   │   │   ├── sqlite.py           # SQLiteStorage implementation
│   │   │   ├── encryption.py       # At-rest encryption
│   │   │   ├── connection_pool.py  # Connection pooling
│   │   │   ├── migration.py        # Schema migrations
│   │   │   └── ...                 # 25+ more: compaction, backup, partitioning, etc.
│   │   ├── metrics/                # Metric system (20 modules)
│   │   │   ├── objective.py        # 76 objective metric templates + handlers
│   │   │   ├── subjective.py       # 60 LLM judge templates
│   │   │   ├── suggester.py        # HeuristicSuggester, LLMSuggester, TemplateSelector
│   │   │   ├── factory.py          # Metric builders
│   │   │   ├── audio_evaluation.py # Audio evaluation metrics
│   │   │   ├── image_evaluation.py # Image evaluation metrics
│   │   │   ├── video_evaluation.py # Video evaluation metrics
│   │   │   └── ...                 # custom_dsl, explanations, snapshot_testing, etc.
│   │   ├── judges/                 # LLM judge implementations (7 modules)
│   │   │   ├── llm_judge.py        # Core judge logic
│   │   │   └── confidence/         # Confidence estimation (logprobs, consistency, verbalized)
│   │   ├── integration/            # External integrations (12 modules)
│   │   │   ├── api_server.py       # REST API server
│   │   │   ├── github_action.py    # GitHub Actions integration
│   │   │   ├── cicd.py             # CI/CD pipeline integration
│   │   │   ├── webhooks.py         # Webhook notifications
│   │   │   └── ...                 # Phoenix export, OpenInference, team collaboration, etc.
│   │   ├── annotation/             # Annotation subsystem (3 modules)
│   │   ├── simulation/             # Synthetic data generation (2 modules)
│   │   ├── testing/                # Test helpers (3 modules: assertion_framework, fuzz_testing)
│   │   ├── utils/                  # Shared utilities (2 modules)
│   │   ├── decorators.py           # @eval decorator
│   │   ├── datasets.py             # load_dataset, save_dataset, dataset_from_calls
│   │   ├── models.py               # Span, FunctionCall, EvalRun, DatasetItem, etc.
│   │   └── ...                     # 190 top-level modules (sampling, simulation, config, etc.)
│   └── pyproject.toml
├── tests/                          # 17,300+ tests in 460 files (gitignored, use git add -f)
│   ├── conftest.py                 # Fixtures, CLIResult helper, test constants
│   └── test_*.py                   # One test file per module, plus integration tests
├── example_agents/                 # SDK integration examples
└── docs/
    └── clis/                       # CLI command documentation
```

## Adding a New Metric

### Objective Metrics

1. Add template to `sdk/evalyn_sdk/metrics/objective.py` (in the `OBJECTIVE_REGISTRY` list):
```python
# Add to OBJECTIVE_REGISTRY list
{
    "id": "my_metric",
    "type": "objective",
    "description": "What it measures",
    "category": "correctness",  # efficiency, structure, robustness, correctness, grounding
    "scope": "overall",         # overall, llm_call, tool_call, trace
    "config": {"threshold": 0.8},
    "requires_reference": False,
}
```

2. Add handler function to `sdk/evalyn_sdk/metrics/objective.py` and register it

### Subjective Metrics (LLM Judges)

1. Add template to `sdk/evalyn_sdk/metrics/subjective.py`:
```python
# Add to SUBJECTIVE_REGISTRY list
{
    "id": "my_judge_metric",
    "type": "subjective",
    "description": "What this metric evaluates",
    "category": "correctness",
    "scope": "overall",
    "prompt": "You are a judge for X. Evaluate whether...",
    "config": {
        "rubric": ["Criterion 1", "Criterion 2"],
        "threshold": 0.5,
    },
    "requires_reference": False,
}
```

## Adding a New CLI Command

1. Create a command module in `sdk/evalyn_sdk/cli/commands/mycommand.py`:
```python
def cmd_mycommand(args: argparse.Namespace) -> None:
    """Description."""
    # implementation

def register_commands(subparsers) -> None:
    p = subparsers.add_parser("mycommand", help="...")
    p.add_argument("--option", ...)
    p.set_defaults(func=cmd_mycommand)
```

2. Import and register in `sdk/evalyn_sdk/cli/main.py`:
```python
from .commands import mycommand
mycommand.register_commands(subparsers)
```

3. Add documentation in `docs/clis/mycommand.md`
4. Add tests in `tests/test_cli_commands.py` or a new test file

## Questions?

Open an issue for questions or feature requests.
