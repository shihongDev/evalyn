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

# Run only fast unit tests (no subprocess CLI tests)
uv run pytest tests/test_analysis.py tests/test_models_extended.py tests/test_storage.py tests/test_cli_utils.py tests/test_metrics_extended.py tests/test_cli_export.py -v
```

Note: the `tests/` directory is gitignored. Use `git add -f tests/<file>` to stage test files.

### Test file overview

| File | What it covers |
|------|---------------|
| `test_analysis.py` | analysis/trends.py, reports.py, core.py (MetricStats, ItemStats, RunAnalysis) |
| `test_models_extended.py` | Span, FunctionCall, DatasetItem, Annotation roundtrips via as_dict/from_dict |
| `test_storage.py` | SQLiteStorage CRUD, ID prefix resolution, annotations |
| `test_cli_utils.py` | formatters, validation, config, command_common utilities |
| `test_metrics_extended.py` | HeuristicSuggester, subjective template structure validation |
| `test_cli_export.py` | Export format builders (markdown, HTML, CSV), trace utility functions |
| `test_cli_commands.py` | CLI commands: analyze, compare, trend, list-runs, show-run (subprocess) |
| `test_cli.py` | CLI argument parsing, help text, basic command invocations (subprocess) |
| `test_cli_integration.py` | End-to-end workflows with real dataset directories (subprocess) |
| `test_insights.py` | Insights engine: correlations, regressions, recommendations |
| `test_sampling.py` | Dataset sampling modes |
| `test_tracing.py` | @eval decorator, span creation, auto-instrumentation |
| `test_metrics.py` | Metric registry, objective metric handlers |
| `test_ape.py` | APE optimizer config, parsing, UCB selection |
| `test_imports.py` | Package import verification |
| `test_*_instrumentors.py` | Provider-specific instrumentation (4 files, mocked frameworks) |

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

```
evalyn/
├── sdk/
│   ├── evalyn_sdk/
│   │   ├── cli/                    # CLI module
│   │   │   ├── main.py             # Entry point & command registration
│   │   │   ├── commands/           # Command implementations
│   │   │   │   ├── analysis.py     # analyze, compare, trend, status, validate
│   │   │   │   ├── traces.py       # list-calls, show-call, show-trace
│   │   │   │   ├── runs.py         # list-runs, show-run
│   │   │   │   ├── export.py       # export (json/csv/md/html), export-for-annotation
│   │   │   │   ├── annotation.py   # annotate, import-annotations, annotation-stats
│   │   │   │   ├── calibration.py  # calibrate, list-calibrations
│   │   │   │   ├── evaluation.py   # run-eval, build-dataset
│   │   │   │   └── simulate.py     # simulate
│   │   │   └── utils/              # CLI utilities
│   │   │       ├── formatters.py   # print_table, format_cost, output_json
│   │   │       ├── config.py       # Config file loading, dataset path resolution
│   │   │       ├── validation.py   # API key checks, extract_project_id
│   │   │       ├── command_common.py # resolve_call_id, resolve_dataset_dir_and_file
│   │   │       ├── ui.py           # Spinner, ProgressBar
│   │   │       ├── loaders.py      # Module/callable loading
│   │   │       └── llm_callers.py  # LLM API callers
│   │   ├── analysis/               # Report & analysis module
│   │   │   ├── core.py             # RunAnalysis, MetricStats, ItemStats, analyze_run()
│   │   │   ├── reports.py          # Text/ASCII reports, ascii_bar, score distributions
│   │   │   ├── insights.py         # Correlations, regressions, recommendations
│   │   │   ├── clustering.py       # Failure clustering (LLM-based)
│   │   │   ├── html_report.py      # HTML dashboard generation with Chart.js
│   │   │   └── trends.py           # Trend analysis over time
│   │   ├── trace/                  # Tracing & instrumentation
│   │   │   ├── tracer.py           # Core tracing logic
│   │   │   ├── auto_instrument.py  # Auto-patching for LLM libraries
│   │   │   └── instrumentation/    # Provider-specific instrumentors
│   │   ├── metrics/
│   │   │   ├── objective.py        # 73 objective metric templates + handlers
│   │   │   ├── subjective.py       # 60 LLM judge templates
│   │   │   ├── suggester.py        # HeuristicSuggester, LLMSuggester, TemplateSelector
│   │   │   └── factory.py          # Metric builders
│   │   ├── evaluation/             # Evaluation engine
│   │   │   ├── runner.py           # EvalRunner with checkpointing and caching
│   │   │   └── units/              # Span-level evaluation units
│   │   ├── judges/                 # LLM judge implementations
│   │   │   ├── llm_judge.py        # Core judge logic
│   │   │   └── confidence/         # Confidence estimation (logprobs, consistency)
│   │   ├── simulation/             # Synthetic data generation
│   │   ├── storage/                # Persistence backends
│   │   │   ├── base.py             # StorageBackend protocol
│   │   │   └── sqlite.py           # SQLiteStorage implementation
│   │   ├── decorators.py           # @eval decorator
│   │   ├── datasets.py             # load_dataset, save_dataset, dataset_from_calls
│   │   └── models.py               # Span, FunctionCall, EvalRun, DatasetItem, etc.
│   └── pyproject.toml
├── tests/                          # Test suite (gitignored, use git add -f)
│   ├── conftest.py                 # Fixtures, CLIResult helper, test constants
│   ├── test_analysis.py            # Analysis engine tests
│   ├── test_models_extended.py     # Model roundtrip tests
│   ├── test_storage.py             # SQLiteStorage tests
│   ├── test_cli_utils.py           # CLI utility tests
│   ├── test_cli_export.py          # Export format tests
│   ├── test_cli_commands.py        # CLI command tests (subprocess)
│   ├── test_cli.py                 # CLI argument/help tests (subprocess)
│   ├── test_metrics_extended.py    # Suggester + template validation tests
│   └── ...                         # Instrumentation, tracing, insights tests
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
