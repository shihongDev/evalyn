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
│   │   ├── cli/              # CLI module
│   │   │   ├── main.py       # CLI entry point & command registration
│   │   │   └── commands/     # Command implementations
│   │   ├── cli_impl.py       # Command implementations
│   │   ├── trace/            # Tracing & instrumentation
│   │   │   ├── tracer.py     # Core tracing logic
│   │   │   └── auto_instrument.py
│   │   ├── decorators.py     # @eval decorator
│   │   ├── runner.py         # Evaluation runner
│   │   ├── analyzer.py       # Report generation
│   │   ├── metrics/
│   │   │   ├── templates.py  # Objective metric templates (30)
│   │   │   ├── subjective.py # LLM judge templates (22)
│   │   │   └── objective.py  # Objective metric handlers
│   │   ├── annotation/       # Human annotation support
│   │   ├── simulation/       # Synthetic data generation
│   │   └── storage/
│   │       └── sqlite.py     # SQLite backend
│   └── pyproject.toml
├── example_agent/            # Reference LangGraph implementation
└── docs/
    └── clis/                 # CLI command documentation
```

## Adding a New Metric

### Objective Metrics

1. Add template to `sdk/evalyn_sdk/metrics/templates.py`:
```python
# Add to OBJECTIVE_TEMPLATES list
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
# Add to SUBJECTIVE_TEMPLATES list
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

1. Add command function in `sdk/evalyn_sdk/cli_impl.py`:
```python
def cmd_mycommand(args: argparse.Namespace) -> None:
    """Description."""
    # implementation
```

2. Import and register in `sdk/evalyn_sdk/cli/main.py`:
```python
from ..cli_impl import cmd_mycommand

# In main(), add to subparsers:
mycommand_parser = subparsers.add_parser("mycommand", help="...")
mycommand_parser.add_argument("--option", ...)
mycommand_parser.set_defaults(func=cmd_mycommand)
```

3. Add documentation in `docs/clis/mycommand.md`

## Questions?

Open an issue for questions or feature requests.
