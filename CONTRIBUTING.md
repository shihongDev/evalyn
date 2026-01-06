# Contributing to Evalyn

## Development Setup

```bash
# Clone and install
git clone https://github.com/anthropics/evalyn.git
cd evalyn/sdk
pip install -e ".[dev,llm]"

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
sdk/
├── evalyn_sdk/
│   ├── cli.py           # CLI commands
│   ├── tracing.py       # @eval decorator & tracing
│   ├── runner.py        # Evaluation runner
│   ├── analyzer.py      # Report generation
│   ├── metrics/
│   │   ├── templates.py # 50+ metric templates
│   │   ├── objective.py # Deterministic metrics
│   │   └── subjective.py# LLM judge metrics
│   └── storage/
│       └── sqlite.py    # SQLite backend
├── tests/
└── example_agent/       # Reference implementation
```

## Adding a New Metric

1. Add template to `metrics/templates.py`:
```python
OBJECTIVE_TEMPLATES["my_metric"] = {
    "id": "my_metric",
    "type": "objective",
    "description": "What it measures",
    "category": "correctness",
    "inputs": ["output"],
    "config": {"threshold": 0.8},
}
```

2. Add handler to `metrics/objective.py` or `metrics/subjective.py`

3. Add test in `tests/test_metrics.py`

## Adding a New CLI Command

1. Add function in `cli.py`:
```python
def cmd_mycommand(args: argparse.Namespace) -> None:
    """Description."""
    # implementation
```

2. Register in `main()`:
```python
parser = subparsers.add_parser("mycommand", help="...")
parser.add_argument("--option", ...)
parser.set_defaults(func=cmd_mycommand)
```

3. Add docs in `docs/clis/mycommand.md`

## Questions?

Open an issue for questions or feature requests.
