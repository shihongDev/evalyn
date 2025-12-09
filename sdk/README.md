# Evalyn Python SDK (work in progress)

This package provides the Python building blocks for Evalyn's automatic evaluation framework:
- `@eval` decorator to trace LLM-facing functions (sync + async)
- pluggable storage (SQLite included)
- metric registry with objective/subjective metrics
- eval runner over datasets
- hooks for LLM judges, human annotation, and calibration

## Local development
1) Create a virtual environment in `sdk/`:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2) Install the package in editable mode:
   ```
   pip install -e ".[dev]"
   ```
3) Run the CLI help:
   ```
   evalyn --help
   ```
