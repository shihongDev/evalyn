# Evalyn SDK (Python)

Python toolkit for automatic evaluation of LLM agents: instrument functions with `@eval`, capture traces (with code metadata), run metrics, involve LLM judges, import human annotations, calibrate judges, and emit OpenTelemetry spans.

## What’s here
- `sdk/`: Python package with tracer/decorator, storage (SQLite), metric registry, runner, judges, suggester/selector, calibration, curation helper, and CLI entrypoint (`evalyn`).
- `example_agent/`: LangGraph-based agent wired with `@eval` plus a one-shot eval script.
- `docs/`: development plan and quick start (`docs/quick_start.md`).

## Quick start
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt   # installs local SDK + otel/llm deps + example_agent deps
evalyn --help
```

See `docs/quick_start.md` for the current pipeline, code snippet, and CLI cheatsheet.

## Housekeeping
- Ignored: node_modules, dist, __pycache__, venvs, coverage files (see .gitignore).
- Default SQLite database is created lazily at first use (`evalyn.sqlite`); delete it to reset stored traces/runs.
