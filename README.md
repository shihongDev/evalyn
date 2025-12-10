# Evalyn SDK (Python)

Python toolkit for automatic evaluation of LLM agents: instrument functions with `@eval`, capture traces (with code metadata), run metrics, involve LLM judges, import human annotations, and calibrate judges. Frontend assets have been removed; this repo now focuses on the SDK.

## What’s here
- `sdk/`: Python package with tracer/decorator, storage (SQLite), metric registry, runner, judges, suggester/selector, calibration, and CLI entrypoint (`evalyn`).
- `docs/`: development plan and quick start (`docs/quick_start.md`).

## Quick start
```
cd sdk
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -e ".[dev]"        # add [llm] if using OpenAI judge
evalyn --help
```

See `docs/quick_start.md` for the current pipeline, code snippet, and CLI cheatsheet.

## Housekeeping
- Ignored: node_modules, dist, __pycache__, venvs, coverage files (see .gitignore).
- Default SQLite database is created lazily at first use (`evalyn.sqlite`); delete it to reset stored traces/runs.
