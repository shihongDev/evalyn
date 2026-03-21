# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-21

### Added
- Initial public release on PyPI
- Auto-instrumentation for OpenAI, Anthropic, Google Gemini, LangChain, CrewAI, AutoGen, DSPy, Haystack, LlamaIndex, Semantic Kernel
- 130+ built-in evaluation metrics (73 objective, 60 LLM judges)
- Tracing with SQLite storage (fully local, no cloud dependencies)
- CLI with 32 commands covering the full evaluation pipeline
- `one-click` command for running the entire pipeline in a single step
- Calibration system with 5 optimizers: GEPA, EvoPrompt, TextGrad, MIPROv2, PromptBreeder
- Human-in-the-loop annotation workflow
- Insights engine with deterministic analysis and LLM expert panel
- HTML dashboard with interactive Chart.js visualizations
- Failure and misalignment clustering
- Synthetic data simulation
- Evaluation trend tracking and run comparison
