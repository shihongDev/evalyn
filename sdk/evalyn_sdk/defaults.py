"""Default model constants for Evalyn SDK.

Centralizes default model names to avoid hardcoded strings throughout the codebase.

Model categories:
- FAST: Lightweight model for quick operations (scoring, basic evaluation)
- SMART: More capable model for complex tasks (generation, optimization)
- DSPY: DSPy-formatted model strings for GEPA integration
"""

# Fast lightweight model for basic operations
DEFAULT_EVAL_MODEL = "gemini-2.5-flash-lite"

# More capable model for generation and optimization
DEFAULT_GENERATOR_MODEL = "gemini-2.5-flash"

# DSPy-formatted model strings for GEPA
DEFAULT_DSPY_MODEL = "gemini/gemini-2.5-flash"

# Provider-specific defaults
DEFAULT_MODELS_BY_PROVIDER = {
    "gemini": DEFAULT_EVAL_MODEL,
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
}

__all__ = [
    "DEFAULT_EVAL_MODEL",
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_DSPY_MODEL",
    "DEFAULT_MODELS_BY_PROVIDER",
]
