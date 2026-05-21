import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

# Demo-wide model default. Users can override per stage with the more
# specific env vars below (e.g. ``EVALYN_DEMO_ANSWER_MODEL``), or override
# every stage at once via ``EVALYN_DEMO_MODEL``.
_DEFAULT_MODEL = os.environ.get("EVALYN_DEMO_MODEL", "gemini-2.5-flash-lite")


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default_factory=lambda: os.environ.get("EVALYN_DEMO_QUERY_MODEL", _DEFAULT_MODEL),
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default_factory=lambda: os.environ.get("EVALYN_DEMO_REFLECTION_MODEL", _DEFAULT_MODEL),
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default_factory=lambda: os.environ.get("EVALYN_DEMO_ANSWER_MODEL", _DEFAULT_MODEL),
        metadata={"description": "The name of the language model to use for the agent's answer."},
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
