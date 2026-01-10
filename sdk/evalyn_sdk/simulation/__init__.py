"""
Simulation module for generating synthetic test data.

Provides tools for:
- Generating query variations (similar, outlier)
- Running agents on synthetic inputs
- Creating versioned simulation datasets
"""

from .simulator import (
    UserSimulator,
    AgentSimulator,
    SimulationConfig,
    GeneratedQuery,
    create_versioned_dataset_dir,
)
from .simulation import (
    synthetic_dataset,
    simulate_agent,
    random_prompt_variations,
)

__all__ = [
    # Simulators
    "UserSimulator",
    "AgentSimulator",
    "SimulationConfig",
    "GeneratedQuery",
    "create_versioned_dataset_dir",
    # Utilities
    "synthetic_dataset",
    "simulate_agent",
    "random_prompt_variations",
]
