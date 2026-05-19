"""
Simulation module for generating synthetic test data.

Provides tools for:
- Generating query variations (similar, outlier) via LLM
- Running agents on synthetic inputs
- Creating versioned simulation datasets
"""

from .simulator import (
    AgentSimulator,
    GeneratedQuery,
    SimulationConfig,
    # LLM-based simulators
    UserSimulator,
    create_versioned_dataset_dir,
    simulate_agent,
    # Simple utilities
    synthetic_dataset,
)

__all__ = [
    # LLM-based simulators
    "UserSimulator",
    "AgentSimulator",
    "SimulationConfig",
    "GeneratedQuery",
    "create_versioned_dataset_dir",
    # Simple utilities
    "synthetic_dataset",
    "simulate_agent",
]
