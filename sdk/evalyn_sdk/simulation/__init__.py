"""
Simulation module for generating synthetic test data.

Provides tools for:
- Generating query variations (similar, outlier) via LLM
- Running agents on synthetic inputs
- Creating versioned simulation datasets
"""

from .simulator import (
    # LLM-based simulators
    UserSimulator,
    AgentSimulator,
    SimulationConfig,
    GeneratedQuery,
    create_versioned_dataset_dir,
    # Simple utilities
    synthetic_dataset,
    simulate_agent,
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
