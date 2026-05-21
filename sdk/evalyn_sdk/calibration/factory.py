"""Optimizer factory: lazy-import, instantiate, and call optimizers by name.

Replaces the if/elif dispatch chain in engine.py. Handles both legacy
optimizers (with inconsistent signatures) and new BaseOptimizer subclasses.
"""

from __future__ import annotations

import inspect
from typing import Any

from .models import PromptOptimizationResult

# Registry: name -> (module_path, class_name)
OPTIMIZER_REGISTRY: dict[str, tuple[str, str]] = {
    "basic": ("evalyn_sdk.calibration.optimizers.basic", "BasicOptimizer"),
    "ape": ("evalyn_sdk.calibration.optimizers.ape", "APEOptimizer"),
    "opro": ("evalyn_sdk.calibration.optimizers.opro", "OPROOptimizer"),
    "gepa": ("evalyn_sdk.calibration.optimizers.gepa", "GEPAOptimizer"),
    "gepa-native": ("evalyn_sdk.calibration.optimizers.gepa_native", "GEPANativeOptimizer"),
    "evoprompt": ("evalyn_sdk.calibration.optimizers.evoprompt", "EvoPromptOptimizer"),
    "textgrad": ("evalyn_sdk.calibration.optimizers.textgrad", "TextGradOptimizer"),
    "miprov2": ("evalyn_sdk.calibration.optimizers.miprov2", "MIPROv2Optimizer"),
    "promptbreeder": ("evalyn_sdk.calibration.optimizers.promptbreeder", "PromptBreederOptimizer"),
}


def create_optimizer(
    name: str,
    config: Any = None,
    api_key: str | None = None,
    **legacy_kwargs,
) -> Any:
    """Lazy-import and instantiate optimizer by name.

    Args:
        name: optimizer name (must be in OPTIMIZER_REGISTRY)
        config: optimizer-specific Config dataclass
        api_key: optional API key
        **legacy_kwargs: for BasicOptimizer compatibility (model=..., etc.)
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: '{name}'. Available: {', '.join(sorted(OPTIMIZER_REGISTRY))}"
        )

    module_path, class_name = OPTIMIZER_REGISTRY[name]

    # Special handling for GEPA (external library)
    if name == "gepa":
        from .optimizers.gepa import GEPA_AVAILABLE

        if not GEPA_AVAILABLE:
            raise ImportError(
                "GEPA optimizer requires the 'gepa' package. Install with: pip install gepa"
            )

    # Lazy import
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Instantiate based on constructor signature
    init_sig = inspect.signature(cls.__init__)
    init_params = {p.name for p in init_sig.parameters.values() if p.name != "self"}

    if "config" in init_params and config is not None:
        kwargs = {"config": config}
        if "api_key" in init_params:
            kwargs["api_key"] = api_key
        return cls(**kwargs)
    elif "model" in init_params:
        # BasicOptimizer takes model, api_key
        kwargs: dict[str, Any] = {}
        if legacy_kwargs.get("model"):
            kwargs["model"] = legacy_kwargs["model"]
        if "api_key" in init_params and api_key:
            kwargs["api_key"] = api_key
        return cls(**kwargs)
    else:
        # Fallback
        try:
            return cls(config=config, api_key=api_key)
        except TypeError:
            try:
                return cls(config=config)
            except TypeError:
                return cls()


def call_optimizer(optimizer: Any, **kwargs) -> PromptOptimizationResult:
    """Call optimizer.optimize() with signature-aware kwarg filtering.

    New optimizers (with **kwargs in optimize()) get all params.
    Legacy optimizers get only the params their signature accepts.
    """
    sig = inspect.signature(optimizer.optimize)
    params = list(sig.parameters.values())

    # If optimize() accepts **kwargs, pass everything
    if any(p.kind == p.VAR_KEYWORD for p in params):
        return optimizer.optimize(**kwargs)

    # Otherwise filter to accepted params only
    accepted = {p.name for p in params if p.name != "self"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return optimizer.optimize(**filtered)
