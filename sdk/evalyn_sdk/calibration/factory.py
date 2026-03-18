"""Factory for creating calibration optimizers by name.

Provides a single entry point ``create_optimizer`` that maps string
names to concrete optimizer instances.
"""

from __future__ import annotations

from typing import Any, Optional


def create_optimizer(name: str, *, config: Any = None, api_key: Optional[str] = None) -> Any:
    """Create an optimizer instance by name.

    Supported names:
    - ``"ape"`` - APE (Automatic Prompt Engineer)
    - ``"opro"`` - OPRO (Optimization by PROmpting)
    - ``"textgrad"`` - TextGrad (critique-and-revise)
    - ``"basic"`` - BasicOptimizer (single-shot)

    Args:
        name: Optimizer name (case-insensitive).
        config: Optimizer-specific config dataclass. If ``None``, defaults
                are used.
        api_key: Optional API key forwarded to the optimizer.

    Returns:
        An optimizer instance.

    Raises:
        ValueError: If the name is not recognized.
    """
    key = name.lower().strip()

    if key == "ape":
        from .ape import APEConfig, APEOptimizer

        return APEOptimizer(config=config or APEConfig(), api_key=api_key)

    if key == "opro":
        from .opro import OPROConfig, OPROOptimizer

        return OPROOptimizer(config=config or OPROConfig(), api_key=api_key)

    if key == "textgrad":
        from .textgrad import TextGradConfig, TextGradOptimizer

        return TextGradOptimizer(config=config or TextGradConfig(), api_key=api_key)

    if key == "basic":
        from .basic import BasicOptimizer

        return BasicOptimizer(api_key=api_key)

    if key in ("gepa-native", "gepa_native"):
        from .gepa_native import GEPANativeConfig, GEPANativeOptimizer

        return GEPANativeOptimizer(config=config or GEPANativeConfig(), api_key=api_key)

    raise ValueError(
        f"Unknown optimizer: {name!r}. "
        f"Supported: ape, opro, textgrad, basic, gepa-native"
    )


__all__ = ["create_optimizer"]
