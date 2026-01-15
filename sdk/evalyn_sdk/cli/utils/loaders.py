"""Module and callable loading utilities for CLI."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from typing import Any, Callable, List


def _get_module_callables(module: Any) -> List[str]:
    """Get all callable names from a module for error suggestions."""
    callables = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if callable(obj) and not inspect.isclass(obj):
            callables.append(name)
    return sorted(callables)


def _suggest_similar(
    name: str, candidates: List[str], max_suggestions: int = 3
) -> List[str]:
    """Find similar names using simple substring matching."""
    name_lower = name.lower()
    # Exact prefix match first
    prefix_matches = [c for c in candidates if c.lower().startswith(name_lower)]
    if prefix_matches:
        return prefix_matches[:max_suggestions]
    # Substring match
    substr_matches = [
        c for c in candidates if name_lower in c.lower() or c.lower() in name_lower
    ]
    return substr_matches[:max_suggestions]


def _load_callable(target: str) -> Callable[..., Any]:
    """
    Load a function reference. Supports:
    - path/to/file.py:function_name   (preferred, no PYTHONPATH fuss)
    - module.path:function_name       (fallback)
    """
    if ":" not in target:
        raise ValueError(
            f"Target must be 'path/to/file.py:function' or 'module:function'\n"
            f"Got: {target}\n"
            f"Example: evalyn suggest-metrics --target example_agent/agent.py:run_agent"
        )

    left, func_name = target.split(":", 1)

    def _get_attr_with_suggestions(
        module: Any, name: str, module_path: str
    ) -> Callable[..., Any]:
        """Get attribute with helpful error message if not found."""
        if hasattr(module, name):
            return getattr(module, name)

        available = _get_module_callables(module)
        similar = _suggest_similar(name, available)

        error_msg = f"Function '{name}' not found in {module_path}"
        if similar:
            error_msg += "\n\nDid you mean one of these?\n"
            for s in similar:
                error_msg += f"  - {left}:{s}\n"
        elif available:
            error_msg += "\n\nAvailable functions:\n"
            for fn in available[:10]:
                error_msg += f"  - {fn}\n"
            if len(available) > 10:
                error_msg += f"  ... and {len(available) - 10} more\n"
        raise AttributeError(error_msg)

    # Path-based load first
    if left.endswith(".py") or os.path.sep in left:
        path = os.path.abspath(left if left.endswith(".py") else left + ".py")
        if not os.path.isfile(path):
            raise ImportError(
                f"Cannot find file: {path}\n"
                f"Make sure the file path is correct and the file exists."
            )
        mod_name = os.path.splitext(os.path.basename(path))[0]
        # Ensure package imports inside the file can resolve (e.g., `from pkg.module import x`)
        pkg_dir = os.path.dirname(path)
        pkg_init = os.path.join(pkg_dir, "__init__.py")
        if os.path.isfile(pkg_init):
            sys.path.insert(0, os.path.dirname(pkg_dir))
        else:
            sys.path.insert(0, pkg_dir)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from path: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(
                f"Failed to load module from {path}:\n{type(e).__name__}: {e}"
            ) from e
        return _get_attr_with_suggestions(module, func_name, path)

    # Fallback: dotted module import
    try:
        module = importlib.import_module(left)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Module '{left}' not found.\n"
            f"If using a file path, make sure it ends with .py\n"
            f"Example: evalyn suggest-metrics --target example_agent/agent.py:run_agent"
        ) from e
    return _get_attr_with_suggestions(module, func_name, left)


__all__ = ["_get_module_callables", "_suggest_similar", "_load_callable"]
