"""Evalyn CLI package.

This package provides the command-line interface for the Evalyn SDK.
The main function is loaded lazily so that importing the cli package
(as a side effect of importing command submodules) does not pay for
main.py's setup cost.
"""


def __getattr__(name: str):
    if name == "main":
        from .main import main

        globals()["main"] = main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]
