"""CLI utility modules.

Individual utilities are imported directly from their submodules
(e.g., ``from ..utils.errors import fatal_error``). This package
__init__ intentionally avoids eagerly importing all submodules to
keep CLI startup fast.
"""
