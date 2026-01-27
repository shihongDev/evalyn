"""Backwards compatibility re-export.

This module has moved to evalyn_sdk.judges.
Please update your imports to:
    from evalyn_sdk.judges import LLMJudge, EchoJudge
"""

# Re-export everything from the new location for backwards compatibility
from ..judges import LLMJudge, EchoJudge
from .subjective import JUDGE_TEMPLATES

__all__ = ["LLMJudge", "EchoJudge", "JUDGE_TEMPLATES"]
