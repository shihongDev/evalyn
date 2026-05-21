"""Quick comparison shortcuts for CLI eval run comparison.

Pure Python, no external dependencies. Resolves shorthand mode strings
into concrete run-pair comparison requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompareMode:
    """Describes a supported comparison mode."""

    mode: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompareMode:
        return cls(
            mode=data["mode"],
            description=data["description"],
        )


@dataclass
class CompareRequest:
    """A resolved (or unresolved) comparison between two runs."""

    run_a_id: str
    run_b_id: str
    mode: str
    resolved: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "mode": self.mode,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompareRequest:
        return cls(
            run_a_id=data["run_a_id"],
            run_b_id=data["run_b_id"],
            mode=data["mode"],
            resolved=data.get("resolved", True),
        )


COMPARE_MODES: dict[str, CompareMode] = {
    "last_2": CompareMode(
        mode="last_2",
        description="Compare the two most recent runs",
    ),
    "latest_vs_pinned": CompareMode(
        mode="latest_vs_pinned",
        description="Compare the latest run against a pinned baseline",
    ),
    "latest_vs_previous": CompareMode(
        mode="latest_vs_previous",
        description="Compare the latest run against the second-to-last run",
    ),
}


def list_modes() -> list[CompareMode]:
    """Return all available comparison modes."""
    return list(COMPARE_MODES.values())


def resolve_last_2(run_ids: list[str]) -> CompareRequest:
    """Pick the two most recent run IDs (last two in list).

    Returns an unresolved request if fewer than 2 runs are available.
    """
    if len(run_ids) < 2:
        return CompareRequest(
            run_a_id="",
            run_b_id="",
            mode="last_2",
            resolved=False,
        )
    return CompareRequest(
        run_a_id=run_ids[-2],
        run_b_id=run_ids[-1],
        mode="last_2",
    )


def resolve_latest_vs_pinned(run_ids: list[str], pinned_id: str) -> CompareRequest:
    """Compare the latest run against a pinned baseline.

    Returns an unresolved request if no runs or pinned_id is empty.
    """
    if not run_ids or not pinned_id:
        return CompareRequest(
            run_a_id="",
            run_b_id="",
            mode="latest_vs_pinned",
            resolved=False,
        )
    return CompareRequest(
        run_a_id=pinned_id,
        run_b_id=run_ids[-1],
        mode="latest_vs_pinned",
    )


def resolve_latest_vs_previous(run_ids: list[str]) -> CompareRequest:
    """Compare the latest run against the second-to-last.

    Returns an unresolved request if fewer than 2 runs are available.
    """
    if len(run_ids) < 2:
        return CompareRequest(
            run_a_id="",
            run_b_id="",
            mode="latest_vs_previous",
            resolved=False,
        )
    return CompareRequest(
        run_a_id=run_ids[-2],
        run_b_id=run_ids[-1],
        mode="latest_vs_previous",
    )


def resolve_shorthand(mode: str, run_ids: list[str], pinned_id: str = "") -> CompareRequest:
    """Dispatch to the correct resolver based on mode string.

    Returns an unresolved request if the mode is unknown or there are
    not enough runs.
    """
    if mode == "last_2":
        return resolve_last_2(run_ids)
    if mode == "latest_vs_pinned":
        return resolve_latest_vs_pinned(run_ids, pinned_id)
    if mode == "latest_vs_previous":
        return resolve_latest_vs_previous(run_ids)
    return CompareRequest(
        run_a_id="",
        run_b_id="",
        mode=mode,
        resolved=False,
    )


def format_compare_request(request: CompareRequest) -> str:
    """Human-readable description of a comparison request."""
    if not request.resolved:
        return f"[unresolved] mode={request.mode} - not enough data to resolve"
    mode_desc = COMPARE_MODES.get(request.mode)
    label = mode_desc.description if mode_desc else request.mode
    return f"{label}: {request.run_a_id} vs {request.run_b_id}"
