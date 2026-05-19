"""Prompt version tracking for calibration.

Tracks judge prompt changes across calibration rounds, providing
version history, diffs, and change detection for each metric.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_prompt_hash(text: str) -> str:
    """SHA-256 of *text*, truncated to the first 16 hex characters."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def has_prompt_changed(current_hash: str, new_text: str) -> bool:
    """Return True when the hash of *new_text* differs from *current_hash*."""
    return compute_prompt_hash(new_text) != current_hash


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    """A single versioned snapshot of a judge prompt."""

    version: int
    prompt_text: str
    prompt_hash: str = ""
    created_at: str = ""
    notes: str = ""
    alignment_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prompt_text": self.prompt_text,
            "prompt_hash": self.prompt_hash,
            "created_at": self.created_at,
            "notes": self.notes,
            "alignment_score": self.alignment_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVersion:
        return cls(
            version=data["version"],
            prompt_text=data["prompt_text"],
            prompt_hash=data.get("prompt_hash", ""),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
            alignment_score=data.get("alignment_score"),
        )


@dataclass
class PromptHistory:
    """Version history for a single metric's judge prompt."""

    metric_id: str
    versions: list[PromptVersion] = field(default_factory=list)

    @property
    def current(self) -> PromptVersion | None:
        """Latest version, or None if no versions exist."""
        if not self.versions:
            return None
        return self.versions[-1]

    @property
    def count(self) -> int:
        return len(self.versions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "versions": [v.as_dict() for v in self.versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptHistory:
        return cls(
            metric_id=data["metric_id"],
            versions=[PromptVersion.from_dict(v) for v in data.get("versions", [])],
        )

    def format_text(self) -> str:
        lines: list[str] = [f"Prompt history for {self.metric_id} ({self.count} versions)"]
        for v in self.versions:
            score_part = f"  score={v.alignment_score}" if v.alignment_score is not None else ""
            notes_part = f"  - {v.notes}" if v.notes else ""
            lines.append(f"  v{v.version}: {v.prompt_hash}{score_part}{notes_part}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Version Store
# ---------------------------------------------------------------------------


class PromptVersionStore:
    """In-memory store that tracks prompt versions for multiple metrics."""

    def __init__(self) -> None:
        self._histories: dict[str, PromptHistory] = {}

    # -- mutations ----------------------------------------------------------

    def add_version(
        self,
        metric_id: str,
        prompt_text: str,
        notes: str = "",
        alignment_score: float | None = None,
    ) -> PromptVersion:
        """Add a new version for *metric_id* with auto-incrementing version number."""
        history = self.get_history(metric_id)
        next_version = history.count + 1
        pv = PromptVersion(
            version=next_version,
            prompt_text=prompt_text,
            prompt_hash=compute_prompt_hash(prompt_text),
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
            alignment_score=alignment_score,
        )
        history.versions.append(pv)
        return pv

    # -- queries ------------------------------------------------------------

    def get_history(self, metric_id: str) -> PromptHistory:
        """Return the history for *metric_id*, creating one if absent."""
        if metric_id not in self._histories:
            self._histories[metric_id] = PromptHistory(metric_id=metric_id)
        return self._histories[metric_id]

    def get_current(self, metric_id: str) -> PromptVersion | None:
        """Return the latest version for *metric_id*, or None."""
        return self.get_history(metric_id).current

    def get_version(self, metric_id: str, version: int) -> PromptVersion | None:
        """Return a specific version, or None if it does not exist."""
        for v in self.get_history(metric_id).versions:
            if v.version == version:
                return v
        return None

    def diff_versions(self, metric_id: str, v1: int, v2: int) -> str:
        """Return a unified diff between two versions of the prompt."""
        pv1 = self.get_version(metric_id, v1)
        pv2 = self.get_version(metric_id, v2)
        if pv1 is None or pv2 is None:
            return ""
        return "\n".join(
            difflib.unified_diff(
                pv1.prompt_text.splitlines(),
                pv2.prompt_text.splitlines(),
                fromfile=f"v{v1}",
                tofile=f"v{v2}",
                lineterm="",
            )
        )

    def list_metrics(self) -> list[str]:
        """Return all tracked metric IDs."""
        return sorted(self._histories.keys())

    # -- serialisation ------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "histories": {mid: h.as_dict() for mid, h in self._histories.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVersionStore:
        store = cls()
        for _mid, h_data in data.get("histories", {}).items():
            history = PromptHistory.from_dict(h_data)
            store._histories[history.metric_id] = history
        return store

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    @classmethod
    def from_json(cls, data: str) -> PromptVersionStore:
        return cls.from_dict(json.loads(data))
