"""Orphan span recovery - detect and attach spans captured outside an active trace context."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import Span


@dataclass
class OrphanSpan:
    """An orphan span with optional match information."""

    span: Span
    matched_to: str | None = None
    match_confidence: float = 0.0
    match_method: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "span": self.span.as_dict(),
            "matched_to": self.matched_to,
            "match_confidence": self.match_confidence,
            "match_method": self.match_method,
        }


@dataclass
class RecoveryResult:
    """Summary of orphan recovery."""

    total_orphans: int
    recovered: int
    truly_lost: int
    recovery_rate: float
    matches: list[OrphanSpan] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_orphans": self.total_orphans,
            "recovered": self.recovered,
            "truly_lost": self.truly_lost,
            "recovery_rate": self.recovery_rate,
            "matches": [m.as_dict() for m in self.matches],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryResult:
        matches = []
        for m in data.get("matches", []):
            span = Span.from_dict(m["span"])
            matches.append(
                OrphanSpan(
                    span=span,
                    matched_to=m.get("matched_to"),
                    match_confidence=m.get("match_confidence", 0.0),
                    match_method=m.get("match_method", ""),
                )
            )
        return cls(
            total_orphans=data["total_orphans"],
            recovered=data["recovered"],
            truly_lost=data["truly_lost"],
            recovery_rate=data["recovery_rate"],
            matches=matches,
        )

    def format_text(self) -> str:
        """Format recovery result as human-readable text."""
        lines = [
            f"Orphan Recovery: {self.recovered}/{self.total_orphans} recovered "
            f"({self.recovery_rate:.0%})",
            f"  Truly lost: {self.truly_lost}",
        ]
        for m in self.matches:
            status = "recovered" if m.matched_to else "lost"
            target = m.matched_to or "-"
            lines.append(
                f"  {m.span.id}: {status} -> {target} "
                f"(confidence={m.match_confidence:.2f}, method={m.match_method or 'none'})"
            )
        return "\n".join(lines)


def find_orphan_spans(spans: list[Span]) -> list[Span]:
    """Find spans whose parent_id references a span not in the list."""
    known_ids = {s.id for s in spans}
    orphans = []
    for s in spans:
        if s.parent_id is not None and s.parent_id not in known_ids:
            orphans.append(s)
    return orphans


def _time_gap_ms(orphan: Span, candidate: Span) -> float:
    """Compute the minimum time gap in ms between an orphan and a candidate's time range."""
    orphan_start_ms = orphan.start_time.timestamp() * 1000
    cand_start_ms = candidate.start_time.timestamp() * 1000
    cand_end_ms = (
        candidate.end_time.timestamp() * 1000
        if candidate.end_time
        else cand_start_ms
    )

    # If orphan starts within the candidate's range, gap is 0
    if cand_start_ms <= orphan_start_ms <= cand_end_ms:
        return 0.0

    # Otherwise pick the closer boundary
    gap_to_start = abs(orphan_start_ms - cand_start_ms)
    gap_to_end = abs(orphan_start_ms - cand_end_ms)
    return min(gap_to_start, gap_to_end)


def match_by_timestamp(
    orphan: Span, candidates: list[Span], max_gap_ms: float = 1000.0
) -> tuple[str, float] | None:
    """Find the candidate span closest in time to the orphan.

    Returns (span_id, confidence) or None if the gap exceeds max_gap_ms.
    Confidence = 1 - (gap_ms / max_gap_ms).
    """
    if not candidates:
        return None

    best_id: str | None = None
    best_gap = float("inf")

    for c in candidates:
        gap = _time_gap_ms(orphan, c)
        if gap < best_gap:
            best_gap = gap
            best_id = c.id

    if best_gap > max_gap_ms or best_id is None:
        return None

    confidence = 1.0 - (best_gap / max_gap_ms)
    return best_id, confidence


def match_by_name(
    orphan: Span, candidates: list[Span]
) -> tuple[str, float] | None:
    """Find a candidate with matching span_type.

    If multiple match, pick the one closest in time.
    Confidence is fixed at 0.5 (name-only match is lower confidence).
    """
    type_matches = [c for c in candidates if c.span_type == orphan.span_type]
    if not type_matches:
        return None

    # Pick the closest in time among type matches
    best_id: str | None = None
    best_gap = float("inf")

    for c in type_matches:
        gap = _time_gap_ms(orphan, c)
        if gap < best_gap:
            best_gap = gap
            best_id = c.id

    if best_id is None:
        return None

    return best_id, 0.5


def recover_orphans(
    orphans: list[Span],
    active_spans: list[Span],
    max_gap_ms: float = 1000.0,
) -> RecoveryResult:
    """Try to recover orphan spans by matching against active spans.

    Tries timestamp match first, then falls back to name match.
    """
    matches: list[OrphanSpan] = []
    recovered = 0

    for orphan in orphans:
        # Try timestamp match first
        ts_match = match_by_timestamp(orphan, active_spans, max_gap_ms)
        if ts_match is not None:
            matched_id, confidence = ts_match
            matches.append(
                OrphanSpan(
                    span=orphan,
                    matched_to=matched_id,
                    match_confidence=confidence,
                    match_method="timestamp",
                )
            )
            recovered += 1
            continue

        # Fall back to name match
        name_match = match_by_name(orphan, active_spans)
        if name_match is not None:
            matched_id, confidence = name_match
            matches.append(
                OrphanSpan(
                    span=orphan,
                    matched_to=matched_id,
                    match_confidence=confidence,
                    match_method="name",
                )
            )
            recovered += 1
            continue

        # Truly lost
        matches.append(OrphanSpan(span=orphan))

    total = len(orphans)
    truly_lost = total - recovered
    rate = recovered / total if total > 0 else 0.0

    return RecoveryResult(
        total_orphans=total,
        recovered=recovered,
        truly_lost=truly_lost,
        recovery_rate=rate,
        matches=matches,
    )


def reattach_orphans(
    spans: list[Span], recovery: RecoveryResult
) -> list[Span]:
    """Return new list of spans with orphans' parent_id updated to their match.

    Does not mutate the originals.
    """
    # Build lookup: orphan span id -> matched parent id
    remap: dict[str, str] = {}
    for m in recovery.matches:
        if m.matched_to is not None:
            remap[m.span.id] = m.matched_to

    result: list[Span] = []
    for s in spans:
        if s.id in remap:
            new_span = copy.deepcopy(s)
            new_span.parent_id = remap[s.id]
            result.append(new_span)
        else:
            result.append(s)

    return result
