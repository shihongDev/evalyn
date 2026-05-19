"""Audio evaluation metrics: file-level checks for audio assets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AudioMetadata:
    """Metadata for an audio file."""

    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    format: str = ""
    size_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioMetadata:
        return cls(
            duration_seconds=data.get("duration_seconds", 0.0),
            sample_rate=data.get("sample_rate", 0),
            channels=data.get("channels", 0),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class AudioScore:
    """Result of a single audio evaluation check."""

    metric_name: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class AudioEvalReport:
    """Aggregated report from multiple audio evaluation checks."""

    scores: List[AudioScore] = field(default_factory=list)
    overall_score: float = 0.0
    pass_rate: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioEvalReport:
        scores = [
            AudioScore(
                metric_name=s["metric_name"],
                score=s["score"],
                passed=s["passed"],
                details=s.get("details", {}),
            )
            for s in data.get("scores", [])
        ]
        return cls(
            scores=scores,
            overall_score=data.get("overall_score", 0.0),
            pass_rate=data.get("pass_rate", 0.0),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Audio Evaluation Report")
        lines.append("-" * 40)
        for s in self.scores:
            status = "PASS" if s.passed else "FAIL"
            lines.append(f"  {s.metric_name}: {s.score:.2f} [{status}]")
        lines.append("-" * 40)
        lines.append(f"  Overall score: {self.overall_score:.2f}")
        lines.append(f"  Pass rate: {self.pass_rate:.1%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_audio_exists(file_path: str) -> AudioScore:
    """Check if file exists and is non-empty."""
    if not file_path:
        return AudioScore(
            metric_name="audio_exists",
            score=0.0,
            passed=False,
            details={"reason": "empty path"},
        )
    if not os.path.isfile(file_path):
        return AudioScore(
            metric_name="audio_exists",
            score=0.0,
            passed=False,
            details={"reason": "file not found"},
        )
    size = os.path.getsize(file_path)
    if size == 0:
        return AudioScore(
            metric_name="audio_exists",
            score=0.0,
            passed=False,
            details={"reason": "file is empty"},
        )
    return AudioScore(
        metric_name="audio_exists",
        score=1.0,
        passed=True,
        details={"size_bytes": size},
    )


def check_audio_format(
    file_path: str,
    expected_formats: Optional[List[str]] = None,
) -> AudioScore:
    """Check file extension matches expected formats."""
    if expected_formats is None:
        expected_formats = ["wav", "mp3", "ogg", "flac"]
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    passed = ext in [f.lower() for f in expected_formats]
    return AudioScore(
        metric_name="audio_format",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"extension": ext, "expected": expected_formats},
    )


def check_audio_duration(
    metadata: AudioMetadata,
    min_seconds: float = 0.1,
    max_seconds: float = 300.0,
) -> AudioScore:
    """Check that audio duration is within bounds."""
    duration = metadata.duration_seconds
    passed = min_seconds <= duration <= max_seconds
    return AudioScore(
        metric_name="audio_duration",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={
            "duration_seconds": duration,
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
        },
    )


def check_audio_size(
    file_path: str,
    min_bytes: int = 100,
    max_bytes: int = 50_000_000,
) -> AudioScore:
    """Check that file size is within bounds."""
    if not os.path.isfile(file_path):
        return AudioScore(
            metric_name="audio_size",
            score=0.0,
            passed=False,
            details={"reason": "file not found"},
        )
    size = os.path.getsize(file_path)
    passed = min_bytes <= size <= max_bytes
    return AudioScore(
        metric_name="audio_size",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"size_bytes": size, "min_bytes": min_bytes, "max_bytes": max_bytes},
    )


def extract_audio_metadata_from_path(file_path: str) -> AudioMetadata:
    """Extract metadata from file path: format from extension, size from os if exists."""
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    size_bytes = 0
    if os.path.isfile(file_path):
        size_bytes = os.path.getsize(file_path)
    return AudioMetadata(
        format=ext,
        size_bytes=size_bytes,
    )


def evaluate_audio(
    file_path: str,
    metadata: Optional[AudioMetadata] = None,
) -> AudioEvalReport:
    """Run all audio checks and return an aggregated report."""
    scores: List[AudioScore] = []

    scores.append(check_audio_exists(file_path))
    scores.append(check_audio_format(file_path))
    scores.append(check_audio_size(file_path))

    if metadata is None:
        metadata = extract_audio_metadata_from_path(file_path)
    # Only check duration when duration is provided (non-zero)
    if metadata.duration_seconds > 0:
        scores.append(check_audio_duration(metadata))

    total = len(scores)
    passed_count = sum(1 for s in scores if s.passed)
    overall = sum(s.score for s in scores) / total if total else 0.0
    pass_rate = passed_count / total if total else 0.0

    return AudioEvalReport(
        scores=scores,
        overall_score=overall,
        pass_rate=pass_rate,
    )
