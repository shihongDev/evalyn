from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    format: str = ""
    size_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": self.format,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoMetadata:
        return cls(
            duration_seconds=data.get("duration_seconds", 0.0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            fps=data.get("fps", 0.0),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class VideoScore:
    """Result of a single video evaluation check."""

    metric_name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class VideoEvalReport:
    """Aggregated report from multiple video evaluation checks."""

    scores: list[VideoScore] = field(default_factory=list)
    overall_score: float = 0.0
    pass_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoEvalReport:
        scores = [
            VideoScore(
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
        lines: list[str] = []
        lines.append("Video Evaluation Report")
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


def check_video_exists(file_path: str) -> VideoScore:
    """Check if file exists and is non-empty."""
    if not file_path:
        return VideoScore(
            metric_name="video_exists",
            score=0.0,
            passed=False,
            details={"reason": "empty path"},
        )
    if not os.path.isfile(file_path):
        return VideoScore(
            metric_name="video_exists",
            score=0.0,
            passed=False,
            details={"reason": "file not found"},
        )
    size = os.path.getsize(file_path)
    if size == 0:
        return VideoScore(
            metric_name="video_exists",
            score=0.0,
            passed=False,
            details={"reason": "file is empty"},
        )
    return VideoScore(
        metric_name="video_exists",
        score=1.0,
        passed=True,
        details={"size_bytes": size},
    )


def check_video_format(
    file_path: str,
    expected_formats: list[str] | None = None,
) -> VideoScore:
    """Check file extension matches expected formats."""
    if expected_formats is None:
        expected_formats = ["mp4", "webm", "avi", "mov"]
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    passed = ext in [f.lower() for f in expected_formats]
    return VideoScore(
        metric_name="video_format",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"extension": ext, "expected": expected_formats},
    )


def check_video_duration(
    metadata: VideoMetadata,
    min_seconds: float = 0.1,
    max_seconds: float = 600.0,
) -> VideoScore:
    """Check that video duration is within bounds."""
    passed = min_seconds <= metadata.duration_seconds <= max_seconds
    return VideoScore(
        metric_name="video_duration",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={
            "duration_seconds": metadata.duration_seconds,
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
        },
    )


def check_video_resolution(
    metadata: VideoMetadata,
    min_width: int = 1,
    min_height: int = 1,
) -> VideoScore:
    """Check that video resolution meets minimum requirements."""
    w_ok = metadata.width >= min_width
    h_ok = metadata.height >= min_height
    passed = w_ok and h_ok
    return VideoScore(
        metric_name="video_resolution",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={
            "width": metadata.width,
            "height": metadata.height,
            "min_width": min_width,
            "min_height": min_height,
        },
    )


def extract_video_metadata_from_path(file_path: str) -> VideoMetadata:
    """Extract metadata from file path: format from extension, size from os if exists."""
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    size_bytes = 0
    if os.path.isfile(file_path):
        size_bytes = os.path.getsize(file_path)
    return VideoMetadata(
        format=ext,
        size_bytes=size_bytes,
    )


def evaluate_video(
    file_path: str,
    metadata: VideoMetadata | None = None,
) -> VideoEvalReport:
    """Run all video checks and return an aggregated report."""
    scores: list[VideoScore] = []

    scores.append(check_video_exists(file_path))
    scores.append(check_video_format(file_path))

    if metadata is None:
        metadata = extract_video_metadata_from_path(file_path)

    # Only check duration when duration is provided (non-zero)
    if metadata.duration_seconds > 0:
        scores.append(check_video_duration(metadata))

    # Only check resolution when width/height are provided (non-zero)
    if metadata.width > 0 or metadata.height > 0:
        scores.append(check_video_resolution(metadata))

    total = len(scores)
    passed_count = sum(1 for s in scores if s.passed)
    overall = sum(s.score for s in scores) / total if total else 0.0
    pass_rate = passed_count / total if total else 0.0

    return VideoEvalReport(
        scores=scores,
        overall_score=overall,
        pass_rate=pass_rate,
    )
