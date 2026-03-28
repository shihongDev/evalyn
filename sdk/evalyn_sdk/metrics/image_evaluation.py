from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageMetadata:
    """Metadata extracted from an image file."""

    width: int = 0
    height: int = 0
    format: str = ""
    size_bytes: int = 0
    has_content: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "has_content": self.has_content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageMetadata:
        return cls(
            width=data.get("width", 0),
            height=data.get("height", 0),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            has_content=data.get("has_content", False),
        )


@dataclass
class ImageScore:
    """Result of a single image evaluation check."""

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
class ImageEvalReport:
    """Aggregated report from multiple image evaluation checks."""

    scores: List[ImageScore] = field(default_factory=list)
    overall_score: float = 0.0
    pass_rate: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageEvalReport:
        scores = [
            ImageScore(
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
        lines.append("Image Evaluation Report")
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


def check_image_exists(file_path: str) -> ImageScore:
    """Check if file exists and is non-empty."""
    if not file_path:
        return ImageScore(
            metric_name="image_exists",
            score=0.0,
            passed=False,
            details={"reason": "empty path"},
        )
    exists = os.path.isfile(file_path)
    if not exists:
        return ImageScore(
            metric_name="image_exists",
            score=0.0,
            passed=False,
            details={"reason": "file not found"},
        )
    size = os.path.getsize(file_path)
    if size == 0:
        return ImageScore(
            metric_name="image_exists",
            score=0.0,
            passed=False,
            details={"reason": "file is empty"},
        )
    return ImageScore(
        metric_name="image_exists",
        score=1.0,
        passed=True,
        details={"size_bytes": size},
    )


def check_image_format(
    file_path: str,
    expected_formats: Optional[List[str]] = None,
) -> ImageScore:
    """Check file extension matches expected formats."""
    if expected_formats is None:
        expected_formats = ["png", "jpg", "jpeg", "webp"]
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    passed = ext in [f.lower() for f in expected_formats]
    return ImageScore(
        metric_name="image_format",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"extension": ext, "expected": expected_formats},
    )


def check_image_size(
    file_path: str,
    min_bytes: int = 100,
    max_bytes: int = 10_000_000,
) -> ImageScore:
    """Check that file size is within bounds."""
    if not os.path.isfile(file_path):
        return ImageScore(
            metric_name="image_size",
            score=0.0,
            passed=False,
            details={"reason": "file not found"},
        )
    size = os.path.getsize(file_path)
    passed = min_bytes <= size <= max_bytes
    return ImageScore(
        metric_name="image_size",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"size_bytes": size, "min_bytes": min_bytes, "max_bytes": max_bytes},
    )


def check_image_dimensions(
    metadata: ImageMetadata,
    min_width: int = 1,
    min_height: int = 1,
    max_width: int = 10000,
    max_height: int = 10000,
) -> ImageScore:
    """Check that image dimensions are within bounds."""
    w_ok = min_width <= metadata.width <= max_width
    h_ok = min_height <= metadata.height <= max_height
    passed = w_ok and h_ok
    return ImageScore(
        metric_name="image_dimensions",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={
            "width": metadata.width,
            "height": metadata.height,
            "bounds": {
                "min_width": min_width,
                "min_height": min_height,
                "max_width": max_width,
                "max_height": max_height,
            },
        },
    )


def extract_image_metadata_from_path(file_path: str) -> ImageMetadata:
    """Extract metadata from file path: format from extension, size from os if exists."""
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    size_bytes = 0
    has_content = False
    if os.path.isfile(file_path):
        size_bytes = os.path.getsize(file_path)
        has_content = size_bytes > 0
    return ImageMetadata(
        format=ext,
        size_bytes=size_bytes,
        has_content=has_content,
    )


def evaluate_image(
    file_path: str,
    metadata: Optional[ImageMetadata] = None,
) -> ImageEvalReport:
    """Run all image checks and return an aggregated report."""
    scores: List[ImageScore] = []

    scores.append(check_image_exists(file_path))
    scores.append(check_image_format(file_path))
    scores.append(check_image_size(file_path))

    if metadata is None:
        metadata = extract_image_metadata_from_path(file_path)
    # Only check dimensions when width/height are provided (non-zero)
    if metadata.width > 0 or metadata.height > 0:
        scores.append(check_image_dimensions(metadata))

    total = len(scores)
    passed_count = sum(1 for s in scores if s.passed)
    overall = sum(s.score for s in scores) / total if total else 0.0
    pass_rate = passed_count / total if total else 0.0

    return ImageEvalReport(
        scores=scores,
        overall_score=overall,
        pass_rate=pass_rate,
    )
