"""Confidence method comparison and calibration analysis.

Compares confidence scores from different methods against actual
correctness. Produces calibration curves and method recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationBin:
    """A single bin in a calibration curve."""

    bin_start: float
    bin_end: float
    mean_confidence: float
    actual_accuracy: float
    count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "bin_start": round(self.bin_start, 2),
            "bin_end": round(self.bin_end, 2),
            "mean_confidence": round(self.mean_confidence, 4),
            "actual_accuracy": round(self.actual_accuracy, 4),
            "count": self.count,
        }


@dataclass
class MethodCalibration:
    """Calibration analysis for one confidence method."""

    method: str
    bins: list[CalibrationBin] = field(default_factory=list)
    ece: float = 0.0  # expected calibration error
    mce: float = 0.0  # maximum calibration error
    brier_score: float = 0.0  # Brier score (lower is better)
    num_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "brier_score": round(self.brier_score, 6),
            "num_samples": self.num_samples,
            "bins": [b.as_dict() for b in self.bins],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MethodCalibration:
        bins = [
            CalibrationBin(
                bin_start=b["bin_start"],
                bin_end=b["bin_end"],
                mean_confidence=b["mean_confidence"],
                actual_accuracy=b["actual_accuracy"],
                count=b["count"],
            )
            for b in data.get("bins", [])
        ]
        return cls(
            method=data["method"],
            ece=data.get("ece", 0.0),
            mce=data.get("mce", 0.0),
            brier_score=data.get("brier_score", 0.0),
            num_samples=data.get("num_samples", 0),
            bins=bins,
        )

    def format_text(self) -> str:
        return (
            f"{self.method}: ECE={self.ece:.4f}, MCE={self.mce:.4f}, "
            f"Brier={self.brier_score:.4f} (n={self.num_samples})"
        )


@dataclass
class ComparisonReport:
    """Side-by-side comparison of confidence methods."""

    calibrations: list[MethodCalibration] = field(default_factory=list)
    recommended_method: str = ""

    @property
    def best_by_ece(self) -> str | None:
        if not self.calibrations:
            return None
        return min(self.calibrations, key=lambda c: c.ece).method

    @property
    def best_by_brier(self) -> str | None:
        if not self.calibrations:
            return None
        return min(self.calibrations, key=lambda c: c.brier_score).method

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommended_method": self.recommended_method,
            "best_by_ece": self.best_by_ece,
            "best_by_brier": self.best_by_brier,
            "methods": [c.as_dict() for c in self.calibrations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonReport:
        calibrations = [MethodCalibration.from_dict(m) for m in data.get("methods", [])]
        return cls(
            calibrations=calibrations,
            recommended_method=data.get("recommended_method", ""),
        )

    def format_text(self) -> str:
        lines = ["Confidence Method Comparison:"]
        for c in sorted(self.calibrations, key=lambda x: x.ece):
            lines.append(f"  {c.format_text()}")
        if self.recommended_method:
            lines.append(f"\n  Recommended: {self.recommended_method}")
        return "\n".join(lines)


def compute_calibration(
    method: str,
    predictions: list[tuple[float, bool]],
    num_bins: int = 10,
) -> MethodCalibration:
    """Compute calibration metrics for a single confidence method.

    Args:
        method: Method name (e.g. "logprobs", "consistency").
        predictions: List of (confidence_score, actually_correct) tuples.
        num_bins: Number of bins for the calibration curve.

    Returns:
        MethodCalibration with ECE, MCE, Brier score, and calibration bins.
    """
    if not predictions:
        return MethodCalibration(method=method)

    n = len(predictions)
    bin_width = 1.0 / num_bins
    bins = []

    total_ece = 0.0
    max_ce = 0.0

    for i in range(num_bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width

        # Collect predictions in this bin
        in_bin = [
            (conf, correct)
            for conf, correct in predictions
            if bin_start <= conf < bin_end or (i == num_bins - 1 and conf == 1.0)
        ]

        if not in_bin:
            continue

        mean_conf = sum(c for c, _ in in_bin) / len(in_bin)
        accuracy = sum(1 for _, c in in_bin if c) / len(in_bin)
        bin_count = len(in_bin)

        bins.append(
            CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                mean_confidence=mean_conf,
                actual_accuracy=accuracy,
                count=bin_count,
            )
        )

        # Calibration error for this bin
        ce = abs(mean_conf - accuracy)
        total_ece += ce * bin_count / n
        max_ce = max(max_ce, ce)

    # Brier score
    brier = sum((conf - (1.0 if correct else 0.0)) ** 2 for conf, correct in predictions) / n

    return MethodCalibration(
        method=method,
        bins=bins,
        ece=total_ece,
        mce=max_ce,
        brier_score=brier,
        num_samples=n,
    )


def compare_methods(
    method_predictions: dict[str, list[tuple[float, bool]]],
    num_bins: int = 10,
) -> ComparisonReport:
    """Compare multiple confidence methods side by side.

    Args:
        method_predictions: Dict of method_name -> list of (confidence, correct).
        num_bins: Number of calibration bins.

    Returns:
        ComparisonReport with calibration for each method and recommendation.
    """
    calibrations = []
    for method, preds in sorted(method_predictions.items()):
        calibrations.append(compute_calibration(method, preds, num_bins))

    # Recommend method with lowest ECE
    recommended = ""
    if calibrations:
        best = min(calibrations, key=lambda c: c.ece)
        recommended = best.method

    return ComparisonReport(
        calibrations=calibrations,
        recommended_method=recommended,
    )
