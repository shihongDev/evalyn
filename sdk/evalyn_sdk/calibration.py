from __future__ import annotations

from statistics import mean
from typing import Dict, List
from uuid import uuid4

from .models import Annotation, CalibrationRecord, MetricResult, now_utc


class CalibrationEngine:
    """
    Lightweight calibration helper. Compares subjective metric results with human annotations
    and proposes a new decision threshold.
    """

    def __init__(self, judge_name: str, current_threshold: float = 0.5):
        self.judge_name = judge_name
        self.current_threshold = current_threshold

    def calibrate(self, metric_results: List[MetricResult], annotations: List[Annotation]) -> CalibrationRecord:
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
        disagreements = []
        aligned = []

        for res in metric_results:
            ann = ann_by_call.get(res.call_id)
            if not ann or res.passed is None:
                continue
            agree = bool(ann.label) == bool(res.passed)
            (aligned if agree else disagreements).append({"call_id": res.call_id, "human": ann.label, "judge": res.passed})

        disagreement_rate = len(disagreements) / max(len(disagreements) + len(aligned), 1)
        suggested_threshold = self._suggest_threshold(metric_results, annotations)

        adjustments = {
            "current_threshold": self.current_threshold,
            "suggested_threshold": suggested_threshold,
            "disagreement_rate": disagreement_rate,
            "sample_disagreements": disagreements[:20],
        }

        return CalibrationRecord(
            id=str(uuid4()),
            judge_config_id=self.judge_name,
            gold_items=list(ann_by_call.keys()),
            adjustments=adjustments,
            created_at=now_utc(),
        )

    def _suggest_threshold(self, metric_results: List[MetricResult], annotations: List[Annotation]) -> float:
        """Simple heuristic: align judge pass-rate with human positive rate."""
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
        human_labels = [bool(ann.label) for ann in annotations]
        judge_passes = [bool(res.passed) for res in metric_results if res.call_id in ann_by_call and res.passed is not None]

        human_rate = mean(human_labels) if human_labels else self.current_threshold
        judge_rate = mean(judge_passes) if judge_passes else self.current_threshold

        # Shift threshold toward reducing the gap; clamp between 0 and 1.
        delta = judge_rate - human_rate
        new_threshold = self.current_threshold + delta
        return max(0.0, min(1.0, new_threshold))
