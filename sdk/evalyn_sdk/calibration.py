from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .models import Annotation, CalibrationRecord, MetricResult, DatasetItem, now_utc


@dataclass
class AlignmentMetrics:
    """
    Alignment metrics between LLM judge and human annotations.
    Uses classification terminology: judge prediction vs human ground truth.
    """
    # Confusion matrix counts
    true_positive: int = 0   # Both judge and human say PASS
    true_negative: int = 0   # Both judge and human say FAIL
    false_positive: int = 0  # Judge says PASS, human says FAIL
    false_negative: int = 0  # Judge says FAIL, human says PASS

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    @property
    def accuracy(self) -> float:
        """Overall agreement rate."""
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        """When judge says PASS, how often is human also PASS."""
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of human PASS cases, how many did judge catch."""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Of human FAIL cases, how many did judge correctly identify."""
        denom = self.true_negative + self.false_positive
        return self.true_negative / denom if denom > 0 else 0.0

    @property
    def cohens_kappa(self) -> float:
        """
        Cohen's kappa: agreement beyond chance.
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        """
        if self.total == 0:
            return 0.0

        # Observed agreement
        p_o = self.accuracy

        # Expected agreement (by chance)
        judge_pass_rate = (self.true_positive + self.false_positive) / self.total
        human_pass_rate = (self.true_positive + self.false_negative) / self.total
        p_e = (judge_pass_rate * human_pass_rate) + ((1 - judge_pass_rate) * (1 - human_pass_rate))

        if p_e >= 1.0:
            return 1.0 if p_o >= 1.0 else 0.0

        return (p_o - p_e) / (1 - p_e)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "confusion_matrix": {
                "true_positive": self.true_positive,
                "true_negative": self.true_negative,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
            },
            "total_samples": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
        }


@dataclass
class DisagreementCase:
    """A single case where judge and human disagree."""
    call_id: str
    input_text: str
    output_text: str
    judge_passed: bool
    judge_reason: str
    human_passed: bool
    human_notes: str
    disagreement_type: str  # "false_positive" or "false_negative"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "input": self.input_text[:500],  # Truncate for readability
            "output": self.output_text[:500],
            "judge_passed": self.judge_passed,
            "judge_reason": self.judge_reason,
            "human_passed": self.human_passed,
            "human_notes": self.human_notes,
            "type": self.disagreement_type,
        }


@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement patterns."""
    false_positives: List[DisagreementCase] = field(default_factory=list)
    false_negatives: List[DisagreementCase] = field(default_factory=list)

    @property
    def total_disagreements(self) -> int:
        return len(self.false_positives) + len(self.false_negatives)

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Summarize common patterns in disagreements."""
        return {
            "false_positive_count": len(self.false_positives),
            "false_negative_count": len(self.false_negatives),
            "false_positive_examples": [d.as_dict() for d in self.false_positives[:5]],
            "false_negative_examples": [d.as_dict() for d in self.false_negatives[:5]],
        }


@dataclass
class PromptOptimizationResult:
    """Result of LLM-based prompt optimization."""
    original_rubric: List[str]
    improved_rubric: List[str]
    improvement_reasoning: str
    suggested_additions: List[str]
    suggested_removals: List[str]
    estimated_improvement: str  # "low", "medium", "high"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_rubric": self.original_rubric,
            "improved_rubric": self.improved_rubric,
            "improvement_reasoning": self.improvement_reasoning,
            "suggested_additions": self.suggested_additions,
            "suggested_removals": self.suggested_removals,
            "estimated_improvement": self.estimated_improvement,
        }


class PromptOptimizer:
    """
    Uses LLM to analyze disagreement patterns and suggest rubric improvements.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set the environment variable or pass api_key to PromptOptimizer."
            )
        return key

    def _call_api(self, prompt: str) -> str:
        """Make direct HTTP call to Gemini API."""
        api_key = self._get_api_key()
        url = self.API_URL.format(model=self.model) + f"?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,  # Low temp for consistent optimization
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini API connection error: {e.reason}") from e

        try:
            candidates = response_data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        except Exception:
            pass

        return ""

    def optimize(
        self,
        metric_id: str,
        current_rubric: List[str],
        disagreements: DisagreementAnalysis,
        alignment_metrics: AlignmentMetrics,
    ) -> PromptOptimizationResult:
        """
        Analyze disagreements and suggest rubric improvements.
        """
        # Build the optimization prompt
        prompt = self._build_optimization_prompt(
            metric_id, current_rubric, disagreements, alignment_metrics
        )

        try:
            response_text = self._call_api(prompt)
            return self._parse_optimization_response(response_text, current_rubric)
        except Exception as e:
            # Return a fallback result on error
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning=f"Optimization failed: {e}",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
            )

    def _build_optimization_prompt(
        self,
        metric_id: str,
        current_rubric: List[str],
        disagreements: DisagreementAnalysis,
        alignment_metrics: AlignmentMetrics,
    ) -> str:
        """Build prompt for rubric optimization."""

        # Format current rubric
        rubric_text = "\n".join([f"- {r}" for r in current_rubric]) if current_rubric else "(no rubric defined)"

        # Format false positives (judge too lenient)
        fp_examples = ""
        for i, case in enumerate(disagreements.false_positives[:3], 1):
            fp_examples += f"""
Example {i} (Judge said PASS, Human said FAIL):
  Input: {case.input_text[:300]}...
  Output: {case.output_text[:300]}...
  Judge reason: {case.judge_reason}
  Human notes: {case.human_notes}
"""

        # Format false negatives (judge too strict)
        fn_examples = ""
        for i, case in enumerate(disagreements.false_negatives[:3], 1):
            fn_examples += f"""
Example {i} (Judge said FAIL, Human said PASS):
  Input: {case.input_text[:300]}...
  Output: {case.output_text[:300]}...
  Judge reason: {case.judge_reason}
  Human notes: {case.human_notes}
"""

        prompt = f"""You are an expert at improving LLM evaluation rubrics. Analyze the following calibration data and suggest improvements.

## METRIC BEING CALIBRATED
Metric ID: {metric_id}

## CURRENT RUBRIC
{rubric_text}

## ALIGNMENT STATISTICS
- Accuracy: {alignment_metrics.accuracy:.1%}
- Precision: {alignment_metrics.precision:.1%} (when judge says PASS, human agrees {alignment_metrics.precision:.1%} of the time)
- Recall: {alignment_metrics.recall:.1%} (judge catches {alignment_metrics.recall:.1%} of human-approved outputs)
- F1 Score: {alignment_metrics.f1:.1%}
- Cohen's Kappa: {alignment_metrics.cohens_kappa:.3f}

## FALSE POSITIVES ({len(disagreements.false_positives)} cases)
Judge was too LENIENT - said PASS when human said FAIL:
{fp_examples if fp_examples else "(none)"}

## FALSE NEGATIVES ({len(disagreements.false_negatives)} cases)
Judge was too STRICT - said FAIL when human said PASS:
{fn_examples if fn_examples else "(none)"}

## YOUR TASK
Analyze the disagreement patterns and suggest an improved rubric. Return a JSON object with:
{{
  "improved_rubric": ["criterion 1", "criterion 2", ...],
  "suggested_additions": ["new criterion to add", ...],
  "suggested_removals": ["criterion to remove or relax", ...],
  "improvement_reasoning": "Explanation of what patterns you found and why these changes help",
  "estimated_improvement": "low|medium|high"
}}

Guidelines:
- If many false positives: make criteria more STRICT or add missing criteria
- If many false negatives: relax overly strict criteria or add nuance
- Be specific and actionable in your rubric criteria
- Aim for 3-7 criteria total

Return ONLY the JSON object, no other text."""

        return prompt

    def _parse_optimization_response(
        self,
        response_text: str,
        original_rubric: List[str],
    ) -> PromptOptimizationResult:
        """Parse the LLM response into a structured result."""

        # Try to extract JSON from response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            # Find JSON object
            start = text.find("{")
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = text[start:i+1]
                            parsed = json.loads(json_str)

                            return PromptOptimizationResult(
                                original_rubric=original_rubric,
                                improved_rubric=parsed.get("improved_rubric", original_rubric),
                                improvement_reasoning=parsed.get("improvement_reasoning", ""),
                                suggested_additions=parsed.get("suggested_additions", []),
                                suggested_removals=parsed.get("suggested_removals", []),
                                estimated_improvement=parsed.get("estimated_improvement", "unknown"),
                            )
                            break
        except Exception:
            pass

        # Fallback if parsing fails
        return PromptOptimizationResult(
            original_rubric=original_rubric,
            improved_rubric=original_rubric,
            improvement_reasoning=f"Could not parse LLM response: {response_text[:200]}",
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement="unknown",
        )


class CalibrationEngine:
    """
    Enhanced calibration engine that:
    1. Computes alignment metrics (precision, recall, F1, kappa)
    2. Analyzes disagreement patterns
    3. Uses LLM to optimize judge rubrics
    """

    def __init__(
        self,
        judge_name: str,
        current_threshold: float = 0.5,
        current_rubric: Optional[List[str]] = None,
        optimize_prompts: bool = True,
        optimizer_model: str = "gemini-2.5-flash-lite",
    ):
        self.judge_name = judge_name
        self.current_threshold = current_threshold
        self.current_rubric = current_rubric or []
        self.optimize_prompts = optimize_prompts
        self.optimizer_model = optimizer_model

    def compute_alignment(
        self,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
    ) -> AlignmentMetrics:
        """Compute alignment metrics between judge and human annotations."""
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}

        metrics = AlignmentMetrics()

        for res in metric_results:
            ann = ann_by_call.get(res.call_id)
            if not ann or res.passed is None:
                continue

            judge_pass = bool(res.passed)
            human_pass = bool(ann.label)

            if judge_pass and human_pass:
                metrics.true_positive += 1
            elif not judge_pass and not human_pass:
                metrics.true_negative += 1
            elif judge_pass and not human_pass:
                metrics.false_positive += 1
            else:  # not judge_pass and human_pass
                metrics.false_negative += 1

        return metrics

    def analyze_disagreements(
        self,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
    ) -> DisagreementAnalysis:
        """Analyze patterns in disagreements between judge and human."""
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
        items_by_call: Dict[str, DatasetItem] = {}
        if dataset_items:
            for item in dataset_items:
                call_id = item.metadata.get("call_id", item.id)
                items_by_call[call_id] = item

        analysis = DisagreementAnalysis()

        for res in metric_results:
            ann = ann_by_call.get(res.call_id)
            if not ann or res.passed is None:
                continue

            judge_pass = bool(res.passed)
            human_pass = bool(ann.label)

            # Skip agreements
            if judge_pass == human_pass:
                continue

            # Get input/output from dataset item if available
            item = items_by_call.get(res.call_id)
            input_text = ""
            output_text = ""
            if item:
                input_text = json.dumps(item.input, default=str) if item.input else ""
                output_text = str(item.output) if item.output else ""

            # Get judge reason from raw_judge or details
            judge_reason = ""
            if res.raw_judge and isinstance(res.raw_judge, dict):
                judge_reason = res.raw_judge.get("reason", "")
            elif res.details and isinstance(res.details, dict):
                judge_reason = res.details.get("reason", "")

            case = DisagreementCase(
                call_id=res.call_id,
                input_text=input_text,
                output_text=output_text,
                judge_passed=judge_pass,
                judge_reason=judge_reason,
                human_passed=human_pass,
                human_notes=ann.rationale or "",
                disagreement_type="false_positive" if judge_pass else "false_negative",
            )

            if judge_pass and not human_pass:
                analysis.false_positives.append(case)
            else:
                analysis.false_negatives.append(case)

        return analysis

    def calibrate(
        self,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
    ) -> CalibrationRecord:
        """
        Full calibration pipeline:
        1. Compute alignment metrics
        2. Analyze disagreement patterns
        3. Optionally optimize prompts via LLM
        4. Return calibration record with all results
        """
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}

        # Step 1: Compute alignment metrics
        alignment = self.compute_alignment(metric_results, annotations)

        # Step 2: Analyze disagreements
        disagreements = self.analyze_disagreements(metric_results, annotations, dataset_items)

        # Step 3: Suggest threshold adjustment (legacy behavior)
        suggested_threshold = self._suggest_threshold(metric_results, annotations)

        # Step 4: Optionally optimize prompts
        prompt_optimization = None
        if self.optimize_prompts and disagreements.total_disagreements > 0:
            try:
                optimizer = PromptOptimizer(model=self.optimizer_model)
                prompt_optimization = optimizer.optimize(
                    metric_id=self.judge_name,
                    current_rubric=self.current_rubric,
                    disagreements=disagreements,
                    alignment_metrics=alignment,
                )
            except Exception as e:
                prompt_optimization = PromptOptimizationResult(
                    original_rubric=self.current_rubric,
                    improved_rubric=self.current_rubric,
                    improvement_reasoning=f"Optimization skipped: {e}",
                    suggested_additions=[],
                    suggested_removals=[],
                    estimated_improvement="unknown",
                )

        # Build adjustments dict with all calibration data
        adjustments = {
            "current_threshold": self.current_threshold,
            "suggested_threshold": suggested_threshold,
            "alignment_metrics": alignment.as_dict(),
            "disagreement_patterns": disagreements.get_pattern_summary(),
        }

        if prompt_optimization:
            adjustments["prompt_optimization"] = prompt_optimization.as_dict()

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
