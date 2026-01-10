from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ..models import Annotation, CalibrationRecord, MetricResult, DatasetItem, now_utc

# GEPA import (optional dependency)
try:
    import gepa

    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False


@dataclass
class AlignmentMetrics:
    """
    Alignment metrics between LLM judge and human annotations.
    Uses classification terminology: judge prediction vs human ground truth.
    """

    # Confusion matrix counts
    true_positive: int = 0  # Both judge and human say PASS
    true_negative: int = 0  # Both judge and human say FAIL
    false_positive: int = 0  # Judge says PASS, human says FAIL
    false_negative: int = 0  # Judge says FAIL, human says PASS

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )

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
        p_e = (judge_pass_rate * human_pass_rate) + (
            (1 - judge_pass_rate) * (1 - human_pass_rate)
        )

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
    """Result of prompt optimization (LLM or GEPA)."""

    original_rubric: List[str]
    improved_rubric: List[str]
    improvement_reasoning: str
    suggested_additions: List[str]
    suggested_removals: List[str]
    estimated_improvement: str  # "low", "medium", "high"
    # New fields for storing the full prompt
    original_preamble: str = ""
    optimized_preamble: str = ""
    full_prompt: str = ""  # Complete prompt ready to use (preamble + rubric + format)

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "original_rubric": self.original_rubric,
            "improved_rubric": self.improved_rubric,
            "improvement_reasoning": self.improvement_reasoning,
            "suggested_additions": self.suggested_additions,
            "suggested_removals": self.suggested_removals,
            "estimated_improvement": self.estimated_improvement,
        }
        # Include preamble fields if they have content
        if self.original_preamble:
            result["original_preamble"] = self.original_preamble
        if self.optimized_preamble:
            result["optimized_preamble"] = self.optimized_preamble
        if self.full_prompt:
            result["full_prompt"] = self.full_prompt
        return result


@dataclass
class ValidationResult:
    """Result of validating optimized prompt against validation set."""

    original_f1: float
    optimized_f1: float
    original_accuracy: float
    optimized_accuracy: float
    improvement_delta: float
    is_better: bool
    confidence: str  # "high", "medium", "low"
    recommendation: str  # "use_optimized", "keep_original", "uncertain"
    validation_samples: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_f1": self.original_f1,
            "optimized_f1": self.optimized_f1,
            "original_accuracy": self.original_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "improvement_delta": self.improvement_delta,
            "is_better": self.is_better,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "validation_samples": self.validation_samples,
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
        key = (
            self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
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
        rubric_text = (
            "\n".join([f"- {r}" for r in current_rubric])
            if current_rubric
            else "(no rubric defined)"
        )

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
                            json_str = text[start : i + 1]
                            parsed = json.loads(json_str)

                            return PromptOptimizationResult(
                                original_rubric=original_rubric,
                                improved_rubric=parsed.get(
                                    "improved_rubric", original_rubric
                                ),
                                improvement_reasoning=parsed.get(
                                    "improvement_reasoning", ""
                                ),
                                suggested_additions=parsed.get(
                                    "suggested_additions", []
                                ),
                                suggested_removals=parsed.get("suggested_removals", []),
                                estimated_improvement=parsed.get(
                                    "estimated_improvement", "unknown"
                                ),
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


@dataclass
class GEPAConfig:
    """Configuration for GEPA optimization."""

    task_lm: str = "gemini/gemini-2.5-flash"  # Model being optimized
    reflection_lm: str = "gemini/gemini-2.5-flash"  # Model for reflection
    max_metric_calls: int = 150  # Budget for optimization
    train_split: float = 0.7  # Train/val split ratio


class GEPAOptimizer:
    """
    Uses GEPA (Generative Evolutionary Prompt Adaptation) for prompt optimization.
    GEPA uses evolutionary search with LLM reflection to optimize prompts.

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.

    Prompt structure:
    - preamble: Optimized by GEPA (e.g., "You are an expert evaluator...")
    - rubric: Fixed, human-defined criteria (e.g., "- No factual errors...")
    - output_format: Fixed instructions for JSON output

    Requires: pip install gepa
    """

    def __init__(self, config: Optional[GEPAConfig] = None):
        if not GEPA_AVAILABLE:
            raise ImportError(
                "GEPA is not installed. Install it with: pip install gepa"
            )
        self.config = config or GEPAConfig()

    def _build_dataset_from_annotations(
        self,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert calibration data to GEPA-compatible train/val sets.
        Each example contains: input, output, expected (human label).
        """
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
        items_by_call: Dict[str, DatasetItem] = {}
        if dataset_items:
            for item in dataset_items:
                call_id = item.metadata.get("call_id", item.id)
                items_by_call[call_id] = item

        examples = []
        for res in metric_results:
            ann = ann_by_call.get(res.call_id)
            if not ann:
                continue

            item = items_by_call.get(res.call_id)
            input_text = ""
            output_text = ""
            if item:
                input_text = json.dumps(item.input, default=str) if item.input else ""
                output_text = str(item.output) if item.output else ""

            examples.append(
                {
                    "input": input_text,
                    "output": output_text,
                    "expected": "PASS" if ann.label else "FAIL",
                    "call_id": res.call_id,
                }
            )

        # Split into train/val
        import random

        random.shuffle(examples)
        split_idx = int(len(examples) * self.config.train_split)
        trainset = examples[:split_idx]
        valset = examples[split_idx:]

        return trainset, valset

    def _build_seed_prompt(
        self, metric_id: str, current_preamble: str
    ) -> Dict[str, str]:
        """
        Build seed prompt with only the preamble (the part to be optimized).
        The rubric will be appended separately after optimization.
        """
        if current_preamble:
            preamble = current_preamble.strip()
        else:
            # Default preamble if none provided
            preamble = f"You are an expert evaluator for the metric: {metric_id}. Carefully analyze the output quality."

        return {"preamble": preamble}

    def _build_full_prompt(self, preamble: str, rubric: List[str]) -> str:
        """
        Combine optimized preamble with fixed rubric to create the full prompt.
        This is what the judge will use.
        """
        rubric_text = ""
        if rubric:
            rubric_lines = "\n".join([f"- {r}" for r in rubric])
            rubric_text = f"\n\nEvaluate using this rubric (PASS only if all criteria met):\n{rubric_lines}"

        output_format = """

After your analysis, provide your verdict as a JSON object:
{"passed": true/false, "reason": "brief explanation", "score": 0.0-1.0}"""

        return preamble + rubric_text + output_format

    def optimize(
        self,
        metric_id: str,
        current_rubric: List[str],
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
        current_preamble: str = "",
    ) -> PromptOptimizationResult:
        """
        Run GEPA optimization to improve the judge preamble.
        The rubric is kept fixed; only the preamble (framing/instructions) is optimized.

        Args:
            metric_id: The metric being calibrated
            current_rubric: Fixed rubric criteria (NOT optimized)
            metric_results: Evaluation results from the judge
            annotations: Human annotations
            dataset_items: Optional dataset items for context
            current_preamble: Current preamble text (the part to optimize)

        Returns:
            PromptOptimizationResult with optimized preamble in improved_rubric[0]
            and the original rubric preserved.
        """
        # Build datasets
        trainset, valset = self._build_dataset_from_annotations(
            metric_results, annotations, dataset_items
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning="Not enough data for GEPA optimization (need at least 5 annotated examples)",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
            )

        # Build seed prompt (only the preamble, not the rubric)
        seed_prompt = self._build_seed_prompt(metric_id, current_preamble)

        # For GEPA, we need to provide the full prompt but only optimize the preamble.
        # We'll inject the fixed rubric into the trainset/valset format instructions.
        rubric_text = ""
        if current_rubric:
            rubric_lines = "\n".join([f"- {r}" for r in current_rubric])
            rubric_text = f"\n\nEvaluate using this rubric (PASS only if all criteria met):\n{rubric_lines}"

        # Add rubric and output format as fixed suffix in the system prompt
        fixed_suffix = (
            rubric_text
            + """

After your analysis, provide your verdict as a JSON object:
{"passed": true/false, "reason": "brief explanation", "score": 0.0-1.0}"""
        )

        # The seed candidate is just the preamble; GEPA will optimize this
        seed_candidate = {"preamble": seed_prompt["preamble"]}

        try:
            # Run GEPA optimization on the preamble only
            # Note: GEPA will optimize the preamble field
            gepa_result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                task_lm=self.config.task_lm,
                max_metric_calls=self.config.max_metric_calls,
                reflection_lm=self.config.reflection_lm,
            )

            # Extract optimized preamble
            optimized_preamble = gepa_result.best_candidate.get(
                "preamble", seed_prompt["preamble"]
            )

            # Build the full optimized prompt (ready to use)
            full_optimized_prompt = self._build_full_prompt(
                optimized_preamble, current_rubric
            )

            # Return result with original rubric preserved and new preamble
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,  # Rubric stays the same
                improvement_reasoning=f"GEPA optimization completed. Best score: {getattr(gepa_result, 'best_score', 'N/A')}",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="high"
                if optimized_preamble != seed_prompt["preamble"]
                else "low",
                # New fields for storing the optimized prompt
                original_preamble=current_preamble,
                optimized_preamble=optimized_preamble,
                full_prompt=full_optimized_prompt,
            )

        except Exception as e:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning=f"GEPA optimization failed: {e}",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )


class CalibrationEngine:
    """
    Enhanced calibration engine that:
    1. Computes alignment metrics (precision, recall, F1, kappa)
    2. Analyzes disagreement patterns
    3. Uses LLM or GEPA to optimize judge prompts

    For GEPA: Only the preamble is optimized; rubric stays fixed.
    For LLM: Both preamble and rubric can be suggested for improvement.
    """

    def __init__(
        self,
        judge_name: str,
        current_threshold: float = 0.5,
        current_rubric: Optional[List[str]] = None,
        current_preamble: str = "",  # Base prompt before rubric
        optimize_prompts: bool = True,
        optimizer_model: str = "gemini-2.5-flash-lite",
        optimizer_type: str = "llm",  # "llm" or "gepa"
        gepa_config: Optional[GEPAConfig] = None,
    ):
        self.judge_name = judge_name
        self.current_threshold = current_threshold
        self.current_rubric = current_rubric or []
        self.current_preamble = current_preamble
        self.optimize_prompts = optimize_prompts
        self.optimizer_model = optimizer_model
        self.optimizer_type = optimizer_type
        self.gepa_config = gepa_config

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

    def _validate_optimization(
        self,
        original_prompt: str,
        optimized_prompt: str,
        metric_id: str,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
        val_split: float = 0.3,
    ) -> Optional[ValidationResult]:
        """
        Validate optimized prompt against validation set.

        1. Split data into train/val (70/30 by default)
        2. Re-run LLM judge with both prompts on val set
        3. Compare F1 scores
        4. Return ValidationResult with recommendation

        Returns None if validation cannot be performed.
        """
        if not dataset_items or len(annotations) < 10:
            # Need at least 10 annotations for meaningful validation
            return None

        # Import judge here to avoid circular dependency
        from ..metrics.judges import GeminiJudge

        # Split annotations into train/val
        import random

        ann_list = list(annotations)
        random.shuffle(ann_list)
        split_idx = int(len(ann_list) * (1 - val_split))
        val_annotations = ann_list[split_idx:]

        if len(val_annotations) < 3:
            # Not enough validation samples
            return None

        # Get validation dataset items
        ann_by_call = {ann.target_id: ann for ann in val_annotations}
        val_items = [
            item
            for item in dataset_items
            if item.metadata.get("call_id") in ann_by_call
        ]

        if not val_items:
            return None

        # Create judges with original and optimized prompts
        try:
            original_judge = GeminiJudge(
                name=f"{metric_id}_original",
                prompt=original_prompt,
                model=self.optimizer_model,
                temperature=0.0,
            )

            optimized_judge = GeminiJudge(
                name=f"{metric_id}_optimized",
                prompt=optimized_prompt,
                model=self.optimizer_model,
                temperature=0.0,
            )
        except Exception:
            return None

        # Evaluate both prompts on validation set
        original_metrics = AlignmentMetrics()
        optimized_metrics = AlignmentMetrics()

        from ..models import FunctionCall

        for item in val_items:
            call_id = item.metadata.get("call_id")
            ann = ann_by_call.get(call_id)
            if not ann:
                continue

            # Create a fake FunctionCall for the judge
            fake_call = FunctionCall(
                id=call_id,
                function_name="validation",
                inputs=item.input or {},
                output=item.output,
                error=None,
                started_at=now_utc(),
                ended_at=now_utc(),
                duration_ms=0.0,
                session_id=None,
                trace=[],
                metadata={},
            )

            human_pass = bool(ann.label)

            # Score with original prompt
            try:
                original_result = original_judge.score(fake_call, item)
                original_pass = bool(original_result.get("passed"))

                if original_pass and human_pass:
                    original_metrics.true_positive += 1
                elif not original_pass and not human_pass:
                    original_metrics.true_negative += 1
                elif original_pass and not human_pass:
                    original_metrics.false_positive += 1
                else:
                    original_metrics.false_negative += 1
            except Exception:
                pass

            # Score with optimized prompt
            try:
                optimized_result = optimized_judge.score(fake_call, item)
                optimized_pass = bool(optimized_result.get("passed"))

                if optimized_pass and human_pass:
                    optimized_metrics.true_positive += 1
                elif not optimized_pass and not human_pass:
                    optimized_metrics.true_negative += 1
                elif optimized_pass and not human_pass:
                    optimized_metrics.false_positive += 1
                else:
                    optimized_metrics.false_negative += 1
            except Exception:
                pass

        # Compare F1 scores
        original_f1 = original_metrics.f1
        optimized_f1 = optimized_metrics.f1
        original_acc = original_metrics.accuracy
        optimized_acc = optimized_metrics.accuracy
        improvement_delta = optimized_f1 - original_f1

        # Determine if optimized is better
        is_better = improvement_delta > 0.02  # At least 2% improvement
        is_worse = improvement_delta < -0.02  # More than 2% degradation

        # Determine confidence and recommendation
        if abs(improvement_delta) < 0.02:
            confidence = "low"
            recommendation = "uncertain"
        elif abs(improvement_delta) < 0.05:
            confidence = "medium"
            recommendation = "use_optimized" if is_better else "keep_original"
        else:
            confidence = "high"
            recommendation = "use_optimized" if is_better else "keep_original"

        return ValidationResult(
            original_f1=original_f1,
            optimized_f1=optimized_f1,
            original_accuracy=original_acc,
            optimized_accuracy=optimized_acc,
            improvement_delta=improvement_delta,
            is_better=is_better and not is_worse,
            confidence=confidence,
            recommendation=recommendation,
            validation_samples=len(val_items),
        )

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
        disagreements = self.analyze_disagreements(
            metric_results, annotations, dataset_items
        )

        # Step 3: Suggest threshold adjustment (legacy behavior)
        suggested_threshold = self._suggest_threshold(metric_results, annotations)

        # Step 4: Optionally optimize prompts
        prompt_optimization = None
        if self.optimize_prompts and disagreements.total_disagreements > 0:
            try:
                if self.optimizer_type == "gepa":
                    # Use GEPA evolutionary optimization (preamble only, rubric stays fixed)
                    if not GEPA_AVAILABLE:
                        raise ImportError(
                            "GEPA is not installed. Install with: pip install gepa"
                        )
                    gepa_optimizer = GEPAOptimizer(config=self.gepa_config)
                    prompt_optimization = gepa_optimizer.optimize(
                        metric_id=self.judge_name,
                        current_rubric=self.current_rubric,
                        metric_results=metric_results,
                        annotations=annotations,
                        dataset_items=dataset_items,
                        current_preamble=self.current_preamble,
                    )
                else:
                    # Use LLM-based optimization (default)
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

        # Step 5: Validate optimized prompt if available
        validation_result = None
        if prompt_optimization and dataset_items:
            # Build original prompt for comparison
            original_prompt = self.current_preamble
            if self.current_rubric:
                original_prompt += (
                    "\n\nEvaluate using this rubric (PASS only if all criteria met):\n"
                )
                original_prompt += "\n".join([f"- {r}" for r in self.current_rubric])

            optimized_prompt = (
                prompt_optimization.full_prompt
                if prompt_optimization.full_prompt
                else original_prompt
            )

            if optimized_prompt != original_prompt:
                try:
                    validation_result = self._validate_optimization(
                        original_prompt=original_prompt,
                        optimized_prompt=optimized_prompt,
                        metric_id=self.judge_name,
                        metric_results=metric_results,
                        annotations=annotations,
                        dataset_items=dataset_items,
                    )
                except Exception:
                    validation_result = None

        # Build adjustments dict with all calibration data
        adjustments = {
            "current_threshold": self.current_threshold,
            "suggested_threshold": suggested_threshold,
            "alignment_metrics": alignment.as_dict(),
            "disagreement_patterns": disagreements.get_pattern_summary(),
            "optimizer_type": self.optimizer_type,
        }

        if prompt_optimization:
            adjustments["prompt_optimization"] = prompt_optimization.as_dict()
            # Include GEPA config if used
            if self.optimizer_type == "gepa" and self.gepa_config:
                adjustments["gepa_config"] = {
                    "task_lm": self.gepa_config.task_lm,
                    "reflection_lm": self.gepa_config.reflection_lm,
                    "max_metric_calls": self.gepa_config.max_metric_calls,
                }

        if validation_result:
            adjustments["validation"] = validation_result.as_dict()

        return CalibrationRecord(
            id=str(uuid4()),
            judge_config_id=self.judge_name,
            gold_items=list(ann_by_call.keys()),
            adjustments=adjustments,
            created_at=now_utc(),
        )

    def _suggest_threshold(
        self, metric_results: List[MetricResult], annotations: List[Annotation]
    ) -> float:
        """Simple heuristic: align judge pass-rate with human positive rate."""
        ann_by_call: Dict[str, Annotation] = {ann.target_id: ann for ann in annotations}
        human_labels = [bool(ann.label) for ann in annotations]
        judge_passes = [
            bool(res.passed)
            for res in metric_results
            if res.call_id in ann_by_call and res.passed is not None
        ]

        human_rate = mean(human_labels) if human_labels else self.current_threshold
        judge_rate = mean(judge_passes) if judge_passes else self.current_threshold

        # Shift threshold toward reducing the gap; clamp between 0 and 1.
        delta = judge_rate - human_rate
        new_threshold = self.current_threshold + delta
        return max(0.0, min(1.0, new_threshold))


def save_calibration(
    record: CalibrationRecord,
    dataset_path: str,
    metric_id: str,
) -> Dict[str, str]:
    """
    Save calibration record and optimized prompts to the dataset's calibrations folder.

    Directory structure:
        <dataset>/
          calibrations/
            <metric_id>/
              <timestamp>_<optimizer>.json     # Full calibration record
              prompts/
                <timestamp>_preamble.txt       # Optimized preamble only
                <timestamp>_full.txt           # Full prompt (ready to use)

    Args:
        record: CalibrationRecord to save
        dataset_path: Path to the dataset folder
        metric_id: Metric ID being calibrated

    Returns:
        Dict with paths to saved files:
        - "calibration": Path to the calibration JSON
        - "preamble": Path to preamble file (if available)
        - "full_prompt": Path to full prompt file (if available)
    """
    from pathlib import Path
    from datetime import datetime

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Create calibrations directory structure
    calibrations_dir = dataset_dir / "calibrations" / metric_id
    prompts_dir = calibrations_dir / "prompts"
    calibrations_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_type = record.adjustments.get("optimizer_type", "llm")

    saved_paths: Dict[str, str] = {}

    # Save the calibration record JSON
    calibration_file = calibrations_dir / f"{timestamp}_{optimizer_type}.json"
    with open(calibration_file, "w", encoding="utf-8") as f:
        json.dump(record.as_dict(), f, indent=2, default=str)
    saved_paths["calibration"] = str(calibration_file)

    # Extract and save optimized prompts if available
    optimization = record.adjustments.get("prompt_optimization", {})
    if optimization:
        # Save optimized preamble
        optimized_preamble = optimization.get("optimized_preamble", "")
        if optimized_preamble:
            preamble_file = prompts_dir / f"{timestamp}_preamble.txt"
            with open(preamble_file, "w", encoding="utf-8") as f:
                f.write(optimized_preamble)
            saved_paths["preamble"] = str(preamble_file)

        # Save full prompt (ready to use)
        full_prompt = optimization.get("full_prompt", "")
        if full_prompt:
            full_file = prompts_dir / f"{timestamp}_full.txt"
            with open(full_file, "w", encoding="utf-8") as f:
                f.write(full_prompt)
            saved_paths["full_prompt"] = str(full_file)

        # If no full_prompt but we have improved_rubric, build it
        if not full_prompt and optimization.get("improved_rubric"):
            preamble = optimization.get("optimized_preamble", "")
            rubric = optimization.get("improved_rubric", [])
            if preamble or rubric:
                rubric_text = ""
                if rubric:
                    rubric_lines = "\n".join([f"- {r}" for r in rubric])
                    rubric_text = f"\n\nEvaluate using this rubric (PASS only if all criteria met):\n{rubric_lines}"

                full_built = (preamble or "") + rubric_text
                if full_built.strip():
                    full_file = prompts_dir / f"{timestamp}_full.txt"
                    with open(full_file, "w", encoding="utf-8") as f:
                        f.write(full_built)
                    saved_paths["full_prompt"] = str(full_file)

    return saved_paths


def load_optimized_prompt(
    dataset_path: str,
    metric_id: str,
    version: Optional[str] = None,
) -> Optional[str]:
    """
    Load an optimized prompt from the calibrations folder.

    Args:
        dataset_path: Path to the dataset folder
        metric_id: Metric ID to load prompt for
        version: Specific version timestamp (e.g., "20250101_120000").
                 If None, loads the most recent.

    Returns:
        The full optimized prompt text, or None if not found.
    """
    from pathlib import Path

    prompts_dir = Path(dataset_path) / "calibrations" / metric_id / "prompts"
    if not prompts_dir.exists():
        return None

    # Find full prompt files
    full_files = sorted(prompts_dir.glob("*_full.txt"), reverse=True)
    if not full_files:
        return None

    if version:
        # Find specific version
        for f in full_files:
            if f.name.startswith(version):
                return f.read_text(encoding="utf-8")
        return None
    else:
        # Return most recent
        return full_files[0].read_text(encoding="utf-8")
