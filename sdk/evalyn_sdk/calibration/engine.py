"""CalibrationEngine: Full calibration pipeline.

Orchestrates the full calibration workflow:
1. Compute alignment metrics (precision, recall, F1, kappa)
2. Analyze disagreement patterns
3. Use optimizer (Basic, GEPA, GEPA-Native, OPRO, APE, or TextGrad) to improve prompts
4. Validate optimized prompts on held-out data

IMPORTANT: All optimizers only optimize the preamble (system prompt/instructions).
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..defaults import DEFAULT_EVAL_MODEL
from ..models import Annotation, CalibrationRecord, DatasetItem, MetricResult, now_utc
from .basic import BasicOptimizer
from .gepa import GEPA_AVAILABLE, GEPAConfig, GEPAOptimizer
from .models import (
    AlignmentMetrics,
    DisagreementAnalysis,
    DisagreementCase,
    PromptOptimizationResult,
    TokenAccumulator,
    ValidationResult,
)


@dataclass
class CalibrationConfig:
    """Configuration for CalibrationEngine.

    Groups all calibration parameters into a single config object.
    """

    judge_name: str
    current_threshold: float = 0.5
    current_rubric: List[str] = field(default_factory=list)
    current_preamble: str = ""  # Base prompt before rubric
    optimize_prompts: bool = True
    optimizer_model: str = DEFAULT_EVAL_MODEL
    optimizer_type: str = "basic"  # "basic", "gepa", "gepa-native", "opro", "ape", or "textgrad"
    gepa_config: Optional[GEPAConfig] = None
    gepa_native_config: Optional[Any] = None  # GEPANativeConfig
    opro_config: Optional[Any] = None  # OPROConfig (import avoided for circular deps)
    ape_config: Optional[Any] = None  # APEConfig (import avoided for circular deps)
    textgrad_config: Optional[Any] = None  # TextGradConfig (import avoided for circular deps)


class CalibrationEngine:
    """
    Enhanced calibration engine that:
    1. Computes alignment metrics (precision, recall, F1, kappa)
    2. Analyzes disagreement patterns
    3. Uses LLM, GEPA, GEPA-Native, OPRO, APE, or TextGrad to optimize judge prompts

    IMPORTANT: All optimizers only optimize the preamble (system prompt/instructions).
    The rubric (evaluation criteria) is kept FIXED as defined by humans.
    This prevents "reward hacking" where the judge optimizes away from human intent.
    """

    def __init__(
        self,
        judge_name: Optional[str] = None,
        current_threshold: float = 0.5,
        current_rubric: Optional[List[str]] = None,
        current_preamble: str = "",  # Base prompt before rubric
        optimize_prompts: bool = True,
        optimizer_model: str = DEFAULT_EVAL_MODEL,
        optimizer_type: str = "basic",  # "basic", "gepa", "gepa-native", "opro", "ape", or "textgrad"
        gepa_config: Optional[GEPAConfig] = None,
        gepa_native_config: Optional[Any] = None,  # GEPANativeConfig
        opro_config: Optional[Any] = None,  # OPROConfig
        ape_config: Optional[Any] = None,  # APEConfig
        textgrad_config: Optional[Any] = None,  # TextGradConfig
        *,
        config: Optional[CalibrationConfig] = None,
    ):
        # Support both config object and individual params (backwards compat)
        if config is not None:
            self.judge_name = config.judge_name
            self.current_threshold = config.current_threshold
            self.current_rubric = config.current_rubric
            self.current_preamble = config.current_preamble
            self.optimize_prompts = config.optimize_prompts
            self.optimizer_model = config.optimizer_model
            self.optimizer_type = config.optimizer_type
            self.gepa_config = config.gepa_config
            self.gepa_native_config = config.gepa_native_config
            self.opro_config = config.opro_config
            self.ape_config = config.ape_config
            self.textgrad_config = config.textgrad_config
        else:
            if judge_name is None:
                raise ValueError("judge_name is required")
            self.judge_name = judge_name
            self.current_threshold = current_threshold
            self.current_rubric = current_rubric or []
            self.current_preamble = current_preamble
            self.optimize_prompts = optimize_prompts
            self.optimizer_model = optimizer_model
            self.optimizer_type = optimizer_type
            self.gepa_config = gepa_config
            self.gepa_native_config = gepa_native_config
            self.opro_config = opro_config
            self.ape_config = ape_config
            self.textgrad_config = textgrad_config

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

    def _build_validation_split(
        self,
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]],
        *,
        val_split: float,
    ) -> Optional[tuple[List[Annotation], List[DatasetItem]]]:
        """Create validation annotation/data split for prompt comparison."""
        if not dataset_items or len(annotations) < 10:
            return None

        import random

        ann_list = list(annotations)
        random.shuffle(ann_list)
        split_idx = int(len(ann_list) * (1 - val_split))
        val_annotations = ann_list[split_idx:]
        if len(val_annotations) < 3:
            return None

        ann_by_call = {ann.target_id: ann for ann in val_annotations}
        val_items = [
            item for item in dataset_items if item.metadata.get("call_id") in ann_by_call
        ]
        if not val_items:
            return None
        return val_annotations, val_items

    def _create_validation_judges(
        self, *, metric_id: str, original_prompt: str, optimized_prompt: str
    ):
        """Construct original and optimized judges for validation scoring."""
        from ..judges import LLMJudge

        try:
            original_judge = LLMJudge(
                name=f"{metric_id}_original",
                prompt=original_prompt,
                model=self.optimizer_model,
                temperature=0.0,
            )
            optimized_judge = LLMJudge(
                name=f"{metric_id}_optimized",
                prompt=optimized_prompt,
                model=self.optimizer_model,
                temperature=0.0,
            )
            return original_judge, optimized_judge
        except Exception:
            return None

    def _build_validation_fake_call(self, item: DatasetItem):
        """Create a minimal FunctionCall for judge scoring on a dataset item."""
        from ..models import FunctionCall

        call_id = item.metadata.get("call_id")
        return FunctionCall(
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

    def _record_validation_prediction(
        self,
        metrics: AlignmentMetrics,
        *,
        predicted_pass: bool,
        human_pass: bool,
    ) -> None:
        """Update alignment counters for one prediction."""
        if predicted_pass and human_pass:
            metrics.true_positive += 1
        elif not predicted_pass and not human_pass:
            metrics.true_negative += 1
        elif predicted_pass and not human_pass:
            metrics.false_positive += 1
        else:
            metrics.false_negative += 1

    def _score_validation_item(
        self,
        judge,
        fake_call,
        item: DatasetItem,
        *,
        human_pass: bool,
        metrics: AlignmentMetrics,
        accumulator: Optional[TokenAccumulator],
    ) -> None:
        """Score one item with one judge and update metrics/token counts."""
        try:
            result = judge.score(fake_call, item)
            predicted_pass = bool(result.get("passed"))
            if accumulator:
                accumulator.add_usage(
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                    result.get("model"),
                )
            self._record_validation_prediction(
                metrics,
                predicted_pass=predicted_pass,
                human_pass=human_pass,
            )
        except Exception:
            pass

    def _build_validation_recommendation(
        self,
        *,
        original_metrics: AlignmentMetrics,
        optimized_metrics: AlignmentMetrics,
        validation_samples: int,
    ) -> ValidationResult:
        """Build ValidationResult from alignment metrics."""
        original_f1 = original_metrics.f1
        optimized_f1 = optimized_metrics.f1
        original_acc = original_metrics.accuracy
        optimized_acc = optimized_metrics.accuracy
        improvement_delta = optimized_f1 - original_f1

        is_better = improvement_delta > 0.02
        is_worse = improvement_delta < -0.02

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
            validation_samples=validation_samples,
        )

    def _validate_optimization(
        self,
        original_prompt: str,
        optimized_prompt: str,
        metric_id: str,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
        val_split: float = 0.3,
        accumulator: Optional[TokenAccumulator] = None,
    ) -> Optional[ValidationResult]:
        """
        Validate optimized prompt against validation set.

        1. Split data into train/val (70/30 by default)
        2. Re-run LLM judge with both prompts on val set
        3. Compare F1 scores
        4. Return ValidationResult with recommendation

        Args:
            original_prompt: The original prompt to compare against
            optimized_prompt: The optimized prompt to validate
            metric_id: The metric being calibrated
            metric_results: Evaluation results from the judge
            annotations: Human annotations
            dataset_items: Optional dataset items for context
            val_split: Fraction of data to use for validation (default: 0.3)
            accumulator: Optional TokenAccumulator to track token usage

        Returns None if validation cannot be performed.
        """
        split = self._build_validation_split(
            annotations,
            dataset_items,
            val_split=val_split,
        )
        if split is None:
            return None
        val_annotations, val_items = split

        judges = self._create_validation_judges(
            metric_id=metric_id,
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
        )
        if judges is None:
            return None
        original_judge, optimized_judge = judges

        original_metrics = AlignmentMetrics()
        optimized_metrics = AlignmentMetrics()
        ann_by_call = {ann.target_id: ann for ann in val_annotations}

        for item in val_items:
            call_id = item.metadata.get("call_id")
            ann = ann_by_call.get(call_id)
            if not ann:
                continue
            human_pass = bool(ann.label)
            fake_call = self._build_validation_fake_call(item)
            self._score_validation_item(
                original_judge,
                fake_call,
                item,
                human_pass=human_pass,
                metrics=original_metrics,
                accumulator=accumulator,
            )
            self._score_validation_item(
                optimized_judge,
                fake_call,
                item,
                human_pass=human_pass,
                metrics=optimized_metrics,
                accumulator=accumulator,
            )

        return self._build_validation_recommendation(
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
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

        # Token accumulator for tracking LLM usage across optimization
        accumulator = TokenAccumulator()

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
                    # Note: GEPA uses external library, token tracking not available
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
                elif self.optimizer_type == "gepa-native":
                    # Use native GEPA with token tracking (no external library)
                    from .gepa_native import GEPANativeOptimizer

                    gepa_native_optimizer = GEPANativeOptimizer(
                        config=self.gepa_native_config
                    )
                    prompt_optimization = gepa_native_optimizer.optimize(
                        metric_id=self.judge_name,
                        current_rubric=self.current_rubric,
                        metric_results=metric_results,
                        annotations=annotations,
                        dataset_items=dataset_items,
                        current_preamble=self.current_preamble,
                        accumulator=accumulator,
                    )
                elif self.optimizer_type == "opro":
                    # Use OPRO optimization (preamble only, rubric stays fixed)
                    from .opro import OPROOptimizer

                    opro_optimizer = OPROOptimizer(config=self.opro_config)
                    prompt_optimization = opro_optimizer.optimize(
                        metric_id=self.judge_name,
                        current_rubric=self.current_rubric,
                        metric_results=metric_results,
                        annotations=annotations,
                        dataset_items=dataset_items,
                        current_preamble=self.current_preamble,
                        accumulator=accumulator,
                    )
                elif self.optimizer_type == "ape":
                    # Use APE optimization (preamble only, rubric stays fixed)
                    from .ape import APEOptimizer

                    ape_optimizer = APEOptimizer(config=self.ape_config)
                    prompt_optimization = ape_optimizer.optimize(
                        metric_id=self.judge_name,
                        current_rubric=self.current_rubric,
                        metric_results=metric_results,
                        annotations=annotations,
                        disagreements=disagreements,
                        dataset_items=dataset_items,
                        current_preamble=self.current_preamble,
                        accumulator=accumulator,
                    )
                elif self.optimizer_type == "textgrad":
                    # Use TextGrad optimization (preamble only, rubric stays fixed)
                    from .textgrad import TextGradOptimizer

                    textgrad_optimizer = TextGradOptimizer(
                        config=self.textgrad_config
                    )
                    prompt_optimization = textgrad_optimizer.optimize(
                        metric_id=self.judge_name,
                        current_rubric=self.current_rubric,
                        current_preamble=self.current_preamble,
                        metric_results=metric_results,
                        annotations=annotations,
                        dataset_items=dataset_items,
                        accumulator=accumulator,
                    )
                else:
                    # Use basic single-shot optimization (default)
                    # Note: Like APE/OPRO, only preamble is optimized; rubric stays fixed
                    optimizer = BasicOptimizer(model=self.optimizer_model)
                    prompt_optimization = optimizer.optimize(
                        metric_id=self.judge_name,
                        current_rubric=self.current_rubric,
                        disagreements=disagreements,
                        alignment_metrics=alignment,
                        current_preamble=self.current_preamble,
                        accumulator=accumulator,
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
                        accumulator=accumulator,
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
            # Include optimizer-specific config
            if self.optimizer_type == "gepa" and self.gepa_config:
                adjustments["gepa_config"] = {
                    "task_lm": self.gepa_config.task_lm,
                    "reflection_lm": self.gepa_config.reflection_lm,
                    "max_metric_calls": self.gepa_config.max_metric_calls,
                }
            elif self.optimizer_type == "gepa-native" and self.gepa_native_config:
                adjustments["gepa_native_config"] = {
                    "task_model": self.gepa_native_config.task_model,
                    "reflection_model": self.gepa_native_config.reflection_model,
                    "max_metric_calls": self.gepa_native_config.max_metric_calls,
                    "num_initial_candidates": self.gepa_native_config.num_initial_candidates,
                    "mini_batch_size": self.gepa_native_config.mini_batch_size,
                }
            elif self.optimizer_type == "ape" and self.ape_config:
                adjustments["ape_config"] = {
                    "num_candidates": self.ape_config.num_candidates,
                    "eval_rounds": self.ape_config.eval_rounds,
                    "eval_samples_per_round": self.ape_config.eval_samples_per_round,
                }
            elif self.optimizer_type == "textgrad" and self.textgrad_config:
                adjustments["textgrad_config"] = {
                    "max_iterations": self.textgrad_config.max_iterations,
                    "improvement_threshold": self.textgrad_config.improvement_threshold,
                    "num_failure_examples": self.textgrad_config.num_failure_examples,
                    "early_stop_patience": self.textgrad_config.early_stop_patience,
                }

        if validation_result:
            adjustments["validation"] = validation_result.as_dict()

        return CalibrationRecord(
            id=str(uuid4()),
            judge_config_id=self.judge_name,
            gold_items=list(ann_by_call.keys()),
            adjustments=adjustments,
            created_at=now_utc(),
            usage_summary=accumulator.as_usage_summary(),
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


__all__ = ["CalibrationConfig", "CalibrationEngine"]
