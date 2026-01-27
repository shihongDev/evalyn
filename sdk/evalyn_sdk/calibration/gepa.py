"""GEPAOptimizer: DSPy-based GEPA prompt optimizer.

Uses GEPA (Generative Evolutionary Prompt Adaptation) for prompt optimization.
GEPA uses evolutionary search with LLM reflection to optimize prompts.

Requires: pip install gepa
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..defaults import DEFAULT_DSPY_MODEL
from ..models import Annotation, DatasetItem, MetricResult
from .models import PromptOptimizationResult
from .utils import build_full_prompt

# GEPA import (optional dependency)
try:
    import gepa

    GEPA_AVAILABLE = True
except ImportError:
    GEPA_AVAILABLE = False


@dataclass
class GEPAConfig:
    """Configuration for GEPA optimization."""

    task_lm: str = DEFAULT_DSPY_MODEL  # Model being optimized
    reflection_lm: str = DEFAULT_DSPY_MODEL  # Model for reflection
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
            full_optimized_prompt = build_full_prompt(
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


__all__ = ["GEPAConfig", "GEPAOptimizer", "GEPA_AVAILABLE"]
