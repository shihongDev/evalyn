"""OPRO: Optimization by PROmpting.

Implements the OPRO algorithm from Google DeepMind for prompt optimization.
Unlike gradient-based methods, OPRO uses LLM as the optimizer itself.

Key concepts:
- Meta-prompt: Contains problem description + optimization trajectory
- Trajectory: Past solutions with scores, sorted worst-to-best
- Iterative loop: Generate candidates -> Evaluate -> Select best -> Repeat

Reference: https://arxiv.org/abs/2309.03409

IMPORTANT: Only the preamble (system prompt/instructions) is optimized.
The rubric (evaluation criteria) is kept FIXED as defined by humans.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..models import Annotation, DatasetItem, MetricResult
from ..utils.api_client import GeminiClient
from .calibration import AlignmentMetrics, PromptOptimizationResult


@dataclass
class TrajectoryEntry:
    """A single entry in the optimization trajectory."""

    preamble: str
    f1_score: float
    accuracy: float
    iteration: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "preamble": self.preamble,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "iteration": self.iteration,
        }


@dataclass
class OPROConfig:
    """Configuration for OPRO optimizer."""

    optimizer_model: str = "gemini-2.5-flash"  # LLM for generating candidates
    scorer_model: str = "gemini-2.5-flash-lite"  # LLM for evaluating (judge)
    max_iterations: int = 10  # Max optimization iterations
    candidates_per_step: int = 4  # Number of new prompts per iteration
    trajectory_length: int = 20  # Max past solutions to keep in meta-prompt
    train_split: float = 0.7  # Train/val split ratio
    temperature: float = 0.7  # Higher for diversity in generation
    early_stop_patience: int = 3  # Stop if no improvement for N iterations
    timeout: int = 120  # API timeout in seconds


class OPROOptimizer:
    """
    OPRO: Optimization by PROmpting.

    Uses LLM to iteratively generate and evaluate prompt candidates.
    The optimizer LLM sees a trajectory of past attempts with their scores,
    and generates new candidates that might score higher.

    IMPORTANT: Only the preamble (framing/instructions) is optimized.
    The rubric (evaluation criteria) is kept fixed as defined by humans.

    Prompt structure:
    - preamble: Optimized by OPRO (e.g., "You are an expert evaluator...")
    - rubric: Fixed, human-defined criteria (e.g., "- No factual errors...")
    - output_format: Fixed instructions for JSON output
    """

    def __init__(
        self,
        config: Optional[OPROConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize OPRO optimizer.

        Args:
            config: OPRO configuration (uses defaults if not provided)
            api_key: Optional API key for Gemini (default: from env)
        """
        self.config = config or OPROConfig()
        self._api_key = api_key
        self._optimizer_client: Optional[GeminiClient] = None
        self._scorer_client: Optional[GeminiClient] = None

    @property
    def optimizer_client(self) -> GeminiClient:
        """Lazy-initialized optimizer LLM client (for generating candidates)."""
        if self._optimizer_client is None:
            self._optimizer_client = GeminiClient(
                model=self.config.optimizer_model,
                temperature=self.config.temperature,
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._optimizer_client

    @property
    def scorer_client(self) -> GeminiClient:
        """Lazy-initialized scorer LLM client (for evaluating prompts)."""
        if self._scorer_client is None:
            self._scorer_client = GeminiClient(
                model=self.config.scorer_model,
                temperature=0.0,  # Deterministic for scoring
                api_key=self._api_key,
                timeout=self.config.timeout,
            )
        return self._scorer_client

    def _build_dataset_from_annotations(
        self,
        metric_results: List[MetricResult],
        annotations: List[Annotation],
        dataset_items: Optional[List[DatasetItem]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert calibration data to train/val sets.
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

        # Shuffle and split into train/val
        random.shuffle(examples)
        split_idx = int(len(examples) * self.config.train_split)
        trainset = examples[:split_idx]
        valset = examples[split_idx:]

        return trainset, valset

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

    def _evaluate_prompt(
        self,
        preamble: str,
        rubric: List[str],
        examples: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """
        Evaluate a prompt by running it as a judge on examples.

        Returns:
            Tuple of (f1_score, accuracy)
        """
        full_prompt = self._build_full_prompt(preamble, rubric)
        metrics = AlignmentMetrics()

        for ex in examples:
            # Build the evaluation prompt for this example
            eval_prompt = f"""{full_prompt}

## Input to evaluate
{ex.get('input', '')[:1000]}

## Output to evaluate
{ex.get('output', '')[:1000]}

Provide your verdict:"""

            try:
                response = self.scorer_client.generate(eval_prompt)
                # Parse the response to extract passed verdict
                judge_pass = self._parse_judge_response(response)
                human_pass = ex.get("expected") == "PASS"
                metrics.record(judge_pass, human_pass)
            except Exception:
                # On error, skip this example
                pass

        return metrics.f1, metrics.accuracy

    def _parse_judge_response(self, response: str) -> bool:
        """Parse judge response to extract pass/fail verdict."""
        text = response.strip().lower()

        # Try to find JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return bool(data.get("passed", False))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for keywords
        if "true" in text or '"passed": true' in text:
            return True
        if "false" in text or '"passed": false' in text:
            return False

        # Default to fail if unclear
        return False

    def _build_meta_prompt(
        self,
        metric_id: str,
        rubric: List[str],
        trajectory: List[TrajectoryEntry],
        train_examples: List[Dict[str, Any]],
        num_candidates: int,
    ) -> str:
        """
        Build the meta-prompt for OPRO optimization.

        The meta-prompt contains:
        1. Task description
        2. Optimization trajectory (sorted worst-to-best)
        3. Training examples
        4. Instructions for generating new candidates
        """
        # Format rubric
        rubric_text = "\n".join([f"- {r}" for r in rubric]) if rubric else "(no rubric defined)"

        # Format trajectory (sorted by F1, worst first)
        sorted_trajectory = sorted(trajectory, key=lambda x: x.f1_score)
        trajectory_text = ""
        for entry in sorted_trajectory[-self.config.trajectory_length :]:
            # Truncate long preambles
            preamble_preview = (
                entry.preamble[:300] + "..."
                if len(entry.preamble) > 300
                else entry.preamble
            )
            trajectory_text += f"""
F1 Score: {entry.f1_score:.3f} | Accuracy: {entry.accuracy:.3f}
Prompt: {preamble_preview}
---"""

        # Format training examples (show a few)
        examples_text = ""
        sample_examples = train_examples[:5]  # Show up to 5 examples
        for i, ex in enumerate(sample_examples, 1):
            input_preview = ex.get("input", "")[:200]
            output_preview = ex.get("output", "")[:200]
            expected = ex.get("expected", "UNKNOWN")
            examples_text += f"""
Example {i} (Human verdict: {expected}):
  Input: {input_preview}...
  Output: {output_preview}...
"""

        meta_prompt = f"""You are an expert prompt optimizer. Your task is to generate improved evaluation prompts.

## TASK DESCRIPTION
Optimize a PREAMBLE (system prompt) for the metric: {metric_id}
The preamble instructs an LLM judge to evaluate outputs and return PASS/FAIL verdicts.

IMPORTANT: The rubric criteria below are FIXED and must NOT be modified.
You are only optimizing the PREAMBLE (instructions/framing before the rubric).

## FIXED RUBRIC (do not modify)
{rubric_text}

## OPTIMIZATION TRAJECTORY
Below are previous preambles and their F1 scores (sorted worst to best).
Study what worked and what didn't:
{trajectory_text if trajectory_text else "(no previous attempts yet)"}

## TRAINING EXAMPLES
Here are examples showing expected behavior:
{examples_text}

## YOUR TASK
Generate {num_candidates} new PREAMBLES that might score higher than the best so far.

Guidelines for good preambles:
- Clear role definition (e.g., "You are an expert evaluator...")
- Specific instructions on how to evaluate
- Guidance on common edge cases
- Emphasis on what matters most for this metric
- Keep it concise but complete (100-300 words ideal)

Return ONLY a JSON array of preamble strings:
["preamble 1", "preamble 2", ...]

Do not include any other text, just the JSON array."""

        return meta_prompt

    def _generate_candidates(
        self,
        metric_id: str,
        rubric: List[str],
        trajectory: List[TrajectoryEntry],
        train_examples: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate new candidate preambles using the optimizer LLM."""
        meta_prompt = self._build_meta_prompt(
            metric_id,
            rubric,
            trajectory,
            train_examples,
            self.config.candidates_per_step,
        )

        try:
            response = self.optimizer_client.generate(meta_prompt)
            return self._parse_candidates_response(response)
        except Exception:
            return []

    def _parse_candidates_response(self, response: str) -> List[str]:
        """Parse the optimizer's response to extract candidate preambles."""
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        # Find JSON array
        try:
            start = text.find("[")
            if start >= 0:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "[":
                        depth += 1
                    elif text[i] == "]":
                        depth -= 1
                        if depth == 0:
                            json_str = text[start : i + 1]
                            candidates = json.loads(json_str)
                            # Ensure all items are strings
                            return [str(c) for c in candidates if c]
        except (json.JSONDecodeError, ValueError):
            pass

        return []

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
        Run OPRO optimization loop.

        The optimization process:
        1. Build train/val sets from annotations
        2. Initialize trajectory with current prompt + score
        3. Iterative loop:
           a. Build meta-prompt with trajectory
           b. Generate N candidate prompts
           c. Evaluate each on training set
           d. Add best to trajectory
           e. Check early stopping
        4. Return best prompt found

        Args:
            metric_id: The metric being calibrated
            current_rubric: Fixed rubric criteria (NOT optimized)
            metric_results: Evaluation results from the judge
            annotations: Human annotations
            dataset_items: Optional dataset items for context
            current_preamble: Current preamble text (the part to optimize)

        Returns:
            PromptOptimizationResult with optimized preamble
        """
        # Build datasets
        trainset, valset = self._build_dataset_from_annotations(
            metric_results, annotations, dataset_items
        )

        if len(trainset) < 3 or len(valset) < 2:
            return PromptOptimizationResult(
                original_rubric=current_rubric,
                improved_rubric=current_rubric,
                improvement_reasoning="Not enough data for OPRO optimization (need at least 5 annotated examples)",
                suggested_additions=[],
                suggested_removals=[],
                estimated_improvement="unknown",
                original_preamble=current_preamble,
            )

        # Initialize with seed prompt if provided, otherwise use default
        if current_preamble:
            seed_preamble = current_preamble.strip()
        else:
            seed_preamble = f"You are an expert evaluator for the metric: {metric_id}. Carefully analyze the output quality and provide an honest assessment."

        # Evaluate seed prompt
        seed_f1, seed_acc = self._evaluate_prompt(seed_preamble, current_rubric, trainset)

        # Initialize trajectory
        trajectory: List[TrajectoryEntry] = [
            TrajectoryEntry(
                preamble=seed_preamble,
                f1_score=seed_f1,
                accuracy=seed_acc,
                iteration=0,
            )
        ]

        best_entry = trajectory[0]
        no_improvement_count = 0

        # Optimization loop
        for iteration in range(1, self.config.max_iterations + 1):
            # Generate candidates
            candidates = self._generate_candidates(
                metric_id, current_rubric, trajectory, trainset
            )

            if not candidates:
                # Failed to generate candidates, continue with next iteration
                no_improvement_count += 1
                if no_improvement_count >= self.config.early_stop_patience:
                    break
                continue

            # Evaluate candidates
            iteration_best: Optional[TrajectoryEntry] = None
            for candidate in candidates:
                f1, acc = self._evaluate_prompt(candidate, current_rubric, trainset)
                entry = TrajectoryEntry(
                    preamble=candidate,
                    f1_score=f1,
                    accuracy=acc,
                    iteration=iteration,
                )
                trajectory.append(entry)

                if iteration_best is None or f1 > iteration_best.f1_score:
                    iteration_best = entry

            # Check for improvement
            if iteration_best and iteration_best.f1_score > best_entry.f1_score:
                best_entry = iteration_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= self.config.early_stop_patience:
                break

        # Validate best prompt on validation set
        val_f1, val_acc = self._evaluate_prompt(
            best_entry.preamble, current_rubric, valset
        )
        seed_val_f1, seed_val_acc = self._evaluate_prompt(
            seed_preamble, current_rubric, valset
        )

        # Determine improvement
        improvement_delta = val_f1 - seed_val_f1
        if improvement_delta > 0.05:
            estimated_improvement = "high"
        elif improvement_delta > 0.02:
            estimated_improvement = "medium"
        elif improvement_delta > 0:
            estimated_improvement = "low"
        else:
            estimated_improvement = "none"

        # Build full optimized prompt
        full_optimized_prompt = self._build_full_prompt(best_entry.preamble, current_rubric)

        # Build reasoning
        reasoning = (
            f"OPRO optimization completed. "
            f"Iterations: {best_entry.iteration}/{self.config.max_iterations}. "
            f"Train F1: {best_entry.f1_score:.3f} (seed: {seed_f1:.3f}). "
            f"Val F1: {val_f1:.3f} (seed: {seed_val_f1:.3f}). "
            f"Improvement: {improvement_delta:+.3f}"
        )

        return PromptOptimizationResult(
            original_rubric=current_rubric,
            improved_rubric=current_rubric,  # Rubric stays the same
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=current_preamble,
            optimized_preamble=best_entry.preamble,
            full_prompt=full_optimized_prompt,
        )


__all__ = ["OPROConfig", "OPROOptimizer", "TrajectoryEntry"]
