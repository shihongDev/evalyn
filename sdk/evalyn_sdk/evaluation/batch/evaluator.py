"""Batch evaluator for large-scale LLM judge evaluation.

Provides batch processing for subjective metrics using provider batch APIs
for 50% cost savings compared to real-time evaluation.

Usage:
    from evalyn_sdk.evaluation.batch import BatchEvaluator

    evaluator = BatchEvaluator(provider="gemini")
    results = evaluator.evaluate(
        items=dataset_items,
        calls=function_calls,
        metrics=subjective_metrics,
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...models import DatasetItem, FunctionCall, Metric, MetricResult
from .providers import create_batch_provider

logger = logging.getLogger(__name__)


@dataclass
class BatchEvalProgress:
    """Progress information for batch evaluation."""

    phase: str  # preparing, submitted, waiting, parsing, complete
    total_requests: int
    completed_requests: int
    elapsed_seconds: float
    job_id: Optional[str] = None
    eta_seconds: Optional[float] = None


ProgressCallback = Callable[[BatchEvalProgress], None]


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text, handling markdown code blocks."""
    text = (text or "").strip()
    if not text:
        return None

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Try to find JSON object by brace matching
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None


def _parse_passed(value: Any) -> Optional[bool]:
    """Parse various representations of pass/fail to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"pass", "passed", "true", "yes", "1"}:
            return True
        if low in {"fail", "failed", "false", "no", "0"}:
            return False
    return None


def _safe_trace_excerpt(
    call: FunctionCall, max_events: int = 20, max_chars: int = 2000
) -> str:
    """Create a safe string excerpt of trace events for the judge prompt."""
    events = call.trace or []
    lines = []
    for ev in events[:max_events]:
        detail = ev.detail or {}
        try:
            detail_str = json.dumps(detail, default=str)
        except Exception:
            detail_str = str(detail)
        lines.append(f"- {ev.kind}: {detail_str[:300]}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


class BatchEvaluator:
    """Evaluates subjective metrics using batch APIs for cost savings.

    Batch evaluation offers:
    - 50% cost reduction compared to real-time API calls
    - Higher rate limits (no throttling)
    - Async processing (minutes to hours)

    Best for:
    - Large datasets (100+ items)
    - Non-urgent evaluations
    - Cost-sensitive workloads
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
    ):
        """Initialize batch evaluator.

        Args:
            provider: Batch provider (gemini, openai, anthropic)
            model: Optional model override
            api_key: Optional API key override
            poll_interval: Seconds between status checks (default: 30)
            timeout: Maximum seconds to wait for completion (default: None)
        """
        self.provider = create_batch_provider(provider, model, api_key)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._model = model or self.provider.model

    def _build_prompt(
        self,
        metric: Metric,
        call: FunctionCall,
        item: DatasetItem,
    ) -> str:
        """Build evaluation prompt for a metric."""
        # Get prompt and rubric from metric spec
        config = metric.spec.config or {}
        prompt = config.get("prompt", metric.spec.description)
        rubric = config.get("rubric", [])

        # Build trace excerpt
        trace_excerpt = _safe_trace_excerpt(call)

        evaluation_input = {
            "input": item.input or item.inputs,
            "output": item.output or call.output,
            "human_label": item.human_label,
            "trace_excerpt": trace_excerpt if trace_excerpt else None,
        }

        # Include rubric if available
        rubric_text = ""
        if rubric:
            rubric_lines = "\n".join(f"- {r}" for r in rubric)
            rubric_text = f"\n\nRUBRIC:\n{rubric_lines}"

        full_prompt = f"""{prompt}{rubric_text}

EVALUATION_INPUT:
{json.dumps(evaluation_input, default=str, ensure_ascii=False, indent=2)}

Evaluate the OUTPUT given the INPUT. Return ONLY a JSON object with:
- "passed": boolean (true if criteria met, false otherwise)
- "reason": string (brief explanation)
- "score": number 0-1 (optional, defaults to 1 if passed, 0 if failed)
"""
        return full_prompt

    def _parse_response(
        self,
        response: str,
        metric: Metric,
        item: DatasetItem,
        call: FunctionCall,
    ) -> MetricResult:
        """Parse batch response into MetricResult."""
        parsed = _extract_json_object(response) or {}

        # Extract passed status
        passed = None
        for key in ("passed", "pass", "verdict"):
            if key in parsed:
                passed = _parse_passed(parsed.get(key))
                if passed is not None:
                    break

        # Extract score
        score = parsed.get("score")
        if isinstance(score, str):
            try:
                score = float(score)
            except Exception:
                score = None

        # Default score based on passed status
        if score is None and passed is not None:
            score = 1.0 if passed else 0.0

        # Clamp score to 0-1
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))

        return MetricResult(
            metric_id=metric.spec.id,
            item_id=item.id,
            call_id=call.id,
            score=score,
            passed=passed,
            details={"reason": parsed.get("reason")},
            raw_judge={"response": parsed, "text": response, "model": self._model},
        )

    def evaluate(
        self,
        prepared: List[Tuple[DatasetItem, FunctionCall]],
        metrics: List[Metric],
        progress_callback: Optional[ProgressCallback] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> List[MetricResult]:
        """Run batch evaluation on prepared items.

        Args:
            prepared: List of (item, call) tuples to evaluate
            metrics: List of subjective metrics to evaluate
            progress_callback: Optional callback for progress updates
            checkpoint_path: Optional path to save/load checkpoint

        Returns:
            List of MetricResult for all evaluations
        """
        start_time = time.time()

        # Filter to subjective metrics only
        subjective_metrics = [m for m in metrics if m.spec.type == "subjective"]
        if not subjective_metrics:
            logger.info("No subjective metrics to evaluate in batch mode")
            return []

        # Phase 1: Prepare batch requests
        if progress_callback:
            progress_callback(
                BatchEvalProgress(
                    phase="preparing",
                    total_requests=len(prepared) * len(subjective_metrics),
                    completed_requests=0,
                    elapsed_seconds=0,
                )
            )

        requests = []
        request_map: Dict[str, Tuple[Metric, DatasetItem, FunctionCall]] = {}

        for item, call in prepared:
            for metric in subjective_metrics:
                custom_id = f"{item.id}_{metric.spec.id}"
                prompt = self._build_prompt(metric, call, item)
                requests.append(
                    {
                        "custom_id": custom_id,
                        "prompt": prompt,
                    }
                )
                request_map[custom_id] = (metric, item, call)

        if not requests:
            return []

        logger.info(f"Submitting {len(requests)} requests to batch API")

        # Phase 2: Submit batch
        if progress_callback:
            progress_callback(
                BatchEvalProgress(
                    phase="submitted",
                    total_requests=len(requests),
                    completed_requests=0,
                    elapsed_seconds=time.time() - start_time,
                )
            )

        job = self.provider.submit(requests, description="Evalyn batch evaluation")
        logger.info(f"Batch job submitted: {job.id}")

        # Save checkpoint with job ID
        if checkpoint_path:
            checkpoint_data = {
                "job_id": job.id,
                "provider": self.provider.provider_name,
                "request_map_keys": list(request_map.keys()),
                "submitted_at": datetime.now().isoformat(),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

        # Phase 3: Wait for completion
        def on_progress(j):
            if progress_callback:
                progress_callback(
                    BatchEvalProgress(
                        phase="waiting",
                        total_requests=j.total_requests,
                        completed_requests=j.completed_requests,
                        elapsed_seconds=time.time() - start_time,
                        job_id=j.id,
                    )
                )

        final_job = self.provider.wait(
            job.id,
            poll_interval=self.poll_interval,
            timeout=self.timeout,
            progress_callback=on_progress,
        )

        if not final_job.is_success():
            raise RuntimeError(f"Batch job failed with status: {final_job.status}")

        logger.info(
            f"Batch job completed: {final_job.completed_requests}/{final_job.total_requests} succeeded"
        )

        # Phase 4: Parse results
        if progress_callback:
            progress_callback(
                BatchEvalProgress(
                    phase="parsing",
                    total_requests=len(requests),
                    completed_requests=final_job.completed_requests,
                    elapsed_seconds=time.time() - start_time,
                    job_id=job.id,
                )
            )

        batch_results = self.provider.get_results(job.id)

        # Convert to MetricResults
        results: List[MetricResult] = []
        for br in batch_results:
            if br.custom_id not in request_map:
                logger.warning(f"Unknown custom_id in results: {br.custom_id}")
                continue

            metric, item, call = request_map[br.custom_id]

            if br.success and br.response:
                result = self._parse_response(br.response, metric, item, call)
            else:
                result = MetricResult(
                    metric_id=metric.spec.id,
                    item_id=item.id,
                    call_id=call.id,
                    score=None,
                    passed=None,
                    details={"error": br.error or "Unknown error"},
                )
            results.append(result)

        # Clean up checkpoint
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        # Final progress
        if progress_callback:
            progress_callback(
                BatchEvalProgress(
                    phase="complete",
                    total_requests=len(requests),
                    completed_requests=len(results),
                    elapsed_seconds=time.time() - start_time,
                    job_id=job.id,
                )
            )

        logger.info(
            f"Batch evaluation complete: {len(results)} results in {time.time() - start_time:.1f}s"
        )
        return results

    def resume(
        self,
        checkpoint_path: Path,
        request_map: Dict[str, Tuple[Metric, DatasetItem, FunctionCall]],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[MetricResult]:
        """Resume evaluation from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            request_map: Mapping of custom_id to (metric, item, call)
            progress_callback: Optional callback for progress updates

        Returns:
            List of MetricResult for all evaluations
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        job_id = checkpoint["job_id"]
        logger.info(f"Resuming batch job: {job_id}")

        start_time = time.time()

        # Wait for completion
        def on_progress(j):
            if progress_callback:
                progress_callback(
                    BatchEvalProgress(
                        phase="waiting",
                        total_requests=j.total_requests,
                        completed_requests=j.completed_requests,
                        elapsed_seconds=time.time() - start_time,
                        job_id=j.id,
                    )
                )

        final_job = self.provider.wait(
            job_id,
            poll_interval=self.poll_interval,
            timeout=self.timeout,
            progress_callback=on_progress,
        )

        if not final_job.is_success():
            raise RuntimeError(f"Batch job failed with status: {final_job.status}")

        # Parse results
        batch_results = self.provider.get_results(job_id)

        results: List[MetricResult] = []
        for br in batch_results:
            if br.custom_id not in request_map:
                continue

            metric, item, call = request_map[br.custom_id]

            if br.success and br.response:
                result = self._parse_response(br.response, metric, item, call)
            else:
                result = MetricResult(
                    metric_id=metric.spec.id,
                    item_id=item.id,
                    call_id=call.id,
                    score=None,
                    passed=None,
                    details={"error": br.error or "Unknown error"},
                )
            results.append(result)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return results
