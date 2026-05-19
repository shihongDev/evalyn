"""Evaluation result signing: cryptographic tamper detection.

Computes a SHA-256 hash of all MetricResults in a run. The hash is
stored in the run metadata. Later, verify_run() checks if results
were manually edited after evaluation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def compute_result_hash(metric_results: List) -> str:
    """Compute a deterministic hash of all metric results.

    The hash covers: metric_id, item_id, score, passed for each result,
    sorted by (metric_id, item_id) for determinism.

    Args:
        metric_results: List of MetricResult objects.

    Returns:
        SHA-256 hash string (32 chars).
    """
    # Build canonical representation
    entries = []
    for r in sorted(metric_results, key=lambda x: (x.metric_id, x.item_id)):
        entries.append({
            "metric_id": r.metric_id,
            "item_id": r.item_id,
            "score": r.score,
            "passed": r.passed,
        })

    content = json.dumps(entries, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def sign_run(run) -> str:
    """Sign an evaluation run by computing and storing the result hash.

    Stores the hash in run.summary["result_hash"].

    Args:
        run: EvalRun object.

    Returns:
        The computed hash string.
    """
    result_hash = compute_result_hash(run.metric_results)
    if not run.summary:
        run.summary = {}
    run.summary["result_hash"] = result_hash
    return result_hash


def verify_run(run) -> Dict[str, Any]:
    """Verify that a run's results haven't been tampered with.

    Recomputes the hash from current results and compares against
    the stored hash.

    Args:
        run: EvalRun object with result_hash in summary.

    Returns:
        Dict with "valid" (bool), "stored_hash", "computed_hash".
    """
    stored_hash = (run.summary or {}).get("result_hash")
    if not stored_hash:
        return {
            "valid": None,
            "reason": "No result_hash found in run summary (run was not signed)",
            "stored_hash": None,
            "computed_hash": None,
        }

    computed_hash = compute_result_hash(run.metric_results)
    is_valid = computed_hash == stored_hash

    return {
        "valid": is_valid,
        "reason": "Results match stored hash" if is_valid else "Results were modified after signing",
        "stored_hash": stored_hash,
        "computed_hash": computed_hash,
    }
