"""Judge ensemble for robust evaluation.

Combine multiple judges via majority vote, weighted average, or
cost-aware strategies. Flag items for human review when judges disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeVote:
    """A single judge's vote on an item."""

    judge_id: str
    score: float
    passed: bool
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "score": self.score,
            "passed": self.passed,
            "weight": self.weight,
        }


@dataclass
class EnsembleResult:
    """Result of combining multiple judge votes for one item."""

    item_id: str
    votes: list[JudgeVote]
    final_score: float
    final_passed: bool
    agreement_rate: float
    method: str
    flagged_for_review: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "votes": [v.as_dict() for v in self.votes],
            "final_score": self.final_score,
            "final_passed": self.final_passed,
            "agreement_rate": self.agreement_rate,
            "method": self.method,
            "flagged_for_review": self.flagged_for_review,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleResult:
        votes = [
            JudgeVote(
                judge_id=v["judge_id"],
                score=v["score"],
                passed=v["passed"],
                weight=v.get("weight", 1.0),
            )
            for v in data["votes"]
        ]
        return cls(
            item_id=data["item_id"],
            votes=votes,
            final_score=data["final_score"],
            final_passed=data["final_passed"],
            agreement_rate=data["agreement_rate"],
            method=data["method"],
            flagged_for_review=data.get("flagged_for_review", False),
        )


@dataclass
class EnsembleConfig:
    """Configuration for ensemble evaluation."""

    method: str = "majority_vote"
    disagreement_threshold: float = 0.3
    min_judges: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "disagreement_threshold": self.disagreement_threshold,
            "min_judges": self.min_judges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleConfig:
        return cls(
            method=data.get("method", "majority_vote"),
            disagreement_threshold=data.get("disagreement_threshold", 0.3),
            min_judges=data.get("min_judges", 3),
        )


@dataclass
class EnsembleReport:
    """Aggregated ensemble evaluation report."""

    results: list[EnsembleResult]
    total_items: int
    flagged_count: int
    avg_agreement: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_items": self.total_items,
            "flagged_count": self.flagged_count,
            "avg_agreement": self.avg_agreement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleReport:
        results = [EnsembleResult.from_dict(r) for r in data["results"]]
        return cls(
            results=results,
            total_items=data["total_items"],
            flagged_count=data["flagged_count"],
            avg_agreement=data["avg_agreement"],
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Ensemble Evaluation Report")
        lines.append("=" * 40)
        lines.append(f"  Total items: {self.total_items}")
        lines.append(f"  Flagged for review: {self.flagged_count}")
        lines.append(f"  Average agreement: {self.avg_agreement:.1%}")
        lines.append("-" * 40)
        for r in self.results:
            status = "PASS" if r.final_passed else "FAIL"
            flag = " [REVIEW]" if r.flagged_for_review else ""
            lines.append(
                f"  {r.item_id}: {r.final_score:.2f} [{status}]"
                f" agreement={r.agreement_rate:.1%}{flag}"
            )
        return "\n".join(lines)


def majority_vote(votes: list[JudgeVote]) -> tuple[bool, float]:
    """Determine pass/fail by majority and compute agreement rate.

    Returns (passed, agreement_rate).
    passed = majority of votes passed.
    agreement = max(pass_count, fail_count) / total.
    """
    if not votes:
        return False, 0.0

    pass_count = sum(1 for v in votes if v.passed)
    fail_count = len(votes) - pass_count
    passed = pass_count > fail_count
    agreement = max(pass_count, fail_count) / len(votes)
    return passed, agreement


def weighted_average(votes: list[JudgeVote]) -> float:
    """Compute weighted average of scores.

    Each vote's contribution is scaled by its weight.
    """
    if not votes:
        return 0.0

    total_weight = sum(v.weight for v in votes)
    if total_weight == 0:
        return 0.0

    return sum(v.score * v.weight for v in votes) / total_weight


def evaluate_ensemble(
    item_id: str,
    votes: list[JudgeVote],
    config: EnsembleConfig,
) -> EnsembleResult:
    """Apply the configured ensemble method to a set of votes.

    Flag for review if agreement < disagreement_threshold.
    """
    if not votes:
        return EnsembleResult(
            item_id=item_id,
            votes=[],
            final_score=0.0,
            final_passed=False,
            agreement_rate=0.0,
            method=config.method,
            flagged_for_review=True,
        )

    if config.method == "weighted_average":
        final_score = weighted_average(votes)
        final_passed = final_score >= 0.5
        # Agreement: fraction of judges on the majority side
        pass_count = sum(1 for v in votes if v.passed)
        fail_count = len(votes) - pass_count
        agreement_rate = max(pass_count, fail_count) / len(votes)
    elif config.method == "cost_aware":
        # Cost-aware: same as weighted average but uses weight as cost proxy
        final_score = weighted_average(votes)
        final_passed = final_score >= 0.5
        pass_count = sum(1 for v in votes if v.passed)
        fail_count = len(votes) - pass_count
        agreement_rate = max(pass_count, fail_count) / len(votes)
    else:
        # Default: majority_vote
        final_passed, agreement_rate = majority_vote(votes)
        final_score = sum(v.score for v in votes) / len(votes)

    flagged = agreement_rate < (1.0 - config.disagreement_threshold)
    return EnsembleResult(
        item_id=item_id,
        votes=list(votes),
        final_score=final_score,
        final_passed=final_passed,
        agreement_rate=agreement_rate,
        method=config.method,
        flagged_for_review=flagged,
    )


def build_ensemble_report(results: list[EnsembleResult]) -> EnsembleReport:
    """Aggregate ensemble results into a report."""
    total = len(results)
    flagged = sum(1 for r in results if r.flagged_for_review)
    if results:
        avg_agreement = sum(r.agreement_rate for r in results) / len(results)
    else:
        avg_agreement = 0.0

    return EnsembleReport(
        results=list(results),
        total_items=total,
        flagged_count=flagged,
        avg_agreement=avg_agreement,
    )
