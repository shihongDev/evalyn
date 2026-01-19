"""
Clustering module for semantic grouping of evaluation failures and disagreements.

Supports two modes:
1. Misalignment clustering: Groups LLM judge vs human disagreements
2. Failure clustering: Groups failed items from eval runs by judge reasoning

Uses LLM-based clustering to group reasons by semantic similarity,
then provides visualization via scatter plots with UMAP dimensionality reduction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..annotation.calibration import DisagreementAnalysis, DisagreementCase
from ..utils.api_client import GeminiClient

if TYPE_CHECKING:
    from ..models import MetricResult, DatasetItem


@dataclass
class ReasonCluster:
    """A cluster of semantically similar disagreement reasons."""

    cluster_id: str
    label: str  # LLM-generated cluster name
    count: int
    reasons: List[str]
    representative_example: DisagreementCase
    disagreement_type: str  # "false_positive" or "false_negative"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "count": self.count,
            "reasons": self.reasons,
            "representative_example": self.representative_example.as_dict(),
            "disagreement_type": self.disagreement_type,
        }


@dataclass
class ClusteringResult:
    """Result of clustering misalignment cases."""

    false_positive_clusters: List[ReasonCluster] = field(default_factory=list)
    false_negative_clusters: List[ReasonCluster] = field(default_factory=list)
    total_cases: int = 0
    # Store embeddings and coordinates for visualization
    embeddings: Optional[List[List[float]]] = None
    coordinates_2d: Optional[List[List[float]]] = None
    case_labels: Optional[List[str]] = None  # Cluster label for each case
    case_types: Optional[List[str]] = None  # "false_positive" or "false_negative"
    case_reasons: Optional[List[str]] = None  # Original reason text
    case_ids: Optional[List[str]] = None  # call_id for each case

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "false_positive_clusters": [c.as_dict() for c in self.false_positive_clusters],
            "false_negative_clusters": [c.as_dict() for c in self.false_negative_clusters],
            "total_cases": self.total_cases,
        }
        if self.coordinates_2d:
            result["coordinates_2d"] = self.coordinates_2d
            result["case_labels"] = self.case_labels
            result["case_types"] = self.case_types
            result["case_reasons"] = self.case_reasons
            result["case_ids"] = self.case_ids
        return result


# --- Failure Clustering (for eval runs without human annotations) ---


@dataclass
class FailureCase:
    """A single failed metric result from an eval run."""

    call_id: str
    input_text: str
    output_text: str
    reason: str  # Judge's reason for failure
    score: float
    metric_id: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "input": self.input_text[:500],
            "output": self.output_text[:500],
            "reason": self.reason,
            "score": self.score,
            "metric_id": self.metric_id,
        }


@dataclass
class FailureCluster:
    """A cluster of semantically similar failure reasons."""

    cluster_id: str
    label: str  # LLM-generated cluster name
    count: int
    reasons: List[str]
    representative_example: FailureCase

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "count": self.count,
            "reasons": self.reasons,
            "representative_example": self.representative_example.as_dict(),
        }


@dataclass
class FailureClusteringResult:
    """Result of clustering failure cases from eval runs."""

    clusters: List[FailureCluster] = field(default_factory=list)
    total_cases: int = 0
    metric_id: str = ""
    # Visualization data
    coordinates_2d: Optional[List[List[float]]] = None
    case_labels: Optional[List[str]] = None
    case_reasons: Optional[List[str]] = None
    case_ids: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "clusters": [c.as_dict() for c in self.clusters],
            "total_cases": self.total_cases,
            "metric_id": self.metric_id,
        }
        if self.coordinates_2d:
            result["coordinates_2d"] = self.coordinates_2d
            result["case_labels"] = self.case_labels
            result["case_reasons"] = self.case_reasons
            result["case_ids"] = self.case_ids
        return result


class ReasonClusterer:
    """
    Clusters disagreement cases by semantic similarity of failure reasons.

    Uses LLM to group reasons into meaningful categories, then optionally
    computes embeddings for visualization.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the clusterer.

        Args:
            model: Gemini model for clustering
            api_key: Optional API key (defaults to env var)
            cache_dir: Directory for caching results (e.g., dataset/calibrations/)
        """
        self.model = model
        self._api_key = api_key
        self._client: Optional[GeminiClient] = None
        self.cache_dir = cache_dir

    @property
    def client(self) -> GeminiClient:
        """Lazy-initialized Gemini API client."""
        if self._client is None:
            self._client = GeminiClient(
                model=self.model,
                temperature=0.0,
                api_key=self._api_key,
                timeout=120,
            )
        return self._client

    def _compute_cache_key(self, disagreements: DisagreementAnalysis) -> str:
        """Compute a cache key based on disagreement content."""
        content = json.dumps(
            {
                "fp": [d.call_id for d in disagreements.false_positives],
                "fn": [d.call_id for d in disagreements.false_negatives],
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached JSON data if available."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"clustering_{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"clustering_{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Cache failures are not critical

    def _deserialize_cluster(
        self, cluster_data: Dict[str, Any], disagreement_type: str
    ) -> ReasonCluster:
        """Deserialize a single cluster from dict."""
        is_fp = disagreement_type == "false_positive"
        example = cluster_data["representative_example"]
        stub_case = DisagreementCase(
            call_id=example.get("call_id", ""),
            input_text=example.get("input", ""),
            output_text=example.get("output", ""),
            judge_passed=example.get("judge_passed", is_fp),
            judge_reason=example.get("judge_reason", ""),
            human_passed=example.get("human_passed", not is_fp),
            human_notes=example.get("human_notes", ""),
            disagreement_type=disagreement_type,
        )
        return ReasonCluster(
            cluster_id=cluster_data["cluster_id"],
            label=cluster_data["label"],
            count=cluster_data["count"],
            reasons=cluster_data["reasons"],
            representative_example=stub_case,
            disagreement_type=disagreement_type,
        )

    def _deserialize_result(self, data: Dict[str, Any]) -> ClusteringResult:
        """Deserialize clustering result from dict."""
        fp_clusters = [
            self._deserialize_cluster(c, "false_positive")
            for c in data.get("false_positive_clusters", [])
        ]
        fn_clusters = [
            self._deserialize_cluster(c, "false_negative")
            for c in data.get("false_negative_clusters", [])
        ]

        return ClusteringResult(
            false_positive_clusters=fp_clusters,
            false_negative_clusters=fn_clusters,
            total_cases=data.get("total_cases", 0),
            coordinates_2d=data.get("coordinates_2d"),
            case_labels=data.get("case_labels"),
            case_types=data.get("case_types"),
            case_reasons=data.get("case_reasons"),
            case_ids=data.get("case_ids"),
        )

    def cluster_reasons(
        self,
        disagreements: DisagreementAnalysis,
        compute_embeddings: bool = True,
    ) -> ClusteringResult:
        """
        Cluster disagreement cases by semantic similarity.

        Args:
            disagreements: DisagreementAnalysis from calibration
            compute_embeddings: Whether to compute embeddings for visualization

        Returns:
            ClusteringResult with clustered cases and optional visualization data
        """
        total = disagreements.total_disagreements

        # Skip if fewer than 3 cases
        if total < 3:
            return ClusteringResult(
                false_positive_clusters=[],
                false_negative_clusters=[],
                total_cases=total,
            )

        # Check cache
        cache_key = self._compute_cache_key(disagreements)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return self._deserialize_result(cached_data)

        # Cluster false positives and false negatives separately
        fp_clusters = self._cluster_cases(
            disagreements.false_positives, "false_positive"
        )
        fn_clusters = self._cluster_cases(
            disagreements.false_negatives, "false_negative"
        )

        result = ClusteringResult(
            false_positive_clusters=fp_clusters,
            false_negative_clusters=fn_clusters,
            total_cases=total,
        )

        # Compute embeddings for visualization if requested
        if compute_embeddings and total > 0:
            all_cases = disagreements.false_positives + disagreements.false_negatives
            self._add_visualization_data(result, all_cases, fp_clusters, fn_clusters)

        # Cache result
        self._save_to_cache(cache_key, result.as_dict())

        return result

    def cluster_failures(
        self,
        metric_results: List["MetricResult"],
        dataset_items: Optional[List["DatasetItem"]] = None,
        compute_embeddings: bool = True,
    ) -> FailureClusteringResult:
        """
        Cluster failed items from eval runs by semantic similarity of judge reasons.

        Args:
            metric_results: List of MetricResult objects (only failed ones are used)
            dataset_items: Optional list of DatasetItem for input/output context
            compute_embeddings: Whether to compute embeddings for visualization

        Returns:
            FailureClusteringResult with clustered failure cases
        """
        # Filter to failed results only
        failed_results = [r for r in metric_results if r.passed is False]

        if not failed_results:
            return FailureClusteringResult(clusters=[], total_cases=0)

        # Get metric_id from first result
        metric_id = failed_results[0].metric_id if failed_results else ""

        # Build item lookup for input/output context
        items_by_call: Dict[str, "DatasetItem"] = {}
        if dataset_items:
            for item in dataset_items:
                call_id = item.metadata.get("call_id", item.id)
                items_by_call[call_id] = item

        # Convert to FailureCase objects
        cases: List[FailureCase] = []
        for res in failed_results:
            # Extract reason from raw_judge or details
            reason = ""
            if res.raw_judge and isinstance(res.raw_judge, dict):
                reason = res.raw_judge.get("reason", "")
            elif res.details and isinstance(res.details, dict):
                reason = res.details.get("reason", "")

            # Get input/output from dataset item
            input_text = ""
            output_text = ""
            item = items_by_call.get(res.call_id)
            if item:
                input_text = json.dumps(item.input, default=str) if item.input else ""
                output_text = str(item.output) if item.output else ""

            cases.append(
                FailureCase(
                    call_id=res.call_id,
                    input_text=input_text,
                    output_text=output_text,
                    reason=reason or "No reason provided",
                    score=res.score or 0.0,
                    metric_id=res.metric_id,
                )
            )

        total = len(cases)

        # Skip clustering if fewer than 3 cases
        if total < 3:
            # Return each case as its own cluster
            clusters = [
                FailureCluster(
                    cluster_id=f"failure_{i}",
                    label=self._truncate_reason(case.reason),
                    count=1,
                    reasons=[case.reason],
                    representative_example=case,
                )
                for i, case in enumerate(cases)
            ]
            return FailureClusteringResult(
                clusters=clusters,
                total_cases=total,
                metric_id=metric_id,
            )

        # Check cache
        cache_key = self._compute_failure_cache_key(cases)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return self._deserialize_failure_result(cached_data)

        # Cluster the failure cases
        clusters = self._cluster_failure_cases(cases)

        result = FailureClusteringResult(
            clusters=clusters,
            total_cases=total,
            metric_id=metric_id,
        )

        # Compute embeddings for visualization if requested
        if compute_embeddings and total > 0:
            self._add_failure_visualization_data(result, cases, clusters)

        # Cache result
        self._save_to_cache(cache_key, result.as_dict())

        return result

    def _compute_failure_cache_key(self, cases: List[FailureCase]) -> str:
        """Compute cache key for failure clustering."""
        content = json.dumps(
            {"failures": [c.call_id for c in cases]},
            sort_keys=True,
        )
        return "fail_" + hashlib.sha256(content.encode()).hexdigest()[:16]

    def _deserialize_failure_result(self, data: Dict[str, Any]) -> FailureClusteringResult:
        """Deserialize failure clustering result from dict."""
        clusters = []
        for c in data.get("clusters", []):
            example = c["representative_example"]
            stub_case = FailureCase(
                call_id=example.get("call_id", ""),
                input_text=example.get("input", ""),
                output_text=example.get("output", ""),
                reason=example.get("reason", ""),
                score=example.get("score", 0.0),
                metric_id=example.get("metric_id", ""),
            )
            clusters.append(
                FailureCluster(
                    cluster_id=c["cluster_id"],
                    label=c["label"],
                    count=c["count"],
                    reasons=c["reasons"],
                    representative_example=stub_case,
                )
            )

        return FailureClusteringResult(
            clusters=clusters,
            total_cases=data.get("total_cases", 0),
            metric_id=data.get("metric_id", ""),
            coordinates_2d=data.get("coordinates_2d"),
            case_labels=data.get("case_labels"),
            case_reasons=data.get("case_reasons"),
            case_ids=data.get("case_ids"),
        )

    def _truncate_reason(self, reason: str, max_len: int = 50) -> str:
        """Truncate a reason string at word boundary."""
        if not reason:
            return "Unknown"
        if len(reason) <= max_len:
            return reason
        truncated = reason[:max_len].rsplit(" ", 1)[0]
        return truncated + "..."

    def _cluster_failure_cases(self, cases: List[FailureCase]) -> List[FailureCluster]:
        """Use LLM to cluster failure cases."""
        if not cases:
            return []

        # Build prompt for LLM clustering
        prompt = self._build_failure_clustering_prompt(cases)

        try:
            response = self.client.generate(prompt)
            clusters = self._parse_failure_clustering_response(response, cases)
            return clusters
        except Exception:
            # Fallback: single cluster with all cases
            reasons = [c.reason for c in cases]
            return [
                FailureCluster(
                    cluster_id="failure_0",
                    label=f"All failures ({len(cases)} cases)",
                    count=len(cases),
                    reasons=reasons,
                    representative_example=cases[0],
                )
            ]

    def _build_failure_clustering_prompt(self, cases: List[FailureCase]) -> str:
        """Build prompt for failure clustering."""
        case_texts = []
        for i, case in enumerate(cases):
            case_texts.append(f"Case {i}: {case.reason[:300]}")

        cases_str = "\n".join(case_texts)

        return f"""You are analyzing FAILED items from an LLM evaluation.

Group the following {len(cases)} failure cases into semantic clusters based on the underlying reason for failure. Each cluster should represent a common failure pattern.

CASES:
{cases_str}

Return a JSON array of clusters. Each cluster has:
- "label": A short descriptive name (3-6 words)
- "case_indices": Array of case numbers that belong to this cluster
- "summary": One sentence explaining why these cases are grouped together

Guidelines:
- Create 2-5 clusters (fewer if cases are very similar)
- Every case must be assigned to exactly one cluster
- Cluster by semantic similarity, not surface keywords

Return ONLY the JSON array, no other text.

Example format:
[
  {{"label": "Incomplete responses", "case_indices": [0, 2, 5], "summary": "Output was truncated or missing key information."}},
  {{"label": "Factual errors", "case_indices": [1, 3, 4], "summary": "Response contained incorrect facts."}}
]"""

    def _parse_failure_clustering_response(
        self,
        response: str,
        cases: List[FailureCase],
    ) -> List[FailureCluster]:
        """Parse LLM response into FailureCluster objects."""
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)

                clusters = []
                assigned = set()

                for i, cluster_data in enumerate(parsed):
                    label = cluster_data.get("label", f"Cluster {i+1}")
                    case_indices = cluster_data.get("case_indices", [])

                    valid_indices = [
                        idx for idx in case_indices
                        if isinstance(idx, int) and 0 <= idx < len(cases)
                    ]

                    if not valid_indices:
                        continue

                    valid_indices = [idx for idx in valid_indices if idx not in assigned]
                    if not valid_indices:
                        continue

                    assigned.update(valid_indices)

                    cluster_cases = [cases[idx] for idx in valid_indices]
                    reasons = [c.reason for c in cluster_cases]

                    clusters.append(
                        FailureCluster(
                            cluster_id=f"failure_{i}",
                            label=label,
                            count=len(cluster_cases),
                            reasons=reasons,
                            representative_example=cluster_cases[0],
                        )
                    )

                # Handle unassigned cases
                unassigned = [i for i in range(len(cases)) if i not in assigned]
                if unassigned:
                    unassigned_cases = [cases[idx] for idx in unassigned]
                    reasons = [c.reason for c in unassigned_cases]
                    clusters.append(
                        FailureCluster(
                            cluster_id=f"failure_{len(clusters)}",
                            label="Other",
                            count=len(unassigned_cases),
                            reasons=reasons,
                            representative_example=unassigned_cases[0],
                        )
                    )

                return clusters

        except Exception:
            pass

        # Fallback: single cluster
        reasons = [c.reason for c in cases]
        return [
            FailureCluster(
                cluster_id="failure_0",
                label="All failures",
                count=len(cases),
                reasons=reasons,
                representative_example=cases[0],
            )
        ]

    def _add_failure_visualization_data(
        self,
        result: FailureClusteringResult,
        cases: List[FailureCase],
        clusters: List[FailureCluster],
    ) -> None:
        """Add embeddings and 2D coordinates for failure visualization."""
        try:
            from sentence_transformers import SentenceTransformer
            import umap
        except ImportError:
            return

        # Build mapping from case call_id to cluster label
        case_to_label: Dict[str, str] = {}
        for cluster in clusters:
            for case in cases:
                if case.reason in cluster.reasons:
                    case_to_label[case.call_id] = cluster.label

        # Get reasons for embedding
        reasons = []
        labels = []
        ids = []
        for case in cases:
            reasons.append(case.reason or "No reason")
            labels.append(case_to_label.get(case.call_id, "Unknown"))
            ids.append(case.call_id)

        if not reasons:
            return

        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(reasons)

            if len(reasons) >= 3:
                n_neighbors = min(15, len(reasons) - 1)
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
                coords_2d = reducer.fit_transform(embeddings)
            else:
                coords_2d = embeddings[:, :2]

            result.coordinates_2d = coords_2d.tolist()
            result.case_labels = labels
            result.case_reasons = reasons
            result.case_ids = ids

        except Exception:
            pass

    def _cluster_cases(
        self,
        cases: List[DisagreementCase],
        disagreement_type: str,
    ) -> List[ReasonCluster]:
        """Use LLM to cluster a list of disagreement cases."""
        if not cases:
            return []

        # For very small sets, each case is its own cluster
        if len(cases) <= 2:
            return [
                ReasonCluster(
                    cluster_id=f"{disagreement_type}_{i}",
                    label=self._truncate_reason(case.judge_reason or case.human_notes),
                    count=1,
                    reasons=[case.judge_reason or case.human_notes],
                    representative_example=case,
                    disagreement_type=disagreement_type,
                )
                for i, case in enumerate(cases)
            ]

        # Build prompt for LLM clustering
        prompt = self._build_clustering_prompt(cases, disagreement_type)

        try:
            response = self.client.generate(prompt)
            clusters = self._parse_clustering_response(response, cases, disagreement_type)
            return clusters
        except Exception:
            # Fallback: single cluster with all cases
            reasons = [c.judge_reason or c.human_notes for c in cases]
            return [
                ReasonCluster(
                    cluster_id=f"{disagreement_type}_0",
                    label=f"Unclustered ({len(cases)} cases)",
                    count=len(cases),
                    reasons=reasons,
                    representative_example=cases[0],
                    disagreement_type=disagreement_type,
                )
            ]


    def _build_clustering_prompt(
        self,
        cases: List[DisagreementCase],
        disagreement_type: str,
    ) -> str:
        """Build prompt for LLM-based clustering."""
        type_desc = (
            "FALSE POSITIVES (judge said PASS but human said FAIL)"
            if disagreement_type == "false_positive"
            else "FALSE NEGATIVES (judge said FAIL but human said PASS)"
        )

        # Format cases for the prompt
        case_texts = []
        for i, case in enumerate(cases):
            reason = case.judge_reason if case.judge_reason else case.human_notes
            case_texts.append(f"Case {i}: {reason[:300]}")

        cases_str = "\n".join(case_texts)

        return f"""You are analyzing {type_desc} in an LLM evaluation.

Group the following {len(cases)} cases into semantic clusters based on the underlying reason for disagreement. Each cluster should represent a common failure pattern.

CASES:
{cases_str}

Return a JSON array of clusters. Each cluster has:
- "label": A short descriptive name (3-6 words)
- "case_indices": Array of case numbers that belong to this cluster
- "summary": One sentence explaining why these cases are grouped together

Guidelines:
- Create 2-5 clusters (fewer if cases are very similar)
- Every case must be assigned to exactly one cluster
- Cluster by semantic similarity, not surface keywords

Return ONLY the JSON array, no other text.

Example format:
[
  {{"label": "Missing context information", "case_indices": [0, 2, 5], "summary": "Judge missed important context."}},
  {{"label": "Factual errors overlooked", "case_indices": [1, 3, 4], "summary": "Judge failed to catch factual mistakes."}}
]"""

    def _parse_clustering_response(
        self,
        response: str,
        cases: List[DisagreementCase],
        disagreement_type: str,
    ) -> List[ReasonCluster]:
        """Parse LLM response into ReasonCluster objects."""
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            # Find JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)

                clusters = []
                assigned = set()

                for i, cluster_data in enumerate(parsed):
                    label = cluster_data.get("label", f"Cluster {i+1}")
                    case_indices = cluster_data.get("case_indices", [])

                    # Validate indices
                    valid_indices = [
                        idx for idx in case_indices
                        if isinstance(idx, int) and 0 <= idx < len(cases)
                    ]

                    if not valid_indices:
                        continue

                    # Avoid double-assignment
                    valid_indices = [idx for idx in valid_indices if idx not in assigned]
                    if not valid_indices:
                        continue

                    assigned.update(valid_indices)

                    cluster_cases = [cases[idx] for idx in valid_indices]
                    reasons = [c.judge_reason or c.human_notes for c in cluster_cases]

                    clusters.append(
                        ReasonCluster(
                            cluster_id=f"{disagreement_type}_{i}",
                            label=label,
                            count=len(cluster_cases),
                            reasons=reasons,
                            representative_example=cluster_cases[0],
                            disagreement_type=disagreement_type,
                        )
                    )

                # Handle unassigned cases
                unassigned = [i for i in range(len(cases)) if i not in assigned]
                if unassigned:
                    unassigned_cases = [cases[idx] for idx in unassigned]
                    reasons = [c.judge_reason or c.human_notes for c in unassigned_cases]
                    clusters.append(
                        ReasonCluster(
                            cluster_id=f"{disagreement_type}_{len(clusters)}",
                            label="Other",
                            count=len(unassigned_cases),
                            reasons=reasons,
                            representative_example=unassigned_cases[0],
                            disagreement_type=disagreement_type,
                        )
                    )

                return clusters

        except Exception:
            pass

        # Fallback: single cluster
        reasons = [c.judge_reason or c.human_notes for c in cases]
        return [
            ReasonCluster(
                cluster_id=f"{disagreement_type}_0",
                label="All cases",
                count=len(cases),
                reasons=reasons,
                representative_example=cases[0],
                disagreement_type=disagreement_type,
            )
        ]

    def _add_visualization_data(
        self,
        result: ClusteringResult,
        all_cases: List[DisagreementCase],
        fp_clusters: List[ReasonCluster],
        fn_clusters: List[ReasonCluster],
    ) -> None:
        """Add embeddings and 2D coordinates for visualization."""
        try:
            # Import optional dependencies
            from sentence_transformers import SentenceTransformer
            import umap
        except ImportError:
            # Optional dependencies not available
            return

        # Build mapping from case call_id to cluster label
        case_to_label: Dict[str, str] = {}
        for cluster in fp_clusters + fn_clusters:
            for i, case in enumerate(
                [c for c in all_cases if c.disagreement_type == cluster.disagreement_type]
            ):
                if (case.judge_reason or case.human_notes) in cluster.reasons:
                    case_to_label[case.call_id] = cluster.label

        # Get reasons for embedding
        reasons = []
        labels = []
        types = []
        ids = []
        for case in all_cases:
            reason = case.judge_reason or case.human_notes or "No reason"
            reasons.append(reason)
            labels.append(case_to_label.get(case.call_id, "Unknown"))
            types.append(case.disagreement_type)
            ids.append(case.call_id)

        if not reasons:
            return

        try:
            # Compute embeddings
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(reasons)

            # Reduce to 2D with UMAP
            if len(reasons) >= 3:
                n_neighbors = min(15, len(reasons) - 1)
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
                coords_2d = reducer.fit_transform(embeddings)
            else:
                # Not enough for UMAP, use first 2 dimensions of embedding
                coords_2d = embeddings[:, :2]

            result.embeddings = embeddings.tolist()
            result.coordinates_2d = coords_2d.tolist()
            result.case_labels = labels
            result.case_types = types
            result.case_reasons = reasons
            result.case_ids = ids

        except Exception:
            # Embedding/UMAP failed, continue without visualization data
            pass


def generate_cluster_html(
    result: ClusteringResult,
    metric_id: str,
) -> str:
    """Generate an interactive HTML scatter plot for clustering visualization.

    Uses Plotly for interactive visualization with hover details.

    Args:
        result: ClusteringResult with 2D coordinates
        metric_id: Name of the metric being analyzed

    Returns:
        HTML string with embedded Plotly scatter plot
    """
    if not result.coordinates_2d:
        return _generate_fallback_html(result, metric_id)

    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return _generate_fallback_html(result, metric_id)

    # Build scatter plot data
    x_coords = [c[0] for c in result.coordinates_2d]
    y_coords = [c[1] for c in result.coordinates_2d]

    # Color by cluster, shape by type
    # Assign colors to unique labels
    unique_labels = sorted(set(result.case_labels or []))
    color_map = {}
    fp_colors = ["#ef4444", "#f87171", "#fca5a5", "#fecaca", "#fee2e2"]  # Red tones
    fn_colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"]  # Blue tones

    fp_idx = 0
    fn_idx = 0
    for label in unique_labels:
        # Find a case with this label to determine type
        for i, l in enumerate(result.case_labels or []):
            if l == label and result.case_types:
                if result.case_types[i] == "false_positive":
                    color_map[label] = fp_colors[fp_idx % len(fp_colors)]
                    fp_idx += 1
                else:
                    color_map[label] = fn_colors[fn_idx % len(fn_colors)]
                    fn_idx += 1
                break

    colors = [color_map.get(l, "#888888") for l in (result.case_labels or [])]
    symbols = [
        "circle" if t == "false_positive" else "diamond"
        for t in (result.case_types or [])
    ]

    # Build hover text
    hover_texts = []
    for i in range(len(x_coords)):
        label = result.case_labels[i] if result.case_labels else "Unknown"
        case_type = result.case_types[i] if result.case_types else "unknown"
        reason = result.case_reasons[i] if result.case_reasons else ""
        # Truncate reason for hover
        reason_short = reason[:150] + "..." if len(reason) > 150 else reason
        type_label = "FP (too lenient)" if case_type == "false_positive" else "FN (too strict)"
        hover_texts.append(
            f"<b>{label}</b><br>"
            f"Type: {type_label}<br>"
            f"Reason: {reason_short}"
        )

    # Create figure
    fig = go.Figure()

    # Add FP trace (circles)
    fp_mask = [t == "false_positive" for t in (result.case_types or [])]
    if any(fp_mask):
        fig.add_trace(
            go.Scatter(
                x=[x for x, m in zip(x_coords, fp_mask) if m],
                y=[y for y, m in zip(y_coords, fp_mask) if m],
                mode="markers",
                marker=dict(
                    size=12,
                    color=[c for c, m in zip(colors, fp_mask) if m],
                    symbol="circle",
                    line=dict(width=1, color="#0a1210"),
                ),
                text=[t for t, m in zip(hover_texts, fp_mask) if m],
                hoverinfo="text",
                name="False Positive (too lenient)",
            )
        )

    # Add FN trace (diamonds)
    fn_mask = [t == "false_negative" for t in (result.case_types or [])]
    if any(fn_mask):
        fig.add_trace(
            go.Scatter(
                x=[x for x, m in zip(x_coords, fn_mask) if m],
                y=[y for y, m in zip(y_coords, fn_mask) if m],
                mode="markers",
                marker=dict(
                    size=12,
                    color=[c for c, m in zip(colors, fn_mask) if m],
                    symbol="diamond",
                    line=dict(width=1, color="#0a1210"),
                ),
                text=[t for t, m in zip(hover_texts, fn_mask) if m],
                hoverinfo="text",
                name="False Negative (too strict)",
            )
        )

    # Layout with dark theme
    fig.update_layout(
        title=dict(
            text=f"Misalignment Clusters: {metric_id}",
            font=dict(size=16, color="#e5e7eb"),
        ),
        paper_bgcolor="#0a1210",
        plot_bgcolor="#0f1a16",
        font=dict(color="#e5e7eb"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1f2d28",
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1f2d28",
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        legend=dict(
            bgcolor="#0f1a16",
            bordercolor="#1f2d28",
            borderwidth=1,
            font=dict(color="#e5e7eb"),
        ),
        hovermode="closest",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Generate HTML
    plot_html = to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False},
    )

    # Build summary stats
    fp_count = len(result.false_positive_clusters)
    fn_count = len(result.false_negative_clusters)
    fp_cases = sum(c.count for c in result.false_positive_clusters)
    fn_cases = sum(c.count for c in result.false_negative_clusters)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Misalignment Clusters - {metric_id}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a1210;
            color: #e5e7eb;
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ margin-bottom: 24px; }}
        .header h1 {{ margin: 0; font-size: 24px; color: #39ff14; }}
        .stats {{
            display: flex;
            gap: 32px;
            margin: 16px 0;
            padding: 16px;
            background: #0f1a16;
            border: 1px solid #1f2d28;
        }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: 700; }}
        .stat-value.fp {{ color: #ef4444; }}
        .stat-value.fn {{ color: #3b82f6; }}
        .stat-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
        .plot-container {{ margin: 24px 0; }}
        .cluster-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 24px;
        }}
        .cluster-table th, .cluster-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #1f2d28;
        }}
        .cluster-table th {{
            background: #0f1a16;
            font-size: 11px;
            text-transform: uppercase;
            color: #6b7280;
        }}
        .cluster-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .cluster-badge.fp {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
        .cluster-badge.fn {{ background: rgba(59, 130, 246, 0.2); color: #3b82f6; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Misalignment Clusters: {metric_id}</h1>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{result.total_cases}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat">
                <div class="stat-value fp">{fp_count}</div>
                <div class="stat-label">FP Clusters ({fp_cases} cases)</div>
            </div>
            <div class="stat">
                <div class="stat-value fn">{fn_count}</div>
                <div class="stat-label">FN Clusters ({fn_cases} cases)</div>
            </div>
        </div>
        <div class="plot-container">
            {plot_html}
        </div>
        {_render_cluster_tables(result)}
    </div>
</body>
</html>"""


def _generate_fallback_html(result: ClusteringResult, metric_id: str) -> str:
    """Generate simple HTML without interactive plot."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Misalignment Clusters - {metric_id}</title>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background: #0a1210;
            color: #e5e7eb;
            padding: 24px;
        }}
        h1 {{ color: #39ff14; }}
        .note {{ color: #6b7280; margin: 16px 0; }}
    </style>
</head>
<body>
    <h1>Misalignment Clusters: {metric_id}</h1>
    <p class="note">Interactive scatter plot requires: pip install evalyn-sdk[clustering]</p>
    {_render_cluster_tables(result)}
</body>
</html>"""


def _render_single_cluster_table(
    clusters: List[ReasonCluster], title: str, color: str, badge_class: str
) -> str:
    """Render a single cluster table section."""
    if not clusters:
        return ""

    rows = []
    for c in clusters:
        example = c.representative_example.judge_reason or c.representative_example.human_notes
        example_short = example[:100] + "..." if len(example) > 100 else example
        rows.append(f"""
            <tr>
                <td><span class="cluster-badge {badge_class}">{c.label}</span></td>
                <td>{c.count}</td>
                <td style="color: #9ca3af;">{example_short}</td>
            </tr>
        """)

    return f"""
        <h2 style="color: {color}; margin-top: 32px;">{title}</h2>
        <table class="cluster-table">
            <thead><tr><th>Cluster</th><th>Count</th><th>Example</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    """


def _render_cluster_tables(result: ClusteringResult) -> str:
    """Render HTML tables for cluster details."""
    fp_table = _render_single_cluster_table(
        result.false_positive_clusters,
        "False Positives (Judge too lenient)",
        "#ef4444",
        "fp",
    )
    fn_table = _render_single_cluster_table(
        result.false_negative_clusters,
        "False Negatives (Judge too strict)",
        "#3b82f6",
        "fn",
    )
    return fp_table + fn_table


def generate_cluster_text(result: ClusteringResult, metric_id: str) -> str:
    """Generate ASCII table output for CLI.

    Args:
        result: ClusteringResult with clusters
        metric_id: Name of the metric

    Returns:
        Formatted text output
    """
    lines = [
        f"\nMISALIGNMENT CLUSTERS: {metric_id}",
        "=" * 60,
        "",
    ]

    if result.total_cases == 0:
        lines.append("No disagreement cases found.")
        return "\n".join(lines)

    if result.total_cases < 3:
        lines.append(f"Only {result.total_cases} disagreement cases - clustering skipped.")
        lines.append("")
        for case in result.false_positive_clusters + result.false_negative_clusters:
            type_label = "FP" if case.disagreement_type == "false_positive" else "FN"
            reason = case.representative_example.judge_reason or case.representative_example.human_notes
            lines.append(f"  [{type_label}] {reason[:70]}...")
        return "\n".join(lines)

    # False positives
    fp_cases = sum(c.count for c in result.false_positive_clusters)
    if result.false_positive_clusters:
        lines.append(
            f"FALSE POSITIVES (Judge too lenient) - {len(result.false_positive_clusters)} clusters, {fp_cases} cases"
        )
        lines.append("-" * 60)
        lines.append(f"{'Cluster':<35} {'Count':<8} Example")
        lines.append("-" * 60)
        for c in result.false_positive_clusters:
            example = c.representative_example.judge_reason or c.representative_example.human_notes
            example_short = example[:30] + "..." if len(example) > 30 else example
            lines.append(f"{c.label:<35} {c.count:<8} {example_short}")
        lines.append("")

    # False negatives
    fn_cases = sum(c.count for c in result.false_negative_clusters)
    if result.false_negative_clusters:
        lines.append(
            f"FALSE NEGATIVES (Judge too strict) - {len(result.false_negative_clusters)} clusters, {fn_cases} cases"
        )
        lines.append("-" * 60)
        lines.append(f"{'Cluster':<35} {'Count':<8} Example")
        lines.append("-" * 60)
        for c in result.false_negative_clusters:
            example = c.representative_example.judge_reason or c.representative_example.human_notes
            example_short = example[:30] + "..." if len(example) > 30 else example
            lines.append(f"{c.label:<35} {c.count:<8} {example_short}")

    return "\n".join(lines)


# --- Failure Clustering Visualization ---


def generate_failure_cluster_html(
    result: FailureClusteringResult,
    metric_id: str,
) -> str:
    """Generate an interactive HTML scatter plot for failure clustering.

    Args:
        result: FailureClusteringResult with clusters
        metric_id: Name of the metric

    Returns:
        HTML string with embedded Plotly scatter plot
    """
    if not result.coordinates_2d:
        return _generate_failure_fallback_html(result, metric_id)

    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return _generate_failure_fallback_html(result, metric_id)

    x_coords = [c[0] for c in result.coordinates_2d]
    y_coords = [c[1] for c in result.coordinates_2d]

    # Assign colors to clusters
    unique_labels = sorted(set(result.case_labels or []))
    colors_palette = [
        "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
        "#3b82f6", "#8b5cf6", "#ec4899", "#6b7280", "#78716c",
    ]
    color_map = {label: colors_palette[i % len(colors_palette)] for i, label in enumerate(unique_labels)}
    colors = [color_map.get(l, "#888888") for l in (result.case_labels or [])]

    # Build hover text
    hover_texts = []
    for i in range(len(x_coords)):
        label = result.case_labels[i] if result.case_labels else "Unknown"
        reason = result.case_reasons[i] if result.case_reasons else ""
        reason_short = reason[:150] + "..." if len(reason) > 150 else reason
        hover_texts.append(f"<b>{label}</b><br>Reason: {reason_short}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                size=12,
                color=colors,
                symbol="circle",
                line=dict(width=1, color="#0a1210"),
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Failures",
        )
    )

    fig.update_layout(
        title=dict(text=f"Failure Clusters: {metric_id}", font=dict(size=16, color="#e5e7eb")),
        paper_bgcolor="#0a1210",
        plot_bgcolor="#0f1a16",
        font=dict(color="#e5e7eb"),
        xaxis=dict(showgrid=True, gridcolor="#1f2d28", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2d28", zeroline=False, showticklabels=False),
        legend=dict(bgcolor="#0f1a16", bordercolor="#1f2d28", borderwidth=1),
        hovermode="closest",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    plot_html = to_html(fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})

    cluster_count = len(result.clusters)
    total_cases = result.total_cases

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Failure Clusters - {metric_id}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a1210;
            color: #e5e7eb;
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ margin-bottom: 24px; }}
        .header h1 {{ margin: 0; font-size: 24px; color: #ef4444; }}
        .stats {{
            display: flex;
            gap: 32px;
            margin: 16px 0;
            padding: 16px;
            background: #0f1a16;
            border: 1px solid #1f2d28;
        }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: 700; color: #ef4444; }}
        .stat-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
        .plot-container {{ margin: 24px 0; }}
        .cluster-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 24px;
        }}
        .cluster-table th, .cluster-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #1f2d28;
        }}
        .cluster-table th {{
            background: #0f1a16;
            font-size: 11px;
            text-transform: uppercase;
            color: #6b7280;
        }}
        .cluster-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Failure Clusters: {metric_id}</h1>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{total_cases}</div>
                <div class="stat-label">Failed Items</div>
            </div>
            <div class="stat">
                <div class="stat-value">{cluster_count}</div>
                <div class="stat-label">Clusters</div>
            </div>
        </div>
        <div class="plot-container">
            {plot_html}
        </div>
        {_render_failure_cluster_table(result)}
    </div>
</body>
</html>"""


def _generate_failure_fallback_html(result: FailureClusteringResult, metric_id: str) -> str:
    """Generate simple HTML for failures without interactive plot."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Failure Clusters - {metric_id}</title>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background: #0a1210;
            color: #e5e7eb;
            padding: 24px;
        }}
        h1 {{ color: #ef4444; }}
        .note {{ color: #6b7280; margin: 16px 0; }}
    </style>
</head>
<body>
    <h1>Failure Clusters: {metric_id}</h1>
    <p class="note">Interactive scatter plot requires: pip install evalyn-sdk[clustering]</p>
    {_render_failure_cluster_table(result)}
</body>
</html>"""


def _render_failure_cluster_table(result: FailureClusteringResult) -> str:
    """Render HTML table for failure clusters."""
    if not result.clusters:
        return "<p>No failure clusters found.</p>"

    rows = []
    for c in result.clusters:
        example = c.representative_example.reason
        example_short = example[:100] + "..." if len(example) > 100 else example
        rows.append(f"""
            <tr>
                <td><span class="cluster-badge">{c.label}</span></td>
                <td>{c.count}</td>
                <td style="color: #9ca3af;">{example_short}</td>
            </tr>
        """)

    return f"""
        <h2 style="color: #ef4444; margin-top: 32px;">Failure Patterns</h2>
        <table class="cluster-table">
            <thead><tr><th>Cluster</th><th>Count</th><th>Example</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    """


def generate_failure_cluster_text(result: FailureClusteringResult, metric_id: str) -> str:
    """Generate ASCII table output for failure clustering.

    Args:
        result: FailureClusteringResult with clusters
        metric_id: Name of the metric

    Returns:
        Formatted text output
    """
    lines = [
        f"\nFAILURE CLUSTERS: {metric_id}",
        "=" * 60,
        "",
    ]

    if result.total_cases == 0:
        lines.append("No failed items found.")
        return "\n".join(lines)

    if result.total_cases < 3:
        lines.append(f"Only {result.total_cases} failures - clustering skipped.")
        lines.append("")
        for cluster in result.clusters:
            reason = cluster.representative_example.reason
            lines.append(f"  {reason[:70]}...")
        return "\n".join(lines)

    lines.append(f"FAILURES - {len(result.clusters)} clusters, {result.total_cases} cases")
    lines.append("-" * 60)
    lines.append(f"{'Cluster':<35} {'Count':<8} Example")
    lines.append("-" * 60)

    for c in result.clusters:
        example = c.representative_example.reason
        example_short = example[:30] + "..." if len(example) > 30 else example
        lines.append(f"{c.label:<35} {c.count:<8} {example_short}")

    return "\n".join(lines)


__all__ = [
    "ReasonCluster",
    "ClusteringResult",
    "ReasonClusterer",
    "generate_cluster_html",
    "generate_cluster_text",
    # Failure clustering
    "FailureCase",
    "FailureCluster",
    "FailureClusteringResult",
    "generate_failure_cluster_html",
    "generate_failure_cluster_text",
]
