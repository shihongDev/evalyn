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

from ..calibration import DisagreementAnalysis, DisagreementCase
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


# --- Shared HTML Styling Constants ---

_FONT_IMPORTS = """<link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">"""

_FONT_FAMILY = "DM Sans, -apple-system, BlinkMacSystemFont, system-ui, sans-serif"
_MONO_FONT = "'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace"

_BASE_STYLES = """* {{ box-sizing: border-box; }}
        body {{
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            background: #FFFBF7;
            color: #1A1A1A;
            margin: 0;
            padding: 48px 24px;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 40px;
            padding-bottom: 24px;
            border-bottom: 1px solid rgba(212, 162, 127, 0.3);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            font-weight: 700;
            color: #1A1A1A;
            letter-spacing: -0.02em;
        }}
        .header .subtitle {{
            font-size: 15px;
            color: #666;
        }}"""

_STATS_STYLES = """.stats {{
            display: flex;
            gap: 48px;
            margin: 32px 0;
        }}
        .stat {{
            display: flex;
            flex-direction: column;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: 700;
            line-height: 1;
        }}
        .stat-label {{
            font-size: 13px;
            color: #666;
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}"""

_NOTE_STYLES = """.note {{
            padding: 16px 20px;
            background: rgba(212, 162, 127, 0.1);
            border-left: 3px solid #D4A27F;
            border-radius: 0 8px 8px 0;
            color: #666;
            font-size: 14px;
            margin: 24px 0;
        }}
        .note code {{
            background: rgba(212, 162, 127, 0.15);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'DM Mono', 'SF Mono', 'Fira Code', Menlo, Monaco, monospace;
            font-size: 13px;
        }}"""

_PLOT_STYLES = """.plot-container {{
            margin: 40px 0;
            background: #FBF7F3;
            border-radius: 12px;
            padding: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.03);
        }}"""

_TABLE_STYLES = """.section-title {{
            font-size: 18px;
            font-weight: 600;
            margin: 40px 0 20px 0;
        }}
        .cluster-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .cluster-table th {{
            text-align: left;
            padding: 12px 16px;
            font-size: 11px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid rgba(212, 162, 127, 0.2);
        }}
        .cluster-table td {{
            padding: 16px;
            border-bottom: 1px solid rgba(212, 162, 127, 0.12);
            vertical-align: top;
        }}
        .cluster-table tr:last-child td {{
            border-bottom: none;
        }}
        .cluster-table tr:hover td {{
            background: rgba(212, 162, 127, 0.05);
        }}
        .cluster-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
        }}
        .cluster-count {{
            font-weight: 600;
        }}
        .cluster-example {{
            font-size: 14px;
            color: #555;
            line-height: 1.6;
        }}"""

_MISALIGNMENT_STYLES = """.stat-value.total {{ color: #1A1A1A; }}
        .stat-value.fp {{ color: #D4A27F; }}
        .stat-value.fn {{ color: #6B8E8E; }}
        .section-title.fp {{ color: #C4836A; }}
        .section-title.fn {{ color: #5F9EA0; }}
        .cluster-badge.fp {{
            background: rgba(212, 162, 127, 0.15);
            color: #8B6914;
        }}
        .cluster-badge.fn {{
            background: rgba(107, 142, 142, 0.15);
            color: #3D6B6B;
        }}
        .cluster-count.fp {{ color: #D4A27F; }}
        .cluster-count.fn {{ color: #6B8E8E; }}"""

_FAILURE_STYLES = """.stat-value {{ color: #D4A27F; }}
        .section-title {{ color: #1A1A1A; }}
        .cluster-badge {{
            background: rgba(212, 162, 127, 0.15);
            color: #8B6914;
        }}
        .cluster-count {{ color: #D4A27F; }}"""


def _build_html_page(
    title: str,
    metric_id: str,
    styles: str,
    body_content: str,
) -> str:
    """Build a complete HTML page with common structure."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}: {metric_id}</title>
    {_FONT_IMPORTS}
    <style>
        {styles}
    </style>
</head>
<body>
    <div class="container">
        {body_content}
    </div>
</body>
</html>"""


def _get_base_plotly_layout(show_legend: bool = False) -> dict:
    """Get common Plotly layout configuration."""
    layout = {
        "title": None,
        "paper_bgcolor": "#FFFBF7",
        "plot_bgcolor": "#FBF7F3",
        "font": {"family": _FONT_FAMILY, "color": "#1A1A1A"},
        "xaxis": {
            "showgrid": True,
            "gridcolor": "rgba(212, 162, 127, 0.15)",
            "zeroline": False,
            "showticklabels": False,
            "title": "",
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "rgba(212, 162, 127, 0.15)",
            "zeroline": False,
            "showticklabels": False,
            "title": "",
        },
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "#FFFBF7",
            "bordercolor": "#D4A27F",
            "font": {"family": _FONT_FAMILY, "color": "#1A1A1A"},
        },
    }
    if show_legend:
        layout["legend"] = {
            "bgcolor": "rgba(255, 251, 247, 0.9)",
            "bordercolor": "rgba(212, 162, 127, 0.3)",
            "borderwidth": 1,
            "font": {"family": _FONT_FAMILY, "color": "#1A1A1A", "size": 12},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        }
        layout["margin"] = {"l": 24, "r": 24, "t": 48, "b": 24}
    else:
        layout["showlegend"] = False
        layout["margin"] = {"l": 24, "r": 24, "t": 24, "b": 24}
    return layout


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
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
        deps_available = True
    except ImportError:
        deps_available = False

    if not result.coordinates_2d:
        if result.total_cases < 3:
            reason = "too_few_cases"
        else:
            reason = "deps_missing"
        return _generate_fallback_html(result, metric_id, reason)

    if not deps_available:
        return _generate_fallback_html(result, metric_id, "deps_missing")

    x_coords = [c[0] for c in result.coordinates_2d]
    y_coords = [c[1] for c in result.coordinates_2d]

    # Build color mapping
    unique_labels = sorted(set(result.case_labels or []))
    color_map = {}
    fp_colors = ["#D4A27F", "#C4836A", "#B87333", "#A0522D", "#CD853F"]
    fn_colors = ["#6B8E8E", "#7A9E7A", "#8FBC8F", "#5F9EA0", "#708090"]

    fp_idx = 0
    fn_idx = 0
    for label in unique_labels:
        for i, l in enumerate(result.case_labels or []):
            if l == label and result.case_types:
                if result.case_types[i] == "false_positive":
                    color_map[label] = fp_colors[fp_idx % len(fp_colors)]
                    fp_idx += 1
                else:
                    color_map[label] = fn_colors[fn_idx % len(fn_colors)]
                    fn_idx += 1
                break

    colors = [color_map.get(l, "#A0A0A0") for l in (result.case_labels or [])]

    # Build hover text
    hover_texts = []
    for i in range(len(x_coords)):
        label = result.case_labels[i] if result.case_labels else "Unknown"
        case_type = result.case_types[i] if result.case_types else "unknown"
        reason = result.case_reasons[i] if result.case_reasons else ""
        reason_short = reason[:150] + "..." if len(reason) > 150 else reason
        type_label = "False Positive" if case_type == "false_positive" else "False Negative"
        hover_texts.append(
            f"<b>{label}</b><br><br>"
            f"<span style='color:#888'>{type_label}</span><br><br>"
            f"{reason_short}"
        )

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
                    size=14,
                    color=[c for c, m in zip(colors, fp_mask) if m],
                    symbol="circle",
                    line=dict(width=2, color="#FFFBF7"),
                    opacity=0.9,
                ),
                text=[t for t, m in zip(hover_texts, fp_mask) if m],
                hoverinfo="text",
                name="False Positive",
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
                    size=14,
                    color=[c for c, m in zip(colors, fn_mask) if m],
                    symbol="diamond",
                    line=dict(width=2, color="#FFFBF7"),
                    opacity=0.9,
                ),
                text=[t for t, m in zip(hover_texts, fn_mask) if m],
                hoverinfo="text",
                name="False Negative",
            )
        )

    fig.update_layout(**_get_base_plotly_layout(show_legend=True))
    plot_html = to_html(
        fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )

    fp_count = len(result.false_positive_clusters)
    fn_count = len(result.false_negative_clusters)
    fp_cases = sum(c.count for c in result.false_positive_clusters)
    fn_cases = sum(c.count for c in result.false_negative_clusters)

    styles = "\n        ".join([
        _BASE_STYLES, _STATS_STYLES, _PLOT_STYLES, _TABLE_STYLES, _MISALIGNMENT_STYLES
    ])

    body = f"""<div class="header">
            <h1>Misalignment Clusters</h1>
            <div class="subtitle">{metric_id}</div>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value total">{result.total_cases}</div>
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
        {_render_cluster_tables(result)}"""

    return _build_html_page("Misalignment Clusters", metric_id, styles, body)


def _generate_fallback_html(
    result: ClusteringResult, metric_id: str, reason: str = "deps_missing"
) -> str:
    """Generate simple HTML without interactive plot."""
    if reason == "too_few_cases":
        note_content = f'Scatter plot requires at least 3 data points. Only {result.total_cases} case(s) found.'
    else:
        note_content = 'Interactive scatter plot requires: <code>pip install evalyn-sdk[clustering]</code>'

    styles = "\n        ".join([
        _BASE_STYLES, _NOTE_STYLES, _TABLE_STYLES, _MISALIGNMENT_STYLES
    ])

    body = f"""<div class="header">
            <h1>Misalignment Clusters</h1>
            <div class="subtitle">{metric_id}</div>
        </div>
        <div class="note">
            {note_content}
        </div>
        {_render_cluster_tables(result)}"""

    return _build_html_page("Misalignment Clusters", metric_id, styles, body)


def _render_single_cluster_table(
    clusters: List[ReasonCluster], title: str, title_class: str, badge_class: str
) -> str:
    """Render a single cluster table section."""
    if not clusters:
        return ""

    rows = []
    for c in clusters:
        example = c.representative_example.judge_reason or c.representative_example.human_notes
        example_short = example[:120] + "..." if len(example) > 120 else example
        rows.append(f"""
            <tr>
                <td><span class="cluster-badge {badge_class}">{c.label}</span></td>
                <td class="cluster-count {badge_class}">{c.count}</td>
                <td class="cluster-example">{example_short}</td>
            </tr>
        """)

    return f"""
        <h2 class="section-title {title_class}">{title}</h2>
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
        "fp",
        "fp",
    )
    fn_table = _render_single_cluster_table(
        result.false_negative_clusters,
        "False Negatives (Judge too strict)",
        "fn",
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
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
        deps_available = True
    except ImportError:
        deps_available = False

    if not result.coordinates_2d:
        if result.total_cases < 3:
            reason = "too_few_cases"
        else:
            reason = "deps_missing"
        return _generate_failure_fallback_html(result, metric_id, reason)

    if not deps_available:
        return _generate_failure_fallback_html(result, metric_id, "deps_missing")

    x_coords = [c[0] for c in result.coordinates_2d]
    y_coords = [c[1] for c in result.coordinates_2d]

    # Build color mapping
    unique_labels = sorted(set(result.case_labels or []))
    colors_palette = [
        "#D4A27F", "#C4836A", "#8B7355", "#6B8E8E",
        "#9B8AA6", "#A68B6B", "#7A9E7A", "#B8A090",
    ]
    color_map = {
        label: colors_palette[i % len(colors_palette)]
        for i, label in enumerate(unique_labels)
    }
    colors = [color_map.get(l, "#A0A0A0") for l in (result.case_labels or [])]

    # Build hover text
    hover_texts = []
    for i in range(len(x_coords)):
        label = result.case_labels[i] if result.case_labels else "Unknown"
        reason = result.case_reasons[i] if result.case_reasons else ""
        reason_short = reason[:150] + "..." if len(reason) > 150 else reason
        hover_texts.append(f"<b>{label}</b><br><br>{reason_short}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                size=14,
                color=colors,
                symbol="circle",
                line=dict(width=2, color="#FFFBF7"),
                opacity=0.9,
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Failures",
        )
    )

    fig.update_layout(**_get_base_plotly_layout(show_legend=False))
    plot_html = to_html(
        fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )

    styles = "\n        ".join([
        _BASE_STYLES, _STATS_STYLES, _PLOT_STYLES, _TABLE_STYLES, _FAILURE_STYLES
    ])

    body = f"""<div class="header">
            <h1>Failure Clusters</h1>
            <div class="subtitle">{metric_id}</div>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{result.total_cases}</div>
                <div class="stat-label">Failed Items</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(result.clusters)}</div>
                <div class="stat-label">Clusters</div>
            </div>
        </div>
        <div class="plot-container">
            {plot_html}
        </div>
        {_render_failure_cluster_table(result)}"""

    return _build_html_page("Failure Clusters", metric_id, styles, body)


def _generate_failure_fallback_html(
    result: FailureClusteringResult, metric_id: str, reason: str = "deps_missing"
) -> str:
    """Generate simple HTML for failures without interactive plot."""
    if reason == "too_few_cases":
        note_content = f'Scatter plot requires at least 3 data points. Only {result.total_cases} failure(s) found.'
    else:
        note_content = 'Interactive scatter plot requires: <code>pip install evalyn-sdk[clustering]</code>'

    styles = "\n        ".join([
        _BASE_STYLES, _NOTE_STYLES, _TABLE_STYLES, _FAILURE_STYLES
    ])

    body = f"""<div class="header">
            <h1>Failure Clusters</h1>
            <div class="subtitle">{metric_id}</div>
        </div>
        <div class="note">
            {note_content}
        </div>
        {_render_failure_cluster_table(result)}"""

    return _build_html_page("Failure Clusters", metric_id, styles, body)


def _render_failure_cluster_table(result: FailureClusteringResult) -> str:
    """Render HTML table for failure clusters."""
    if not result.clusters:
        return '<p style="color: #666;">No failure clusters found.</p>'

    rows = []
    for c in result.clusters:
        example = c.representative_example.reason
        example_short = example[:120] + "..." if len(example) > 120 else example
        rows.append(f"""
            <tr>
                <td><span class="cluster-badge">{c.label}</span></td>
                <td class="cluster-count">{c.count}</td>
                <td class="cluster-example">{example_short}</td>
            </tr>
        """)

    return f"""
        <h2 class="section-title">Failure Patterns</h2>
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
