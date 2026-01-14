"""
Agent Simulator for generating synthetic test data.

Uses LLM to generate variations of seed inputs (similar or outlier),
runs the target agent, and saves results as a new dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from ..models import DatasetItem, FunctionCall, now_utc
from ..trace.tracer import EvalTracer
from ..utils.api_client import GeminiClient


# =============================================================================
# Simple Utilities
# =============================================================================


def synthetic_dataset(prompts: Iterable[str]) -> List[DatasetItem]:
    """Create a toy dataset from a list of prompts."""
    return [
        DatasetItem(
            id=str(uuid4()),
            inputs={"user_input": prompt},
            expected=None,
            metadata={"tag": "synthetic"},
        )
        for prompt in prompts
    ]


def simulate_agent(
    tracer: EvalTracer,
    handler: Callable[[str], str],
    prompts: Iterable[str],
) -> List[FunctionCall]:
    """Run a handler over prompts to produce traced FunctionCalls."""
    wrapped = tracer.instrument(handler)
    calls: List[FunctionCall] = []
    for prompt in prompts:
        wrapped(prompt)
        if tracer.last_call:
            calls.append(tracer.last_call)
    return calls


# =============================================================================
# LLM-Based Simulators
# =============================================================================


@dataclass
class SimulationConfig:
    """Configuration for synthetic data generation."""

    num_similar: int = 3  # Variations similar to seed per item
    num_outlier: int = 1  # Outlier/edge cases per item
    model: str = "gemini-2.5-flash-lite"
    temperature_similar: float = 0.3  # Lower temp for similar variations
    temperature_outlier: float = 0.8  # Higher temp for creative outliers
    max_seed_items: int = 50  # Max seed items to use


@dataclass
class GeneratedQuery:
    """A generated query from the simulator."""

    query: str
    mode: str  # "similar" or "outlier"
    seed_id: str  # ID of the seed item this was generated from
    seed_input: Dict[str, Any]
    generation_reason: str  # Why this was generated


class UserSimulator:
    """
    LLM-based user simulator that generates synthetic queries.

    Two modes:
    1. Similar: Generate queries similar to seed inputs (test robustness)
    2. Outlier: Generate edge cases, adversarial inputs, unusual requests
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client: Optional[GeminiClient] = None

    @property
    def client(self) -> GeminiClient:
        """Lazy-initialized Gemini API client."""
        if self._client is None:
            self._client = GeminiClient(
                model=self.model,
                api_key=self._api_key,
            )
        return self._client

    def generate_similar(
        self,
        seed_items: List[DatasetItem],
        num_per_seed: int = 3,
    ) -> List[GeneratedQuery]:
        """Generate queries similar to seed inputs."""

        # Build prompt with seed examples
        seed_examples = []
        for item in seed_items[:10]:  # Use up to 10 examples
            input_data = item.input or item.inputs
            if isinstance(input_data, dict):
                # Extract the main query/question
                query = (
                    input_data.get("kwargs", {}).get("question")
                    or input_data.get("question")
                    or input_data.get("query")
                    or input_data.get("input")
                    or json.dumps(input_data)
                )
            else:
                query = str(input_data)
            seed_examples.append(query)

        prompt = f"""You are a user simulator. Given these example user queries, generate {num_per_seed * len(seed_items)} NEW similar queries.

EXAMPLE QUERIES FROM REAL USERS:
{chr(10).join(f"- {q}" for q in seed_examples)}

Generate variations that:
1. Ask about similar topics but with different phrasing
2. Have similar complexity and style
3. Are realistic user queries
4. Cover slight variations (different aspects, follow-ups, related topics)

Return a JSON array of objects, each with:
- "query": the generated query string
- "reason": brief explanation of how it relates to the examples

Return ONLY the JSON array, no other text.
Generate exactly {num_per_seed * len(seed_items)} queries."""

        try:
            response = self.client.generate(prompt, temperature=0.3)
            queries = self._parse_query_response(response)
        except Exception as e:
            print(f"Warning: Failed to generate similar queries: {e}")
            queries = []

        # Map generated queries back to seed items
        results = []
        for i, q in enumerate(queries):
            seed_idx = i % len(seed_items)
            seed_item = seed_items[seed_idx]
            results.append(
                GeneratedQuery(
                    query=q.get("query", ""),
                    mode="similar",
                    seed_id=seed_item.id,
                    seed_input=seed_item.input or seed_item.inputs or {},
                    generation_reason=q.get("reason", "similar variation"),
                )
            )

        return results

    def generate_outliers(
        self,
        seed_items: List[DatasetItem],
        num_per_seed: int = 1,
    ) -> List[GeneratedQuery]:
        """Generate edge case / outlier queries."""

        # Build prompt with seed examples
        seed_examples = []
        for item in seed_items[:10]:
            input_data = item.input or item.inputs
            if isinstance(input_data, dict):
                query = (
                    input_data.get("kwargs", {}).get("question")
                    or input_data.get("question")
                    or input_data.get("query")
                    or json.dumps(input_data)
                )
            else:
                query = str(input_data)
            seed_examples.append(query)

        prompt = f"""You are a QA engineer designing edge case tests. Given these example user queries, generate {num_per_seed * len(seed_items)} OUTLIER/EDGE CASE queries.

EXAMPLE QUERIES (normal usage):
{chr(10).join(f"- {q}" for q in seed_examples)}

Generate edge cases that test:
1. Ambiguous or unclear requests
2. Very long or very short queries
3. Multiple questions in one query
4. Queries with typos or unusual formatting
5. Boundary conditions (empty, special characters)
6. Adversarial but non-malicious queries (trying to confuse the system)
7. Out-of-domain but related queries
8. Queries requiring clarification

Return a JSON array of objects, each with:
- "query": the generated query string
- "reason": what edge case this tests
- "category": one of [ambiguous, long, short, multi, typo, boundary, adversarial, out_of_domain, clarification]

Return ONLY the JSON array, no other text.
Generate exactly {num_per_seed * len(seed_items)} queries."""

        try:
            response = self.client.generate(prompt, temperature=0.8)
            queries = self._parse_query_response(response)
        except Exception as e:
            print(f"Warning: Failed to generate outlier queries: {e}")
            queries = []

        # Map generated queries back to seed items
        results = []
        for i, q in enumerate(queries):
            seed_idx = i % len(seed_items)
            seed_item = seed_items[seed_idx]
            category = q.get("category", "edge_case")
            results.append(
                GeneratedQuery(
                    query=q.get("query", ""),
                    mode="outlier",
                    seed_id=seed_item.id,
                    seed_input=seed_item.input or seed_item.inputs or {},
                    generation_reason=f"[{category}] {q.get('reason', 'edge case')}",
                )
            )

        return results

    def _parse_query_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into list of query dicts."""
        text = response.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            # Try direct parse
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Try to find JSON array
        start = text.find("[")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start : i + 1])
                            if isinstance(parsed, list):
                                return parsed
                        except Exception:
                            pass
                        break

        return []


class AgentSimulator:
    """
    Full simulation pipeline:
    1. Load seed dataset
    2. Generate synthetic queries (similar + outlier)
    3. Run target agent on generated queries
    4. Save results as new dataset
    """

    def __init__(
        self,
        target_fn: Callable,
        config: Optional[SimulationConfig] = None,
        model: str = "gemini-2.5-flash-lite",
    ):
        self.target_fn = target_fn
        self.config = config or SimulationConfig()
        self.user_simulator = UserSimulator(model=model)

    def run(
        self,
        seed_dataset: List[DatasetItem],
        output_dir: Path,
        modes: List[str] = ["similar", "outlier"],
    ) -> Dict[str, Path]:
        """
        Run simulation and save results.

        Returns dict mapping mode -> output path.
        """
        results = {}

        # Limit seed items
        seed_items = seed_dataset[: self.config.max_seed_items]

        for mode in modes:
            if mode == "similar":
                generated = self.user_simulator.generate_similar(
                    seed_items,
                    num_per_seed=self.config.num_similar,
                )
            elif mode == "outlier":
                generated = self.user_simulator.generate_outliers(
                    seed_items,
                    num_per_seed=self.config.num_outlier,
                )
            else:
                continue

            if not generated:
                print(f"No queries generated for mode={mode}")
                continue

            # Run agent on generated queries
            dataset_items = self._run_agent_on_queries(generated)

            # Save to output directory
            mode_dir = (
                output_dir / f"sim-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            mode_dir.mkdir(parents=True, exist_ok=True)

            # Save dataset
            dataset_path = mode_dir / "dataset.jsonl"
            with open(dataset_path, "w", encoding="utf-8") as f:
                for item in dataset_items:
                    f.write(
                        json.dumps(item.as_dict(), ensure_ascii=False, default=str)
                        + "\n"
                    )

            # Save meta.json
            meta = {
                "version": "synthetic",
                "created_at": now_utc().isoformat(),
                "source": "synthetic",
                "mode": mode,
                "seed_dataset": str(output_dir.parent),
                "num_items": len(dataset_items),
                "config": {
                    "num_similar": self.config.num_similar,
                    "num_outlier": self.config.num_outlier,
                    "model": self.config.model,
                },
            }
            with open(mode_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results[mode] = mode_dir
            print(f"Generated {len(dataset_items)} items for mode={mode} -> {mode_dir}")

        return results

    def _run_agent_on_queries(
        self,
        queries: List[GeneratedQuery],
    ) -> List[DatasetItem]:
        """Run the target agent on generated queries."""
        results = []

        for i, gq in enumerate(queries):
            print(f"  Running query {i + 1}/{len(queries)}: {gq.query[:50]}...")

            try:
                # Prepare input - try to match the original input format
                if gq.seed_input and isinstance(gq.seed_input, dict):
                    if "kwargs" in gq.seed_input:
                        # Function was called with kwargs
                        kwargs = dict(gq.seed_input.get("kwargs", {}))
                        # Replace the query field
                        for key in ["question", "query", "input", "prompt"]:
                            if key in kwargs:
                                kwargs[key] = gq.query
                                break
                        else:
                            # No known query field, use first string arg
                            kwargs["question"] = gq.query

                        output = self.target_fn(**kwargs)
                        input_data = {"args": [], "kwargs": kwargs}
                    else:
                        # Direct dict input
                        output = self.target_fn(gq.query)
                        input_data = {"query": gq.query}
                else:
                    output = self.target_fn(gq.query)
                    input_data = {"query": gq.query}

                error = None
            except Exception as e:
                output = None
                error = str(e)
                input_data = {"query": gq.query}

            item = DatasetItem(
                id=str(uuid4()),
                input=input_data,
                output=output,
                metadata={
                    "mode": gq.mode,
                    "seed_id": gq.seed_id,
                    "generation_reason": gq.generation_reason,
                    "error": error,
                },
            )
            results.append(item)

        return results


def create_versioned_dataset_dir(
    base_dir: Path,
    project: str,
    version: str,
    suffix: str = "",
) -> Path:
    """
    Create a versioned dataset directory.

    Structure: base_dir/project/version[-suffix]/
    """
    if suffix:
        dir_name = f"{version}-{suffix}"
    else:
        dir_name = version

    dataset_dir = base_dir / project / dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dir
