from __future__ import annotations

import random
from typing import Callable, Iterable, List
from uuid import uuid4

from ..models import DatasetItem, FunctionCall
from ..trace.tracer import EvalTracer


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


def random_prompt_variations(base_prompt: str, n: int = 5) -> List[str]:
    suffixes = [
        "as a step-by-step guide",
        "with an example",
        "in bullet points",
        "with edge cases",
        "with caveats",
    ]
    return [f"{base_prompt} {random.choice(suffixes)}" for _ in range(n)]
