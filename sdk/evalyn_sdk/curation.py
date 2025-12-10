from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .models import DatasetItem
from .tracing import eval_session


def curate_dataset(
    target_fn: Callable,
    prompts: Iterable[str],
    *,
    expected_strategy: str = "as_output",
    build_metadata: Optional[Callable[[str, str], dict]] = None,
    session_id: Optional[str] = None,
    store_path: Optional[str | Path] = None,
) -> List[DatasetItem]:
    """
    Run the target function over prompts to build a dataset with traces captured via @eval.

    expected_strategy:
      - "as_output": use the produced output as the expected value (baseline/regression).
      - "none": leave expected as None.
    build_metadata(prompt, output) -> dict to attach metadata per item (optional).
    If store_path is provided, writes JSONL to that path.
    """
    items: List[DatasetItem] = []
    with eval_session(session_id or "curation"):
        for idx, prompt in enumerate(prompts, start=1):
            output = target_fn(prompt)
            expected = output if expected_strategy == "as_output" else None
            metadata = build_metadata(prompt, output) if build_metadata else {}
            items.append(
                DatasetItem(
                    id=f"item-{idx}",
                    inputs={"query": prompt},
                    expected=expected,
                    metadata=metadata,
                )
            )

    if store_path:
        path = Path(store_path)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.__dict__) + "\n")

    return items
