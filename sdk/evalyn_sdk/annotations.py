from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import Annotation


def export_annotations(annotations: Iterable[Annotation], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann.as_dict()) + "\n")


def import_annotations(path: str | Path) -> List[Annotation]:
    annotations: List[Annotation] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(Annotation.from_dict(json.loads(line)))
    return annotations
