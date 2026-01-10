from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

from ..models import Annotation, AnnotationItem, HumanLabel, DatasetItem


def export_annotations(annotations: Iterable[Annotation], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann.as_dict()) + "\n")


def export_annotation_items(items: Iterable[AnnotationItem], path: str | Path) -> None:
    """Export annotation items in the new format."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")


def import_annotations(path: str | Path) -> List[Annotation]:
    """Import annotations - supports both old and new formats."""
    annotations: List[Annotation] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Check if it's the new AnnotationItem format
                if "human_label" in data and "eval_results" in data:
                    # Convert AnnotationItem to Annotation
                    human_label = data.get("human_label")
                    if human_label:
                        annotations.append(
                            Annotation(
                                id=str(uuid.uuid4()),
                                target_id=data.get("id", ""),
                                label=human_label.get("passed"),
                                rationale=human_label.get("notes"),
                                annotator=human_label.get("annotator", "unknown"),
                                source="human",
                                confidence=None,
                            )
                        )
                else:
                    # Old format
                    annotations.append(Annotation.from_dict(data))
    return annotations


def import_annotation_items(path: str | Path) -> List[AnnotationItem]:
    """Import annotation items in the new format."""
    items: List[AnnotationItem] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                items.append(AnnotationItem.from_dict(data))
    return items


def merge_annotations_into_dataset(
    dataset_items: List[DatasetItem],
    annotation_items: List[AnnotationItem],
) -> List[DatasetItem]:
    """Merge human_labels from annotations back into dataset items."""
    ann_map = {item.id: item for item in annotation_items}
    merged = []
    for item in dataset_items:
        ann = ann_map.get(item.id)
        if ann and ann.human_label:
            item.human_label = ann.human_label.as_dict()
        merged.append(item)
    return merged
