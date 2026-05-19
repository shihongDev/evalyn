"""
Dataset cross-contamination check.

Verify no item leakage between train/test/calibration splits by
comparing content hashes across all split pairs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class LeakageItem:
    """An item found in multiple splits."""

    item_id: str
    found_in: list[str] = field(default_factory=list)
    input_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "found_in": self.found_in,
            "input_hash": self.input_hash,
        }


@dataclass
class ContaminationCheckResult:
    """Result of a cross-contamination check across splits."""

    leaks: list[LeakageItem] = field(default_factory=list)
    total_checked: int = 0
    leaked_count: int = 0
    clean: bool = True
    splits_checked: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "leaks": [lk.as_dict() for lk in self.leaks],
            "total_checked": self.total_checked,
            "leaked_count": self.leaked_count,
            "clean": self.clean,
            "splits_checked": self.splits_checked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContaminationCheckResult:
        return cls(
            leaks=[
                LeakageItem(
                    item_id=lk["item_id"],
                    found_in=lk.get("found_in", []),
                    input_hash=lk.get("input_hash", ""),
                )
                for lk in data.get("leaks", [])
            ],
            total_checked=data.get("total_checked", 0),
            leaked_count=data.get("leaked_count", 0),
            clean=data.get("clean", True),
            splits_checked=data.get("splits_checked", []),
        )

    def format_text(self) -> str:
        """Human-readable summary of contamination check."""
        lines: list[str] = []
        lines.append("Cross-Contamination Check")
        lines.append(f"Splits checked: {', '.join(self.splits_checked) or 'none'}")
        lines.append(f"Total items checked: {self.total_checked}")
        if self.clean:
            lines.append("Result: CLEAN - no leakage detected")
        else:
            lines.append(f"Result: CONTAMINATED - {self.leaked_count} leaked item(s)")
            for lk in self.leaks:
                lines.append(f"  - {lk.item_id}: found in {', '.join(lk.found_in)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def hash_item_content(text: str) -> str:
    """SHA-256 hash of normalized text (lowercased, stripped)."""
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_cross_contamination(
    splits: dict[str, list[dict[str, Any]]],
    input_field: str = "input",
) -> ContaminationCheckResult:
    """Check all splits for overlapping items by content hash.

    Args:
        splits: mapping of split name to list of item dicts,
            e.g. {"train": [...], "test": [...], "calibration": [...]}.
        input_field: dict key used to extract the text content for hashing.

    Returns:
        ContaminationCheckResult with any leaks found.
    """
    split_names = sorted(splits.keys())
    # Map: content_hash -> {split_name: [item_ids]}
    hash_to_splits: dict[str, dict[str, list[str]]] = {}
    total_checked = 0

    for split_name in split_names:
        for idx, item in enumerate(splits[split_name]):
            total_checked += 1
            raw = item.get(input_field, "")
            text = raw if isinstance(raw, str) else str(raw)
            h = hash_item_content(text)
            item_id = item.get("id", f"{split_name}-{idx}")
            if h not in hash_to_splits:
                hash_to_splits[h] = {}
            if split_name not in hash_to_splits[h]:
                hash_to_splits[h][split_name] = []
            hash_to_splits[h][split_name].append(item_id)

    leaks: list[LeakageItem] = []
    for h, split_map in hash_to_splits.items():
        if len(split_map) > 1:
            # This content appears in multiple splits
            found_in = sorted(split_map.keys())
            # Use the first item id from the first split as representative
            first_split = found_in[0]
            item_id = split_map[first_split][0]
            leaks.append(LeakageItem(item_id=item_id, found_in=found_in, input_hash=h))

    return ContaminationCheckResult(
        leaks=leaks,
        total_checked=total_checked,
        leaked_count=len(leaks),
        clean=len(leaks) == 0,
        splits_checked=split_names,
    )


def find_duplicate_items(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    input_field: str = "input",
) -> list[LeakageItem]:
    """Find items with the same input hash in both lists."""
    hashes_a: dict[str, str] = {}  # hash -> item_id
    for idx, item in enumerate(items_a):
        raw = item.get(input_field, "")
        text = raw if isinstance(raw, str) else str(raw)
        h = hash_item_content(text)
        hashes_a[h] = item.get("id", f"a-{idx}")

    duplicates: list[LeakageItem] = []
    for idx, item in enumerate(items_b):
        raw = item.get(input_field, "")
        text = raw if isinstance(raw, str) else str(raw)
        h = hash_item_content(text)
        if h in hashes_a:
            item_id = item.get("id", f"b-{idx}")
            duplicates.append(
                LeakageItem(
                    item_id=item_id,
                    found_in=["a", "b"],
                    input_hash=h,
                )
            )

    return duplicates


def compute_overlap_rate(
    splits: dict[str, list[dict[str, Any]]],
    input_field: str = "input",
) -> dict[str, float]:
    """Per-pair overlap percentage.

    Returns a dict keyed by "splitA-splitB" with overlap as a fraction
    of the smaller split's size (0.0 to 1.0).
    """
    split_names = sorted(splits.keys())
    # Pre-compute hash sets per split
    split_hashes: dict[str, set] = {}
    for name in split_names:
        hashes = set()
        for item in splits[name]:
            raw = item.get(input_field, "")
            text = raw if isinstance(raw, str) else str(raw)
            hashes.add(hash_item_content(text))
        split_hashes[name] = hashes

    result: dict[str, float] = {}
    for i, name_a in enumerate(split_names):
        for name_b in split_names[i + 1 :]:
            overlap = len(split_hashes[name_a] & split_hashes[name_b])
            smaller = min(len(split_hashes[name_a]), len(split_hashes[name_b]))
            rate = overlap / smaller if smaller > 0 else 0.0
            result[f"{name_a}-{name_b}"] = rate

    return result


def suggest_decontamination(result: ContaminationCheckResult) -> list[str]:
    """Suggestions for removing leaked items."""
    if result.clean:
        return ["No decontamination needed - splits are clean."]

    suggestions: list[str] = []
    suggestions.append(
        f"Found {result.leaked_count} leaked item(s) across splits."
    )
    suggestions.append(
        "Remove duplicates from all splits except the training set."
    )
    suggestions.append(
        "Re-run the contamination check after decontamination to verify."
    )

    split_set: set = set()
    for lk in result.leaks:
        for s in lk.found_in:
            split_set.add(s)

    if "test" in split_set and "train" in split_set:
        suggestions.append(
            "Priority: remove leaked items from the test split to preserve evaluation integrity."
        )
    if "calibration" in split_set:
        suggestions.append(
            "Also remove leaked items from the calibration split."
        )

    return suggestions
