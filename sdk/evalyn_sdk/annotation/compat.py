"""Cross-surface annotation record normalization.

Annotation records land in ``<dataset>/annotations.jsonl`` from at least
three historical surfaces:

- CLI ``Annotation`` (``sdk.evalyn_sdk.models.Annotation``): keyed by
  ``target_id``, per-metric verdicts in a ``metric_labels`` dict,
  ``human_label: bool`` semantics.
- CLI ``AnnotationItem`` (``sdk.evalyn_sdk.models.AnnotationItem``):
  the export format with full ``input``/``output``/``eval_results`` plus
  a single ``human_label`` sub-object.
- Dashboard verdict (``dashboard.evalyn_dashboard.api.v2.annotation``):
  keyed by ``item_id``, per-metric verdicts in a ``labels`` list with
  string enum values ``"pass" | "fail" | "skip"``, plus ``session_id``,
  ``evidence`` snippets, and ``used_ai_verdict`` per label.

The dashboard test at ``dashboard/tests/test_api_v2_annotation.py``
explicitly comments on the divergence ("a pre-existing annotations.jsonl
with a different shape"). This module is the single place where the
three shapes are reconciled. Readers (``annotation-stats`` CLI,
calibration, dashboard finalize dedup) call ``normalize`` and get a
canonical view; writers stay surface-specific.

When a 4th shape arrives, extend ``detect_shape`` and ``normalize`` here
rather than teaching every reader about the new format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Shape identifiers. Public constants so callers can pattern-match
# without importing string literals.
SHAPE_CLI_ANNOTATION = "cli_annotation"
SHAPE_CLI_ANNOTATION_ITEM = "cli_annotation_item"
SHAPE_DASHBOARD = "dashboard"
SHAPE_UNKNOWN = "unknown"

# Canonical label values. Mirrors the dashboard's enum because it's
# strictly more expressive than the CLI's bool (skip cannot be
# represented by a bool).
LABEL_PASS = "pass"
LABEL_FAIL = "fail"
LABEL_SKIP = "skip"


def detect_shape(rec: Any) -> str:
    """Return the on-disk shape of an annotation JSONL record.

    Distinguishes the three known shapes by fields unique to each.
    Returns ``SHAPE_UNKNOWN`` for anything else - never raises.
    """
    if not isinstance(rec, dict):
        return SHAPE_UNKNOWN
    # Dashboard verdict: ``labels`` is a list-of-dicts and ``item_id`` is
    # the per-item key. The CLI ``Annotation`` shape also has ``label``
    # (singular) but never ``labels`` as a list.
    if isinstance(rec.get("labels"), list) and "item_id" in rec:
        return SHAPE_DASHBOARD
    # CLI Annotation: ``metric_labels`` is a dict, keyed on metric_id.
    if isinstance(rec.get("metric_labels"), dict) and "target_id" in rec:
        return SHAPE_CLI_ANNOTATION
    # CLI AnnotationItem (the export format): carries ``eval_results``
    # and a ``human_label`` sub-object.
    if "eval_results" in rec and "human_label" in rec:
        return SHAPE_CLI_ANNOTATION_ITEM
    return SHAPE_UNKNOWN


@dataclass
class CanonicalLabel:
    """One metric verdict in canonical form."""

    metric_id: str
    # None means "no verdict on this metric for this item". A blank
    # entry for a metric that was scored by a judge means the human
    # chose to defer (different from "skip" which is explicit).
    label: str | None = None
    used_ai_verdict: bool | None = None
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "metric_id": self.metric_id,
            "label": self.label,
            "used_ai_verdict": self.used_ai_verdict,
            "note": self.note,
        }


@dataclass
class CanonicalAnnotation:
    """Annotation record normalized across the three known surfaces.

    Canonical = the union of fields any consumer might need. Surface-
    specific extras (CLI ``rationale``, dashboard ``evidence``) are
    preserved in ``extras`` so a downstream reader that cares about them
    can still get at the raw values without re-parsing.
    """

    item_id: str
    labels: list[CanonicalLabel] = field(default_factory=list)
    annotator: str = ""
    created_at_iso: str | None = None
    session_id: str | None = None
    rationale: str | None = None
    confidence: float | None = None
    source_shape: str = SHAPE_UNKNOWN
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "labels": [lbl.as_dict() for lbl in self.labels],
            "annotator": self.annotator,
            "created_at_iso": self.created_at_iso,
            "session_id": self.session_id,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "source_shape": self.source_shape,
            "extras": self.extras,
        }


def _bool_to_label(value: Any) -> str | None:
    """Map CLI's bool ``human_label`` to canonical string enum."""
    if value is True:
        return LABEL_PASS
    if value is False:
        return LABEL_FAIL
    return None


def _from_cli_annotation(rec: dict) -> CanonicalAnnotation:
    """CLI ``Annotation`` -> canonical."""
    labels: list[CanonicalLabel] = []
    for metric_id, ml in (rec.get("metric_labels") or {}).items():
        if not isinstance(ml, dict):
            continue
        labels.append(
            CanonicalLabel(
                metric_id=metric_id,
                label=_bool_to_label(ml.get("human_label")),
                # CLI ``agree_with_llm`` is the structural twin of
                # dashboard ``used_ai_verdict`` - both record whether
                # the human deferred to the LLM's judgment.
                used_ai_verdict=ml.get("agree_with_llm"),
                note=ml.get("notes") or None,
            )
        )
    confidence: float | None = None
    raw_conf = rec.get("confidence")
    if isinstance(raw_conf, (int, float)):
        confidence = float(raw_conf)
    return CanonicalAnnotation(
        item_id=str(rec.get("target_id") or ""),
        labels=labels,
        annotator=str(rec.get("annotator") or ""),
        created_at_iso=rec.get("created_at"),
        session_id=None,
        rationale=rec.get("rationale"),
        confidence=confidence,
        source_shape=SHAPE_CLI_ANNOTATION,
        extras={"id": rec.get("id"), "source": rec.get("source")},
    )


def _from_cli_annotation_item(rec: dict) -> CanonicalAnnotation:
    """CLI ``AnnotationItem`` -> canonical.

    The export format carries the full evaluation payload. We pull just
    the annotation-relevant slice (id + human_label) and stash the
    larger payload in ``extras`` so a downstream tool that needs the
    eval_results can recover them.
    """
    human_label = rec.get("human_label") or {}
    labels: list[CanonicalLabel] = []
    # ``human_label`` here is a HumanLabel sub-object with a single
    # ``passed: bool``. There is no per-metric breakdown in this shape.
    passed = human_label.get("passed") if isinstance(human_label, dict) else None
    # Fabricate a single synthetic label so downstream consumers see at
    # least one verdict. The metric_id is empty (caller can choose to
    # map it to "overall" or treat as item-level).
    if passed is not None:
        labels.append(
            CanonicalLabel(
                metric_id="",
                label=_bool_to_label(passed),
                used_ai_verdict=None,
                note=human_label.get("notes") if isinstance(human_label, dict) else None,
            )
        )
    return CanonicalAnnotation(
        item_id=str(rec.get("id") or ""),
        labels=labels,
        annotator=str(
            human_label.get("annotator") if isinstance(human_label, dict) else ""
        )
        or "",
        created_at_iso=None,
        session_id=None,
        rationale=human_label.get("notes") if isinstance(human_label, dict) else None,
        confidence=None,
        source_shape=SHAPE_CLI_ANNOTATION_ITEM,
        extras={
            "input": rec.get("input"),
            "output": rec.get("output"),
            "eval_results": rec.get("eval_results") or {},
            "metadata": rec.get("metadata") or {},
        },
    )


def _from_dashboard(rec: dict) -> CanonicalAnnotation:
    """Dashboard verdict -> canonical."""
    labels: list[CanonicalLabel] = []
    for entry in rec.get("labels") or []:
        if not isinstance(entry, dict):
            continue
        labels.append(
            CanonicalLabel(
                metric_id=str(entry.get("metric_id") or ""),
                label=entry.get("label"),
                used_ai_verdict=entry.get("used_ai_verdict"),
                note=entry.get("note"),
            )
        )
    confidence: float | None = None
    # Dashboard sometimes carries per-label confidence; average across
    # labels for an item-level scalar. Skipping confidence entirely is
    # also defensible - keeping the average for symmetry with the CLI.
    confs = [
        entry.get("confidence")
        for entry in rec.get("labels") or []
        if isinstance(entry, dict)
        and isinstance(entry.get("confidence"), (int, float))
    ]
    if confs:
        confidence = float(sum(confs) / len(confs))
    return CanonicalAnnotation(
        item_id=str(rec.get("item_id") or ""),
        labels=labels,
        annotator=str(rec.get("annotator_id") or ""),
        created_at_iso=rec.get("ts_iso"),
        session_id=rec.get("session_id"),
        rationale=rec.get("note"),
        confidence=confidence,
        source_shape=SHAPE_DASHBOARD,
        extras={
            "evidence": rec.get("evidence") or [],
            "skipped_metrics": rec.get("skipped_metrics") or [],
        },
    )


def normalize(rec: Any) -> CanonicalAnnotation | None:
    """Normalize any supported annotation shape to canonical.

    Returns ``None`` for records that don't match a known shape. Callers
    decide whether to skip + warn (the usual choice for ``annotation-stats``)
    or raise.
    """
    shape = detect_shape(rec)
    if shape == SHAPE_CLI_ANNOTATION:
        return _from_cli_annotation(rec)
    if shape == SHAPE_CLI_ANNOTATION_ITEM:
        return _from_cli_annotation_item(rec)
    if shape == SHAPE_DASHBOARD:
        return _from_dashboard(rec)
    return None


def derive_overall_label(annotation: CanonicalAnnotation) -> bool | None:
    """Roll the per-metric labels up to an overall pass/fail bool.

    Returns ``True`` if every concrete label is pass, ``False`` if any
    label is fail, ``None`` if no labels carry a verdict (all skipped /
    deferred). Matches the CLI ``Annotation.label`` semantics where the
    overall is the conjunction of per-metric humans.
    """
    saw_any = False
    for lbl in annotation.labels:
        if lbl.label == LABEL_FAIL:
            return False
        if lbl.label == LABEL_PASS:
            saw_any = True
    return True if saw_any else None


__all__ = [
    "CanonicalAnnotation",
    "CanonicalLabel",
    "LABEL_FAIL",
    "LABEL_PASS",
    "LABEL_SKIP",
    "SHAPE_CLI_ANNOTATION",
    "SHAPE_CLI_ANNOTATION_ITEM",
    "SHAPE_DASHBOARD",
    "SHAPE_UNKNOWN",
    "derive_overall_label",
    "detect_shape",
    "normalize",
]
