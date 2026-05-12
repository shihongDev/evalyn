"""Tests for ``sdk.evalyn_sdk.annotation.compat``.

The compat layer is the seam between the CLI annotation flow and the
dashboard ``/annotate`` UI. These tests pin the shape detection and
normalization for all three on-disk formats so any reader of
``<dataset>/annotations.jsonl`` can rely on a single canonical view.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.annotation.compat import (
    CanonicalAnnotation,
    CanonicalLabel,
    LABEL_FAIL,
    LABEL_PASS,
    LABEL_SKIP,
    SHAPE_CLI_ANNOTATION,
    SHAPE_CLI_ANNOTATION_ITEM,
    SHAPE_DASHBOARD,
    SHAPE_UNKNOWN,
    derive_overall_label,
    detect_shape,
    normalize,
)


# ---------------------------------------------------------------------------
# detect_shape: classify a record by its disk shape
# ---------------------------------------------------------------------------


class TestDetectShape:
    def test_dashboard_record(self):
        rec = {
            "item_id": "call_abc",
            "labels": [{"metric_id": "helpfulness", "label": "pass"}],
            "annotator_id": "shiho",
        }
        assert detect_shape(rec) == SHAPE_DASHBOARD

    def test_cli_annotation_record(self):
        rec = {
            "id": "ann-x-1",
            "target_id": "call_abc",
            "label": True,
            "annotator": "human",
            "metric_labels": {
                "helpfulness": {
                    "metric_id": "helpfulness",
                    "agree_with_llm": True,
                    "human_label": True,
                    "notes": "",
                }
            },
        }
        assert detect_shape(rec) == SHAPE_CLI_ANNOTATION

    def test_cli_annotation_item_record(self):
        rec = {
            "id": "item_x",
            "input": {},
            "output": "hi",
            "eval_results": {"helpfulness": {"passed": True}},
            "human_label": {"passed": True, "annotator": "alice"},
        }
        assert detect_shape(rec) == SHAPE_CLI_ANNOTATION_ITEM

    def test_unknown_returns_unknown_not_raise(self):
        assert detect_shape({"random": "thing"}) == SHAPE_UNKNOWN
        assert detect_shape("not a dict") == SHAPE_UNKNOWN
        assert detect_shape(None) == SHAPE_UNKNOWN
        assert detect_shape([]) == SHAPE_UNKNOWN

    def test_dashboard_disambiguation_requires_item_id(self):
        # Has labels-as-list but no item_id: not a dashboard record.
        rec = {"labels": [{"metric_id": "x"}]}
        assert detect_shape(rec) == SHAPE_UNKNOWN

    def test_cli_annotation_disambiguation_requires_target_id(self):
        rec = {"metric_labels": {"x": {}}}
        # No target_id -> falls through to unknown.
        assert detect_shape(rec) == SHAPE_UNKNOWN


# ---------------------------------------------------------------------------
# normalize: each shape -> canonical
# ---------------------------------------------------------------------------


class TestNormalizeDashboard:
    def test_basic_dashboard_record(self):
        rec = {
            "item_id": "call_42",
            "labels": [
                {"metric_id": "h", "label": "pass", "used_ai_verdict": True, "note": None},
                {"metric_id": "f", "label": "fail", "used_ai_verdict": False, "note": "wrong"},
            ],
            "annotator_id": "shiho",
            "session_id": "ann-20260512-001122_aabbcc",
            "ts_iso": "2026-05-12T00:11:22+00:00",
            "note": "item-level rationale",
            "evidence": [{"snippet": "x"}],
            "skipped_metrics": ["t"],
        }
        c = normalize(rec)
        assert isinstance(c, CanonicalAnnotation)
        assert c.item_id == "call_42"
        assert c.annotator == "shiho"
        assert c.session_id == "ann-20260512-001122_aabbcc"
        assert c.rationale == "item-level rationale"
        assert c.created_at_iso == "2026-05-12T00:11:22+00:00"
        assert c.source_shape == SHAPE_DASHBOARD
        assert len(c.labels) == 2
        assert c.labels[0].metric_id == "h"
        assert c.labels[0].label == LABEL_PASS
        assert c.labels[0].used_ai_verdict is True
        assert c.labels[1].label == LABEL_FAIL
        assert c.labels[1].note == "wrong"
        assert c.extras["evidence"] == [{"snippet": "x"}]
        assert c.extras["skipped_metrics"] == ["t"]

    def test_per_label_confidence_averages(self):
        rec = {
            "item_id": "x",
            "labels": [
                {"metric_id": "a", "label": "pass", "confidence": 0.8},
                {"metric_id": "b", "label": "fail", "confidence": 0.4},
            ],
        }
        c = normalize(rec)
        assert c is not None
        assert c.confidence == pytest.approx(0.6)

    def test_empty_labels_list(self):
        rec = {"item_id": "x", "labels": [], "annotator_id": "alice"}
        c = normalize(rec)
        assert c is not None
        assert c.labels == []
        assert c.annotator == "alice"


class TestNormalizeCliAnnotation:
    def test_basic_cli_annotation(self):
        rec = {
            "id": "ann-call42-3",
            "target_id": "call_42",
            "label": True,
            "rationale": "looks right",
            "annotator": "human",
            "source": "human",
            "confidence": 4,
            "metric_labels": {
                "helpfulness": {
                    "metric_id": "helpfulness",
                    "agree_with_llm": True,
                    "human_label": True,
                    "notes": "",
                },
                "factuality": {
                    "metric_id": "factuality",
                    "agree_with_llm": False,
                    "human_label": False,
                    "notes": "URL doesn't say what agent claims",
                },
            },
            "created_at": "2026-05-04T14:30:00+00:00",
        }
        c = normalize(rec)
        assert c is not None
        assert c.item_id == "call_42"
        assert c.annotator == "human"
        assert c.rationale == "looks right"
        assert c.confidence == 4.0
        assert c.created_at_iso == "2026-05-04T14:30:00+00:00"
        assert c.session_id is None
        assert c.source_shape == SHAPE_CLI_ANNOTATION
        assert len(c.labels) == 2
        by_metric = {lbl.metric_id: lbl for lbl in c.labels}
        assert by_metric["helpfulness"].label == LABEL_PASS
        assert by_metric["helpfulness"].used_ai_verdict is True
        assert by_metric["factuality"].label == LABEL_FAIL
        assert by_metric["factuality"].used_ai_verdict is False
        assert by_metric["factuality"].note == "URL doesn't say what agent claims"


class TestNormalizeCliAnnotationItem:
    def test_with_human_label(self):
        rec = {
            "id": "item_x",
            "input": {"q": "hi"},
            "output": "hello",
            "eval_results": {"helpfulness": {"passed": True}},
            "human_label": {"passed": False, "annotator": "alice", "notes": "off-topic"},
        }
        c = normalize(rec)
        assert c is not None
        assert c.item_id == "item_x"
        assert c.annotator == "alice"
        assert c.rationale == "off-topic"
        assert c.source_shape == SHAPE_CLI_ANNOTATION_ITEM
        assert len(c.labels) == 1
        assert c.labels[0].label == LABEL_FAIL
        # Eval results preserved for downstream calibration consumers.
        assert c.extras["eval_results"] == {"helpfulness": {"passed": True}}

    def test_without_human_label_keeps_payload(self):
        rec = {
            "id": "item_x",
            "input": {},
            "output": "hi",
            "eval_results": {"h": {"passed": True}},
            "human_label": None,
        }
        c = normalize(rec)
        assert c is not None
        assert c.labels == []
        assert c.extras["output"] == "hi"


class TestNormalizeUnknownReturnsNone:
    def test_returns_none_no_raise(self):
        assert normalize({"random": "thing"}) is None
        assert normalize(None) is None
        assert normalize("string") is None


# ---------------------------------------------------------------------------
# derive_overall_label: roll metric labels up to a single bool
# ---------------------------------------------------------------------------


class TestDeriveOverallLabel:
    def test_all_pass_is_true(self):
        ann = CanonicalAnnotation(
            item_id="x",
            labels=[
                CanonicalLabel("a", LABEL_PASS),
                CanonicalLabel("b", LABEL_PASS),
            ],
        )
        assert derive_overall_label(ann) is True

    def test_any_fail_is_false(self):
        ann = CanonicalAnnotation(
            item_id="x",
            labels=[
                CanonicalLabel("a", LABEL_PASS),
                CanonicalLabel("b", LABEL_FAIL),
            ],
        )
        assert derive_overall_label(ann) is False

    def test_only_skip_returns_none(self):
        ann = CanonicalAnnotation(
            item_id="x",
            labels=[CanonicalLabel("a", LABEL_SKIP)],
        )
        assert derive_overall_label(ann) is None

    def test_empty_labels_returns_none(self):
        ann = CanonicalAnnotation(item_id="x", labels=[])
        assert derive_overall_label(ann) is None

    def test_pass_plus_skip_treats_as_pass(self):
        # If every concrete verdict is pass and some are skipped, the
        # overall pass-rate signal is still pass.
        ann = CanonicalAnnotation(
            item_id="x",
            labels=[
                CanonicalLabel("a", LABEL_PASS),
                CanonicalLabel("b", LABEL_SKIP),
            ],
        )
        assert derive_overall_label(ann) is True


# ---------------------------------------------------------------------------
# Round-trip via as_dict (used by CLI annotation-stats to convert back)
# ---------------------------------------------------------------------------


class TestCanonicalAsDict:
    def test_label_as_dict(self):
        lbl = CanonicalLabel("h", LABEL_PASS, used_ai_verdict=True, note="x")
        assert lbl.as_dict() == {
            "metric_id": "h",
            "label": "pass",
            "used_ai_verdict": True,
            "note": "x",
        }

    def test_annotation_as_dict_includes_extras(self):
        ann = CanonicalAnnotation(
            item_id="x",
            labels=[CanonicalLabel("a", LABEL_PASS)],
            annotator="alice",
            session_id="sess1",
            extras={"evidence": []},
        )
        d = ann.as_dict()
        assert d["item_id"] == "x"
        assert d["session_id"] == "sess1"
        assert d["extras"] == {"evidence": []}
        assert d["labels"][0]["label"] == "pass"


# ---------------------------------------------------------------------------
# End-to-end: `evalyn annotation-stats` reads dashboard-shaped records
# ---------------------------------------------------------------------------


class TestAnnotationStatsCrossSurfaceRead:
    """The original bug: CLI `annotation-stats` blindly parsed every JSONL
    line as `AnnotationItem.from_dict`. Dashboard-shaped lines either
    crashed or silently produced bogus stats. After wiring through the
    compat layer, both shapes are counted correctly.
    """

    def _run_stats(self, tmp_path, lines):
        """Helper: write annotations.jsonl in tmp_path, invoke cmd directly."""
        import argparse
        import json

        from evalyn_sdk.cli.commands.annotation import cmd_annotation_stats

        ann_path = tmp_path / "annotations.jsonl"
        with ann_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        ns = argparse.Namespace(dataset=str(tmp_path), quiet=True)
        # Should not raise. We just need to confirm the command completes.
        cmd_annotation_stats(ns)

    def test_pure_dashboard_records_no_crash(self, tmp_path):
        records = [
            {
                "item_id": "call_a",
                "labels": [
                    {"metric_id": "h", "label": "pass", "used_ai_verdict": True},
                ],
                "annotator_id": "dash",
                "session_id": "sess1",
                "ts_iso": "2026-05-12T00:00:00+00:00",
            },
            {
                "item_id": "call_b",
                "labels": [
                    {"metric_id": "h", "label": "fail", "used_ai_verdict": False},
                ],
                "annotator_id": "dash",
                "session_id": "sess1",
                "ts_iso": "2026-05-12T00:00:01+00:00",
            },
        ]
        self._run_stats(tmp_path, records)

    def test_mixed_shapes_no_crash(self, tmp_path):
        records = [
            # CLI Annotation shape
            {
                "id": "ann-x-1",
                "target_id": "call_a",
                "label": True,
                "annotator": "cli",
                "source": "human",
                "metric_labels": {
                    "helpfulness": {
                        "metric_id": "helpfulness",
                        "agree_with_llm": True,
                        "human_label": True,
                        "notes": "",
                    }
                },
            },
            # Dashboard shape
            {
                "item_id": "call_b",
                "labels": [{"metric_id": "helpfulness", "label": "fail", "used_ai_verdict": False}],
                "annotator_id": "dash",
                "session_id": "sess1",
            },
            # Unknown shape - should not crash, should be reported
            {"foo": "bar"},
        ]
        self._run_stats(tmp_path, records)
