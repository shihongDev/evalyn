"""Tests for calibration human validation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.human_validation import (
    ValidationDecision,
    ValidationReport,
    ValidationRequest,
    build_validation_report,
    create_validation_request,
    record_decision,
    render_validation_summary,
)


# -------------------------------------------------------------------
# ValidationRequest
# -------------------------------------------------------------------

class TestValidationRequest:
    def test_as_dict_from_dict_roundtrip(self):
        req = ValidationRequest(
            metric_id="accuracy",
            original_prompt="Rate the answer",
            calibrated_prompt="Rate the answer carefully",
            sample_items=[{"q": "hello"}],
            alignment_before=0.6,
            alignment_after=0.85,
        )
        d = req.as_dict()
        restored = ValidationRequest.from_dict(d)
        assert restored.metric_id == "accuracy"
        assert restored.original_prompt == "Rate the answer"
        assert restored.calibrated_prompt == "Rate the answer carefully"
        assert restored.sample_items == [{"q": "hello"}]
        assert restored.alignment_before == 0.6
        assert restored.alignment_after == 0.85

    def test_from_dict_defaults(self):
        req = ValidationRequest.from_dict({
            "metric_id": "m1",
            "original_prompt": "orig",
            "calibrated_prompt": "cal",
        })
        assert req.sample_items == []
        assert req.alignment_before == 0.0
        assert req.alignment_after == 0.0

    def test_defaults(self):
        req = ValidationRequest(
            metric_id="m", original_prompt="o", calibrated_prompt="c"
        )
        assert req.sample_items == []
        assert req.alignment_before == 0.0
        assert req.alignment_after == 0.0


# -------------------------------------------------------------------
# ValidationDecision
# -------------------------------------------------------------------

class TestValidationDecision:
    def test_as_dict_from_dict_roundtrip(self):
        ts = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        dec = ValidationDecision(
            approved=True, reviewer="alice", notes="looks good", timestamp=ts
        )
        d = dec.as_dict()
        restored = ValidationDecision.from_dict(d)
        assert restored.approved is True
        assert restored.reviewer == "alice"
        assert restored.notes == "looks good"
        assert restored.timestamp == ts

    def test_from_dict_defaults(self):
        dec = ValidationDecision.from_dict({"approved": False})
        assert dec.reviewer == ""
        assert dec.notes == ""
        assert dec.timestamp is None

    def test_as_dict_none_timestamp(self):
        dec = ValidationDecision(approved=True)
        d = dec.as_dict()
        assert d["timestamp"] is None

    def test_defaults(self):
        dec = ValidationDecision(approved=False)
        assert dec.reviewer == ""
        assert dec.notes == ""
        assert dec.timestamp is None


# -------------------------------------------------------------------
# ValidationReport
# -------------------------------------------------------------------

class TestValidationReport:
    def test_as_dict_from_dict_roundtrip(self):
        req = ValidationRequest(metric_id="m", original_prompt="o", calibrated_prompt="c")
        dec = ValidationDecision(approved=True, reviewer="bob")
        report = ValidationReport(
            requests=[req],
            decisions=[dec],
            approved_count=1,
            rejected_count=0,
            pending_count=0,
        )
        d = report.as_dict()
        restored = ValidationReport.from_dict(d)
        assert len(restored.requests) == 1
        assert restored.requests[0].metric_id == "m"
        assert len(restored.decisions) == 1
        assert restored.decisions[0].approved is True
        assert restored.approved_count == 1
        assert restored.rejected_count == 0
        assert restored.pending_count == 0

    def test_from_dict_defaults(self):
        report = ValidationReport.from_dict({})
        assert report.requests == []
        assert report.decisions == []
        assert report.approved_count == 0

    def test_format_text(self):
        req = ValidationRequest(
            metric_id="accuracy",
            original_prompt="o",
            calibrated_prompt="c",
            alignment_before=0.5,
            alignment_after=0.8,
        )
        report = ValidationReport(
            requests=[req], decisions=[], approved_count=0, rejected_count=0, pending_count=1
        )
        text = report.format_text()
        assert "Validation Report" in text
        assert "accuracy" in text
        assert "Pending: 1" in text
        assert "0.500" in text
        assert "0.800" in text

    def test_format_text_empty(self):
        report = ValidationReport()
        text = report.format_text()
        assert "Requests: 0" in text
        assert "Decisions: 0" in text


# -------------------------------------------------------------------
# create_validation_request
# -------------------------------------------------------------------

class TestCreateValidationRequest:
    def test_factory(self):
        req = create_validation_request(
            metric_id="helpfulness",
            original_prompt="original",
            calibrated_prompt="calibrated",
            sample_items=[{"a": 1}],
            alignment_before=0.3,
            alignment_after=0.7,
        )
        assert req.metric_id == "helpfulness"
        assert req.original_prompt == "original"
        assert req.calibrated_prompt == "calibrated"
        assert req.sample_items == [{"a": 1}]
        assert req.alignment_before == 0.3
        assert req.alignment_after == 0.7

    def test_factory_defaults(self):
        req = create_validation_request(
            metric_id="m", original_prompt="o", calibrated_prompt="c"
        )
        assert req.sample_items == []
        assert req.alignment_before == 0.0
        assert req.alignment_after == 0.0

    def test_factory_copies_sample_items(self):
        items = [{"x": 1}]
        req = create_validation_request(
            metric_id="m", original_prompt="o", calibrated_prompt="c", sample_items=items
        )
        items.append({"y": 2})
        assert len(req.sample_items) == 1  # Not affected by mutation


# -------------------------------------------------------------------
# render_validation_summary
# -------------------------------------------------------------------

class TestRenderValidationSummary:
    def test_shows_both_prompts(self):
        req = ValidationRequest(
            metric_id="m",
            original_prompt="prompt A",
            calibrated_prompt="prompt B",
        )
        text = render_validation_summary(req)
        assert "prompt A" in text
        assert "prompt B" in text

    def test_shows_delta(self):
        req = ValidationRequest(
            metric_id="m",
            original_prompt="o",
            calibrated_prompt="c",
            alignment_before=0.4,
            alignment_after=0.7,
        )
        text = render_validation_summary(req)
        assert "0.400" in text
        assert "0.700" in text
        assert "+0.300" in text

    def test_negative_delta(self):
        req = ValidationRequest(
            metric_id="m",
            original_prompt="o",
            calibrated_prompt="c",
            alignment_before=0.8,
            alignment_after=0.5,
        )
        text = render_validation_summary(req)
        assert "-0.300" in text

    def test_shows_sample_items(self):
        req = ValidationRequest(
            metric_id="m",
            original_prompt="o",
            calibrated_prompt="c",
            sample_items=[{"input": "hello"}, {"input": "world"}],
        )
        text = render_validation_summary(req)
        assert "Sample items (2)" in text
        assert "hello" in text
        assert "world" in text

    def test_no_sample_items(self):
        req = ValidationRequest(
            metric_id="m", original_prompt="o", calibrated_prompt="c"
        )
        text = render_validation_summary(req)
        assert "Sample items" not in text


# -------------------------------------------------------------------
# record_decision
# -------------------------------------------------------------------

class TestRecordDecision:
    def test_with_timestamp(self):
        dec = record_decision(approved=True, reviewer="alice", notes="ok")
        assert dec.approved is True
        assert dec.reviewer == "alice"
        assert dec.notes == "ok"
        assert dec.timestamp is not None
        assert dec.timestamp.tzinfo is not None

    def test_defaults(self):
        dec = record_decision(approved=False)
        assert dec.approved is False
        assert dec.reviewer == ""
        assert dec.notes == ""
        assert dec.timestamp is not None


# -------------------------------------------------------------------
# build_validation_report
# -------------------------------------------------------------------

class TestBuildValidationReport:
    def test_counts(self):
        requests = [
            ValidationRequest(metric_id="m1", original_prompt="o", calibrated_prompt="c"),
            ValidationRequest(metric_id="m2", original_prompt="o", calibrated_prompt="c"),
            ValidationRequest(metric_id="m3", original_prompt="o", calibrated_prompt="c"),
        ]
        decisions = [
            ValidationDecision(approved=True),
            ValidationDecision(approved=False),
        ]
        report = build_validation_report(requests, decisions)
        assert report.approved_count == 1
        assert report.rejected_count == 1
        assert report.pending_count == 1

    def test_all_approved(self):
        requests = [
            ValidationRequest(metric_id="m1", original_prompt="o", calibrated_prompt="c"),
        ]
        decisions = [ValidationDecision(approved=True)]
        report = build_validation_report(requests, decisions)
        assert report.approved_count == 1
        assert report.rejected_count == 0
        assert report.pending_count == 0

    def test_no_decisions(self):
        requests = [
            ValidationRequest(metric_id="m1", original_prompt="o", calibrated_prompt="c"),
            ValidationRequest(metric_id="m2", original_prompt="o", calibrated_prompt="c"),
        ]
        report = build_validation_report(requests, [])
        assert report.approved_count == 0
        assert report.rejected_count == 0
        assert report.pending_count == 2

    def test_empty(self):
        report = build_validation_report([], [])
        assert report.approved_count == 0
        assert report.rejected_count == 0
        assert report.pending_count == 0

    def test_copies_lists(self):
        requests = [
            ValidationRequest(metric_id="m1", original_prompt="o", calibrated_prompt="c"),
        ]
        decisions = [ValidationDecision(approved=True)]
        report = build_validation_report(requests, decisions)
        requests.append(
            ValidationRequest(metric_id="m2", original_prompt="o", calibrated_prompt="c")
        )
        assert len(report.requests) == 1  # Not affected by mutation
