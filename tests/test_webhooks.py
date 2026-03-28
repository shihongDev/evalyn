"""
Tests for evalyn_sdk.integration.webhooks.

Run with: uv run pytest tests/test_webhooks.py -v
"""

from __future__ import annotations

import json

import pytest

from evalyn_sdk.integration.webhooks import (
    WebhookConfig,
    WebhookDelivery,
    WebhookEvent,
    WebhookReport,
    build_delivery_payload,
    build_webhook_report,
    create_event,
    should_send,
    sign_payload,
    simulate_delivery,
)


# ---------------------------------------------------------------------------
# create_event
# ---------------------------------------------------------------------------


def test_create_event_has_timestamp():
    event = create_event("completion")
    assert event.timestamp != ""
    assert "T" in event.timestamp  # ISO format


def test_create_event_type():
    event = create_event("failure")
    assert event.event_type == "failure"


def test_create_event_default_payload():
    event = create_event("completion")
    assert event.payload == {}


def test_create_event_custom_payload():
    event = create_event("regression", {"metric": "accuracy", "delta": -0.05})
    assert event.payload["metric"] == "accuracy"
    assert event.payload["delta"] == -0.05


# ---------------------------------------------------------------------------
# should_send
# ---------------------------------------------------------------------------


def test_should_send_matching():
    cfg = WebhookConfig(url="https://example.com", events=["completion", "failure"])
    event = create_event("completion")
    assert should_send(event, cfg) is True


def test_should_send_non_matching():
    cfg = WebhookConfig(url="https://example.com", events=["completion"])
    event = create_event("regression")
    assert should_send(event, cfg) is False


def test_should_send_empty_events():
    cfg = WebhookConfig(url="https://example.com", events=[])
    event = create_event("completion")
    assert should_send(event, cfg) is False


# ---------------------------------------------------------------------------
# sign_payload
# ---------------------------------------------------------------------------


def test_sign_payload_deterministic():
    sig1 = sign_payload("data", "secret")
    sig2 = sign_payload("data", "secret")
    assert sig1 == sig2


def test_sign_payload_different_secret():
    sig1 = sign_payload("data", "secret1")
    sig2 = sign_payload("data", "secret2")
    assert sig1 != sig2


def test_sign_payload_different_data():
    sig1 = sign_payload("data_a", "secret")
    sig2 = sign_payload("data_b", "secret")
    assert sig1 != sig2


def test_sign_payload_hex_format():
    sig = sign_payload("test", "key")
    assert all(c in "0123456789abcdef" for c in sig)
    assert len(sig) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# build_delivery_payload
# ---------------------------------------------------------------------------


def test_build_delivery_payload_valid_json():
    event = create_event("completion", {"run_id": "abc"})
    payload = build_delivery_payload(event)
    parsed = json.loads(payload)
    assert parsed["event_type"] == "completion"
    assert parsed["payload"]["run_id"] == "abc"


def test_build_delivery_payload_sorted_keys():
    event = create_event("failure")
    payload = build_delivery_payload(event)
    parsed = json.loads(payload)
    keys = list(parsed.keys())
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# simulate_delivery
# ---------------------------------------------------------------------------


def test_simulate_delivery_marks_delivered():
    cfg = WebhookConfig(url="https://hooks.example.com/eval")
    event = create_event("completion")
    delivery = simulate_delivery(event, cfg)
    assert delivery.status == "delivered"
    assert delivery.response_code == 200
    assert delivery.attempts == 1
    assert delivery.url == cfg.url


def test_simulate_delivery_empty_url_marks_failed():
    cfg = WebhookConfig(url="")
    event = create_event("completion")
    delivery = simulate_delivery(event, cfg)
    assert delivery.status == "failed"
    assert delivery.error == "empty URL"
    assert delivery.attempts == 1


def test_simulate_delivery_preserves_event():
    cfg = WebhookConfig(url="https://example.com")
    event = create_event("regression", {"delta": -0.1})
    delivery = simulate_delivery(event, cfg)
    assert delivery.event.event_type == "regression"
    assert delivery.event.payload["delta"] == -0.1


# ---------------------------------------------------------------------------
# build_webhook_report
# ---------------------------------------------------------------------------


def test_build_webhook_report_counts():
    cfg_ok = WebhookConfig(url="https://example.com")
    cfg_fail = WebhookConfig(url="")
    d1 = simulate_delivery(create_event("completion"), cfg_ok)
    d2 = simulate_delivery(create_event("failure"), cfg_ok)
    d3 = simulate_delivery(create_event("regression"), cfg_fail)
    report = build_webhook_report([d1, d2, d3])
    assert report.total_sent == 3
    assert report.total_delivered == 2
    assert report.total_failed == 1


def test_build_webhook_report_empty():
    report = build_webhook_report([])
    assert report.total_sent == 0
    assert report.total_delivered == 0
    assert report.total_failed == 0


# ---------------------------------------------------------------------------
# WebhookConfig
# ---------------------------------------------------------------------------


def test_webhook_config_defaults():
    cfg = WebhookConfig()
    assert cfg.url == ""
    assert cfg.events == ["completion"]
    assert cfg.secret == ""
    assert cfg.timeout_seconds == 10.0
    assert cfg.retry_count == 2


def test_webhook_config_as_dict():
    cfg = WebhookConfig(url="https://x.com", secret="s3c")
    d = cfg.as_dict()
    assert d["url"] == "https://x.com"
    assert d["secret"] == "s3c"


def test_webhook_config_from_dict():
    d = {
        "url": "https://hooks.io",
        "events": ["failure", "regression"],
        "secret": "abc",
        "timeout_seconds": 5.0,
    }
    cfg = WebhookConfig.from_dict(d)
    assert cfg.url == "https://hooks.io"
    assert cfg.events == ["failure", "regression"]
    assert cfg.secret == "abc"
    assert cfg.timeout_seconds == 5.0
    assert cfg.retry_count == 2  # default


def test_webhook_config_from_dict_empty():
    cfg = WebhookConfig.from_dict({})
    assert cfg.url == ""
    assert cfg.events == ["completion"]


def test_webhook_config_roundtrip():
    cfg = WebhookConfig(
        url="https://x.com",
        events=["failure"],
        secret="key",
        headers={"X-Custom": "val"},
        timeout_seconds=3.0,
        retry_count=5,
    )
    restored = WebhookConfig.from_dict(cfg.as_dict())
    assert restored.url == cfg.url
    assert restored.events == cfg.events
    assert restored.secret == cfg.secret
    assert restored.headers == cfg.headers
    assert restored.timeout_seconds == cfg.timeout_seconds
    assert restored.retry_count == cfg.retry_count


# ---------------------------------------------------------------------------
# WebhookEvent
# ---------------------------------------------------------------------------


def test_webhook_event_as_dict():
    e = WebhookEvent(event_type="completion", payload={"x": 1}, timestamp="t")
    d = e.as_dict()
    assert d["event_type"] == "completion"
    assert d["payload"] == {"x": 1}


def test_webhook_event_from_dict():
    d = {"event_type": "failure", "payload": {"err": "timeout"}, "timestamp": "ts"}
    e = WebhookEvent.from_dict(d)
    assert e.event_type == "failure"
    assert e.payload["err"] == "timeout"


def test_webhook_event_roundtrip():
    e = WebhookEvent(event_type="regression", payload={"d": -0.1}, timestamp="now")
    restored = WebhookEvent.from_dict(e.as_dict())
    assert restored.event_type == e.event_type
    assert restored.payload == e.payload
    assert restored.timestamp == e.timestamp


# ---------------------------------------------------------------------------
# WebhookDelivery
# ---------------------------------------------------------------------------


def test_webhook_delivery_as_dict():
    e = WebhookEvent(event_type="completion")
    d = WebhookDelivery(event=e, url="https://x.com", status="delivered")
    result = d.as_dict()
    assert result["status"] == "delivered"
    assert result["event"]["event_type"] == "completion"


def test_webhook_delivery_defaults():
    d = WebhookDelivery()
    assert d.status == "pending"
    assert d.response_code == 0
    assert d.attempts == 0


# ---------------------------------------------------------------------------
# WebhookReport
# ---------------------------------------------------------------------------


def test_webhook_report_format_text():
    cfg = WebhookConfig(url="https://example.com")
    d1 = simulate_delivery(create_event("completion"), cfg)
    report = build_webhook_report([d1])
    text = report.format_text()
    assert "Webhook Report" in text
    assert "Delivered: 1" in text
    assert "Total sent: 1" in text


def test_webhook_report_format_text_empty():
    report = build_webhook_report([])
    text = report.format_text()
    assert "Total sent: 0" in text


def test_webhook_report_as_dict():
    report = WebhookReport(total_sent=2, total_delivered=1, total_failed=1)
    d = report.as_dict()
    assert d["total_sent"] == 2
    assert d["total_delivered"] == 1


def test_webhook_report_from_dict():
    d = {
        "deliveries": [
            {
                "event": {"event_type": "completion", "payload": {}, "timestamp": "t"},
                "url": "https://x.com",
                "status": "delivered",
                "response_code": 200,
                "error": "",
                "attempts": 1,
            }
        ],
        "total_sent": 1,
        "total_failed": 0,
        "total_delivered": 1,
    }
    report = WebhookReport.from_dict(d)
    assert report.total_sent == 1
    assert len(report.deliveries) == 1
    assert report.deliveries[0].status == "delivered"


def test_webhook_report_roundtrip():
    cfg = WebhookConfig(url="https://x.com")
    d1 = simulate_delivery(create_event("completion"), cfg)
    original = build_webhook_report([d1])
    restored = WebhookReport.from_dict(original.as_dict())
    assert restored.total_sent == original.total_sent
    assert restored.total_delivered == original.total_delivered
    assert len(restored.deliveries) == len(original.deliveries)
