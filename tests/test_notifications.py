"""Tests for integration.notifications module."""

from __future__ import annotations

import pytest

from evalyn_sdk.integration.notifications import (
    NotificationConfig,
    NotificationMessage,
    NotificationResult,
    build_completion_message,
    build_failure_message,
    build_regression_message,
    format_discord_payload,
    format_slack_payload,
    should_notify,
    simulate_send,
)


# ---------------------------------------------------------------------------
# build_completion_message
# ---------------------------------------------------------------------------


class TestBuildCompletionMessage:
    def test_fields_present(self):
        msg = build_completion_message("run-1", 0.85, 100)
        assert msg.title == "Evaluation completed: run-1"
        assert "85.0%" in msg.body
        assert "100" in msg.body
        assert msg.fields["run_id"] == "run-1"
        assert msg.fields["pass_rate"] == "85.0%"
        assert msg.fields["total_items"] == "100"

    def test_severity_info_when_high_pass_rate(self):
        msg = build_completion_message("r", 0.9, 10)
        assert msg.severity == "info"

    def test_severity_warning_when_low_pass_rate(self):
        msg = build_completion_message("r", 0.3, 10)
        assert msg.severity == "warning"

    def test_duration_field_included(self):
        msg = build_completion_message("r", 0.8, 50, duration_ms=1234.0)
        assert "duration_ms" in msg.fields
        assert msg.fields["duration_ms"] == "1234"

    def test_duration_field_omitted_when_zero(self):
        msg = build_completion_message("r", 0.8, 50, duration_ms=0.0)
        assert "duration_ms" not in msg.fields

    def test_timestamp_populated(self):
        msg = build_completion_message("r", 1.0, 1)
        assert msg.timestamp != ""


# ---------------------------------------------------------------------------
# build_failure_message
# ---------------------------------------------------------------------------


class TestBuildFailureMessage:
    def test_fields_present(self):
        msg = build_failure_message("run-2", "timeout")
        assert msg.severity == "error"
        assert "timeout" in msg.body
        assert msg.fields["error"] == "timeout"
        assert msg.fields["run_id"] == "run-2"

    def test_metric_id_included(self):
        msg = build_failure_message("run-2", "bad score", metric_id="accuracy")
        assert msg.fields["metric_id"] == "accuracy"
        assert "accuracy" in msg.body

    def test_metric_id_omitted_when_empty(self):
        msg = build_failure_message("run-2", "err")
        assert "metric_id" not in msg.fields

    def test_timestamp_populated(self):
        msg = build_failure_message("r", "e")
        assert msg.timestamp != ""


# ---------------------------------------------------------------------------
# build_regression_message
# ---------------------------------------------------------------------------


class TestBuildRegressionMessage:
    def test_fields_present(self):
        msg = build_regression_message("run-3", ["accuracy", "safety"])
        assert "2 regression(s)" in msg.body
        assert msg.fields["regression_count"] == "2"
        assert "accuracy" in msg.fields["regressions"]

    def test_severity_warning_for_few(self):
        msg = build_regression_message("r", ["m1"])
        assert msg.severity == "warning"

    def test_severity_error_for_many(self):
        msg = build_regression_message("r", ["m1", "m2", "m3"])
        assert msg.severity == "error"

    def test_pass_rate_delta_included(self):
        msg = build_regression_message("r", ["m1"], pass_rate_delta=-0.15)
        assert "pass_rate_delta" in msg.fields
        assert "-15.0%" in msg.fields["pass_rate_delta"]

    def test_pass_rate_delta_omitted_when_zero(self):
        msg = build_regression_message("r", ["m1"], pass_rate_delta=0.0)
        assert "pass_rate_delta" not in msg.fields

    def test_empty_regressions_list(self):
        msg = build_regression_message("r", [])
        assert "0 regression(s)" in msg.body


# ---------------------------------------------------------------------------
# format_slack_payload
# ---------------------------------------------------------------------------


class TestFormatSlackPayload:
    def test_has_blocks(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(platform="slack")
        payload = format_slack_payload(msg, config)
        assert "blocks" in payload
        assert len(payload["blocks"]) >= 2

    def test_channel_included(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(channel="#evals")
        payload = format_slack_payload(msg, config)
        assert payload["channel"] == "#evals"

    def test_channel_omitted_when_empty(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig()
        payload = format_slack_payload(msg, config)
        assert "channel" not in payload

    def test_mention_on_failure(self):
        msg = build_failure_message("r", "err")
        config = NotificationConfig(mention_on_failure="@oncall")
        payload = format_slack_payload(msg, config)
        texts = [
            b["text"]["text"]
            for b in payload["blocks"]
            if b["type"] == "section" and "text" in b
        ]
        assert any("@oncall" in t for t in texts)

    def test_color_attachment(self):
        msg = build_failure_message("r", "err")
        config = NotificationConfig()
        payload = format_slack_payload(msg, config)
        assert "attachments" in payload
        assert payload["attachments"][0]["color"] == "#cc0000"


# ---------------------------------------------------------------------------
# format_discord_payload
# ---------------------------------------------------------------------------


class TestFormatDiscordPayload:
    def test_has_embeds(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(platform="discord")
        payload = format_discord_payload(msg, config)
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1

    def test_embed_fields(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(platform="discord")
        payload = format_discord_payload(msg, config)
        embed = payload["embeds"][0]
        assert "fields" in embed
        assert embed["title"] == msg.title

    def test_mention_on_failure_in_content(self):
        msg = build_failure_message("r", "err")
        config = NotificationConfig(mention_on_failure="@team")
        payload = format_discord_payload(msg, config)
        assert "@team" in payload["content"]

    def test_no_mention_for_info(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(mention_on_failure="@team")
        payload = format_discord_payload(msg, config)
        assert payload["content"] == ""


# ---------------------------------------------------------------------------
# should_notify
# ---------------------------------------------------------------------------


class TestShouldNotify:
    def test_matching_event(self):
        config = NotificationConfig(events=["completion", "failure"])
        assert should_notify("completion", config) is True

    def test_non_matching_event(self):
        config = NotificationConfig(events=["completion"])
        assert should_notify("regression", config) is False

    def test_empty_events(self):
        config = NotificationConfig(events=[])
        assert should_notify("completion", config) is False


# ---------------------------------------------------------------------------
# simulate_send
# ---------------------------------------------------------------------------


class TestSimulateSend:
    def test_sent_with_url(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(
            webhook_url="https://hooks.example.com/abc"
        )
        result = simulate_send(msg, config)
        assert result.sent is True
        assert result.platform == "slack"
        assert result.error == ""
        assert result.message_preview == msg.title

    def test_not_sent_without_url(self):
        msg = build_completion_message("r", 0.9, 10)
        config = NotificationConfig(webhook_url="")
        result = simulate_send(msg, config)
        assert result.sent is False
        assert "no webhook URL" in result.error


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_notification_config_from_dict(self):
        data = {
            "platform": "discord",
            "webhook_url": "https://discord.com/api/webhooks/x",
            "channel": "#alerts",
            "events": ["failure"],
            "mention_on_failure": "@here",
        }
        config = NotificationConfig.from_dict(data)
        assert config.platform == "discord"
        assert config.webhook_url == "https://discord.com/api/webhooks/x"
        assert config.events == ["failure"]

    def test_notification_config_from_dict_defaults(self):
        config = NotificationConfig.from_dict({})
        assert config.platform == "slack"
        assert config.webhook_url == ""
        assert len(config.events) == 3

    def test_notification_config_roundtrip(self):
        original = NotificationConfig(
            platform="discord",
            webhook_url="https://example.com",
            channel="#ch",
            events=["failure", "regression"],
            mention_on_failure="@team",
        )
        restored = NotificationConfig.from_dict(original.as_dict())
        assert restored.platform == original.platform
        assert restored.webhook_url == original.webhook_url
        assert restored.events == original.events

    def test_notification_result_as_dict(self):
        result = NotificationResult(
            sent=True, platform="slack", error="", message_preview="title"
        )
        d = result.as_dict()
        assert d["sent"] is True
        assert d["platform"] == "slack"
        assert d["message_preview"] == "title"

    def test_notification_message_as_dict(self):
        msg = NotificationMessage(
            title="t", body="b", severity="info", fields={"k": "v"}
        )
        d = msg.as_dict()
        assert d["title"] == "t"
        assert d["fields"] == {"k": "v"}
