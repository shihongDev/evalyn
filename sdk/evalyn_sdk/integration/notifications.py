"""
Slack/Discord notification support for evaluation events.

Builds notification messages for completion, failure, and regression events,
formats them as platform-specific payloads, and provides simulated sending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class NotificationConfig:
    """Configuration for notification delivery."""

    platform: str = "slack"  # "slack", "discord", "generic"
    webhook_url: str = ""
    channel: str = ""
    events: list[str] = field(
        default_factory=lambda: ["completion", "failure", "regression"]
    )
    mention_on_failure: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "webhook_url": self.webhook_url,
            "channel": self.channel,
            "events": list(self.events),
            "mention_on_failure": self.mention_on_failure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotificationConfig:
        return cls(
            platform=data.get("platform", "slack"),
            webhook_url=data.get("webhook_url", ""),
            channel=data.get("channel", ""),
            events=data.get("events", ["completion", "failure", "regression"]),
            mention_on_failure=data.get("mention_on_failure", ""),
        )


@dataclass
class NotificationMessage:
    """A notification message ready for formatting and delivery."""

    title: str
    body: str
    severity: str = "info"  # "info", "warning", "error"
    fields: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "severity": self.severity,
            "fields": dict(self.fields),
            "timestamp": self.timestamp,
        }


@dataclass
class NotificationResult:
    """Result of a notification send attempt."""

    sent: bool
    platform: str
    error: str = ""
    message_preview: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "platform": self.platform,
            "error": self.error,
            "message_preview": self.message_preview,
        }


# ---------------------------------------------------------------------------
# Message Builders
# ---------------------------------------------------------------------------

_SEVERITY_MAP = {
    "info": "info",
    "warning": "warning",
    "error": "error",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_completion_message(
    run_id: str,
    pass_rate: float,
    total_items: int,
    duration_ms: float = 0.0,
) -> NotificationMessage:
    """Build a notification for evaluation completion."""
    fields: dict[str, str] = {
        "run_id": run_id,
        "pass_rate": f"{pass_rate:.1%}",
        "total_items": str(total_items),
    }
    if duration_ms > 0:
        fields["duration_ms"] = f"{duration_ms:.0f}"

    severity = "info" if pass_rate >= 0.5 else "warning"

    return NotificationMessage(
        title=f"Evaluation completed: {run_id}",
        body=f"Pass rate {pass_rate:.1%} across {total_items} items",
        severity=severity,
        fields=fields,
        timestamp=_now_iso(),
    )


def build_failure_message(
    run_id: str,
    error: str,
    metric_id: str = "",
) -> NotificationMessage:
    """Build a notification for evaluation failure."""
    fields: dict[str, str] = {
        "run_id": run_id,
        "error": error,
    }
    if metric_id:
        fields["metric_id"] = metric_id

    body = f"Evaluation failed: {error}"
    if metric_id:
        body = f"Metric {metric_id} failed: {error}"

    return NotificationMessage(
        title=f"Evaluation failure: {run_id}",
        body=body,
        severity="error",
        fields=fields,
        timestamp=_now_iso(),
    )


def build_regression_message(
    run_id: str,
    regressions: list[str],
    pass_rate_delta: float = 0.0,
) -> NotificationMessage:
    """Build a notification for detected regressions."""
    count = len(regressions)
    fields: dict[str, str] = {
        "run_id": run_id,
        "regression_count": str(count),
        "regressions": ", ".join(regressions),
    }
    if pass_rate_delta != 0.0:
        fields["pass_rate_delta"] = f"{pass_rate_delta:+.1%}"

    severity = "error" if count >= 3 else "warning"

    body_parts = [f"{count} regression(s) detected"]
    if regressions:
        body_parts.append(f"affected metrics: {', '.join(regressions)}")
    if pass_rate_delta != 0.0:
        body_parts.append(f"overall delta: {pass_rate_delta:+.1%}")

    return NotificationMessage(
        title=f"Regressions detected: {run_id}",
        body="; ".join(body_parts),
        severity=severity,
        fields=fields,
        timestamp=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Payload Formatters
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {
    "info": "#36a64f",
    "warning": "#ff9900",
    "error": "#cc0000",
}

_DISCORD_SEVERITY_COLORS = {
    "info": 0x36A64F,
    "warning": 0xFF9900,
    "error": 0xCC0000,
}


def format_slack_payload(
    message: NotificationMessage,
    config: NotificationConfig,
) -> dict[str, Any]:
    """Format a notification as a Slack webhook payload with blocks."""
    color = _SEVERITY_COLORS.get(message.severity, "#36a64f")

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": message.title},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": message.body},
        },
    ]

    if message.fields:
        field_elements = []
        for key, value in message.fields.items():
            field_elements.append(
                {"type": "mrkdwn", "text": f"*{key}:* {value}"}
            )
        blocks.append({"type": "section", "fields": field_elements})

    if config.mention_on_failure and message.severity == "error":
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"cc {config.mention_on_failure}",
                },
            }
        )

    payload: dict[str, Any] = {
        "blocks": blocks,
        "attachments": [{"color": color}],
    }

    if config.channel:
        payload["channel"] = config.channel

    return payload


def format_discord_payload(
    message: NotificationMessage,
    config: NotificationConfig,
) -> dict[str, Any]:
    """Format a notification as a Discord webhook payload with embeds."""
    color = _DISCORD_SEVERITY_COLORS.get(message.severity, 0x36A64F)

    embed_fields: list[dict[str, Any]] = []
    for key, value in message.fields.items():
        embed_fields.append({"name": key, "value": value, "inline": True})

    embed: dict[str, Any] = {
        "title": message.title,
        "description": message.body,
        "color": color,
    }

    if embed_fields:
        embed["fields"] = embed_fields

    if message.timestamp:
        embed["timestamp"] = message.timestamp

    content = ""
    if config.mention_on_failure and message.severity == "error":
        content = f"cc {config.mention_on_failure}"

    return {
        "content": content,
        "embeds": [embed],
    }


# ---------------------------------------------------------------------------
# Event Filtering and Simulated Send
# ---------------------------------------------------------------------------


def should_notify(event_type: str, config: NotificationConfig) -> bool:
    """Check if event type is in config.events."""
    return event_type in config.events


def simulate_send(
    message: NotificationMessage,
    config: NotificationConfig,
) -> NotificationResult:
    """Simulate sending a notification (no actual HTTP).

    Marks as sent if webhook_url is non-empty. Returns error otherwise.
    """
    if not config.webhook_url:
        return NotificationResult(
            sent=False,
            platform=config.platform,
            error="no webhook URL configured",
            message_preview=message.title,
        )

    return NotificationResult(
        sent=True,
        platform=config.platform,
        error="",
        message_preview=message.title,
    )
