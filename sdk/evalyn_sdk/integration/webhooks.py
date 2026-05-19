"""
Webhook notifications: trigger HTTP webhooks on eval events.

Provides dataclasses and pure functions for creating, signing, filtering,
and simulating webhook deliveries. No actual HTTP calls are made - the
simulate_delivery function produces a WebhookDelivery record.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""

    url: str = ""
    events: list[str] = field(default_factory=lambda: ["completion"])
    secret: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    retry_count: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "events": list(self.events),
            "secret": self.secret,
            "headers": dict(self.headers),
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookConfig:
        return cls(
            url=data.get("url", ""),
            events=data.get("events", ["completion"]),
            secret=data.get("secret", ""),
            headers=data.get("headers", {}),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            retry_count=data.get("retry_count", 2),
        )


@dataclass
class WebhookEvent:
    """An event that may trigger a webhook delivery."""

    event_type: str = ""  # "completion" / "failure" / "regression" / "improvement"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookEvent:
        return cls(
            event_type=data.get("event_type", ""),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class WebhookDelivery:
    """Record of a single webhook delivery attempt."""

    event: WebhookEvent = field(default_factory=WebhookEvent)
    url: str = ""
    status: str = "pending"  # "pending" / "delivered" / "failed"
    response_code: int = 0
    error: str = ""
    attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.as_dict(),
            "url": self.url,
            "status": self.status,
            "response_code": self.response_code,
            "error": self.error,
            "attempts": self.attempts,
        }


@dataclass
class WebhookReport:
    """Aggregate report of webhook deliveries."""

    deliveries: list[WebhookDelivery] = field(default_factory=list)
    total_sent: int = 0
    total_failed: int = 0
    total_delivered: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "deliveries": [d.as_dict() for d in self.deliveries],
            "total_sent": self.total_sent,
            "total_failed": self.total_failed,
            "total_delivered": self.total_delivered,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookReport:
        deliveries = []
        for d in data.get("deliveries", []):
            event = WebhookEvent.from_dict(d.get("event", {}))
            deliveries.append(
                WebhookDelivery(
                    event=event,
                    url=d.get("url", ""),
                    status=d.get("status", "pending"),
                    response_code=d.get("response_code", 0),
                    error=d.get("error", ""),
                    attempts=d.get("attempts", 0),
                )
            )
        return cls(
            deliveries=deliveries,
            total_sent=data.get("total_sent", 0),
            total_failed=data.get("total_failed", 0),
            total_delivered=data.get("total_delivered", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Webhook Report")
        lines.append("-" * 40)
        lines.append(f"  Total sent: {self.total_sent}")
        lines.append(f"  Delivered: {self.total_delivered}")
        lines.append(f"  Failed: {self.total_failed}")
        if self.deliveries:
            lines.append("")
            lines.append("  Deliveries:")
            for d in self.deliveries:
                lines.append(
                    f"    [{d.status}] {d.event.event_type} -> {d.url}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_event(
    event_type: str, payload: dict[str, Any] | None = None
) -> WebhookEvent:
    """Create a webhook event with the current UTC timestamp."""
    return WebhookEvent(
        event_type=event_type,
        payload=payload if payload is not None else {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def should_send(event: WebhookEvent, config: WebhookConfig) -> bool:
    """Check if the event type matches one in the config's event list."""
    return event.event_type in config.events


def sign_payload(payload: str, secret: str) -> str:
    """Compute HMAC-SHA256 of payload using secret. Return hex digest."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_delivery_payload(event: WebhookEvent) -> str:
    """JSON-serialize an event for delivery."""
    return json.dumps(event.as_dict(), sort_keys=True)


def simulate_delivery(
    event: WebhookEvent, config: WebhookConfig
) -> WebhookDelivery:
    """Simulate a webhook delivery (no actual HTTP).

    Marks as 'delivered' if the URL is non-empty, otherwise 'failed'.
    """
    if config.url:
        return WebhookDelivery(
            event=event,
            url=config.url,
            status="delivered",
            response_code=200,
            attempts=1,
        )
    return WebhookDelivery(
        event=event,
        url="",
        status="failed",
        response_code=0,
        error="empty URL",
        attempts=1,
    )


def build_webhook_report(deliveries: list[WebhookDelivery]) -> WebhookReport:
    """Aggregate a list of deliveries into a WebhookReport."""
    total_sent = len(deliveries)
    total_delivered = sum(1 for d in deliveries if d.status == "delivered")
    total_failed = sum(1 for d in deliveries if d.status == "failed")
    return WebhookReport(
        deliveries=list(deliveries),
        total_sent=total_sent,
        total_failed=total_failed,
        total_delivered=total_delivered,
    )
