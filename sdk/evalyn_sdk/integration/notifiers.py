"""Webhook notifiers for Slack, Discord, and generic JSON endpoints.

Renders evaluation results into webhook-shaped payloads and posts them with
light retry behavior. Uses only stdlib `urllib.request` so the SDK does not
pick up a `requests` dependency.

The format functions accept:
    run_summary: dict with keys {id, dataset_name, created_at, metrics}
        where metrics is {metric_id: {avg_score, pass_rate?, count?, ...}}
    breaches: list of breach-like objects exposing
        `.expression.metric_id`, `.expression.raw`, `.actual_score`
        (matches `evalyn_sdk.evaluation.guards.ThresholdBreach`).

Payload strings may contain Slack/Discord emoji shortcodes (e.g. ":warning:")
because the receiving platform renders them as content; that is not stylistic
emoji in our own code.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

VALID_FORMATS = ("slack", "discord", "generic")


def auto_detect_format(url: str) -> str:
    """Classify a webhook URL by hostname.

    - hooks.slack.com -> "slack"
    - discord.com / discordapp.com -> "discord"
    - everything else -> "generic"
    """
    if not isinstance(url, str) or not url.strip():
        return "generic"
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return "generic"
    if not host:
        return "generic"
    if "slack.com" in host:
        return "slack"
    if "discord.com" in host or "discordapp.com" in host:
        return "discord"
    return "generic"


# ---------------------------------------------------------------------------
# Payload formatters
# ---------------------------------------------------------------------------


def _breach_lines(breaches: list[Any]) -> list[str]:
    """Render breaches as a list of `metric op threshold (actual=N)` lines."""
    lines: list[str] = []
    for b in breaches:
        expr = getattr(b, "expression", None)
        raw = getattr(expr, "raw", None) or getattr(b, "raw", "?")
        actual = getattr(b, "actual_score", None)
        if isinstance(actual, (int, float)):
            lines.append(f"{raw} (actual={actual:.4f})")
        else:
            lines.append(f"{raw}")
    return lines


def _metric_lines(metrics: dict[str, Any], limit: int = 10) -> list[str]:
    """Render top-N metric lines for inclusion in a payload body."""
    if not metrics:
        return []
    items = list(metrics.items())[:limit]
    rendered: list[str] = []
    for metric_id, stats in items:
        if not isinstance(stats, dict):
            continue
        avg = stats.get("avg_score")
        pass_rate = stats.get("pass_rate")
        bits = [metric_id]
        if isinstance(avg, (int, float)):
            bits.append(f"avg={avg:.4f}")
        if isinstance(pass_rate, (int, float)):
            bits.append(f"pass_rate={pass_rate:.2%}")
        rendered.append(" ".join(bits))
    return rendered


def _summary_fields(run_summary: dict) -> dict[str, Any]:
    """Extract canonical fields from a run summary dict."""
    return {
        "run_id": str(run_summary.get("id", "unknown")),
        "dataset_name": str(run_summary.get("dataset_name", "unknown")),
        "created_at": str(run_summary.get("created_at", "")),
        "metrics": run_summary.get("metrics", {}) or {},
    }


def format_slack_message(
    run_summary: dict, breaches: list[Any], custom_message: str | None = None
) -> dict:
    """Build a Slack webhook payload using Block Kit.

    Slack renders `text` as the notification fallback (also used in
    push/notification previews), so it always carries a human summary.
    The `blocks` array provides the rich in-channel rendering.
    """
    fields = _summary_fields(run_summary)
    run_id = fields["run_id"]
    dataset = fields["dataset_name"]
    metrics = fields["metrics"]

    if breaches:
        header_text = f":rotating_light: Evalyn alert: {len(breaches)} threshold breach(es)"
        fallback = (
            f"Evalyn breach in run {run_id[:8]} on '{dataset}': "
            f"{len(breaches)} threshold(s) failed"
        )
    else:
        header_text = ":white_check_mark: Evalyn run completed"
        fallback = f"Evalyn run {run_id[:8]} on '{dataset}' completed"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header_text, "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Run*\n`{run_id}`"},
                {"type": "mrkdwn", "text": f"*Dataset*\n{dataset}"},
            ],
        },
    ]

    if custom_message:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": custom_message}}
        )

    breach_lines = _breach_lines(breaches)
    if breach_lines:
        body = "\n".join(f"- `{line}`" for line in breach_lines)
        # Slack enforces a 3000-char hard cap per `text` field
        # (https://api.slack.com/reference/block-kit/blocks#section).
        body = _truncate_slack_text(f"*Breaches*\n{body}")
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": body}}
        )

    metric_lines = _metric_lines(metrics)
    if metric_lines:
        body = "\n".join(f"- {line}" for line in metric_lines)
        body = _truncate_slack_text(f"*Metrics*\n{body}")
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": body}}
        )

    return {"text": fallback, "blocks": blocks}


# Slack docs: section block `text` field hard caps at 3000 chars and
# rejects the whole payload (HTTP 400) when exceeded. Cap at 2900 to
# leave headroom for the prefix label ("*Breaches*\n", etc).
_SLACK_TEXT_MAX = 2900


def _truncate_slack_text(text: str) -> str:
    if len(text) <= _SLACK_TEXT_MAX:
        return text
    return text[:_SLACK_TEXT_MAX - 3] + "..."


def format_discord_message(
    run_summary: dict, breaches: list[Any], custom_message: str | None = None
) -> dict:
    """Build a Discord webhook payload using an embed.

    Discord embeds carry a color tint: red for breaches, green when the
    run is clean. Top metrics ride in the embed fields.
    """
    fields = _summary_fields(run_summary)
    run_id = fields["run_id"]
    dataset = fields["dataset_name"]
    metrics = fields["metrics"]
    created_at = fields["created_at"]

    if breaches:
        title = f"Evalyn alert: {len(breaches)} threshold breach(es)"
        color = 15158332  # red
        content = (
            f":rotating_light: Eval run `{run_id[:8]}` on `{dataset}` "
            f"breached {len(breaches)} threshold(s)."
        )
    else:
        title = "Evalyn run completed"
        color = 3066993  # green
        content = (
            f":white_check_mark: Eval run `{run_id[:8]}` on `{dataset}` completed."
        )

    if custom_message:
        content = custom_message

    embed_fields: list[dict] = [
        {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
        {"name": "Dataset", "value": dataset, "inline": True},
    ]

    breach_lines = _breach_lines(breaches)
    if breach_lines:
        # Discord limits a single field value to 1024 chars; keep it short.
        text = "\n".join(f"- `{line}`" for line in breach_lines)
        if len(text) > 1000:
            text = text[:997] + "..."
        embed_fields.append({"name": "Breaches", "value": text, "inline": False})

    metric_lines = _metric_lines(metrics)
    if metric_lines:
        text = "\n".join(f"- {line}" for line in metric_lines)
        if len(text) > 1000:
            text = text[:997] + "..."
        embed_fields.append({"name": "Metrics", "value": text, "inline": False})

    embed: dict[str, Any] = {
        "title": title,
        "color": color,
        "fields": embed_fields,
    }
    if created_at:
        embed["timestamp"] = created_at

    return {"content": content, "embeds": [embed]}


def format_generic_message(
    run_summary: dict, breaches: list[Any], custom_message: str | None = None
) -> dict:
    """Build a raw JSON payload usable by any custom webhook endpoint.

    Shape:
        {
            "run_id": str,
            "dataset_name": str,
            "created_at": str,
            "breached": bool,
            "breaches": [
                {"expression": "<raw>", "metric_id": str, "actual_score": float}
            ],
            "metrics": {metric_id: {avg_score, pass_rate, ...}},
            "message": <optional custom_message>
        }
    """
    fields = _summary_fields(run_summary)
    breach_list: list[dict[str, Any]] = []
    for b in breaches:
        expr = getattr(b, "expression", None)
        breach_list.append(
            {
                "expression": getattr(expr, "raw", None) or getattr(b, "raw", ""),
                "metric_id": getattr(expr, "metric_id", None)
                or getattr(b, "metric_id", ""),
                "threshold": getattr(expr, "threshold", None),
                "op": getattr(expr, "op", None),
                "actual_score": getattr(b, "actual_score", None),
            }
        )

    payload: dict[str, Any] = {
        "run_id": fields["run_id"],
        "dataset_name": fields["dataset_name"],
        "created_at": fields["created_at"],
        "breached": bool(breaches),
        "breaches": breach_list,
        "metrics": fields["metrics"],
    }
    if custom_message:
        payload["message"] = custom_message
    return payload


# ---------------------------------------------------------------------------
# HTTP delivery
# ---------------------------------------------------------------------------


@dataclass
class WebhookResponse:
    """Outcome of a `send_webhook` call.

    `status` is None when the request never produced an HTTP status code
    (DNS failure, connection refused, etc.). `attempts` reflects the
    number of POSTs actually issued, including the final one.
    """

    status: int | None
    body: str
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


def send_webhook(
    url: str,
    payload: dict,
    *,
    timeout: int = 10,
    max_attempts: int = 3,
    backoff: float = 0.5,
) -> WebhookResponse:
    """POST `payload` as JSON to `url` with retry on transient failure.

    Retries on:
    - HTTP 5xx responses (server-side temporary failure)
    - URLError (network/DNS/connection failures)

    Does NOT retry on 4xx responses; the payload itself is rejected and
    retrying will not help. The final response (success or last failure)
    is returned to the caller for logging.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("send_webhook requires a non-empty URL")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "evalyn-notify/1.0",
    }
    last: WebhookResponse | None = None

    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    status = resp.getcode()
                resp_body = resp.read().decode("utf-8", errors="replace")
                last = WebhookResponse(
                    status=status, body=resp_body, attempts=attempt
                )
                if 200 <= status < 300:
                    return last
                # 3xx returned by urlopen is rare; treat as final.
                return last
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            last = WebhookResponse(
                status=exc.code,
                body=err_body,
                attempts=attempt,
                error=f"HTTP {exc.code}: {exc.reason}",
            )
            if 500 <= exc.code < 600 and attempt < max_attempts:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            # 4xx or last attempt: stop.
            return last
        except urllib.error.URLError as exc:
            # Do NOT format `exc.reason` directly: Python sometimes embeds the
            # full URL in DNS-failure reasons, leaking the webhook URL (a
            # secret for Slack/Discord) into stderr / CI logs.
            reason_class = type(exc.reason).__name__ if hasattr(exc, "reason") else "URLError"
            last = WebhookResponse(
                status=None,
                body="",
                attempts=attempt,
                error=f"URLError ({reason_class}): connection failed",
            )
            if attempt < max_attempts:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            return last

    # Defensive fallback; the loop above always returns or assigns `last`.
    return last or WebhookResponse(
        status=None, body="", attempts=0, error="No attempt made"
    )


__all__ = [
    "VALID_FORMATS",
    "WebhookResponse",
    "auto_detect_format",
    "format_discord_message",
    "format_generic_message",
    "format_slack_message",
    "send_webhook",
]
