"""Tests for `evalyn notify`: format helpers, webhook delivery, and CLI dispatch.

Covers:
- format_slack_message / format_discord_message / format_generic_message
  produce the expected payload shape from a run summary and breaches.
- auto_detect_format classifies Slack/Discord/generic URLs.
- send_webhook retries on 5xx and URLError, does not retry on 4xx, and
  reports status to the caller.
- CLI dispatch: --threshold breach triggers a send, no breach + no --always
  skips the send, and --always sends regardless.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# conftest already inserts sdk/ into sys.path.
from evalyn_sdk.evaluation.guards import (
    ThresholdBreach,
    parse_threshold_expression,
)
from evalyn_sdk.integration.notifiers import (
    WebhookResponse,
    auto_detect_format,
    format_discord_message,
    format_generic_message,
    format_slack_message,
    send_webhook,
)

from conftest import run_cli


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_run_summary() -> dict:
    """Build a sample run summary dict (matches what _build_run_summary emits)."""
    return {
        "id": "run_abc12345",
        "dataset_name": "test_dataset",
        "created_at": "2026-05-23T10:00:00+00:00",
        "metrics": {
            "helpfulness": {
                "count": 10,
                "avg_score": 0.75,
                "pass_rate": 0.7,
            },
            "safety": {
                "count": 10,
                "avg_score": 0.95,
                "pass_rate": 1.0,
            },
        },
    }


def _make_breach(metric_id: str, op: str, threshold: float, actual: float) -> ThresholdBreach:
    expr = parse_threshold_expression(f"{metric_id}{op}{threshold}")
    return ThresholdBreach(expression=expr, actual_score=actual)


# ---------------------------------------------------------------------------
# auto_detect_format
# ---------------------------------------------------------------------------


class TestAutoDetectFormat:
    """URL host classification picks the right format."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://hooks.slack.com/services/T00/B00/abc", "slack"),
            ("https://discord.com/api/webhooks/123/abc", "discord"),
            ("https://discordapp.com/api/webhooks/123/abc", "discord"),
            ("https://example.com/webhook", "generic"),
            ("https://my-server.io/notify", "generic"),
            ("", "generic"),
            ("   ", "generic"),
            ("not-a-url", "generic"),
        ],
    )
    def test_classification(self, url, expected):
        assert auto_detect_format(url) == expected


# ---------------------------------------------------------------------------
# format_slack_message
# ---------------------------------------------------------------------------


class TestFormatSlackMessage:
    """Slack payloads carry text fallback + Block Kit blocks."""

    def test_breach_payload_shape(self):
        summary = _make_run_summary()
        breaches = [_make_breach("helpfulness", "<", 0.8, 0.75)]
        payload = format_slack_message(summary, breaches)

        assert isinstance(payload, dict)
        assert "text" in payload and "Evalyn" in payload["text"]
        assert "blocks" in payload and isinstance(payload["blocks"], list)

        # Must contain a header block and a section with breach details.
        block_types = [b["type"] for b in payload["blocks"]]
        assert "header" in block_types
        assert "section" in block_types

        # Breach text appears somewhere in a section.
        rendered = json.dumps(payload)
        assert "helpfulness<0.8" in rendered
        assert "0.7500" in rendered

    def test_clean_run_payload(self):
        summary = _make_run_summary()
        payload = format_slack_message(summary, [])

        assert "completed" in payload["text"].lower()
        rendered = json.dumps(payload)
        # No breach text present.
        assert "Breaches" not in rendered

    def test_custom_message_included(self):
        summary = _make_run_summary()
        payload = format_slack_message(summary, [], custom_message="hello team")
        rendered = json.dumps(payload)
        assert "hello team" in rendered

    def test_print_sample_for_reporting(self, capsys):
        """Capture a sample Slack payload to attach to the task report."""
        summary = _make_run_summary()
        breaches = [
            _make_breach("helpfulness", "<", 0.8, 0.75),
            _make_breach("latency_p95", ">", 1000, 1250.5),
        ]
        payload = format_slack_message(summary, breaches)
        print("\n--- SAMPLE SLACK PAYLOAD ---")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("--- END SAMPLE ---")
        # Sanity: the print succeeded and payload is non-trivial.
        assert len(payload["blocks"]) >= 3


# ---------------------------------------------------------------------------
# format_discord_message
# ---------------------------------------------------------------------------


class TestFormatDiscordMessage:
    """Discord payloads carry content + a single embed."""

    def test_breach_payload_shape(self):
        summary = _make_run_summary()
        breaches = [_make_breach("helpfulness", "<", 0.8, 0.75)]
        payload = format_discord_message(summary, breaches)

        assert isinstance(payload, dict)
        assert "content" in payload
        assert "embeds" in payload and len(payload["embeds"]) == 1
        embed = payload["embeds"][0]
        assert embed["color"] == 15158332  # red
        assert "Breaches" in [f["name"] for f in embed["fields"]]

    def test_clean_run_payload(self):
        summary = _make_run_summary()
        payload = format_discord_message(summary, [])
        embed = payload["embeds"][0]
        assert embed["color"] == 3066993  # green
        # No breach field when there are no breaches.
        assert "Breaches" not in [f["name"] for f in embed["fields"]]

    def test_custom_message_overrides_content(self):
        summary = _make_run_summary()
        payload = format_discord_message(
            summary, [], custom_message="custom alert text"
        )
        assert payload["content"] == "custom alert text"

    def test_print_sample_for_reporting(self):
        """Capture a sample Discord payload to attach to the task report."""
        summary = _make_run_summary()
        breaches = [
            _make_breach("helpfulness", "<", 0.8, 0.75),
            _make_breach("latency_p95", ">", 1000, 1250.5),
        ]
        payload = format_discord_message(summary, breaches)
        print("\n--- SAMPLE DISCORD PAYLOAD ---")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("--- END SAMPLE ---")
        assert payload["embeds"][0]["title"]


# ---------------------------------------------------------------------------
# format_generic_message
# ---------------------------------------------------------------------------


class TestFormatGenericMessage:
    """Generic payload exposes raw fields for custom webhooks."""

    def test_payload_shape(self):
        summary = _make_run_summary()
        breaches = [_make_breach("helpfulness", "<", 0.8, 0.75)]
        payload = format_generic_message(summary, breaches)

        assert payload["run_id"] == "run_abc12345"
        assert payload["dataset_name"] == "test_dataset"
        assert payload["breached"] is True
        assert len(payload["breaches"]) == 1
        breach = payload["breaches"][0]
        assert breach["metric_id"] == "helpfulness"
        assert breach["op"] == "<"
        assert breach["threshold"] == 0.8
        assert breach["actual_score"] == 0.75
        assert "metrics" in payload and "helpfulness" in payload["metrics"]

    def test_no_breaches(self):
        summary = _make_run_summary()
        payload = format_generic_message(summary, [])
        assert payload["breached"] is False
        assert payload["breaches"] == []

    def test_custom_message_appended(self):
        summary = _make_run_summary()
        payload = format_generic_message(summary, [], custom_message="x")
        assert payload["message"] == "x"


# ---------------------------------------------------------------------------
# send_webhook
# ---------------------------------------------------------------------------


def _make_http_error(code: int, body: str = "error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg=f"HTTP {code}",
        hdrs={},
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _make_success_resp(body: str = '{"ok": true}', status: int = 200):
    resp = MagicMock()
    resp.read.return_value = body.encode("utf-8")
    resp.status = status
    resp.getcode.return_value = status
    resp.__enter__ = lambda self: self
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestSendWebhook:
    """send_webhook retries on transient errors, not on client errors."""

    @patch("evalyn_sdk.integration.notifiers.urllib.request.urlopen")
    def test_success_no_retry(self, mock_urlopen):
        mock_urlopen.return_value = _make_success_resp()

        result = send_webhook("https://example.com/hook", {"a": 1})

        assert result.ok is True
        assert result.status == 200
        assert result.attempts == 1
        assert mock_urlopen.call_count == 1

    @patch("evalyn_sdk.integration.notifiers.time.sleep")
    @patch("evalyn_sdk.integration.notifiers.urllib.request.urlopen")
    def test_retry_on_500_then_success(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            _make_http_error(500, "boom"),
            _make_http_error(500, "boom"),
            _make_success_resp(),
        ]

        result = send_webhook("https://example.com/hook", {"a": 1})

        assert result.ok is True
        assert result.attempts == 3
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("evalyn_sdk.integration.notifiers.time.sleep")
    @patch("evalyn_sdk.integration.notifiers.urllib.request.urlopen")
    def test_exhaust_retries_on_persistent_5xx(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            _make_http_error(503, "down"),
            _make_http_error(503, "down"),
            _make_http_error(503, "down"),
        ]

        result = send_webhook("https://example.com/hook", {"a": 1})

        assert result.ok is False
        assert result.status == 503
        assert result.attempts == 3
        assert mock_urlopen.call_count == 3

    @patch("evalyn_sdk.integration.notifiers.time.sleep")
    @patch("evalyn_sdk.integration.notifiers.urllib.request.urlopen")
    def test_no_retry_on_4xx(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [_make_http_error(400, "bad request")]

        result = send_webhook("https://example.com/hook", {"a": 1})

        assert result.ok is False
        assert result.status == 400
        assert result.attempts == 1
        assert mock_urlopen.call_count == 1
        # 4xx should not sleep before giving up.
        assert mock_sleep.call_count == 0

    @patch("evalyn_sdk.integration.notifiers.time.sleep")
    @patch("evalyn_sdk.integration.notifiers.urllib.request.urlopen")
    def test_retry_on_urlerror_then_success(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            urllib.error.URLError("connection refused"),
            _make_success_resp(),
        ]

        result = send_webhook("https://example.com/hook", {"a": 1})

        assert result.ok is True
        assert result.attempts == 2
        assert mock_sleep.call_count == 1

    def test_empty_url_raises(self):
        with pytest.raises(ValueError):
            send_webhook("", {"a": 1})

    def test_max_attempts_validated(self):
        with pytest.raises(ValueError):
            send_webhook("https://example.com", {"a": 1}, max_attempts=0)


# ---------------------------------------------------------------------------
# CLI dispatch via subprocess
# ---------------------------------------------------------------------------


def _seed_eval_run_dir(tmp_path: Path, breaching: bool) -> Path:
    """Create a dataset dir with a single eval_runs/<id>/results.json file.

    Returns the dataset directory. When `breaching` is True, helpfulness
    avg_score is 0.6 so `helpfulness<0.8` breaches.
    """
    dataset_dir = tmp_path / "ds"
    runs_dir = dataset_dir / "eval_runs" / "run_test001"
    runs_dir.mkdir(parents=True)
    (dataset_dir / "dataset.jsonl").write_text("", encoding="utf-8")

    avg = 0.6 if breaching else 0.95
    run_data = {
        "id": "run_test001",
        "dataset_name": "ds",
        "created_at": datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
        "metric_results": [
            {
                "metric_id": "helpfulness",
                "item_id": "item1",
                "call_id": "call_test_001",
                "score": avg,
                "passed": breaching is False,
                "details": {},
            }
        ],
        "metrics": [{"id": "helpfulness", "type": "objective"}],
        "judge_configs": [],
        "summary": {
            "metrics": {
                "helpfulness": {
                    "count": 1,
                    "avg_score": avg,
                    "pass_rate": 0.0 if breaching else 1.0,
                }
            },
            "failed_items": [],
        },
        "usage_summary": {},
    }
    (runs_dir / "results.json").write_text(
        json.dumps(run_data), encoding="utf-8"
    )
    return dataset_dir


def _notify_registered() -> bool:
    """Check whether main.py routes 'notify' to a real command module.

    The task explicitly forbids editing main.py; the caller will wire the
    registration themselves. Until then, CLI subprocess tests skip cleanly.
    """
    try:
        from evalyn_sdk.cli.main import _resolve_command_module
    except ImportError:
        return False
    return _resolve_command_module("notify") is not None


_skip_if_not_registered = pytest.mark.skipif(
    not _notify_registered(),
    reason="`notify` not yet registered in cli/main.py; CLI tests skipped",
)


@_skip_if_not_registered
class TestCliDispatch:
    """End-to-end: invoking `evalyn notify` via the CLI binary."""

    def test_no_breach_no_always_skips(self, tmp_path):
        ds = _seed_eval_run_dir(tmp_path, breaching=False)
        result = run_cli(
            "notify",
            "--dataset",
            str(ds),
            "--webhook",
            "https://example.com/hook",
            "--threshold",
            "helpfulness<0.8",
            "--dry-run",
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "skipping webhook" in result.stdout
        # --dry-run path is short-circuited by the skip; no payload emitted.
        assert "run_id" not in result.stdout

    def test_breach_triggers_dry_run_payload(self, tmp_path):
        ds = _seed_eval_run_dir(tmp_path, breaching=True)
        result = run_cli(
            "notify",
            "--dataset",
            str(ds),
            "--webhook",
            "https://example.com/hook",
            "--threshold",
            "helpfulness<0.8",
            "--dry-run",
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        # Generic payload printed because example.com -> generic.
        payload = json.loads(result.stdout)
        assert payload["breached"] is True
        assert payload["run_id"] == "run_test001"
        assert payload["breaches"][0]["metric_id"] == "helpfulness"

    def test_always_sends_without_breach(self, tmp_path):
        ds = _seed_eval_run_dir(tmp_path, breaching=False)
        result = run_cli(
            "notify",
            "--dataset",
            str(ds),
            "--webhook",
            "https://example.com/hook",
            "--always",
            "--dry-run",
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        payload = json.loads(result.stdout)
        assert payload["breached"] is False
        assert payload["breaches"] == []

    def test_slack_type_emits_blocks(self, tmp_path):
        ds = _seed_eval_run_dir(tmp_path, breaching=True)
        result = run_cli(
            "notify",
            "--dataset",
            str(ds),
            "--webhook",
            "https://example.com/hook",
            "--type",
            "slack",
            "--threshold",
            "helpfulness<0.8",
            "--dry-run",
        )
        assert result.returncode == 0, (
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        payload = json.loads(result.stdout)
        assert "blocks" in payload
        assert "text" in payload
