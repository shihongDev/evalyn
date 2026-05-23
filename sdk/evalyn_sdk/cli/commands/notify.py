"""Notify command: send webhook alerts when an eval run breaches thresholds.

Pairs with `run-eval --fail-on` so CI pipelines can both fail AND notify on
quality regressions. Supports Slack, Discord, and generic JSON endpoints.

When `--type` is omitted, the format is auto-detected from the URL host
(hooks.slack.com -> slack, discord.com -> discord, else generic). If no
`--threshold` is given, the run summary is checked against the optional
`notify:` block in `evalyn.yaml`. Without thresholds AND without `--always`,
nothing is sent.
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Notify"

# Most-toggled knobs for the dashboard form.
ESSENTIAL = {"run", "webhook", "always"}

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...evaluation.guards import (
    ThresholdReport,
    evaluate_thresholds,
    parse_threshold_expression,
)
from ...integration.notifiers import (
    VALID_FORMATS,
    auto_detect_format,
    format_discord_message,
    format_generic_message,
    format_slack_message,
    send_webhook,
)
from ..utils.command_common import load_eval_run_for_command
from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error


def _parse_threshold_expressions(raw_exprs: list[str]) -> list[Any]:
    """Parse all --threshold expressions; fatal on invalid syntax."""
    parsed: list[Any] = []
    for expr in raw_exprs or []:
        try:
            parsed.append(parse_threshold_expression(expr))
        except ValueError as exc:
            fatal_error(f"Invalid --threshold expression: {exc}")
    return parsed


def _build_run_summary(run: Any) -> dict:
    """Build a small dict the formatters consume.

    Pulls metric stats from `run.summary["metrics"]` when present, falling
    back to recomputing from `metric_results` if the summary is empty.
    """
    summary = getattr(run, "summary", {}) or {}
    metrics = summary.get("metrics") if isinstance(summary, dict) else None

    if not metrics:
        from collections import defaultdict

        by_metric: dict[str, list] = defaultdict(list)
        for mr in getattr(run, "metric_results", []) or []:
            by_metric[mr.metric_id].append(mr)
        metrics = {}
        for mid, results in by_metric.items():
            scores = [r.score for r in results if r.score is not None]
            passes = [r.passed for r in results if r.passed is not None]
            metrics[mid] = {
                "count": len(results),
                "avg_score": (sum(scores) / len(scores)) if scores else None,
                "pass_rate": (sum(1 for p in passes if p) / len(passes))
                if passes
                else None,
            }

    created_at = getattr(run, "created_at", None)
    if hasattr(created_at, "isoformat"):
        created_str = created_at.isoformat()
    else:
        created_str = str(created_at or "")
    return {
        "id": getattr(run, "id", "unknown"),
        "dataset_name": getattr(run, "dataset_name", "unknown"),
        "created_at": created_str,
        "metrics": metrics or {},
    }


def _select_format(args: argparse.Namespace, url: str) -> str:
    """Return the explicit `--type` choice or auto-detect from URL."""
    explicit = getattr(args, "type", None)
    if explicit:
        if explicit not in VALID_FORMATS:
            fatal_error(
                f"Invalid --type: {explicit!r}",
                f"Choose one of: {', '.join(VALID_FORMATS)}",
            )
        return explicit
    return auto_detect_format(url)


def _build_payload(
    fmt: str,
    run_summary: dict,
    breaches: list[Any],
    custom_message: str | None,
) -> dict:
    if fmt == "slack":
        return format_slack_message(run_summary, breaches, custom_message)
    if fmt == "discord":
        return format_discord_message(run_summary, breaches, custom_message)
    return format_generic_message(run_summary, breaches, custom_message)


def _resolve_webhook_url(args: argparse.Namespace, config: dict) -> tuple[str, str | None]:
    """Pick a webhook URL from --webhook flag or evalyn.yaml `notify:` block.

    Returns (url, hint_format). hint_format is "slack"/"discord" when the
    URL came from a typed config key; that overrides URL-based detection.
    """
    url = getattr(args, "webhook", None)
    if url:
        return url, None

    notify_cfg = config.get("notify") if isinstance(config, dict) else None
    if isinstance(notify_cfg, dict):
        slack_url = notify_cfg.get("slack")
        discord_url = notify_cfg.get("discord")
        generic_url = notify_cfg.get("webhook") or notify_cfg.get("url")
        # Preference: explicit slack/discord typed keys beat the generic URL.
        if slack_url:
            return str(slack_url), "slack"
        if discord_url:
            return str(discord_url), "discord"
        if generic_url:
            return str(generic_url), None

    fatal_error(
        "No webhook URL provided",
        "Pass --webhook <url> or set notify.{slack,discord,webhook} in evalyn.yaml",
    )
    return "", None  # unreachable


def _should_send(report: ThresholdReport, always: bool, config: dict) -> bool:
    """Return True when we should fire the webhook.

    Rules:
    - If --always is set, send unconditionally.
    - If any threshold expression breached, send.
    - Otherwise, honor evalyn.yaml `notify.on` ("always" sends regardless).
    """
    if always:
        return True
    if report.any_breached:
        return True
    notify_cfg = config.get("notify") if isinstance(config, dict) else None
    if isinstance(notify_cfg, dict):
        on = notify_cfg.get("on")
        if isinstance(on, str) and on.lower() == "always":
            return True
        if isinstance(on, list) and any(
            isinstance(x, str) and x.lower() == "always" for x in on
        ):
            return True
    return False


def cmd_notify(args: argparse.Namespace) -> None:
    """Evaluate thresholds against a run and send a webhook alert.

    Exit codes: 0 success, 2 resolution failure (via fatal_error), 3 delivery
    failure (network/HTTP error on send).
    """
    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, "dataset", None),
        getattr(args, "latest", False),
        config,
    )

    loaded = load_eval_run_for_command(
        run_id=getattr(args, "run", None),
        dataset_path=dataset_path,
        error_hint="Specify --run <run_id>, --dataset <path>, or --latest",
    )
    run = loaded.run

    threshold_exprs = _parse_threshold_expressions(getattr(args, "threshold", None) or [])
    report = evaluate_thresholds(threshold_exprs, run)

    for warning in report.missing:
        print(warning.format(), file=sys.stderr, flush=True)

    always = bool(getattr(args, "always", False))
    if not _should_send(report, always, config):
        print(
            f"No breaches for run {run.id[:8]} on '{run.dataset_name}'; "
            "skipping webhook (use --always to send anyway)."
        )
        return

    url, hint_format = _resolve_webhook_url(args, config)
    fmt = hint_format or _select_format(args, url)

    run_summary = _build_run_summary(run)
    custom_message = getattr(args, "message", None)
    payload = _build_payload(fmt, run_summary, report.breaches, custom_message)

    if getattr(args, "dry_run", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    response = send_webhook(url, payload)
    if response.ok:
        print(
            f"Notification sent ({fmt}) to webhook: HTTP {response.status} "
            f"after {response.attempts} attempt(s)."
        )
        if report.any_breached:
            for breach in report.breaches:
                print(breach.format(), file=sys.stderr, flush=True)
        return

    err_msg = response.error or f"HTTP {response.status}"
    print(
        f"Webhook delivery failed ({fmt}) after {response.attempts} attempt(s): {err_msg}",
        file=sys.stderr,
        flush=True,
    )
    if response.body:
        print(f"Response body: {response.body[:500]}", file=sys.stderr, flush=True)
    sys.exit(3)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "notify",
        help="Send webhook alert (Slack/Discord/generic) when an eval breaches thresholds",
    )
    # Run / dataset resolution
    parser.add_argument("--run", help="Eval run ID to evaluate")
    parser.add_argument("--dataset", help="Dataset path (uses latest run in eval_runs/)")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )

    # Delivery
    parser.add_argument(
        "--webhook",
        help="Webhook URL. Falls back to notify.{slack,discord,webhook} in evalyn.yaml.",
    )
    parser.add_argument(
        "--type",
        choices=list(VALID_FORMATS),
        default=None,
        help="Payload format. Auto-detected from URL host when omitted.",
    )

    # Gating
    parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        metavar="EXPR",
        help=(
            "Threshold expression like 'helpfulness_accuracy<0.8' "
            "(operators: <, <=, >, >=, ==, !=). Repeatable."
        ),
    )
    parser.add_argument(
        "--always",
        action="store_true",
        help="Send notification regardless of threshold breaches.",
    )

    # Content
    parser.add_argument(
        "--message",
        help="Custom message text appended to the payload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the payload JSON instead of POSTing it.",
    )

    parser.set_defaults(func=cmd_notify)


__all__ = ["cmd_notify", "register_commands"]
