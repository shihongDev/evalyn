"""Export-OTLP command: verify an OTLP collector and emit the config users need.

Sends a single test span to the user's target collector. On success, prints
the env-var + evalyn.yaml + programmatic snippet they need to route evalyn
traces to it. On failure, prints a useful error.
"""

from __future__ import annotations

import argparse
import sys

# Dashboard catalog group
GROUP = "Tracing"
ESSENTIAL = {"endpoint"}


def _try_send_test_span(endpoint: str, protocol: str) -> tuple[bool, str]:
    """Send a single test span via OTLP and return (ok, message)."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        return False, f"OpenTelemetry SDK missing: {e}"

    try:
        if protocol == "http":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
    except ImportError as e:
        pkg = "opentelemetry-exporter-otlp-proto-http" if protocol == "http" else "opentelemetry-exporter-otlp"
        return False, f"OTLP exporter not installed ({e}). Install: pip install {pkg}"

    try:
        resource = Resource.create({"service.name": "evalyn-export-otlp-test"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        tracer = trace.get_tracer("evalyn.export_otlp")
        with tracer.start_as_current_span("evalyn.export_otlp.test"):
            pass

        # Force flush; BatchSpanProcessor returns False if it hits its
        # internal timeout, which is our best signal of a reachability
        # failure without parsing exporter internals.
        flushed = provider.force_flush(timeout_millis=5000)
        provider.shutdown()
        if not flushed:
            return False, "flush timed out; endpoint may be unreachable"
        return True, "test span accepted"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _print_config_snippet(endpoint: str, protocol: str) -> None:
    """Print the env-var + yaml snippet the user needs."""
    print()
    print("Endpoint verified. To enable OTLP export in your traces:")
    print()
    print("Option 1: environment variables (per-shell)")
    print("  export EVALYN_OTEL_EXPORTER=otlp")
    print(f"  export EVALYN_OTEL_ENDPOINT={endpoint}")
    if protocol == "http":
        print("  export EVALYN_OTEL_PROTOCOL=http")
    print()
    print("Option 2: evalyn.yaml")
    print("  tracing:")
    print("    exporter: otlp")
    print(f"    endpoint: {endpoint}")
    if protocol == "http":
        print("    protocol: http")
    print()
    print("Option 3: programmatic")
    print("  from evalyn_sdk.trace.otel import configure_otel")
    print(f'  configure_otel(exporter="otlp", endpoint="{endpoint}")')


def cmd_export_otlp(args: argparse.Namespace) -> None:
    endpoint = args.endpoint
    protocol = args.protocol

    if args.dry_run:
        print(f"Dry run: would test endpoint={endpoint} protocol={protocol}")
        _print_config_snippet(endpoint, protocol)
        return

    print(f"Testing OTLP endpoint: {endpoint} ({protocol})")
    ok, msg = _try_send_test_span(endpoint, protocol)
    if ok:
        print(f"OK: {msg}")
        _print_config_snippet(endpoint, protocol)
        return
    print(f"FAIL: {msg}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Common causes:", file=sys.stderr)
    print("  - Collector is not running or not reachable from this host", file=sys.stderr)
    print("  - Wrong protocol (try --protocol http if your collector uses HTTP/4318)", file=sys.stderr)
    print("  - Wrong endpoint format (gRPC: host:4317, HTTP: http://host:4318/v1/traces)", file=sys.stderr)
    print("  - Missing dependency: pip install opentelemetry-exporter-otlp", file=sys.stderr)
    raise SystemExit(1)


def register_commands(subparsers) -> None:
    parser = subparsers.add_parser(
        "export-otlp",
        help="Verify an OTLP collector and print the config to enable export",
        description=(
            "Send a single test span to an OTLP collector to verify it is "
            "reachable, then print the env vars / evalyn.yaml snippet needed "
            "to route evalyn traces to it."
        ),
    )
    parser.add_argument("--endpoint", required=True, help="OTLP endpoint (e.g. https://otel.example.com:4317)")
    parser.add_argument(
        "--protocol",
        choices=["grpc", "http"],
        default="grpc",
        help="OTLP protocol (default: grpc)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip sending the test span, just print what would be done",
    )
    parser.set_defaults(func=cmd_export_otlp)
