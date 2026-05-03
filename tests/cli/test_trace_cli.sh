#!/bin/bash
# =============================================================================
# Trace CLI Tests
# =============================================================================
# Tests for trace inspection commands:
#   - list-calls
#   - show-call
#   - show-trace
#   - show-span
#   - show-projects
# =============================================================================

set +e  # Continue on errors

# Load configuration and helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/helpers.sh"

parse_args "$@"

# =============================================================================
# Setup
# =============================================================================

log_header "Trace CLI Tests"
ensure_test_dir
ensure_test_traces

# =============================================================================
# Test: Basic Commands
# =============================================================================

log_header "Basic Help and Version"

run_cmd_check "evalyn --help" "Evalyn CLI" --help
run_cmd_check "evalyn --version" "evalyn" --version
run_cmd_check "evalyn help" "QUICK START" help

# =============================================================================
# Test: list-calls
# =============================================================================

log_header "list-calls Command"

run_cmd "list-calls (basic)" run_evalyn list-calls --limit 5
run_cmd "list-calls --format json" run_evalyn list-calls --limit 3 --format json
run_cmd "list-calls --project filter" run_evalyn list-calls --project "$TEST_PROJECT" --limit 5
run_cmd "list-calls --simulation" run_evalyn list-calls --simulation --limit 5
run_cmd "list-calls --production" run_evalyn list-calls --production --limit 5

# =============================================================================
# Test: show-projects
# =============================================================================

log_header "show-projects Command"

run_cmd "show-projects" run_evalyn show-projects

# =============================================================================
# Test: show-call
# =============================================================================

log_header "show-call Command"

run_cmd "show-call --last" run_evalyn show-call --last

# Test with specific call ID
CALL_ID=$(get_call_id)
if [ -n "$CALL_ID" ]; then
    run_cmd "show-call --id (specific)" run_evalyn show-call --id "$CALL_ID"
fi

# =============================================================================
# Test: show-trace
# =============================================================================

log_header "show-trace Command"

run_cmd "show-trace --last" run_evalyn show-trace --last
run_cmd "show-trace --last --verbose" run_evalyn show-trace --last --verbose

if [ -n "$CALL_ID" ]; then
    run_cmd "show-trace --id (specific)" run_evalyn show-trace --id "$CALL_ID"
    run_cmd "show-trace --id --verbose" run_evalyn show-trace --id "$CALL_ID" --verbose
fi

# =============================================================================
# Test: show-span
# =============================================================================

log_header "show-span Command"

# Get a span ID from the trace
SPAN_ID=""
if [ -n "$CALL_ID" ]; then
    # Try to extract a span ID from show-trace output
    SPAN_ID=$(run_evalyn show-trace --id "$CALL_ID" 2>/dev/null | grep -oE '\[[a-f0-9-]+\]' | head -1 | tr -d '[]')
fi

if [ -n "$SPAN_ID" ]; then
    run_cmd "show-span --id" run_evalyn show-span --id "$SPAN_ID"
fi

# =============================================================================
# Summary
# =============================================================================

print_summary "Trace CLI"
exit $?
