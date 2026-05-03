#!/bin/bash
# =============================================================================
# CLI Test Configuration
# =============================================================================
# Shared configuration for all CLI test scripts.
# Source this file at the beginning of each test script.
# =============================================================================

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SDK_ROOT="$PROJECT_ROOT/sdk"
TEST_OUTPUT_DIR="$PROJECT_ROOT/data/test-output/cli-tests"
TEST_DB="$PROJECT_ROOT/data/test/traces.sqlite"
TEST_FIXTURES="$PROJECT_ROOT/tests/fixtures"

# Test project configuration
TEST_PROJECT="oneclick-test"
TEST_VERSION="v1"

# Environment setup
export PYTHONPATH="$SDK_ROOT:$PYTHONPATH"
export EVALYN_DB="$TEST_DB"

# Load .env if exists (for API keys)
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# =============================================================================
# Metric Scopes (used by test_eval_cli.sh)
# =============================================================================
METRIC_SCOPES=("overall" "llm_call" "tool_call" "trace")

# =============================================================================
# Shared Test Data
# =============================================================================

# Create a standard objective-only metrics file (no LLM required)
# Usage: create_objective_metrics <output_file> [--with-latency]
create_objective_metrics() {
    local output_file="$1"
    local with_latency="${2:-}"

    if [ "$with_latency" = "--with-latency" ]; then
        cat > "$output_file" << 'EOF'
[
    {"id": "output_nonempty", "type": "objective", "description": "Check output is not empty"},
    {"id": "output_length", "type": "objective", "description": "Measure output length"},
    {"id": "latency_ms", "type": "objective", "description": "Measure latency if available"}
]
EOF
    else
        cat > "$output_file" << 'EOF'
[
    {"id": "output_nonempty", "type": "objective", "description": "Check output is not empty"},
    {"id": "output_length", "type": "objective", "description": "Measure output length"}
]
EOF
    fi
}
