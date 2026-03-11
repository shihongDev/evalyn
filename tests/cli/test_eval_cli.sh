#!/bin/bash
# =============================================================================
# Evaluation CLI Tests
# =============================================================================
# Tests for evaluation and metric commands:
#   - suggest-metrics (all modes: basic, bundle, llm-registry, llm-brainstorm)
#   - select-metrics
#   - list-metrics
#   - run-eval
#   - list-runs
#   - show-run
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

log_header "Evaluation CLI Tests"
ensure_test_dir
ensure_test_traces

DATASET_DIR="$TEST_OUTPUT_DIR/eval-dataset"
METRICS_DIR="$TEST_OUTPUT_DIR/metrics"
rm -rf "$DATASET_DIR" "$METRICS_DIR"
mkdir -p "$METRICS_DIR"

# Create test dataset
log_info "Creating test dataset..."
create_test_dataset "$DATASET_DIR" 5

# =============================================================================
# Test: list-metrics
# =============================================================================

log_header "list-metrics Command"

run_cmd_check "list-metrics" "Objective|Subjective" list-metrics

# =============================================================================
# Test: suggest-metrics (basic mode - no LLM)
# =============================================================================

log_header "suggest-metrics: Basic Mode"

run_cmd "suggest-metrics --mode basic" run_evalyn suggest-metrics \
    --project "$TEST_PROJECT" \
    --dataset "$DATASET_DIR" \
    --mode basic \
    --metrics-name "test-basic"

# Test with --target (analyze source code)
run_cmd_allow_fail "suggest-metrics --target" run_evalyn suggest-metrics \
    --target "example_agents/gemini_research_agent/agent.py:run_agent" \
    --mode basic \
    --dataset "$DATASET_DIR" \
    --metrics-name "test-target"

# =============================================================================
# Test: suggest-metrics (bundle mode)
# =============================================================================

log_header "suggest-metrics: Bundle Mode"

# Test a few representative bundles
for bundle in "research-agent" "summarization" "orchestrator" "chatbot"; do
    run_cmd "suggest-metrics --bundle $bundle" run_evalyn suggest-metrics \
        --project "$TEST_PROJECT" \
        --dataset "$DATASET_DIR" \
        --mode bundle \
        --bundle "$bundle" \
        --metrics-name "bundle-$bundle"
done

# =============================================================================
# Test: suggest-metrics (scope filters)
# =============================================================================

log_header "suggest-metrics: Scope Filters"

for scope in "${METRIC_SCOPES[@]}"; do
    run_cmd "suggest-metrics --scope $scope" run_evalyn suggest-metrics \
        --project "$TEST_PROJECT" \
        --dataset "$DATASET_DIR" \
        --mode basic \
        --scope "$scope" \
        --metrics-name "scope-$scope"
done

# =============================================================================
# Test: suggest-metrics (LLM modes - requires API key)
# =============================================================================

log_header "suggest-metrics: LLM Modes"

if require_api_key "suggest-metrics --mode llm-registry"; then
    run_cmd "suggest-metrics --mode llm-registry" run_evalyn suggest-metrics \
        --project "$TEST_PROJECT" \
        --dataset "$DATASET_DIR" \
        --mode llm-registry \
        --num-metrics 3 \
        --metrics-name "llm-registry"
fi

if require_api_key "suggest-metrics --mode llm-brainstorm"; then
    run_cmd "suggest-metrics --mode llm-brainstorm" run_evalyn suggest-metrics \
        --project "$TEST_PROJECT" \
        --dataset "$DATASET_DIR" \
        --mode llm-brainstorm \
        --num-metrics 3 \
        --metrics-name "llm-brainstorm"
fi

# =============================================================================
# Test: select-metrics (requires API key)
# =============================================================================

log_header "select-metrics Command"

if require_api_key "select-metrics"; then
    # select-metrics requires --target and --llm-caller (advanced usage)
    run_cmd_allow_fail "select-metrics" run_evalyn select-metrics \
        --target "example_agents/gemini_research_agent/agent.py:run_agent" \
        --llm-caller "evalyn_sdk.metrics.suggesters:suggest_metrics_llm" \
        --limit 3
fi

# =============================================================================
# Test: run-eval
# =============================================================================

log_header "run-eval Command"

# Find a metrics file from suggest-metrics
METRICS_FILE=$(find "$DATASET_DIR" -name "*.json" -path "*/metrics/*" 2>/dev/null | head -1)

if [ -n "$METRICS_FILE" ]; then
    run_cmd "run-eval (with metrics file)" run_evalyn run-eval \
        --dataset "$DATASET_DIR" \
        --metrics "$METRICS_FILE"

    run_cmd "run-eval --format json" run_evalyn run-eval \
        --dataset "$DATASET_DIR" \
        --metrics "$METRICS_FILE" \
        --format json
else
    # Create a simple metrics file
    cat > "$METRICS_DIR/test-metrics.json" << 'EOF'
[
    {"id": "output_nonempty", "type": "objective", "description": "Check output is not empty"},
    {"id": "latency_ms", "type": "objective", "description": "Measure latency"}
]
EOF
    run_cmd "run-eval (objective only)" run_evalyn run-eval \
        --dataset "$DATASET_DIR" \
        --metrics "$METRICS_DIR/test-metrics.json"
fi

# Auto-detect metrics
run_cmd_allow_fail "run-eval (auto-detect)" run_evalyn run-eval --dataset "$DATASET_DIR"

# Latest dataset
run_cmd_allow_fail "run-eval --latest" run_evalyn run-eval --latest

# Custom run name
run_cmd_allow_fail "run-eval --dataset-name" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --dataset-name "custom-test-run"

# Use calibrated prompts
run_cmd_allow_fail "run-eval --use-calibrated" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --use-calibrated

# =============================================================================
# Test: list-runs
# =============================================================================

log_header "list-runs Command"

run_cmd "list-runs" run_evalyn list-runs --limit 10
run_cmd "list-runs --format json" run_evalyn list-runs --format json --limit 5

# =============================================================================
# Test: show-run
# =============================================================================

log_header "show-run Command"

RUN_ID=$(get_run_id)
if [ -n "$RUN_ID" ]; then
    run_cmd "show-run --id" run_evalyn show-run --id "$RUN_ID"
fi
run_cmd_allow_fail "show-run --last" run_evalyn show-run --last

# =============================================================================
# Test: run-eval advanced options
# =============================================================================

log_header "run-eval: Advanced Options"

# Create objective metrics for deterministic testing
create_objective_metrics "$METRICS_DIR/objective-only.json"

# Test --verbose flag (should show cost breakdown)
run_cmd_check "run-eval --verbose" "output_nonempty|Avg Score|Pass Rate" run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/objective-only.json" \
    --verbose

# Test workers with different values
run_cmd "run-eval -w 1 (single worker)" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/objective-only.json" \
    -w 1

run_cmd "run-eval -w 2 (parallel)" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/objective-only.json" \
    -w 2

# Test custom dataset name
run_cmd "run-eval --dataset-name" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/objective-only.json" \
    --dataset-name "custom-name-test"

# Test table vs json output explicitly
run_cmd_check "run-eval --format table output" "output_nonempty|Pass|Fail|Score" run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/objective-only.json" \
    --format table

run_cmd_check "run-eval --format json output" "output_nonempty|score|passed" run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/objective-only.json" \
    --format json

# =============================================================================
# Test: Error handling
# =============================================================================

log_header "run-eval: Error Handling"

# Missing dataset directory
run_cmd_allow_fail "run-eval missing dataset" run_evalyn run-eval \
    --dataset "/nonexistent/path/dataset" \
    --metrics "$METRICS_DIR/objective-only.json"

# Invalid metrics file
run_cmd_allow_fail "run-eval missing metrics" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "/nonexistent/metrics.json"

# Invalid metrics format (create malformed file)
echo "not valid json" > "$METRICS_DIR/invalid.json"
run_cmd_allow_fail "run-eval invalid metrics JSON" run_evalyn run-eval \
    --dataset "$DATASET_DIR" \
    --metrics "$METRICS_DIR/invalid.json"

# =============================================================================
# Summary
# =============================================================================

print_summary "Evaluation CLI"
exit $?
