#!/bin/bash
# =============================================================================
# Pipeline CLI Tests
# =============================================================================
# Tests for pipeline commands:
#   - init
#   - workflow
#   - one-click (dry-run and actual runs)
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

log_header "Pipeline CLI Tests"
ensure_test_dir
ensure_test_traces

PIPELINE_DIR="$TEST_OUTPUT_DIR/pipeline"
rm -rf "$PIPELINE_DIR"
mkdir -p "$PIPELINE_DIR"

# =============================================================================
# Test: init
# =============================================================================

log_header "init Command"

run_cmd "init (new config)" run_evalyn init \
    --output "$PIPELINE_DIR/evalyn.yaml" \
    --force

# Verify config file
log_test "Verify config file created"
((TESTS_RUN++))
if [ -f "$PIPELINE_DIR/evalyn.yaml" ]; then
    log_pass "Config file created"
else
    log_fail "Config file not created"
fi

# =============================================================================
# Test: workflow
# =============================================================================

log_header "workflow Command"

run_cmd "workflow" run_evalyn workflow

# =============================================================================
# Test: one-click (dry-run with different modes)
# =============================================================================

log_header "one-click: Dry Run"

run_cmd "one-click --dry-run (basic)" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --dry-run \
    --skip-annotation \
    --skip-calibration \
    --auto-yes \
    --metric-mode basic

run_cmd "one-click --dry-run (llm-registry)" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --dry-run \
    --skip-annotation \
    --skip-calibration \
    --auto-yes \
    --metric-mode llm-registry

run_cmd "one-click --dry-run (llm-brainstorm)" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --dry-run \
    --skip-annotation \
    --skip-calibration \
    --auto-yes \
    --metric-mode llm-brainstorm

# Test bundle mode with various bundles
for bundle in "research-agent" "summarization" "orchestrator"; do
    run_cmd "one-click --dry-run (bundle: $bundle)" run_evalyn one-click \
        --project "$TEST_PROJECT" \
        --dry-run \
        --skip-annotation \
        --skip-calibration \
        --auto-yes \
        --metric-mode bundle \
        --bundle "$bundle"
done

# =============================================================================
# Test: one-click (dry-run with filters)
# =============================================================================

log_header "one-click: Filter Options"

run_cmd "one-click --production-only" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --dry-run \
    --production-only \
    --skip-annotation \
    --skip-calibration \
    --auto-yes

run_cmd "one-click --output-dir" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --dry-run \
    --output-dir "$PIPELINE_DIR/custom-output" \
    --skip-annotation \
    --skip-calibration \
    --auto-yes

run_cmd "one-click --dataset-limit" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --dry-run \
    --dataset-limit 3 \
    --skip-annotation \
    --skip-calibration \
    --auto-yes

# =============================================================================
# Test: one-click (actual execution)
# =============================================================================

log_header "one-click: Actual Execution"

OUTPUT_DIR="$PIPELINE_DIR/oneclick-basic"
run_cmd "one-click (basic mode)" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-limit 3 \
    --skip-annotation \
    --skip-calibration \
    --auto-yes \
    --metric-mode basic

# Verify outputs
log_test "Verify pipeline outputs"
((TESTS_RUN++))
if [ -f "$OUTPUT_DIR/dataset/dataset.jsonl" ]; then
    log_pass "Dataset created"
else
    log_fail "Dataset not created"
fi

log_test "Verify metrics created"
((TESTS_RUN++))
if [ -d "$OUTPUT_DIR/metrics" ] && [ -n "$(ls -A $OUTPUT_DIR/metrics 2>/dev/null)" ]; then
    log_pass "Metrics directory populated"
else
    log_fail "Metrics not created"
fi

log_test "Verify pipeline state"
((TESTS_RUN++))
if [ -f "$OUTPUT_DIR/pipeline_state.json" ]; then
    log_pass "Pipeline state saved"
else
    log_fail "Pipeline state not saved"
fi

# =============================================================================
# Test: one-click (resume)
# =============================================================================

log_header "one-click: Resume"

# Create interrupted state
RESUME_DIR="$PIPELINE_DIR/oneclick-resume"
mkdir -p "$RESUME_DIR/dataset"

cat > "$RESUME_DIR/pipeline_state.json" << 'EOF'
{
    "started_at": "2024-01-01T00:00:00",
    "config": {"project": "oneclick-test"},
    "steps": {
        "dataset": {"status": "success", "output": "dataset/dataset.jsonl"}
    },
    "output_dir": "."
}
EOF

echo '{"id": "test1", "inputs": {"query": "test"}, "output": "response"}' > "$RESUME_DIR/dataset/dataset.jsonl"
echo '{"project": "oneclick-test", "item_count": 1}' > "$RESUME_DIR/dataset/meta.json"

run_cmd "one-click --resume" run_evalyn one-click \
    --project "$TEST_PROJECT" \
    --metric-mode basic \
    --skip-annotation \
    --skip-calibration \
    --auto-yes \
    --resume \
    --output-dir "$RESUME_DIR"

# =============================================================================
# Summary
# =============================================================================

print_summary "Pipeline CLI"
exit $?
