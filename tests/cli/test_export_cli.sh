#!/bin/bash
# =============================================================================
# Export CLI Tests
# =============================================================================
# Tests for export commands:
#   - export (json, csv, markdown, html)
#   - export-for-annotation
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

log_header "Export CLI Tests"
ensure_test_dir
ensure_test_traces

DATASET_DIR="$TEST_OUTPUT_DIR/export-dataset"
EXPORT_DIR="$TEST_OUTPUT_DIR/exports"
rm -rf "$DATASET_DIR" "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

# Create test dataset and run eval
log_info "Creating test dataset..."
create_test_dataset "$DATASET_DIR" 5

# Create metrics and run eval for export data
mkdir -p "$DATASET_DIR/metrics"
cat > "$DATASET_DIR/metrics/test-metrics.json" << 'EOF'
[
    {"id": "output_nonempty", "type": "objective"},
    {"id": "latency_ms", "type": "objective"}
]
EOF

log_info "Running evaluation..."
run_evalyn run-eval --dataset "$DATASET_DIR" --metrics "$DATASET_DIR/metrics/test-metrics.json" >/dev/null 2>&1

# =============================================================================
# Test: export (all formats)
# =============================================================================

log_header "export Command"

run_cmd "export --format json" run_evalyn export \
    --dataset "$DATASET_DIR" \
    --format json \
    --output "$EXPORT_DIR/results.json"

run_cmd "export --format csv" run_evalyn export \
    --dataset "$DATASET_DIR" \
    --format csv \
    --output "$EXPORT_DIR/results.csv"

run_cmd "export --format markdown" run_evalyn export \
    --dataset "$DATASET_DIR" \
    --format markdown \
    --output "$EXPORT_DIR/results.md"

run_cmd "export --format html" run_evalyn export \
    --dataset "$DATASET_DIR" \
    --format html \
    --output "$EXPORT_DIR/results.html"

# =============================================================================
# Test: export-for-annotation
# =============================================================================

log_header "export-for-annotation Command"

run_cmd "export-for-annotation" run_evalyn export-for-annotation \
    --dataset "$DATASET_DIR" \
    --output "$EXPORT_DIR/for_annotation.jsonl"

# =============================================================================
# Verify Export Files
# =============================================================================

log_header "Verify Exports"

for file in results.json results.csv results.md results.html for_annotation.jsonl; do
    log_test "Verify $file exists"
    ((TESTS_RUN++))
    if [ -f "$EXPORT_DIR/$file" ]; then
        SIZE=$(stat -c%s "$EXPORT_DIR/$file" 2>/dev/null || stat -f%z "$EXPORT_DIR/$file" 2>/dev/null || echo "0")
        log_pass "$file exists (${SIZE} bytes)"
    else
        log_fail "$file not created"
    fi
done

# =============================================================================
# Summary
# =============================================================================

print_summary "Export CLI"
exit $?
