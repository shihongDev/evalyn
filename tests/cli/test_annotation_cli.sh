#!/bin/bash
# =============================================================================
# Annotation CLI Tests
# =============================================================================
# Tests for annotation commands:
#   - import-annotations
#   - annotation-stats
#   - annotate (skipped - interactive)
#   - annotate-spans (skipped - interactive)
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

log_header "Annotation CLI Tests"
ensure_test_dir
ensure_test_traces

DATASET_DIR="$TEST_OUTPUT_DIR/annotation-dataset"
ANNOTATIONS_DIR="$TEST_OUTPUT_DIR/annotations"
rm -rf "$DATASET_DIR" "$ANNOTATIONS_DIR"
mkdir -p "$ANNOTATIONS_DIR"

# Create test dataset
log_info "Creating test dataset..."
create_test_dataset "$DATASET_DIR" 5

# =============================================================================
# Create Test Annotations
# =============================================================================

log_header "Create Test Annotations"

# Simple annotations file
SIMPLE_ANNOTATIONS="$ANNOTATIONS_DIR/simple.jsonl"
cat > "$SIMPLE_ANNOTATIONS" << 'EOF'
{"id": "ann-1", "target_id": "test-1", "label": true, "rationale": "Good response", "annotator": "test", "source": "human", "confidence": 5}
{"id": "ann-2", "target_id": "test-2", "label": false, "rationale": "Incomplete", "annotator": "test", "source": "human", "confidence": 4}
{"id": "ann-3", "target_id": "test-3", "label": true, "rationale": "Accurate", "annotator": "test", "source": "human", "confidence": 3}
EOF

# Metric-specific annotations
METRIC_ANNOTATIONS="$ANNOTATIONS_DIR/metric_specific.jsonl"
create_test_annotations "$METRIC_ANNOTATIONS" "$DATASET_DIR"

# =============================================================================
# Test: import-annotations
# =============================================================================

log_header "import-annotations Command"

run_cmd "import-annotations (simple)" run_evalyn import-annotations --path "$SIMPLE_ANNOTATIONS"
run_cmd "import-annotations (metric-specific)" run_evalyn import-annotations --path "$METRIC_ANNOTATIONS"

# =============================================================================
# Test: annotation-stats
# =============================================================================

log_header "annotation-stats Command"

run_cmd "annotation-stats (file)" run_evalyn annotation-stats --dataset "$SIMPLE_ANNOTATIONS"
run_cmd "annotation-stats (dataset)" run_evalyn annotation-stats --dataset "$DATASET_DIR"

# Note: annotate and annotate-spans commands require user input (interactive)

# =============================================================================
# Summary
# =============================================================================

print_summary "Annotation CLI"
exit $?
