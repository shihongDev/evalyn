#!/bin/bash
# =============================================================================
# Simulate CLI Tests
# =============================================================================
# Tests for simulation commands:
#   - simulate (similar, outlier modes)
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

log_header "Simulate CLI Tests"
ensure_test_dir
ensure_test_traces

DATASET_DIR="$TEST_OUTPUT_DIR/simulate-dataset"
SIMULATION_DIR="$TEST_OUTPUT_DIR/simulations"
rm -rf "$DATASET_DIR" "$SIMULATION_DIR"
mkdir -p "$SIMULATION_DIR"

# Create test dataset
log_info "Creating test dataset..."
create_test_dataset "$DATASET_DIR" 5

# =============================================================================
# Test: simulate (similar mode)
# =============================================================================

log_header "simulate: Similar Mode"

if require_api_key "simulate --modes similar"; then
    run_cmd "simulate --modes similar" run_evalyn simulate \
        --dataset "$DATASET_DIR" \
        --target "example_agents/gemini_research_agent/agent.py:run_agent" \
        --output "$SIMULATION_DIR/similar" \
        --modes "similar" \
        --num-similar 2 \
        --max-seeds 2
fi

# =============================================================================
# Test: simulate (outlier mode)
# =============================================================================

log_header "simulate: Outlier Mode"

if require_api_key "simulate --modes outlier"; then
    run_cmd "simulate --modes outlier" run_evalyn simulate \
        --dataset "$DATASET_DIR" \
        --target "example_agents/gemini_research_agent/agent.py:run_agent" \
        --output "$SIMULATION_DIR/outlier" \
        --modes "outlier" \
        --num-outlier 2 \
        --max-seeds 2
fi

# =============================================================================
# Test: simulate (combined modes)
# =============================================================================

log_header "simulate: Combined Modes"

if require_api_key "simulate --modes similar,outlier"; then
    run_cmd "simulate --modes similar,outlier" run_evalyn simulate \
        --dataset "$DATASET_DIR" \
        --target "example_agents/gemini_research_agent/agent.py:run_agent" \
        --output "$SIMULATION_DIR/combined" \
        --modes "similar,outlier" \
        --num-similar 1 \
        --num-outlier 1 \
        --max-seeds 1
fi

# =============================================================================
# Test: simulate (temperature options)
# =============================================================================

log_header "simulate: Temperature Options"

if require_api_key "simulate --temp-similar"; then
    run_cmd "simulate --temp-similar 0.5" run_evalyn simulate \
        --dataset "$DATASET_DIR" \
        --target "example_agents/gemini_research_agent/agent.py:run_agent" \
        --output "$SIMULATION_DIR/temp" \
        --modes "similar" \
        --num-similar 1 \
        --max-seeds 1 \
        --temp-similar 0.5

    run_cmd "simulate --temp-outlier 0.9" run_evalyn simulate \
        --dataset "$DATASET_DIR" \
        --target "example_agents/gemini_research_agent/agent.py:run_agent" \
        --output "$SIMULATION_DIR/temp-outlier" \
        --modes "outlier" \
        --num-outlier 1 \
        --max-seeds 1 \
        --temp-outlier 0.9
fi

# =============================================================================
# Summary
# =============================================================================

print_summary "Simulate CLI"
exit $?
