#!/bin/bash
# =============================================================================
# Evalyn CLI Test Suite - Master Runner
# =============================================================================
# Runs all CLI test scripts and aggregates results.
#
# Usage:
#   ./tests/cli/run_all.sh              # Run all tests
#   ./tests/cli/run_all.sh --quick      # Skip LLM-dependent tests
#   ./tests/cli/run_all.sh --verbose    # Show verbose output
#   ./tests/cli/run_all.sh trace eval   # Run only specified test groups
#
# Available test groups:
#   trace      - Trace inspection (list-calls, show-call, show-trace, etc.)
#   dataset    - Dataset building (build-dataset, validate, status)
#   eval       - Evaluation (suggest-metrics, run-eval, list-runs, etc.)
#   eval-prod  - Evaluation with production data (calibration fixture)
#   eval-results - Evaluation results validation
#   analysis   - Analysis (analyze, compare, trend)
#   annotation - Annotation (import-annotations, annotation-stats)
#   calibration - Calibration (calibrate, list-calibrations, clustering)
#   export     - Export (export, export-for-annotation)
#   simulate   - Simulation (simulate)
#   pipeline   - Pipeline (init, workflow, one-click)
#
# Prerequisites:
#   1. Run: python tests/fixtures/oneclick/setup_test_traces.py
#   2. Set GEMINI_API_KEY for LLM-dependent tests
# =============================================================================

set +e  # Continue on errors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

# =============================================================================
# Parse Arguments
# =============================================================================

VERBOSE=""
QUICK_MODE=false
SPECIFIED_TESTS=()

for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE="--verbose"
            ;;
        --quick|-q)
            QUICK_MODE=true
            ;;
        --help|-h)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *)
            SPECIFIED_TESTS+=("$arg")
            ;;
    esac
done

# =============================================================================
# Available Tests
# =============================================================================

ALL_TESTS=(
    "trace:test_trace_cli.sh"
    "dataset:test_dataset_cli.sh"
    "eval:test_eval_cli.sh"
    "eval-prod:test_eval_prod.sh"
    "eval-results:test_eval_results.sh"
    "analysis:test_analysis_cli.sh"
    "annotation:test_annotation_cli.sh"
    "calibration:test_calibration_cli.sh"
    "export:test_export_cli.sh"
    "simulate:test_simulate_cli.sh"
    "pipeline:test_pipeline_cli.sh"
)

# Tests that require LLM API key
LLM_TESTS=("eval" "eval-prod" "calibration" "simulate")

# Check if test requires LLM
is_llm_test() {
    local name="$1"
    for llm_test in "${LLM_TESTS[@]}"; do
        [ "$name" = "$llm_test" ] && return 0
    done
    return 1
}

# =============================================================================
# Determine Which Tests to Run
# =============================================================================

TESTS_TO_RUN=()

if [ ${#SPECIFIED_TESTS[@]} -eq 0 ]; then
    # Run all tests
    for test_entry in "${ALL_TESTS[@]}"; do
        name="${test_entry%%:*}"

        # Skip LLM tests in quick mode
        if [ "$QUICK_MODE" = true ] && is_llm_test "$name"; then
            echo -e "${YELLOW}[SKIP]${NC} $name (quick mode)"
            continue
        fi

        TESTS_TO_RUN+=("$test_entry")
    done
else
    # Run only specified tests
    for specified in "${SPECIFIED_TESTS[@]}"; do
        found=false
        for test_entry in "${ALL_TESTS[@]}"; do
            name="${test_entry%%:*}"
            if [ "$name" = "$specified" ]; then
                TESTS_TO_RUN+=("$test_entry")
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo -e "${RED}Unknown test group: $specified${NC}"
            echo "Available: trace, dataset, eval, eval-prod, eval-results, analysis, annotation, calibration, export, simulate, pipeline"
            exit 1
        fi
    done
fi

# =============================================================================
# Run Tests
# =============================================================================

log_header "Evalyn CLI Test Suite"
echo "Quick mode: $QUICK_MODE"
echo "Verbose: ${VERBOSE:-false}"
echo "Tests to run: ${#TESTS_TO_RUN[@]}"
echo ""

FAILED_SCRIPTS=()

for test_entry in "${TESTS_TO_RUN[@]}"; do
    name="${test_entry%%:*}"
    script="${test_entry#*:}"

    echo -e "\n${BLUE}>>> Running: $name${NC}"
    echo "    Script: $script"

    if bash "$SCRIPT_DIR/$script" $VERBOSE; then
        echo -e "${GREEN}>>> $name: PASSED${NC}"
    else
        echo -e "${RED}>>> $name: FAILED${NC}"
        FAILED_SCRIPTS+=("$name")
    fi
done

# =============================================================================
# Summary
# =============================================================================

echo ""
log_header "Test Suite Summary"
echo "Test groups run: ${#TESTS_TO_RUN[@]}"
echo "Failed groups: ${#FAILED_SCRIPTS[@]}"

if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed test groups:${NC}"
    for failed in "${FAILED_SCRIPTS[@]}"; do
        echo "  - $failed"
    done
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}All test groups passed!${NC}"
    exit 0
fi
