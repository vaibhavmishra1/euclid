#!/bin/bash
# =============================================================================
# RENTROPY SMOKE TEST: Run all 4 diversity reward variants
# =============================================================================
# This script runs a minimal viable experiment to test all 4 Rentropy variants.
#
# Usage:
#   bash scripts/smoketest_all_variants.sh [base_model]
#
# Example:
#   bash scripts/smoketest_all_variants.sh Qwen/Qwen3-0.6B-Base
#
# Requirements:
#   - Set STORAGE_PATH environment variable
#   - Set HUGGINGFACENAME environment variable (optional, for dataset upload)
#   - GPU with at least 8GB VRAM
#   - Cluster centroids built (run cluster_space/build_clusters.py first)
# =============================================================================

set -e  # Exit on error

BASE_MODEL=${1:-"Qwen/Qwen3-0.6B-Base"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_PREFIX="rentropy_smoketest_${TIMESTAMP}"

echo "=============================================="
echo "RENTROPY SMOKE TEST - ALL 4 VARIANTS"
echo "=============================================="
echo "Base Model: $BASE_MODEL"
echo "Timestamp: $TIMESTAMP"
echo "Storage: $STORAGE_PATH"
echo "=============================================="

# Check environment
if [ -z "$STORAGE_PATH" ]; then
    echo "ERROR: STORAGE_PATH not set. Please set it first:"
    echo "  export STORAGE_PATH=/path/to/storage"
    exit 1
fi

# Create output directories
mkdir -p ${STORAGE_PATH}/models
mkdir -p ${STORAGE_PATH}/generated_question
mkdir -p ${STORAGE_PATH}/temp_results
mkdir -p ${STORAGE_PATH}/smoketest_results

# Results file
RESULTS_FILE="${STORAGE_PATH}/smoketest_results/${EXPERIMENT_PREFIX}_results.txt"
echo "Rentropy Smoke Test Results" > $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "Base Model: $BASE_MODEL" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Function to run a single variant
run_variant() {
    local mode=$1
    local mode_name=$2
    local exp_name="${EXPERIMENT_PREFIX}_mode${mode}"
    
    echo ""
    echo "=============================================="
    echo "Running Variant $mode: $mode_name"
    echo "Experiment: $exp_name"
    echo "=============================================="
    
    START_TIME=$(date +%s)
    
    # Update config with this mode
    CONFIG_FILE="../rentropy_config.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        sed -i.bak "s/^diversity_mode:.*/diversity_mode: $mode/" "$CONFIG_FILE"
    fi
    
    # Run questioner training (1 iteration)
    echo "[Mode $mode] Training questioner..."
    bash scripts/smoketest_questioner.sh \
        $BASE_MODEL \
        $BASE_MODEL \
        ${exp_name}_questioner \
        $mode 2>&1 | tee ${STORAGE_PATH}/smoketest_results/${exp_name}_questioner.log
    
    # Check if questioner model was created
    QUESTIONER_PATH="${STORAGE_PATH}/models/${exp_name}_questioner/global_step_1/actor/huggingface"
    if [ ! -d "$QUESTIONER_PATH" ]; then
        QUESTIONER_PATH=$BASE_MODEL
        echo "[Mode $mode] Using base model as questioner (checkpoint not found)"
    fi
    
    # Run solver training (optional - can skip for pure questioner test)
    # echo "[Mode $mode] Training solver..."
    # bash scripts/smoketest_solver.sh \
    #     $BASE_MODEL \
    #     $QUESTIONER_PATH \
    #     ${exp_name}_solver 2>&1 | tee ${STORAGE_PATH}/smoketest_results/${exp_name}_solver.log
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "" >> $RESULTS_FILE
    echo "Mode $mode ($mode_name):" >> $RESULTS_FILE
    echo "  Duration: ${DURATION}s" >> $RESULTS_FILE
    echo "  Questioner log: ${STORAGE_PATH}/smoketest_results/${exp_name}_questioner.log" >> $RESULTS_FILE
    
    echo "[Mode $mode] Completed in ${DURATION}s"
}

# =============================================================================
# Run all 4 variants
# =============================================================================

echo ""
echo "Starting smoke test for all 4 variants..."
echo ""

# Mode 1: Vanilla (R-Zero baseline)
run_variant 1 "Vanilla (baseline)"

# Mode 2: + Rarity reward
run_variant 2 "Rarity reward"

# Mode 3: + Batch uniqueness
run_variant 3 "Rarity + Batch uniqueness"

# Mode 4: + Within-cluster uniqueness
run_variant 4 "Full Rentropy"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "SMOKE TEST COMPLETED"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
cat $RESULTS_FILE
echo ""
echo "Logs saved to: ${STORAGE_PATH}/smoketest_results/"
echo ""
echo "Next steps:"
echo "  1. Check logs for errors"
echo "  2. Compare generated questions across modes"
echo "  3. If smoke test passes, run full experiment with main_rentropy.sh"
