#!/bin/bash
# =============================================================================
# exp0_0: Run Baseline Experiment
# =============================================================================
# 
# Usage:
#   bash run_baseline.sh                    # Default: 50 seeds
#   bash run_baseline.sh --num_seeds 10     # Quick test with 10 seeds
#   bash run_baseline.sh --gpu 1            # Use GPU 1
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Default values
GPU_ID=0
NUM_SEEDS=""
VERBOSE=""

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --num_seeds)
            NUM_SEEDS="--num_seeds $2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu N         GPU device ID (default: 0)"
            echo "  --num_seeds N   Number of seeds to process"
            echo "  --verbose       Enable verbose output"
            echo "  --config FILE   Config file path"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

echo "============================================================"
echo "exp0_0: Zero-Shot Baseline Experiment"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Script Directory: ${SCRIPT_DIR}"
echo "  Config File:      ${CONFIG_FILE}"
echo "  Results Dir:      ${RESULTS_DIR}"
echo "  GPU:              ${GPU_ID}"
echo ""

# Check if config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Set environment variables
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false

# Check for HuggingFace token
if [ -z "${HF_TOKEN}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
fi

# =============================================================================
# Run Pipeline
# =============================================================================

echo "============================================================"
echo "Starting Baseline Pipeline"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

# Run the main pipeline
python baseline_pipeline.py \
    --config "${CONFIG_FILE}" \
    ${NUM_SEEDS} \
    ${VERBOSE}

PIPELINE_EXIT_CODE=$?

if [ ${PIPELINE_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Pipeline failed with exit code ${PIPELINE_EXIT_CODE}"
    exit ${PIPELINE_EXIT_CODE}
fi

# =============================================================================
# Run Analysis
# =============================================================================

echo ""
echo "============================================================"
echo "Running Analysis"
echo "============================================================"
echo ""

python analyze_results.py \
    --results_dir "${RESULTS_DIR}" \
    --plots

ANALYSIS_EXIT_CODE=$?

if [ ${ANALYSIS_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "WARNING: Analysis failed with exit code ${ANALYSIS_EXIT_CODE}"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Experiment Complete"
echo "============================================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Output files:"
ls -la "${RESULTS_DIR}/" 2>/dev/null || echo "  (no files yet)"
echo ""
echo "To view the report:"
echo "  cat ${RESULTS_DIR}/report.txt"
echo ""
echo "To view plots (if generated):"
echo "  ls ${RESULTS_DIR}/plots/"
echo ""
