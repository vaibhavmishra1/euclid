#!/bin/bash
set -euo pipefail

# ============================================================================
# GRPO Training: vibhuiitj/darwin_iter3_try3_solver_step10
# Datasets:      ALL benchmarks (Math500, GSM8K, AMC23, Minerva, OlympiadBench,
#                AIME2024/2025, MMLU-Pro, BBEH, SuperGPQA, GPQA Diamond)
# Hardware:      8x A100 80GB
#
# Usage:
#   bash benchmark_grpo/train.sh
#
# This script will:
#   1. Prepare the dataset (download all benchmarks + merge) if not already done
#   2. Run GRPO training using the verl framework
#   3. Merge the final checkpoint into a HuggingFace-compatible model
#   4. Run evaluation on standard math benchmarks
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export VLLM_DISABLE_COMPILE_CACHE=1

# Helper function to find latest checkpoint path
find_latest_checkpoint() {
    local checkpoint_dir="$1"
    local latest_step=$(ls -d ${checkpoint_dir}/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -z "$latest_step" ]; then
        echo "ERROR: No checkpoint found in ${checkpoint_dir}" >&2
        return 1
    fi
    echo "${checkpoint_dir}/global_step_${latest_step}/actor"
}

# ======================== Step 1: Prepare Dataset ========================
if [ ! -f "$SCRIPT_DIR/data/train.parquet" ]; then
    echo "============================================"
    echo "Step 1: Downloading & merging all benchmark datasets"
    echo "============================================"
    python "$SCRIPT_DIR/prepare_data.py"
else
    echo "Dataset already prepared at $SCRIPT_DIR/data/"
fi

# ======================== Step 2: GRPO Training ========================
echo ""
echo "============================================"
echo "Step 2: Starting GRPO training"
echo "  Model:    vibhuiitj/darwin_iter3_try3_solver_step10"
echo "  Datasets: All benchmarks (merged)"
echo "  GPUs:     8x A100 80GB"
echo "  Rollouts: 8 per question"
echo "  Config:   benchmark_grpo/config.yaml"
echo "============================================"
echo ""

python3 -m verl.trainer.main \
    config=benchmark_grpo/config.yaml

echo ""
echo "============================================"
echo "Step 2: GRPO training complete"
echo "============================================"

# ======================== Step 3: Merge Checkpoint ========================
echo ""
echo "============================================"
echo "Step 3: Merging model checkpoint"
echo "============================================"

LATEST_CHECKPOINT=$(find_latest_checkpoint "$SCRIPT_DIR/checkpoints/darwin_all_benchmarks_grpo")
if [ $? -eq 0 ]; then
    echo "Found checkpoint at: $LATEST_CHECKPOINT"
    python scripts/model_merger.py --local_dir "$LATEST_CHECKPOINT"
    MERGED_MODEL="${LATEST_CHECKPOINT}/huggingface"
    echo "Merged model saved to: $MERGED_MODEL"
else
    echo "ERROR: Could not find checkpoint"
    exit 1
fi

# ======================== Step 4: Evaluate ========================
echo ""
echo "============================================"
echo "Step 4: Running evaluation"
echo "============================================"

bash evaluation/evaluate.bash "$MERGED_MODEL"

echo ""
echo "============================================"
echo "All done!"
echo "  Merged model: $MERGED_MODEL"
echo "============================================"
