#!/bin/bash
# Rentropy: Main training loop with cluster-entropy diversity reward
# Usage: bash scripts/main_rentropy.sh <base_model> <model_abbr> [diversity_mode]
#
# Arguments:
#   base_model: Base model path (e.g., Qwen/Qwen3-4B-Base)
#   model_abbr: Model abbreviation for naming experiments (e.g., qwen3-4b)
#   diversity_mode: (optional) Diversity reward mode (1-4, default: 4)
#     1: Vanilla majority voting reward only (R-Zero baseline)
#     2: Mode 1 + reward for choosing a rare cluster
#     3: Mode 2 + reward for uniqueness from other n-1 questions in batch
#     4: Mode 3 + within-cluster uniqueness reward (full Rentropy)
#
# Example: 
#   bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b 1 > tempf.txt
#   bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b 2  # Use mode 2

Base_model=$1
Model_abbr=$2
Diversity_mode=${3:-4}  # Default to mode 4 (full Rentropy)
export HUGGINGFACENAME="vibhuiitj"
export STORAGE_PATH="/workspace/euclid/rentropy/R-Zero-main/storage"
export PYTHONPATH="/workspace/euclid/rentropy/R-Zero-main:$PYTHONPATH"

# Validate diversity mode
if [[ ! "$Diversity_mode" =~ ^[1-4]$ ]]; then
    echo "ERROR: diversity_mode must be 1, 2, 3, or 4. Got: $Diversity_mode"
    exit 1
fi

echo "Model_abbr: $Model_abbr"
echo "Diversity Mode: $Diversity_mode"
echo "Running Rentropy experiment with cluster-entropy diversity reward (mode $Diversity_mode)"
echo "STORAGE_PATH: $STORAGE_PATH"
echo "HUGGINGFACENAME: $HUGGINGFACENAME"
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"
# Check if cluster centroids exist
CENTROIDS_PATH="../cluster_space/cluster_data/centroids.npy"
if [ ! -f "$CENTROIDS_PATH" ]; then
    echo "WARNING: Cluster centroids not found at $CENTROIDS_PATH"
    echo "Please run: python cluster_space/build_clusters.py first"
    echo "Continuing anyway (will fall back to mode 1 if diversity_mode > 1)"
fi

# Helper function to find latest checkpoint path
find_latest_checkpoint() {
    local checkpoint_dir="$1"
    if [ -f "${checkpoint_dir}/latest_global_step.txt" ]; then
        # Use tracker file if available
        local latest_step=$(cat "${checkpoint_dir}/latest_global_step.txt")
        echo "${checkpoint_dir}/global_step_${latest_step}/actor/huggingface"
    else
        # Fallback: find the highest global_step_* directory
        local latest_step=$(ls -d ${checkpoint_dir}/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
        if [ -z "$latest_step" ]; then
            echo "ERROR: No checkpoint found in ${checkpoint_dir}" >&2
            return 1
        fi
        echo "${checkpoint_dir}/global_step_${latest_step}/actor/huggingface"
    fi
}

# Initialize first iteration with base model
bash scripts/questioner_train_rentropy.sh $Base_model $Base_model ${Model_abbr}_questioner_v1 $Diversity_mode
# QUESTIONER_V1_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_questioner_v1")
# bash scripts/solver_train.sh $Base_model "$QUESTIONER_V1_CHECKPOINT" ${Model_abbr}_solver_v1

# for i in {2..5}; do
#     prev=$((i-1))
    
#     # Train questioner with rentropy reward
#     SOLVER_PREV_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}")
#     QUESTIONER_PREV_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}")
#     bash scripts/questioner_train_rentropy.sh \
#         "$SOLVER_PREV_CHECKPOINT" \
#         "$QUESTIONER_PREV_CHECKPOINT" \
#         ${Model_abbr}_questioner_v${i} \
#         $Diversity_mode

#     # Train solver
#     SOLVER_PREV_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}")
#     QUESTIONER_CURR_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}")
#     bash scripts/solver_train.sh \
#         "$SOLVER_PREV_CHECKPOINT" \
#         "$QUESTIONER_CURR_CHECKPOINT" \
#         ${Model_abbr}_solver_v${i}
# done

# bash evaluation/evaluate.bash $Base_model
