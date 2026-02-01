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
#   bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b 4
#   bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b 2  # Use mode 2

Base_model=$1
Model_abbr=$2
Diversity_mode=${3:-4}  # Default to mode 4 (full Rentropy)

# Validate diversity mode
if [[ ! "$Diversity_mode" =~ ^[1-4]$ ]]; then
    echo "ERROR: diversity_mode must be 1, 2, 3, or 4. Got: $Diversity_mode"
    exit 1
fi

echo "Model_abbr: $Model_abbr"
echo "Diversity Mode: $Diversity_mode"
echo "Running Rentropy experiment with cluster-entropy diversity reward (mode $Diversity_mode)"

# Check if cluster centroids exist
CENTROIDS_PATH="../cluster_space/cluster_data/centroids.npy"
if [ ! -f "$CENTROIDS_PATH" ]; then
    echo "WARNING: Cluster centroids not found at $CENTROIDS_PATH"
    echo "Please run: python cluster_space/build_clusters.py first"
    echo "Continuing anyway (will fall back to mode 1 if diversity_mode > 1)"
fi

# Initialize first iteration with base model
bash scripts/questioner_train_rentropy.sh $Base_model $Base_model ${Model_abbr}_questioner_v1 $Diversity_mode
bash scripts/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1

# for i in {2..5}; do
#     prev=$((i-1))
    
#     # Train questioner with rentropy reward
#     bash scripts/questioner_train_rentropy.sh \
#         ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
#         ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
#         ${Model_abbr}_questioner_v${i} \
#         $Diversity_mode

#     # Train solver
#     bash scripts/solver_train.sh \
#         ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
#         ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
#         ${Model_abbr}_solver_v${i}
# done

# bash evaluation/evaluate.bash $Base_model
