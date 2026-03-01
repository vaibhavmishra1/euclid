#!/bin/bash
# Rentropy: Main training loop with cluster-entropy diversity reward
# Usage: bash scripts/main_rentropy.sh <base_model> <model_abbr>
#
# Example: bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b

Base_model=$1
Model_abbr=$2
echo "Model_abbr: $Model_abbr"
echo "Running Rentropy experiment with cluster-entropy diversity reward"

# Check if cluster centroids exist
CENTROIDS_PATH="../cluster_space/cluster_data/centroids.npy"
if [ ! -f "$CENTROIDS_PATH" ]; then
    echo "WARNING: Cluster centroids not found at $CENTROIDS_PATH"
    echo "Please run: python cluster_space/build_clusters.py first"
    echo "Continuing anyway (will fall back to mode 1 if diversity_mode > 1)"
fi

# Initialize first iteration with base model
bash scripts/questioner_train_rentropy.sh $Base_model $Base_model ${Model_abbr}_questioner_v1
bash scripts/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1

for i in {2..5}; do
    prev=$((i-1))
    
    # Train questioner with rentropy reward
    bash scripts/questioner_train_rentropy.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    # Train solver
    bash scripts/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${Model_abbr}_solver_v${i}
done

bash evaluation/evaluate.bash $Base_model
