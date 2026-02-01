#!/bin/bash
# Rentropy: Questioner training with cluster-entropy diversity reward
# Usage: bash scripts/questioner_train_rentropy.sh <solver_model> <questioner_model> <save_name> [diversity_mode]
#
# Arguments:
#   solver_model: Path to solver model
#   questioner_model: Path to questioner model (being trained)
#   save_name: Name for saving checkpoints
#   diversity_mode: (optional) Diversity reward mode (1-4, default: 4)
#     1: Vanilla majority voting reward only (R-Zero baseline)
#     2: Mode 1 + reward for choosing a rare cluster
#     3: Mode 2 + reward for uniqueness from other n-1 questions in batch
#     4: Mode 3 + within-cluster uniqueness reward (full Rentropy)

solver_model_path=$1
questioner_model_path=$2
save_path=$3
diversity_mode=${4:-4}  # Default to mode 4 (full Rentropy)

# Validate diversity mode
if [[ ! "$diversity_mode" =~ ^[1-4]$ ]]; then
    echo "ERROR: diversity_mode must be 1, 2, 3, or 4. Got: $diversity_mode"
    exit 1
fi

echo "save_path: $save_path"
echo "diversity_mode: $diversity_mode"

# Update rentropy config with the specified diversity mode
CONFIG_FILE="../rentropy_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    # Create backup
    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak" 2>/dev/null || true
    # Update diversity_mode in config
    sed -i.bak "s/^diversity_mode:.*/diversity_mode: $diversity_mode/" "$CONFIG_FILE"
    echo "Updated rentropy_config.yaml with diversity_mode: $diversity_mode"
else
    echo "WARNING: rentropy_config.yaml not found at $CONFIG_FILE"
    echo "Diversity mode $diversity_mode will be ignored (using defaults)"
fi

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Start vLLM services (for majority voting)
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# Train Questioner with Rentropy reward function
echo "Start training questioner with Rentropy: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=1024 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=6 \
    worker.reward.reward_function=./examples/reward_function/caller_rentropy.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=6 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5
echo "Stopping vLLM service (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
sleep 3
if ps -p $VLLM_PID > /dev/null 2>&1; then
    echo "Force killing vLLM..."
    kill -9 $VLLM_PID 2>/dev/null
fi
# Clean up any orphaned vLLM processes on port 5000
pkill -f "vllm_service_init.*port 5000" 2>/dev/null || true
sleep 2

# Merge model
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10


echo "questioner training finished"
echo "Questioner training finished for mode $diversity_mode"