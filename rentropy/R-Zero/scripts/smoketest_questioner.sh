#!/bin/bash
# Smoke test: Questioner training with minimal compute
# Usage: bash scripts/smoketest_questioner.sh <solver_model> <questioner_model> <save_name> <diversity_mode>

solver_model_path=$1
questioner_model_path=$2
save_path=$3
diversity_mode=${4:-1}  # Default to mode 1

echo "=============================================="
echo "SMOKE TEST: Questioner Training"
echo "Model: $questioner_model_path"
echo "Save: $save_path"
echo "Diversity Mode: $diversity_mode"
echo "=============================================="

# Update rentropy config with the specified diversity mode
CONFIG_FILE="../rentropy_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    # Use sed to update diversity_mode in config
    sed -i.bak "s/^diversity_mode:.*/diversity_mode: $diversity_mode/" "$CONFIG_FILE"
    echo "Updated rentropy_config.yaml with diversity_mode: $diversity_mode"
fi

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Use single vLLM server for smoke test
export RENTROPY_NUM_SERVERS=1

# Clean up any existing vLLM processes first
echo "Cleaning up any existing vLLM processes..."
pkill -f "vllm_service_init.*port 5000" 2>/dev/null || true
sleep 2

# Start single vLLM service (smoke test uses 1 server)
echo "Starting single vLLM service..."
CUDA_VISIBLE_DEVICES=0 python3 vllm_service_init/start_vllm_server.py \
    --port 5000 \
    --model_path $solver_model_path \
    --gpu_mem_util 0.2 &
VLLM_PID=$!
sleep 30  # Wait for vLLM to load

# Train Questioner with minimal settings
echo "Start training questioner (smoke test)..."

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config_smoketest.yaml \
    data.max_response_length=512 \
    data.rollout_batch_size=16 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=2 \
    worker.reward.reward_function=./examples/reward_function/caller_rentropy.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=2 \
    worker.rollout.gpu_memory_utilization=0.25 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps=2 \
    trainer.save_freq=1 \
    trainer.logger='["console"]'

# Kill vLLM service (aggressive cleanup)
echo "Stopping vLLM service (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
sleep 3

# Force kill if still running
if ps -p $VLLM_PID > /dev/null 2>&1; then
    echo "Force killing vLLM..."
    kill -9 $VLLM_PID 2>/dev/null
fi

# Clean up any orphaned vLLM processes on port 5000
pkill -f "vllm_service_init.*port 5000" 2>/dev/null || true
sleep 2

# Merge model (if checkpoint exists)
if [ -d "${STORAGE_PATH}/models/$save_path/global_step_1/actor" ]; then
    echo "Merging model..."
    python3 scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_1/actor
fi

echo "Questioner smoke test finished for mode $diversity_mode"
