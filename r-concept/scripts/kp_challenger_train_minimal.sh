#!/bin/bash
# MINIMAL Challenger Training Script (Single GPU, Fewer Steps)
#
# Usage: bash scripts/kp_challenger_train_minimal.sh <solver_model> <challenger_model> <save_name> <kp_path> <alpha> <beta>

solver_model_path=$1
challenger_model_path=$2
save_path=$3
knowledge_points_path=$4
alpha=${5:-0.7}
beta=${6:-0.3}

echo "=============================================="
echo "MINIMAL Challenger Training (Single GPU)"
echo "=============================================="
echo "Solver Model: $solver_model_path"
echo "Challenger Model: $challenger_model_path"
echo "Save Path: $save_path"
echo "=============================================="

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Start vLLM service for solver (single instance)
echo "Starting vLLM service (single GPU)..."
bash vllm_service_init/start.sh $solver_model_path $RUN_ID || echo "Warning: vLLM service start failed"

echo "vLLM service started with RUN_ID=$RUN_ID"

# Train Challenger with MINIMAL settings
echo "Start training challenger (MINIMAL): $challenger_model_path -> $save_path"

# MINIMAL: Single GPU, fewer steps, smaller batch
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.model.model_path=$challenger_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./knowledge_curriculum/reward_function.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    data.format_prompt=./examples/format_prompt/knowledge_challenger.jinja \
    worker.rollout.n=2 \
    worker.actor.global_batch_size=4 \
    trainer.max_steps=2 \
    trainer.save_freq=1 || {
    echo "Warning: Challenger training failed"
    exit 1
}

sleep 5

# Merge model checkpoints
echo "Merging model..."
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_2/actor || {
    echo "Warning: Model merge failed"
}

sleep 5

# Cleanup
pkill python 2>/dev/null || true

echo "MINIMAL Challenger training finished"
echo "Model saved to: ${STORAGE_PATH}/models/$save_path/global_step_2/actor/huggingface"
