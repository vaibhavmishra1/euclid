#!/bin/bash
# Knowledge-Point-Based Challenger Training Script
#
# Usage: bash scripts/kp_challenger_train.sh <solver_model> <challenger_model> <save_name> <kp_path> <alpha> <beta>

solver_model_path=$1
challenger_model_path=$2
save_path=$3
knowledge_points_path=$4
alpha=${5:-0.7}
beta=${6:-0.3}

echo "=============================================="
echo "Knowledge-Point-Based Challenger Training"
echo "=============================================="
echo "Solver Model: $solver_model_path"
echo "Challenger Model: $challenger_model_path"
echo "Save Path: $save_path"
echo "Knowledge Points: $knowledge_points_path"
echo "Alpha: $alpha"
echo "Beta: $beta"
echo "=============================================="

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Start vLLM services for solver (for uncertainty computation)
echo "Starting vLLM services..."
bash vllm_service_init/start.sh $solver_model_path $RUN_ID

echo "vLLM services started with RUN_ID=$RUN_ID"

# Train Challenger with knowledge-point-based prompts
echo "Start training challenger: $challenger_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$challenger_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./knowledge_curriculum/reward_function.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/knowledge_challenger.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=16 \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5

# Merge model checkpoints
echo "Merging model..."
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

# Cleanup
pkill python

echo "Challenger training finished"
echo "Model saved to: ${STORAGE_PATH}/models/$save_path/global_step_5/actor/huggingface"
