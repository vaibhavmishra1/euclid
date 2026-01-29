#!/bin/bash
# Rentropy: Questioner training with cluster-entropy diversity reward
# Usage: bash scripts/questioner_train_rentropy.sh <solver_model> <questioner_model> <save_name>

solver_model_path=$1
questioner_model_path=$2
save_path=$3
echo "save_path: $save_path"

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
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller_rentropy.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=16 \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5

# Merge model
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python

echo "questioner training finished"
