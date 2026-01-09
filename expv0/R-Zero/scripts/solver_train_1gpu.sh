#!/bin/bash
set -euo pipefail

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo "STORAGE_PATH=${STORAGE_PATH:-}"
echo "Start 1-GPU solver iteration: $experiment_name"

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Generate 100 questions (1 GPU)"
bash question_generate/question_generate_1gpu.bash "$questioner_model_path" 100 "$experiment_name"

echo "Evaluate generated questions (1 GPU)"
bash question_evaluate/evaluate_1gpu.sh "$solver_model_path" "$experiment_name"

echo "Upload filtered dataset to HF (1 GPU shard)"
python question_evaluate/upload_1gpu.py \
  --repo_name "$experiment_name" \
  --max_score 0.8 \
  --min_score 0.3 \
  --experiment_name "$experiment_name"

echo "Train solver (1 GPU)"
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.max_response_length=4096 \
  worker.actor.model.model_path="$solver_model_path" \
  trainer.experiment_name="$experiment_name" \
  trainer.save_checkpoint_path="${STORAGE_PATH}/models/${experiment_name}/" \
  data.train_files="${HUGGINGFACENAME}/${experiment_name}@train" \
  trainer.total_epochs=100 \
  trainer.max_steps=20 \
  data.format_prompt=./examples/format_prompt/solver.jinja \
  trainer.val_freq=4 \
  trainer.n_gpus_per_node=1 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.30 \
  worker.rollout.max_num_batched_tokens=8192 \
  worker.actor.padding_free=false \
  worker.ref.padding_free=false \
  worker.actor.use_torch_compile=false \
  worker.ref.use_torch_compile=false \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1

echo "Merging solver checkpoint..."
python scripts/model_merger.py --local_dir "${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor"

echo "solver training finished"

