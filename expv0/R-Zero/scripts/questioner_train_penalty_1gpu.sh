#!/bin/bash
set -euo pipefail

solver_model_path=$1
questioner_model_path=$2
save_path=$3

echo "STORAGE_PATH=${STORAGE_PATH:-}"
echo "Start 1-GPU questioner training: $questioner_model_path -> $save_path"

# Start a single local grading server (on the same GPU).
# For Qwen3-0.6B this is usually OK; if you OOM, reduce gpu_mem_util further (e.g. 0.15).
bash vllm_service_init/start_single.sh "$solver_model_path" 0.25 5000
VLLM_PID="$(cat /tmp/rzero_vllm_server.pid)"

cleanup() {
  echo "Cleaning up vLLM server (PID ${VLLM_PID})..."
  kill -9 "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.max_response_length=4096 \
  worker.actor.model.model_path="$questioner_model_path" \
  trainer.experiment_name="$save_path" \
  trainer.save_checkpoint_path="${STORAGE_PATH}/models/${save_path}/" \
  trainer.total_epochs=1000 \
  trainer.max_steps=6 \
  trainer.save_freq=1 \
  trainer.val_freq=-1 \
  trainer.n_gpus_per_node=1 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.30 \
  worker.rollout.max_num_batched_tokens=8192 \
  worker.actor.padding_free=false \
  worker.ref.padding_free=false \
  worker.actor.use_torch_compile=false \
  worker.ref.use_torch_compile=false \
  worker.reward.reward_function=./examples/reward_function/caller_penalty_single.py:compute_score \
  data.format_prompt=./examples/format_prompt/questioner.jinja \
  worker.rollout.n=4 \
  worker.actor.global_batch_size=4 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1

echo "Merging questioner checkpoint..."
python scripts/model_merger.py --local_dir "${STORAGE_PATH}/models/${save_path}/global_step_5/actor"

echo "questioner training finished"

