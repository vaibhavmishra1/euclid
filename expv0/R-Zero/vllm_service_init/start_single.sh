#!/bin/bash
# Start a single local vLLM grading server on GPU 0 (port 5000).
#
# Usage:
#   bash vllm_service_init/start_single.sh <model_path> [gpu_mem_util] [port]
#
model_path=$1
gpu_mem_util=${2:-0.25}
port=${3:-5000}

export VLLM_DISABLE_COMPILE_CACHE=1

CUDA_VISIBLE_DEVICES=0 python vllm_service_init/start_vllm_server.py \
  --port "${port}" \
  --model_path "${model_path}" \
  --gpu_mem_util "${gpu_mem_util}" &

echo $! > /tmp/rzero_vllm_server.pid
echo "Started vLLM server PID $(cat /tmp/rzero_vllm_server.pid) on port ${port}"

