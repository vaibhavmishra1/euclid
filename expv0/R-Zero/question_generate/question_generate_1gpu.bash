#!/bin/bash
# Single-GPU question generation (writes one shard: suffix=0)

model_name=$1
num_samples=$2
save_name=$3

export VLLM_DISABLE_COMPILE_CACHE=1

CUDA_VISIBLE_DEVICES=0 python question_generate/question_generate.py \
  --model "$model_name" \
  --suffix 0 \
  --num_samples "$num_samples" \
  --save_name "$save_name"

