#!/bin/bash
# Single-GPU evaluation of generated questions (reads one shard: suffix=0)

model_name=$1
save_name=$2

CUDA_VISIBLE_DEVICES=0 python question_evaluate/evaluate.py \
  --model "$model_name" \
  --suffix 0 \
  --save_name "$save_name"

