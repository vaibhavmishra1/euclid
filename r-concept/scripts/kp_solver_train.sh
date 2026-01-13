#!/bin/bash
# Knowledge-Point-Based Solver Training Script
#
# Usage: bash scripts/kp_solver_train.sh <solver_model> <questions_path> <save_name> <min_score> <max_score>

solver_model_path=$1
questions_path=$2
experiment_name=$3
min_score=${4:-0.3}
max_score=${5:-0.7}

echo "SCRIPT - =============================================="
echo "SCRIPT - Knowledge-Point-Based Solver Training"
echo "SCRIPT - =============================================="
echo "SCRIPT - Solver Model: $solver_model_path"
echo "SCRIPT - Questions Path: $questions_path"
echo "SCRIPT - Experiment Name: $experiment_name"
echo "SCRIPT - Min Score (beta): $min_score"
echo "SCRIPT - Max Score (alpha): $max_score"
echo "SCRIPT - =============================================="

export VLLM_DISABLE_COMPILE_CACHE=1

# Prepare training data by filtering questions
echo "SCRIPT - Preparing training data..."
python3 << EOF
import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import login

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME", "default_user")

# Load tokens
try:
    with open('tokens.json', 'r') as f:
        token = json.load(f)['huggingface']
    login(token=token)
except:
    print("Warning: Could not login to HuggingFace")

# Load questions
try:
    with open("${questions_path}", 'r') as f:
        data = json.load(f)
except:
    print(f"Error loading {questions_path}")
    exit(1)

# Filter by score
filtered = []
for item in data:
    score = item.get('score', 0)
    if ${min_score} <= score <= ${max_score}:
        if item.get('question') and item.get('answer'):
            filtered.append({
                'problem': item['question'],
                'answer': item['answer'],
                'score': score,
                'knowledge_points': item.get('knowledge_points', []),
                'set_id': item.get('set_id', -1),
                'difficulty': item.get('difficulty', 1),
            })

print(f"Filtered {len(filtered)} questions from {len(data)} total")

if len(filtered) == 0:
    print("Warning: No questions passed the filter!")
    exit(1)

# Create HuggingFace dataset
dataset = Dataset.from_list(filtered)
dataset_dict = DatasetDict({"train": dataset})

# Push to hub
try:
    dataset_dict.push_to_hub(f"{HUGGINGFACENAME}/${experiment_name}", private=True)
    print(f"Uploaded dataset to {HUGGINGFACENAME}/${experiment_name}")
except Exception as e:
    print(f"Warning: Could not upload to HuggingFace: {e}")
    # Save locally instead
    local_path = f"{STORAGE_PATH}/datasets/${experiment_name}"
    os.makedirs(local_path, exist_ok=True)
    dataset_dict.save_to_disk(local_path)
    print(f"Saved dataset locally to {local_path}")

EOF

# Train solver
echo "SCRIPT - Training solver..."
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=100 \
    trainer.max_steps=20 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1

# Merge model
echo "SCRIPT - Merging model..."
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor

sleep 10

echo "Solver training finished"
echo "SCRIPT - Model saved to: ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface"

# Optional: Run evaluation
# bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
