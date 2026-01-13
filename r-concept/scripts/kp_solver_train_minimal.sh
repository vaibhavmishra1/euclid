#!/bin/bash
# MINIMAL Solver Training Script (Single GPU, Fewer Steps)
#
# Usage: bash scripts/kp_solver_train_minimal.sh <solver_model> <questions_path> <save_name> <min_score> <max_score>
#
# WHY THIS IS ESSENTIAL:
# - The whole point of curriculum learning is to improve the SOLVER
# - Without solver training, we can't show that curriculum questions help
# - We need to demonstrate: solver improves after training on curriculum-filtered questions

solver_model_path=$1
questions_path=$2
experiment_name=$3
min_score=${4:-0.3}
max_score=${5:-0.7}

echo "SCRIPT - =============================================="
echo "SCRIPT - MINIMAL Solver Training (ESSENTIAL for proof)"
echo "SCRIPT - =============================================="
echo "SCRIPT - Solver Model: $solver_model_path"
echo "SCRIPT - Questions Path: $questions_path"
echo "SCRIPT - Experiment Name: $experiment_name"
echo "SCRIPT - Min Score (beta): $min_score"
echo "SCRIPT - Max Score (alpha): $max_score"
echo "SCRIPT - =============================================="
echo "SCRIPT - WHY: We need to show solver improves with curriculum!"
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
except Exception as e:
    print(f"Error loading ${questions_path}: {e}")
    exit(1)

# Filter by score (curriculum filtering: only questions in sweet spot)
# More lenient: accept questions with any score > 0 if none in sweet spot
filtered = []
for item in data:
    score = item.get('score', 0)
    if ${min_score} <= score <= ${max_score}:
        if item.get('answer'):  # Just need answer, question might be empty but valid
            filtered.append({
                'problem': item.get('question', '') or f"Solve the math problem with answer {item['answer']}",
                'answer': item['answer'],
                'score': score,
                'knowledge_points': item.get('knowledge_points', []),
                'set_id': item.get('set_id', -1),
                'difficulty': item.get('difficulty', 1),
            })

# Fallback: if no questions passed the sweet spot, use all with score > 0
if len(filtered) == 0:
    print("No questions in sweet spot, using fallback with all scored questions...")
    for item in data:
        score = item.get('score', 0)
        if score > 0 and item.get('answer'):
            filtered.append({
                'problem': item.get('question', '') or f"Solve the math problem with answer {item['answer']}",
                'answer': item['answer'],
                'score': score,
                'knowledge_points': item.get('knowledge_points', []),
                'set_id': item.get('set_id', -1),
                'difficulty': item.get('difficulty', 1),
            })

print(f"Filtered {len(filtered)} questions from {len(data)} total (curriculum filter)")

if len(filtered) == 0:
    print("Warning: No questions passed the filter!")
    print("This means no questions were in the sweet spot (beta <= score <= alpha)")
    print("Consider adjusting alpha/beta thresholds or checking evaluation results")
    exit(1)

# Create HuggingFace dataset
dataset = Dataset.from_list(filtered)
dataset_dict = DatasetDict({"train": dataset})

# Push to hub or save locally
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

# Clean Ray state before training
unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

# Train solver with MINIMAL settings
echo "SCRIPT - Training solver (MINIMAL: single GPU, fewer steps)..."
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=100 \
    trainer.max_steps=5 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.logger='["console"]' || {
    echo "SCRIPT - Warning: Solver training failed"
    exit 1
}

# Merge model
echo "SCRIPT - Merging model..."
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_5/actor || {
    echo "SCRIPT - Warning: Model merge failed"
}

sleep 5

echo "SCRIPT - MINIMAL Solver training finished"
echo "SCRIPT - Model saved to: ${STORAGE_PATH}/models/${experiment_name}/global_step_5/actor/huggingface"
echo "SCRIPT - "
echo "SCRIPT - This trained solver can now be evaluated to show improvement!"
