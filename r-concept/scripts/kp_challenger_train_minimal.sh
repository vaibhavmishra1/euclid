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

echo "SCRIPT - =============================================="
echo "SCRIPT - MINIMAL Challenger Training (Single GPU)"
echo "SCRIPT - =============================================="
echo "SCRIPT - Solver Model: $solver_model_path"
echo "SCRIPT - Challenger Model: $challenger_model_path"
echo "SCRIPT - Save Path: $save_path"
echo "SCRIPT - =============================================="

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "SCRIPT - RUN_ID=$RUN_ID"

# Prepare training dataset from knowledge points
echo "SCRIPT - Preparing training dataset from knowledge points..."
state_path="${STORAGE_PATH}/ks_state/$(basename $knowledge_points_path .jsonl)_minimal_state.json"
dataset_dir="${STORAGE_PATH}/datasets/${save_path}_challenger_data"

python3 << EOF
import json
import os
from datasets import Dataset
from knowledge_curriculum.knowledge_manager import KnowledgeSetManager

STORAGE_PATH = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")
knowledge_points_path = "$knowledge_points_path"
state_path = "$state_path"
dataset_dir = "$dataset_dir"
questions_per_set = 1  # Minimal: 1 question per set

# Load knowledge set manager
manager = KnowledgeSetManager(
    knowledge_points_path=knowledge_points_path,
    alpha=${alpha},
    beta=${beta},
    questions_per_set=questions_per_set,
    state_save_path=state_path,
)

# Create training dataset entries
samples = []
for set_id, state in manager.knowledge_sets.items():
    # Create entry with knowledge points and difficulty
    # Format for verl: problem field will be used by format_prompt template
    kp_str = " | ".join(state.knowledge_points)  # Join KPs for display
    entry = {
        "problem": f"Knowledge Points: {kp_str} | Difficulty: {state.difficulty}",
        "answer": "",  # Empty answer - challenger generates the question
        "knowledge_points": json.dumps(state.knowledge_points),  # Serialize list as JSON string
        "difficulty": state.difficulty,
        "set_id": set_id,
    }
    samples.append(entry)

# Create HuggingFace dataset and save as parquet
dataset = Dataset.from_list(samples)
os.makedirs(dataset_dir, exist_ok=True)
dataset.to_parquet(os.path.join(dataset_dir, "train.parquet"))

print(f"Created dataset with {len(samples)} entries")
print(f"Saved to {dataset_dir}")

# DEBUG: Print one example prompt that will be sent to challenger
if samples:
    from jinja2 import Template
    example = samples[0]
    kp_list = json.loads(example['knowledge_points'])
    difficulty = example['difficulty']
    
    # Load the template to show formatted prompt
    template_path = "./examples/format_prompt/knowledge_challenger.jinja"
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        template = Template(template_content)
        formatted_prompt = template.render(
            knowledge_points=kp_list,
            difficulty=difficulty,
            problem=example['problem']
        )
        print("\n" + "="*80)
        print("DEBUG - Example Prompt Sent to Challenger:")
        print("="*80)
        print(f"Knowledge Points: {kp_list}")
        print(f"Difficulty: {difficulty}")
        print(f"\nFormatted Prompt:\n{formatted_prompt}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Could not format prompt template: {e}")
        print(f"Raw example: {example}")
EOF

echo "SCRIPT - Dataset prepared: $dataset_dir"

# Start vLLM service for solver (single instance)
echo "SCRIPT - Starting vLLM service (single GPU)..."
bash vllm_service_init/start.sh $solver_model_path $RUN_ID || echo "SCRIPT - Warning: vLLM service start failed"

echo "SCRIPT - vLLM service started with RUN_ID=$RUN_ID"

# Train Challenger with MINIMAL settings
echo "SCRIPT - Start training challenger (MINIMAL): $challenger_model_path -> $save_path"

# Clean Ray state
unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

# MINIMAL: Single GPU, fewer steps, smaller batch
# Use knowledge points dataset instead of default
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=$dataset_dir \
    data.max_response_length=2048 \
    data.rollout_batch_size=50 \
    worker.actor.model.model_path=$challenger_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./knowledge_curriculum/reward_function.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    data.format_prompt=./examples/format_prompt/knowledge_challenger.jinja \
    worker.rollout.n=2 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.actor.global_batch_size=4 \
    trainer.max_steps=2 \
    trainer.save_freq=1 \
    trainer.logger='["console"]' || {
    echo "SCRIPT - Warning: Challenger training failed"
    exit 1
}

sleep 5

# Merge model checkpoints
echo "SCRIPT - Merging model..."
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_2/actor || {
    echo "SCRIPT - Warning: Model merge failed"
}

sleep 5

# Cleanup
pkill python 2>/dev/null || true

echo "SCRIPT - MINIMAL Challenger training finished"
echo "SCRIPT - Model saved to: ${STORAGE_PATH}/models/$save_path/global_step_2/actor/huggingface"
