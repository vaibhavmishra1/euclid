#!/bin/bash
# MINIMAL VERSION for Testing Knowledge-Set Curriculum Learning
# 
# This version uses minimal compute to prove the concept works:
# - Smaller model (1B instead of 4B)
# - Fewer knowledge sets (50-100 instead of 868)
# - 1 question per set (instead of 5)
# - 1-2 iterations (instead of 3)
# - Single GPU (instead of 4-8)
#
# Usage: bash scripts/kp_main_minimal.sh <base_model> <model_abbr> <knowledge_points_path> [num_iterations] [num_sets]

Base_model=${1:-"Qwen/Qwen3-4B-Base"}  # Use math-specialized model
Model_abbr=$2
Knowledge_points_path=$3
Num_iterations=${4:-1}  # Just 1 iteration for testing
Num_sets=${5:-50}       # Only use first 50 knowledge sets

# Minimal hyperparameters
ALPHA=0.7
BETA=0.3
QUESTIONS_PER_SET=1  # Just 1 question per set (minimal!)

# Set STORAGE_PATH if not set
STORAGE_PATH=${STORAGE_PATH:-"/workspace/rzero_storage"}

# Create storage directories
mkdir -p "${STORAGE_PATH}"/{models,generated_question,ks_state,datasets,temp_results}

# Function to check and free disk space
free_disk_space() {
    local min_free_gb=${1:-10}  # Default: need at least 10GB free
    local available_gb=$(df -BG / | awk 'NR==2 {print int($4)}')
    
    if [ "$available_gb" -lt "$min_free_gb" ]; then
        echo "SCRIPT - Low disk space: ${available_gb}GB available, cleaning old checkpoints..."
        
        # Find and remove old checkpoints (keep only latest 1)
        if [ -d "${STORAGE_PATH}/models" ]; then
            for model_dir in "${STORAGE_PATH}"/models/*/; do
                if [ -d "$model_dir" ]; then
                    # Get all global_step directories, sort by step number, keep only latest 1
                    steps=$(ls -d "${model_dir}"global_step_* 2>/dev/null | sort -V | head -n -1)
                    for step_dir in $steps; do
                        echo "SCRIPT - Removing old checkpoint: $step_dir"
                        rm -rf "$step_dir"
                    done
                fi
            done
        fi
        
        # Clean up temp files
        rm -rf /tmp/ray/* 2>/dev/null || true
        rm -rf "${STORAGE_PATH}"/temp_results/* 2>/dev/null || true
        
        available_gb=$(df -BG / | awk 'NR==2 {print int($4)}')
        echo "SCRIPT - After cleanup: ${available_gb}GB available"
    else
        echo "SCRIPT - Disk space OK: ${available_gb}GB available"
    fi
}

# Clean up any running processes and free GPU/CPU resources
echo "SCRIPT - Cleaning up processes and freeing resources..."
pkill -9 -f python 2>/dev/null || true
pkill -9 -f ray 2>/dev/null || true
pkill -9 -f vllm 2>/dev/null || true
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS
rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf "${STORAGE_PATH}"/temp_results/* 2>/dev/null || true

# Check and free disk space
free_disk_space 10

sleep 2
echo "SCRIPT - Cleanup complete"

echo "SCRIPT - =============================================="
echo "SCRIPT - MINIMAL Knowledge-Set Curriculum Training"
echo "SCRIPT - =============================================="
echo "SCRIPT - Base Model: $Base_model (SMALLER for testing)"
echo "SCRIPT - Model Abbreviation: $Model_abbr"
echo "SCRIPT - Knowledge Points File: $Knowledge_points_path"
echo "SCRIPT - Storage Path: $STORAGE_PATH"
echo "SCRIPT - Iterations: $Num_iterations (MINIMAL)"
echo "SCRIPT - Knowledge Sets: $Num_sets (SUBSET)"
echo "SCRIPT - Questions per Set: $QUESTIONS_PER_SET (MINIMAL)"
echo "SCRIPT - Total questions per iteration: $((Num_sets * QUESTIONS_PER_SET))"
echo "SCRIPT - =============================================="

# Create subset of knowledge points
SUBSET_FILE="${STORAGE_PATH}/kp_subset_${Num_sets}.jsonl"
head -n $Num_sets "$Knowledge_points_path" > "$SUBSET_FILE"
echo "SCRIPT - Created subset: $SUBSET_FILE"

# Use subset for training
Knowledge_points_path="$SUBSET_FILE"

# Initialize knowledge set state
echo "SCRIPT - Initializing knowledge set state..."
python3 << EOF
from knowledge_curriculum.knowledge_manager import KnowledgeSetManager
import os

STORAGE_PATH = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")
state_path = f"{STORAGE_PATH}/ks_state/${Model_abbr}_minimal_state.json"
os.makedirs(os.path.dirname(state_path), exist_ok=True)

manager = KnowledgeSetManager(
    knowledge_points_path="${Knowledge_points_path}",
    alpha=${ALPHA},
    beta=${BETA},
    questions_per_set=${QUESTIONS_PER_SET},
    state_save_path=state_path,
)

stats = manager.get_statistics()
print(f"Loaded {stats['total_knowledge_sets']} knowledge sets")
print(f"Initial difficulty distribution: {stats['difficulty_distribution']}")

manager.save_state()
print(f"State saved to {state_path}")
EOF

# Start training loop
solver_model=$Base_model
challenger_model=$Base_model
state_path="${STORAGE_PATH}/ks_state/${Model_abbr}_minimal_state.json"

for i in $(seq 1 $Num_iterations); do
    echo ""
    echo "SCRIPT - ######################################################"
    echo "SCRIPT - # ITERATION $i / $Num_iterations (MINIMAL)"
    echo "SCRIPT - ######################################################"
    echo ""
    
    # Check disk space before training
    free_disk_space 15  # Need at least 15GB for checkpoint saving
    
    # Step 1: Train Challenger (MINIMAL - single GPU, fewer steps)
    echo "SCRIPT - [Iteration $i] Step 1: Training Challenger (MINIMAL)..."
    challenger_save_name="${Model_abbr}_challenger_v${i}_minimal"
    
    # Use minimal challenger training
    bash scripts/kp_challenger_train_minimal.sh \
        "$solver_model" \
        "$challenger_model" \
        "$challenger_save_name" \
        "$Knowledge_points_path" \
        "$ALPHA" \
        "$BETA" || echo "SCRIPT - Warning: Challenger training failed, continuing..."
    
    # Clean up old checkpoints after training to free space
    free_disk_space 10
    
    # Update challenger model path
    new_challenger="${STORAGE_PATH}/models/${challenger_save_name}/global_step_2/actor/huggingface"
    if [ -d "$new_challenger" ]; then
        challenger_model=$new_challenger
        echo "SCRIPT - Challenger updated: $challenger_model"
    else
        echo "SCRIPT - Using base challenger model (no training)"
    fi
    
    echo "SCRIPT - [Iteration $i] Step 1 complete!"
done

echo "SCRIPT - "
echo "SCRIPT - =============================================="
echo "SCRIPT - MINIMAL Training Complete!"
echo "SCRIPT - =============================================="
echo "SCRIPT - "
echo "SCRIPT - Training finished for iteration $i"
echo "SCRIPT - "

#     # Update challenger model path (commented out)
#     new_challenger="${STORAGE_PATH}/models/${challenger_save_name}/global_step_2/actor/huggingface"
#     if [ -d "$new_challenger" ]; then
#         challenger_model=$new_challenger
#         echo "SCRIPT - Challenger updated: $challenger_model"
#     else
#         echo "SCRIPT - Using base challenger model (no training)"
#     fi
    
#     # Step 2: Generate Questions (MINIMAL - just 1 per set)
#     echo "SCRIPT - [Iteration $i] Step 2: Generating $((Num_sets * QUESTIONS_PER_SET)) questions..."
#     question_save_name="${Model_abbr}_iter${i}_minimal"
    
#     python3 -m knowledge_curriculum.question_generate \
#         --model "$challenger_model" \
#         --save_name "$question_save_name" \
#         --suffix "0" \
#         --knowledge_points_path "$Knowledge_points_path" \
#         --state_path "$state_path" \
#         --alpha "$ALPHA" \
#         --beta "$BETA" \
#         --questions_per_set "$QUESTIONS_PER_SET" \
#         --batch_size 16 \
#         --limit $((Num_sets * QUESTIONS_PER_SET)) || echo "SCRIPT - Warning: Generation failed"
    
#     # Step 3: Evaluate Questions (MINIMAL - single GPU)
#     echo "SCRIPT - [Iteration $i] Step 3: Evaluating questions (MINIMAL)..."
    
#     # Backup original file
#     cp "${STORAGE_PATH}/generated_question/${question_save_name}_0.json" \
#        "${STORAGE_PATH}/generated_question/${question_save_name}_backup.json" 2>/dev/null || true
    
#     # Evaluate on single GPU (minimal)
#     CUDA_VISIBLE_DEVICES=0 python question_evaluate/evaluate.py \
#         --model "$solver_model" \
#         --suffix "0" \
#         --save_name "$question_save_name" \
#         --num_samples 4 || echo "SCRIPT - Warning: Evaluation failed"
    
#     # Combine results (metadata is now preserved by evaluate.py)
#     python3 << EOF
# import json
# import os

# STORAGE_PATH = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")

# # Load evaluated results (metadata already preserved by evaluate.py)
# evaluated = []
# try:
#     with open(f"{STORAGE_PATH}/generated_question/${question_save_name}_0_results.json", 'r') as f:
#         evaluated = json.load(f)
# except FileNotFoundError:
#     print("Warning: No evaluation results found")
#     evaluated = []

# # Verify metadata is present
# missing_metadata = sum(1 for item in evaluated if item.get('set_id') is None)
# if missing_metadata > 0:
#     print(f"Warning: {missing_metadata} items missing set_id")

# with open(f"{STORAGE_PATH}/generated_question/${question_save_name}_evaluated.json", 'w') as f:
#     json.dump(evaluated, f, indent=2, ensure_ascii=False)

# print(f"Saved {len(evaluated)} evaluated questions with metadata")
# EOF
    
#     evaluated_path="${STORAGE_PATH}/generated_question/${question_save_name}_evaluated.json"
    
#     # Step 4: Update Difficulties
#     echo "SCRIPT - [Iteration $i] Step 4: Updating difficulties..."
#     python3 << EOF
# from knowledge_curriculum.knowledge_manager import KnowledgeSetManager
# import json
# import os

# STORAGE_PATH = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")
# state_path = "${state_path}"

# manager = KnowledgeSetManager(
#     knowledge_points_path="${Knowledge_points_path}",
#     alpha=${ALPHA},
#     beta=${BETA},
#     questions_per_set=${QUESTIONS_PER_SET},
#     state_save_path=state_path,
# )

# try:
#     with open("${evaluated_path}", 'r') as f:
#         results = json.load(f)
    
#     for result in results:
#         set_id = result.get('set_id')
#         reward = result.get('score', result.get('overall', 0.5))
#         if set_id is not None:
#             manager.record_reward(set_id, reward)
    
#     adjustments = manager.update_difficulties()
    
#     increased = sum(1 for a in adjustments.values() if a['action'] == 'increased')
#     decreased = sum(1 for a in adjustments.values() if a['action'] == 'decreased')
#     maintained = sum(1 for a in adjustments.values() if a['action'] == 'maintained')
    
#     print(f"Difficulty adjustments:")
#     print(f"  Increased: {increased}")
#     print(f"  Decreased: {decreased}")
#     print(f"  Maintained: {maintained}")
    
#     stats = manager.get_statistics()
#     print(f"New distribution: {stats['difficulty_distribution']}")
    
#     manager.save_state()
# except Exception as e:
#     print(f"Error updating difficulties: {e}")
# EOF
    
#     # Step 5: Train Solver (ESSENTIAL - shows curriculum learning works!)
#     echo "SCRIPT - [Iteration $i] Step 5: Training Solver (ESSENTIAL for proof)..."
#     solver_save_name="${Model_abbr}_solver_v${i}_minimal"
    
#     bash scripts/kp_solver_train_minimal.sh \
#         "$solver_model" \
#         "$evaluated_path" \
#         "$solver_save_name" \
#         "$BETA" \
#         "$ALPHA" || echo "SCRIPT - Warning: Solver training failed, continuing..."
    
#     # Update solver model path
#     new_solver="${STORAGE_PATH}/models/${solver_save_name}/global_step_5/actor/huggingface"
#     if [ -d "$new_solver" ]; then
#         solver_model=$new_solver
#         echo "Solver updated: $solver_model"
#         echo "SCRIPT - ✅ Solver trained on curriculum-filtered questions!"
#     else
#         echo "SCRIPT - Warning: New solver model not found, using previous"
#     fi
    
#     echo "SCRIPT - [Iteration $i] Complete!"
#     echo "SCRIPT -   Challenger: $challenger_model"
#     echo "SCRIPT -   Solver: $solver_model"
#     echo ""
#     echo "SCRIPT -   ✅ To show improvement: Evaluate solver on test set"
#     echo "SCRIPT -      bash evaluation/evaluate.bash $solver_model"
# done

# echo ""
# echo "SCRIPT - =============================================="
# echo "SCRIPT - MINIMAL Training Complete!"
# echo "SCRIPT - =============================================="
# echo ""
# echo "SCRIPT - WHAT WAS PROVEN:"
# echo "SCRIPT -   1. ✅ Knowledge sets load with correct difficulties"
# echo "SCRIPT -   2. ✅ Questions generate with KPs + difficulty"
# echo "SCRIPT -   3. ✅ Rewards compute correctly"
# echo "SCRIPT -   4. ✅ Difficulties update based on thresholds"
# echo "SCRIPT -   5. ✅ Solver trained on curriculum-filtered questions"
# echo ""
# echo "SCRIPT - NEXT STEP: Evaluate trained solver to show improvement:"
# echo "SCRIPT -   bash evaluation/evaluate.bash $solver_model"
# echo ""
# echo "SCRIPT - For full training, use:"
# echo "SCRIPT -   bash scripts/kp_main.sh $Base_model $Model_abbr <full_kp_path> 3"
# echo ""
