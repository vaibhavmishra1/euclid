#!/bin/bash
# Main Script for Knowledge-Set-Based Curriculum Training
#
# Usage: bash scripts/kp_main.sh <base_model> <model_abbr> <knowledge_points_path> [num_iterations]
#
# Example:
#   bash scripts/kp_main.sh Qwen/Qwen3-4B qwen3-4b-ks /path/to/knowledge_points.jsonl 3

Base_model=$1
Model_abbr=$2
Knowledge_points_path=$3
Num_iterations=${4:-3}

# Curriculum Learning Hyperparameters
ALPHA=0.7           # Increase difficulty if avg_reward > alpha
BETA=0.3            # Decrease difficulty if avg_reward < beta
QUESTIONS_PER_SET=5 # Questions to generate per knowledge set

echo "SCRIPT - =============================================="
echo "SCRIPT - Knowledge-Set-Based Curriculum Training"
echo "SCRIPT - =============================================="
echo "SCRIPT - Base Model: $Base_model"
echo "SCRIPT - Model Abbreviation: $Model_abbr"
echo "SCRIPT - Knowledge Points File: $Knowledge_points_path"
echo "SCRIPT - Iterations: $Num_iterations"
echo "SCRIPT - Alpha threshold: $ALPHA"
echo "SCRIPT - Beta threshold: $BETA"
echo "SCRIPT - Questions per Set: $QUESTIONS_PER_SET"
echo "SCRIPT - Difficulty Scale: 1-5 (from MATH dataset levels)"
echo "Storage Path: $STORAGE_PATH"
echo "SCRIPT - =============================================="

# Verify knowledge points file exists
if [ ! -f "$Knowledge_points_path" ]; then
    echo "SCRIPT - Error: Knowledge points file not found: $Knowledge_points_path"
    exit 1
fi

# Count knowledge sets (= number of lines in JSONL)
NUM_SETS=$(wc -l < "$Knowledge_points_path" | tr -d ' ')
TOTAL_QUESTIONS=$((NUM_SETS * QUESTIONS_PER_SET))
echo "SCRIPT - Knowledge Sets: $NUM_SETS"
echo "SCRIPT - Total questions per iteration: $TOTAL_QUESTIONS"
echo "SCRIPT - =============================================="

# Initialize knowledge set state
echo "SCRIPT - Initializing knowledge set state..."
python3 << EOF
from knowledge_curriculum.knowledge_manager import KnowledgeSetManager
import os

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")
state_path = f"{STORAGE_PATH}/ks_state/${Model_abbr}_state.json"
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
print(f"Initial difficulty distribution (from problem levels 1-5): {stats['difficulty_distribution']}")
print(f"Average initial difficulty: {stats['avg_difficulty']:.2f}")

manager.save_state()
print(f"State saved to {state_path}")
EOF

# Start training loop
solver_model=$Base_model
challenger_model=$Base_model
state_path="${STORAGE_PATH}/ks_state/${Model_abbr}_state.json"

for i in $(seq 1 $Num_iterations); do
    echo ""
    echo "SCRIPT - ######################################################"
    echo "SCRIPT - # ITERATION $i / $Num_iterations"
    echo "SCRIPT - ######################################################"
    echo ""
    
    # Step 1: Train Challenger
    echo "SCRIPT - [Iteration $i] Step 1: Training Challenger..."
    challenger_save_name="${Model_abbr}_challenger_v${i}"
    
    bash scripts/kp_challenger_train.sh \
        "$solver_model" \
        "$challenger_model" \
        "$challenger_save_name" \
        "$Knowledge_points_path" \
        "$ALPHA" \
        "$BETA"
    
    # Update challenger model path
    new_challenger="${STORAGE_PATH}/models/${challenger_save_name}/global_step_5/actor/huggingface"
    if [ -d "$new_challenger" ]; then
        challenger_model=$new_challenger
        echo "SCRIPT - Challenger updated: $challenger_model"
    else
        echo "SCRIPT - Warning: New challenger model not found, using previous"
    fi
    
    # Step 2: Generate Questions (5 per knowledge set)
    echo "SCRIPT - [Iteration $i] Step 2: Generating $TOTAL_QUESTIONS questions..."
    question_save_name="${Model_abbr}_iter${i}"
    
    python3 -m knowledge_curriculum.question_generate \
        --model "$challenger_model" \
        --save_name "$question_save_name" \
        --suffix "0" \
        --knowledge_points_path "$Knowledge_points_path" \
        --state_path "$state_path" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --questions_per_set "$QUESTIONS_PER_SET"
    
    # Step 3: Evaluate Questions with Solver (uses R-Zero's reward)
    echo "SCRIPT - [Iteration $i] Step 3: Evaluating questions..."
    
    bash question_evaluate/evaluate.sh "$solver_model" "$question_save_name"
    
    # Combine results from all GPUs (metadata is now preserved by evaluate.py)
    python3 << EOF
import json
import os

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")

# Load evaluated results from all GPUs (metadata already preserved by evaluate.py)
evaluated = []
for gpu_id in range(8):
    try:
        result_file = f"{STORAGE_PATH}/generated_question/${question_save_name}_{gpu_id}_results.json"
        with open(result_file, 'r') as f:
            evaluated.extend(json.load(f))
    except FileNotFoundError:
        pass

print(f"Loaded {len(evaluated)} evaluated results")

# Verify metadata is present
missing_metadata = sum(1 for item in evaluated if item.get('set_id') is None)
if missing_metadata > 0:
    print(f"Warning: {missing_metadata} items missing set_id")

with open(f"{STORAGE_PATH}/generated_question/${question_save_name}_evaluated.json", 'w') as f:
    json.dump(evaluated, f, indent=2, ensure_ascii=False)

print(f"Saved {len(evaluated)} evaluated questions with metadata")
EOF
    
    evaluated_path="${STORAGE_PATH}/generated_question/${question_save_name}_evaluated.json"
    
    # Step 4: Update Difficulties based on average reward per set
    echo "SCRIPT - [Iteration $i] Step 4: Updating difficulties (scale 1-5)..."
    python3 << EOF
from knowledge_curriculum.knowledge_manager import KnowledgeSetManager
import json
import os

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")
state_path = "${state_path}"

manager = KnowledgeSetManager(
    knowledge_points_path="${Knowledge_points_path}",
    alpha=${ALPHA},
    beta=${BETA},
    questions_per_set=${QUESTIONS_PER_SET},
    state_save_path=state_path,
)

# Load results and record rewards per set
with open("${evaluated_path}", 'r') as f:
    results = json.load(f)

for result in results:
    set_id = result.get('set_id')
    # R-Zero's reward is in 'score' or 'overall' field
    reward = result.get('score', result.get('overall', result.get('reward', 0.5)))
    if set_id is not None:
        manager.record_reward(set_id, reward)

# Update difficulties based on average rewards
adjustments = manager.update_difficulties()

# Print summary
increased = sum(1 for a in adjustments.values() if a['action'] == 'increased')
decreased = sum(1 for a in adjustments.values() if a['action'] == 'decreased')
maintained = sum(1 for a in adjustments.values() if a['action'] == 'maintained')
at_max = sum(1 for a in adjustments.values() if a['action'] == 'at_max')
at_min = sum(1 for a in adjustments.values() if a['action'] == 'at_min')

print(f"Difficulty adjustments (scale 1-5):")
print(f"  Increased: {increased}")
print(f"  Decreased: {decreased}")
print(f"  Maintained: {maintained}")
print(f"  At max (5): {at_max}")
print(f"  At min (1): {at_min}")

stats = manager.get_statistics()
print(f"New distribution: {stats['difficulty_distribution']}")
print(f"New average difficulty: {stats['avg_difficulty']:.2f}")

manager.save_state()
EOF
    
    # Step 5: Train Solver
    echo "SCRIPT - [Iteration $i] Step 5: Training Solver..."
    solver_save_name="${Model_abbr}_solver_v${i}"
    
    bash scripts/kp_solver_train.sh \
        "$solver_model" \
        "$evaluated_path" \
        "$solver_save_name" \
        "$BETA" \
        "$ALPHA"
    
    # Update solver model path
    new_solver="${STORAGE_PATH}/models/${solver_save_name}/global_step_15/actor/huggingface"
    if [ -d "$new_solver" ]; then
        solver_model=$new_solver
        echo "Solver updated: $solver_model"
    else
        echo "SCRIPT - Warning: New solver model not found, using previous"
    fi
    
    echo "SCRIPT - [Iteration $i] Complete!"
    echo "SCRIPT -   Challenger: $challenger_model"
    echo "SCRIPT -   Solver: $solver_model"
done

echo "SCRIPT - "
echo "SCRIPT - =============================================="
echo "SCRIPT - Training Complete!"
echo "SCRIPT - =============================================="
echo "SCRIPT - Final Solver: $solver_model"
echo "SCRIPT - Final Challenger: $challenger_model"
echo "SCRIPT - "

# Run final evaluation
echo "SCRIPT - Running final evaluation..."
bash evaluation/evaluate.bash "$solver_model"
