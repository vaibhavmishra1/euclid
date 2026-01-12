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

echo "=============================================="
echo "Knowledge-Set-Based Curriculum Training"
echo "=============================================="
echo "Base Model: $Base_model"
echo "Model Abbreviation: $Model_abbr"
echo "Knowledge Points File: $Knowledge_points_path"
echo "Iterations: $Num_iterations"
echo "Alpha threshold: $ALPHA"
echo "Beta threshold: $BETA"
echo "Questions per Set: $QUESTIONS_PER_SET"
echo "Difficulty Scale: 1-5 (from MATH dataset levels)"
echo "Storage Path: $STORAGE_PATH"
echo "=============================================="

# Verify knowledge points file exists
if [ ! -f "$Knowledge_points_path" ]; then
    echo "Error: Knowledge points file not found: $Knowledge_points_path"
    exit 1
fi

# Count knowledge sets (= number of lines in JSONL)
NUM_SETS=$(wc -l < "$Knowledge_points_path" | tr -d ' ')
TOTAL_QUESTIONS=$((NUM_SETS * QUESTIONS_PER_SET))
echo "Knowledge Sets: $NUM_SETS"
echo "Total questions per iteration: $TOTAL_QUESTIONS"
echo "=============================================="

# Initialize knowledge set state
echo "Initializing knowledge set state..."
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
    echo "######################################################"
    echo "# ITERATION $i / $Num_iterations"
    echo "######################################################"
    echo ""
    
    # Step 1: Train Challenger
    echo "[Iteration $i] Step 1: Training Challenger..."
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
        echo "Challenger updated: $challenger_model"
    else
        echo "Warning: New challenger model not found, using previous"
    fi
    
    # Step 2: Generate Questions (5 per knowledge set)
    echo "[Iteration $i] Step 2: Generating $TOTAL_QUESTIONS questions..."
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
    echo "[Iteration $i] Step 3: Evaluating questions..."
    
    # Backup original file before evaluation (evaluate.py deletes input files)
    cp "${STORAGE_PATH}/generated_question/${question_save_name}_0.json" \
       "${STORAGE_PATH}/generated_question/${question_save_name}_backup.json"
    
    bash question_evaluate/evaluate.sh "$solver_model" "$question_save_name"
    
    # Combine results and preserve set_id by merging with original data
    python3 << EOF
import json
import os

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/rzero_storage")

# First, load original generated questions (with set_id, knowledge_points, difficulty)
# Note: evaluate.py deletes input files, so we need a backup
original_data = []
backup_file = f"{STORAGE_PATH}/generated_question/${question_save_name}_backup.json"
try:
    with open(backup_file, 'r') as f:
        original_data = json.load(f)
    print(f"Loaded {len(original_data)} original questions from backup")
except FileNotFoundError:
    print("Warning: No backup file found")

# Create question -> metadata mapping
question_to_metadata = {}
for item in original_data:
    q = item.get('question', '').strip()
    if q:
        question_to_metadata[q] = {
            'set_id': item.get('set_id'),
            'knowledge_points': item.get('knowledge_points', []),
            'difficulty': item.get('difficulty', 1),
        }

# Load evaluated results from all GPUs
evaluated = []
for gpu_id in range(8):
    try:
        result_file = f"{STORAGE_PATH}/generated_question/${question_save_name}_{gpu_id}_results.json"
        with open(result_file, 'r') as f:
            evaluated.extend(json.load(f))
    except FileNotFoundError:
        pass

print(f"Loaded {len(evaluated)} evaluated results")

# Merge metadata back into evaluated results
combined = []
matched = 0
for item in evaluated:
    q = item.get('question', '').strip()
    if q in question_to_metadata:
        item['set_id'] = question_to_metadata[q]['set_id']
        item['knowledge_points'] = question_to_metadata[q]['knowledge_points']
        item['difficulty'] = question_to_metadata[q]['difficulty']
        matched += 1
    combined.append(item)

print(f"Merged metadata for {matched}/{len(combined)} questions")

# If no evaluated results, use original with default scores
if not combined and original_data:
    print("Warning: No evaluated results, using original data with score=0.5")
    for item in original_data:
        if item.get('valid', False):
            item['score'] = 0.5  # Default score
            combined.append(item)

with open(f"{STORAGE_PATH}/generated_question/${question_save_name}_evaluated.json", 'w') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"Saved {len(combined)} evaluated questions")
EOF
    
    evaluated_path="${STORAGE_PATH}/generated_question/${question_save_name}_evaluated.json"
    
    # Step 4: Update Difficulties based on average reward per set
    echo "[Iteration $i] Step 4: Updating difficulties (scale 1-5)..."
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
    echo "[Iteration $i] Step 5: Training Solver..."
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
        echo "Warning: New solver model not found, using previous"
    fi
    
    echo "[Iteration $i] Complete!"
    echo "  Challenger: $challenger_model"
    echo "  Solver: $solver_model"
done

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Final Solver: $solver_model"
echo "Final Challenger: $challenger_model"
echo ""

# Run final evaluation
echo "Running final evaluation..."
bash evaluation/evaluate.bash "$solver_model"
