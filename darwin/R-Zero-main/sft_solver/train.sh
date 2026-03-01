#!/bin/bash
# Training script for SFT Solver
# This script trains a 4B solver model on the OpenAI validated dataset

set -e  # Exit on error

# Configuration
MODEL_NAME="vibhuiitj/darwin_iter2_dataset_verified_matched"  # Base model to fine-tune
DATA_PATH="/workspace/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json"
OUTPUT_DIR="./outputs/sft_solver_darwin_iter2"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-5

# Create output directory
mkdir -p $OUTPUT_DIR

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed
export PYTHONUNBUFFERED=1

# Load tokens
if [ -f "../tokens.json" ]; then
    export HF_TOKEN=$(python3 -c "import json; print(json.load(open('../tokens.json'))['huggingface'])")
    
fi

echo "========================================"
echo "SFT Solver Training"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run training
python train_sft_solver.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --use_lora 

echo ""
echo "========================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "========================================"
#!/bin/bash
# Training script for SFT Solver
# This script trains a 4B solver model on the OpenAI validated dataset

set -e  # Exit on error

# Configuration
MODEL_NAME="vibhuiitj/darwin_iter2_dataset_verified_matched"  # Base model to fine-tune
DATA_PATH="/workspace/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json"
OUTPUT_DIR="./outputs/sft_solver_darwin_iter2"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-5

# Create output directory
mkdir -p $OUTPUT_DIR

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed
export PYTHONUNBUFFERED=1

# Load tokens
if [ -f "../tokens.json" ]; then
    export HF_TOKEN=$(python3 -c "import json; print(json.load(open('../tokens.json'))['huggingface'])")
    
fi

echo "========================================"
echo "SFT Solver Training"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run training
python train_sft_solver.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --use_lora 

echo ""
echo "========================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "========================================"
#!/bin/bash
# Training script for SFT Solver
# This script trains a 4B solver model on the OpenAI validated dataset

set -e  # Exit on error

# Configuration
MODEL_NAME="vibhuiitj/darwin_iter2_dataset_verified_matched"  # Base model to fine-tune
DATA_PATH="/workspace/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json"
OUTPUT_DIR="./outputs/sft_solver_darwin_iter2"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-5

# Create output directory
mkdir -p $OUTPUT_DIR

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed
export PYTHONUNBUFFERED=1

# Load tokens
if [ -f "../tokens.json" ]; then
    export HF_TOKEN=$(python3 -c "import json; print(json.load(open('../tokens.json'))['huggingface'])")
    
fi

echo "========================================"
echo "SFT Solver Training"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run training
python train_sft_solver.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --use_lora 

echo ""
echo "========================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "========================================"
#!/bin/bash
# Training script for SFT Solver
# This script trains a 4B solver model on the OpenAI validated dataset

set -e  # Exit on error

# Configuration
MODEL_NAME="vibhuiitj/darwin_iter2_dataset_verified_matched"  # Base model to fine-tune
DATA_PATH="/workspace/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json"
OUTPUT_DIR="./outputs/sft_solver_darwin_iter2"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-5

# Create output directory
mkdir -p $OUTPUT_DIR

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed
export PYTHONUNBUFFERED=1

# Load tokens
if [ -f "../tokens.json" ]; then
    export HF_TOKEN=$(python3 -c "import json; print(json.load(open('../tokens.json'))['huggingface'])")
    
fi

echo "========================================"
echo "SFT Solver Training"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run training
python train_sft_solver.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --use_lora 

echo ""
echo "========================================"
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "========================================"
