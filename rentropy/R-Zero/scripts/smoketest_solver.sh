#!/bin/bash
# Smoke test: Solver training with minimal compute
# Usage: bash scripts/smoketest_solver.sh <solver_model> <questioner_model> <experiment_name>

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo "=============================================="
echo "SMOKE TEST: Solver Training"
echo "Solver: $solver_model_path"
echo "Questioner: $questioner_model_path"
echo "Experiment: $experiment_name"
echo "=============================================="

export VLLM_DISABLE_COMPILE_CACHE=1

# Generate minimal questions (100 instead of 1000)
echo "Generating questions (minimal)..."
bash question_generate/question_generate.bash $questioner_model_path 100 $experiment_name

# Evaluate generated questions
echo "Evaluating generated questions..."
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name

# Upload to HuggingFace (if configured)
echo "Uploading dataset..."
python question_evaluate/upload.py \
    --repo_name ${experiment_name} \
    --max_score 0.8 \
    --min_score 0.3 \
    --experiment_name ${experiment_name} || echo "Upload skipped (no HF credentials)"

# Train solver with minimal steps
echo "Training solver..."
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config_smoketest.yaml \
    data.max_response_length=1024 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=10 \
    trainer.max_steps=2 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.logger='["console"]'

# Merge model
if [ -d "${STORAGE_PATH}/models/${experiment_name}/global_step_1/actor" ]; then
    echo "Merging model..."
    python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_1/actor
fi

echo "Solver smoke test finished"
