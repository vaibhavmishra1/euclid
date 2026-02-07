solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

# Add project root to PYTHONPATH
export PYTHONPATH="/workspace/euclid/rentropy/R-Zero-main:$PYTHONPATH"

# Helper function to find latest checkpoint path
find_latest_checkpoint() {
    local checkpoint_dir="$1"
    # Find the highest global_step_* directory
    local latest_step=$(ls -d ${checkpoint_dir}/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -z "$latest_step" ]; then
        echo "ERROR: No checkpoint found in ${checkpoint_dir}" >&2
        return 1
    fi
    echo "${checkpoint_dir}/global_step_${latest_step}/actor"
}

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path 1000 $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'start upload'
python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.99 --min_score 0.5 --experiment_name ${experiment_name}
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=10 \
    trainer.max_steps=10 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=4 \
    trainer.save_freq=2 \
    worker.rollout.n=4 \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=16 

echo "merging model"
# Find the latest checkpoint dynamically instead of hardcoding global_step_15
LATEST_CHECKPOINT=$(find_latest_checkpoint "${STORAGE_PATH}/models/${experiment_name}")
if [ $? -eq 0 ]; then
    echo "Found checkpoint at: $LATEST_CHECKPOINT"
    python scripts/model_merger.py --local_dir "$LATEST_CHECKPOINT"
else
    echo "ERROR: Could not find checkpoint for ${experiment_name}"
    exit 1
fi

sleep 10
sleep 5
echo "Stopping vLLM service (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
sleep 3
if ps -p $VLLM_PID > /dev/null 2>&1; then
    echo "Force killing vLLM..."
    kill -9 $VLLM_PID 2>/dev/null
fi
pkill -f "vllm_service_init.*port 5000" 2>/dev/null || true


sleep 10

echo "solver training finished"
echo "Solver training finished"
# bash evaluation/evaluate.bash ${LATEST_CHECKPOINT}/huggingface
