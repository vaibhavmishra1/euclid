solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
# Knobs (defaults match upstream behavior):
# - RZERO_NUM_QUESTIONS_PER_GPU: questions generated per GPU process (default 1000)
# - RZERO_NUM_GPUS: how many GPUs to use for gen/eval (default 8)
RZERO_NUM_QUESTIONS_PER_GPU=${RZERO_NUM_QUESTIONS_PER_GPU:-1000}
bash question_generate/question_generate.bash $questioner_model_path $RZERO_NUM_QUESTIONS_PER_GPU $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'start upload'
python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name}
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
    trainer.total_epochs=100 \
    trainer.max_steps=${RZERO_SOLVER_MAX_STEPS:-20} \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.n=${RZERO_SOLVER_RL_ROLLOUT_N:-5} \

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor

sleep 10

echo "solver training finished"

bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
