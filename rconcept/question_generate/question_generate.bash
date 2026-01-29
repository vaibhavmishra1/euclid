# load the model name from the command line
model_name=$1
num_samples=$2
save_name=$3
export VLLM_DISABLE_COMPILE_CACHE=1

# Simple scaling knobs (no extra orchestration):
# - RZERO_NUM_GPUS: how many local GPUs to use for question generation (default 8)
# - num_samples is interpreted as "questions per GPU process"
RZERO_NUM_GPUS=${RZERO_NUM_GPUS:-8}

pids=()
for ((i=0; i<RZERO_NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i python question_generate/question_generate.py --model $model_name --suffix $i --num_samples $num_samples --save_name $save_name &
  pids[$i]=$!
done

for ((i=0; i<RZERO_NUM_GPUS; i++)); do
  wait ${pids[$i]}
done

wait
