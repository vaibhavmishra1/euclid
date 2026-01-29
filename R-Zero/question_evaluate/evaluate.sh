#!/bin/bash

model_name=$1
save_name=$2

pids=()

# Simple scaling knob:
# - RZERO_NUM_GPUS: how many local GPUs to use for evaluation (default 8)
# - RZERO_SOLVER_ROLLOUT_N: number of solver samples per question (default 9)
RZERO_NUM_GPUS=${RZERO_NUM_GPUS:-8}
RZERO_SOLVER_ROLLOUT_N=${RZERO_SOLVER_ROLLOUT_N:-9}

for ((i=0; i<RZERO_NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py --model $model_name --suffix $i --save_name $save_name --num_samples $RZERO_SOLVER_ROLLOUT_N &
  pids[$i]=$!
done

wait ${pids[0]}
echo "Task 0 finished."

timeout_duration=3600

(
  sleep $timeout_duration
  echo "Timeout reached. Killing remaining tasks..."
  for ((i=1; i<RZERO_NUM_GPUS; i++)); do
    if kill -0 ${pids[$i]} 2>/dev/null; then
      kill -9 ${pids[$i]} 2>/dev/null
      echo "Killed task $i"
    fi
  done
) &

for ((i=1; i<RZERO_NUM_GPUS; i++)); do
  wait ${pids[$i]} 2>/dev/null
done
