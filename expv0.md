experiment v0-
run baseline R-zero and verify results

cd /root/euclid/expv0/R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/some/big/path"   # where results will be written
mkdir -p "$STORAGE_PATH/evaluation"
cd /root/euclid/expv0/R-Zero
python evaluation/generate.py --model "Qwen/Qwen3-4B-Base" --dataset math

cd /root/euclid/expv0/R-Zero
python evaluation/generate.py --model "$STORAGE_PATH/models/qwen3-4b_solver_v3/global_step_15/actor/huggingface" --dataset math

