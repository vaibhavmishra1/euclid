#!/bin/bash
set -euo pipefail

# 1-GPU, 1-iteration R-Zero run with a small backbone + 100-question generation + 100-example MATH eval.
#
# Requirements:
# - export STORAGE_PATH="/path/to/big/storage"
# - export HUGGINGFACENAME="your_hf_username_or_org"   (needed for dataset upload)
# - tokens.json must contain a valid "huggingface" token for private dataset upload
#
# This script trains:
# - questioner_v1 (penalty reward, 6 steps)
# - solver_v1 (20 steps)
# Then evaluates:
# - base model on MATH (first 100 examples)
# - trained solver model on MATH (first 100 examples)

BASE_MODEL="Qwen/Qwen3-0.6B"
ABBR="qwen3-0p6b"

if [[ -z "${STORAGE_PATH:-}" ]]; then
  echo "STORAGE_PATH is not set" >&2
  exit 1
fi
if [[ -z "${HUGGINGFACENAME:-}" ]]; then
  echo "HUGGINGFACENAME is not set (needed for dataset upload)" >&2
  exit 1
fi

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

echo "=== Install 1-GPU training dependencies (no flash_attn) ==="
# Some base images ship an old distutils-installed `blinker` that pip cannot uninstall.
# Flask>=3 needs a newer blinker, so we install it in a way that overrides the system package.
python -m pip install --ignore-installed blinker==1.9.0
python -m pip install -r requirements-train-1gpu.txt

echo "=== Train questioner v1 ==="
bash scripts/questioner_train_penalty_1gpu.sh "$BASE_MODEL" "$BASE_MODEL" "${ABBR}_questioner_v1"

QUESTIONER_HF_DIR="$STORAGE_PATH/models/${ABBR}_questioner_v1/global_step_5/actor/huggingface"

echo "=== Train solver v1 ==="
bash scripts/solver_train_1gpu.sh "$BASE_MODEL" "$QUESTIONER_HF_DIR" "${ABBR}_solver_v1"

SOLVER_HF_DIR="$STORAGE_PATH/models/${ABBR}_solver_v1/global_step_15/actor/huggingface"

echo "=== Eval base model on MATH (first 100) ==="
python -m evaluation.generate --model "$BASE_MODEL" --dataset math --limit 100

echo "=== Eval trained solver on MATH (first 100) ==="
python -m evaluation.generate --model "$SOLVER_HF_DIR" --dataset math --limit 100

echo "Done."

