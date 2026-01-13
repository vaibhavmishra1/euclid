cd /workspace/euclid/r-concept && source /venv/main/bin/activate && \
unset RAY_ADDRESS && \
export STORAGE_PATH="/workspace/rzero_storage" && \
export HUGGINGFACENAME="vibhuiitj" && \
export VLLM_DISABLE_COMPILE_CACHE=1 && \
rm -rf $STORAGE_PATH/models/kp_number_theory_* \
       $STORAGE_PATH/generated_question/kp_number_theory_* \
       $STORAGE_PATH/ks_state/kp_number_theory_* 2>/dev/null || true && \
echo "Starting training..." && \
export DUMP_DEBUG_DATA=1 && \
pip install -r requirements.txt && \

git config --global user.email "vaibhavm209625@gmail.com" && git config --global user.name "vaibhavmishra1" && \

bash scripts/kp_main_minimal.sh \
    "Qwen/Qwen3-4B-Base" \
    "kp_number_theory" \
    "/workspace/math_concepts_openai_all_number_theory.jsonl" \
    1 \
    50