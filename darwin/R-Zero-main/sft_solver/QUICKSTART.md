# Quick Start Guide - SFT Solver Training

This guide will help you train your solver model in 5 minutes.

## Prerequisites

```bash
cd /Users/vaibhav/Desktop/brahma/tree/euclid/darwin/R-Zero-main/sft_solver
pip install -r requirements.txt
```

## Step 1: Verify Data

Make sure your dataset exists:

```bash
ls -lh /Users/vaibhav/Desktop/brahma/tree/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json
```

Expected: ~1,661 examples (all verified correct with OpenAI)

## Step 2: Configure Training

Edit `train.sh` to set your model:

```bash
# For 4B models, use one of:
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"           # Llama 8B (closest to 4B)
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"              # Qwen Math 7B (math-specialized)
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"         # Mistral 7B
```

**Note**: True 4B models are rare. The 7-8B models above are closest and work well with 4-bit quantization.

## Step 3: Run Training

```bash
./train.sh
```

This will:
- Load the OpenAI-validated dataset (1,661 verified examples)
- Apply 4-bit quantization for memory efficiency
- Use LoRA for parameter-efficient training
- Train for 3 epochs (~2-4 hours on a single GPU)
- Save checkpoints every 100 steps
- Log to Wandb

## Step 4: Monitor Progress

Training metrics are logged to Wandb. You can monitor:
- Training loss (should decrease)
- Validation loss (should decrease)
- Learning rate schedule
- Tokens/second

Access Wandb at: `wandb.ai/<your-username>/darwin-sft-solver`

## Step 5: Evaluate Model

After training:

```bash
python evaluate.py \
    --model_path ./outputs/sft_solver_darwin_iter2/final_model \
    --base_model Qwen/Qwen2.5-Math-7B-Instruct \
    --data_path /Users/vaibhav/Desktop/brahma/tree/euclid/darwin/question_generation_clustering/balanced_questions__darwin_iter2_openai_validated_matched.json \
    --num_samples 100
```

This will evaluate on 100 samples and show accuracy.

## Step 6: Merge and Upload

Merge LoRA weights with base model:

```bash
python merge_and_upload.py \
    --base_model Qwen/Qwen2.5-Math-7B-Instruct \
    --adapter_path ./outputs/sft_solver_darwin_iter2/final_model \
    --output_path ./merged_model \
    --push_to_hub \
    --repo_name vibhuiitj/darwin_iter2_solver_sft
```

This creates a standalone model that can be used without PEFT.

## Expected Results

With the OpenAI-validated dataset:
- Training loss: ~1.5-2.0 (final)
- Validation loss: ~1.6-2.1 (final)
- Accuracy on validation set: ~60-80% (depending on base model)

## File Structure

After training:

```
sft_solver/
├── train_sft_solver.py       # Main training script
├── train.sh                  # Easy training launcher
├── evaluate.py               # Evaluation script
├── merge_and_upload.py       # Merge LoRA and upload
├── requirements.txt          # Dependencies
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
└── outputs/                  # Created after training
    └── sft_solver_darwin_iter2/
        ├── checkpoint-100/
        ├── checkpoint-200/
        ├── checkpoint-300/
        └── final_model/      # Your trained model
            ├── adapter_config.json
            ├── adapter_model.bin
            └── tokenizer files
```

## GPU Memory Requirements

| Model Size | Quantization | GPU Memory | Batch Size | Time (est.) |
|-----------|--------------|------------|------------|-------------|
| 7B        | 4-bit        | ~12 GB     | 2          | 3-4 hours   |
| 7B        | 8-bit        | ~20 GB     | 2          | 4-6 hours   |
| 7B        | None         | ~40 GB     | 1          | 8-12 hours  |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in train.sh
BATCH_SIZE=1
GRAD_ACCUM=16  # Increase to maintain effective batch size
```

### Slow Training
```bash
# Increase batch size if you have memory
BATCH_SIZE=4
GRAD_ACCUM=4
```

### Model Not Learning
- Check learning rate (try 1e-5 to 5e-5)
- Increase epochs to 5
- Verify data quality (all should have openai_match=1)

## Next Steps

1. **Evaluate thoroughly**: Test on MATH, GSM8K benchmarks
2. **Compare baselines**: Compare with base model and previous iterations
3. **Use in Darwin loop**: Deploy for next iteration of question generation
4. **Iterate**: Use new solver to generate better training data

## Support

For issues:
1. Check README.md for detailed documentation
2. Review training logs in `outputs/sft_solver_darwin_iter2/`
3. Check Wandb for training curves
4. Verify GPU availability: `nvidia-smi`

## Example Commands

**Full training:**
```bash
./train.sh
```

**Training with custom settings:**
```bash
python train_sft_solver.py \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --data_path "/path/to/data.json" \
    --output_dir "./outputs/my_experiment" \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --use_lora \
    --use_4bit
```

**Evaluation:**
```bash
python evaluate.py \
    --model_path ./outputs/sft_solver_darwin_iter2/final_model \
    --base_model Qwen/Qwen2.5-Math-7B-Instruct \
    --data_path /path/to/data.json \
    --num_samples 100
```

**Merge and upload:**
```bash
python merge_and_upload.py \
    --base_model Qwen/Qwen2.5-Math-7B-Instruct \
    --adapter_path ./outputs/sft_solver_darwin_iter2/final_model \
    --output_path ./merged_model \
    --push_to_hub \
    --repo_name vibhuiitj/darwin_iter2_solver_sft
```

---

**Ready to train?** Run `./train.sh` now! 🚀
