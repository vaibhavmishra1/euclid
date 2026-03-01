# SFT Solver Training

Supervised Fine-Tuning (SFT) for training a math solver model on OpenAI-validated questions.

## Overview

This training script fine-tunes a 4B parameter language model on the OpenAI-validated dataset of math questions. Each question in the dataset has been verified for correctness by OpenAI's API, ensuring high-quality training data.

## Features

- ✅ **4-bit Quantization**: Memory-efficient training with QLoRA
- ✅ **LoRA Fine-tuning**: Parameter-efficient fine-tuning
- ✅ **Flash Attention 2**: Fast and memory-efficient attention
- ✅ **Gradient Checkpointing**: Reduced memory usage
- ✅ **Wandb Integration**: Comprehensive experiment tracking
- ✅ **Automatic Data Filtering**: Only trains on verified correct examples

## Installation

```bash
cd /Users/vaibhav/Desktop/brahma/tree/euclid/darwin/R-Zero-main/sft_solver
pip install -r requirements.txt
```

## Dataset Format

The training script expects a JSON file with the following structure:

```json
[
  {
    "question": "Let \\( f(x) \\) be a polynomial...",
    "openai_response": "To solve this problem...",
    "openai_answer": "5",
    "openai_match": 1,
    "score": 0.875,
    "cluster_id": 42
  },
  ...
]
```

**Key fields:**
- `question`: The math problem to solve
- `openai_response`: The full solution from OpenAI (used as ground truth)
- `openai_match`: 1 if verified correct, 0 otherwise (only examples with 1 are used)

## Quick Start

### 1. Basic Training

```bash
chmod +x train.sh
./train.sh
```

### 2. Custom Configuration

Edit the variables in `train.sh`:

```bash
MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct"  # Change to your 4B model
DATA_PATH="/path/to/your/dataset.json"
OUTPUT_DIR="./outputs/my_experiment"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-5
```

### 3. Advanced Usage

Run with Python directly for more control:

```bash
python train_sft_solver.py \
    --model_name "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --data_path "/path/to/dataset.json" \
    --output_dir "./outputs/sft_solver" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_lora \
    --use_4bit \
    --wandb_run_name "my-experiment"
```

## Configuration

### Model Selection

The script supports any HuggingFace causal language model. For 4B models, consider:

- `Qwen/Qwen2.5-Math-7B-Instruct` (math-specialized)
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### Memory Requirements

| Configuration | GPU Memory | Training Time (est.) |
|--------------|------------|---------------------|
| 4-bit + LoRA | ~12 GB | 2-4 hours |
| 8-bit + LoRA | ~20 GB | 3-6 hours |
| Full fine-tune | ~40 GB | 8-12 hours |

### LoRA Configuration

Default LoRA settings (in `train_sft_solver.py`):

```python
lora_r = 64              # LoRA rank
lora_alpha = 128         # LoRA scaling factor
lora_dropout = 0.05      # Dropout rate
lora_target_modules = [  # Modules to apply LoRA
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

## Training Process

The script performs the following steps:

1. **Load Dataset**: Reads the JSON file
2. **Filter Data**: Keeps only examples with `openai_match == 1`
3. **Format Prompts**: Converts to instruction format
4. **Split Data**: 95% train, 5% validation
5. **Tokenize**: Converts text to model inputs
6. **Train**: Supervised fine-tuning with causal language modeling objective
7. **Evaluate**: Periodic evaluation on validation set
8. **Save**: Saves model checkpoints and final model

## Output Structure

```
outputs/sft_solver_darwin_iter2/
├── checkpoint-100/           # Periodic checkpoints
├── checkpoint-200/
├── checkpoint-300/
├── final_model/             # Final trained model
│   ├── adapter_config.json  # LoRA config
│   ├── adapter_model.bin    # LoRA weights
│   └── tokenizer files...
└── runs/                    # Tensorboard logs
```

## Monitoring Training

### Wandb Dashboard

Training metrics are automatically logged to Wandb:
- Training loss
- Validation loss
- Learning rate
- Gradient norms
- Tokens per second

View at: `https://wandb.ai/<your-username>/darwin-sft-solver`

### Key Metrics to Watch

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without large gap from train loss
- **Perplexity**: Lower is better
- **Gradient Norm**: Should be stable (not exploding)

## Using the Trained Model

### Load for Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./outputs/sft_solver_darwin_iter2/final_model")
tokenizer = AutoTokenizer.from_pretrained("./outputs/sft_solver_darwin_iter2/final_model")

# Generate
question = "What is the derivative of x^2 + 3x + 5?"
prompt = f"""<|im_start|>system
You are a helpful assistant that solves mathematical problems step by step.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Merge LoRA and Push to HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "./outputs/sft_solver_darwin_iter2/final_model")

# Merge and unload
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# Push to HuggingFace
merged_model.push_to_hub("vibhuiitj/darwin_iter2_solver_sft")
tokenizer.push_to_hub("vibhuiitj/darwin_iter2_solver_sft")
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Reduce `max_seq_length` to 1024
4. Enable `gradient_checkpointing` (already default)

### Slow Training

1. Increase `per_device_train_batch_size`
2. Use Flash Attention 2 (requires `flash-attn` package)
3. Reduce `gradient_accumulation_steps`
4. Use multiple GPUs with `torchrun`

### Poor Performance

1. Increase `num_train_epochs`
2. Adjust `learning_rate` (try 1e-5 to 5e-5)
3. Increase LoRA `r` and `alpha`
4. Check data quality (verify `openai_match` filtering)

## Dataset Statistics

**Current Dataset**: `balanced_questions__darwin_iter2_openai_validated_matched.json`

- Total examples: 1,661 (all verified correct)
- Average score: 0.768
- Clusters covered: 89 out of 128
- Average question length: ~150 tokens
- Average response length: ~800 tokens

## Next Steps

After training:

1. **Evaluate**: Test on held-out math benchmarks (MATH, GSM8K, etc.)
2. **Compare**: Compare with the base model and previous iterations
3. **Iterate**: Use this solver in the next Darwin iteration for question generation
4. **Deploy**: Serve the model using vLLM for fast inference

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
