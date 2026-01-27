# ExpV1_1: GRPO Training for Mathematical Reasoning

This experiment compares **GRPO (Group Relative Policy Optimization)** vs **SFT (ExpV1_0)** on the same synthetic math dataset.

## Overview

### Research Question
> Does GRPO outperform SFT when training on the same dataset of solver-generated solutions?

### Key Insight from ExpV1_0
In ExpV1_0, SFT was trained on the **solver's own best COT** (selected via self-consistency from multiple rollouts), NOT on teacher-provided solutions. This makes the comparison with GRPO more interesting:

| Aspect | ExpV1_0 (SFT) | ExpV1_1 (GRPO) |
|--------|---------------|----------------|
| COT Source | Solver's best rollout | Solver's all rollouts |
| Selection | Self-consistency (offline) | Reward-weighted (online) |
| Signal | "Best" solution as target | Binary correctness reward |
| Learning | Imitation (from best) | Trial and error (from all) |

## What is GRPO?

**Group Relative Policy Optimization** (from DeepSeek-Math) is a simplified RL algorithm:

1. Generate **n rollouts** per problem (e.g., n=4)
2. Score each rollout with **reward function** (correct=1, incorrect=0)
3. Compute **group-relative advantage**:
   ```
   advantage_i = (reward_i - mean) / (std + ε)
   ```
4. Update policy to **increase probability of high-advantage responses**

### Why GRPO over PPO?
- No value network needed (uses group statistics as baseline)
- Simpler and more stable
- Works well for outcome-supervised tasks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python -m euclid.expv1.expv1_1.run_grpo \
    --config euclid/expv1/expv1_1/config.yaml
```

### With Command-Line Overrides

```bash
python -m tree.euclid.expv1.expv1_1.run_grpo \
    --config tree/euclid/expv1/expv1_1/config.yaml \
    --model Qwen/Qwen3-1.7B-Base \
    --dataset tree/euclid/expv1/expv1_0/output_Qwen3-1.7B-Base_accepted/accepted_verified_retried.jsonl \
    --output ./output_grpo \
    --num-generations 4 \
    --epochs 1 \
    --kl-coef 0.05
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config.yaml | Required |
| `--model` | Model name or path | Qwen/Qwen3-1.7B-Base |
| `--dataset` | Path to ExpV1_0 dataset | - |
| `--output` | Output directory | ./output_grpo |
| `--num-generations` | Rollouts per prompt | 4 |
| `--epochs` | Training epochs | 1 |
| `--lr` | Learning rate | 1e-6 |
| `--kl-coef` | KL penalty coefficient | 0.05 |
| `--batch-size` | Per-device batch size | 2 |
| `--temperature` | Sampling temperature | 0.8 |

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
grpo:
  # Model
  model_name_or_path: "Qwen/Qwen3-1.7B-Base"
  
  # GRPO specific
  num_generations: 4      # Group size for advantage
  temperature: 0.8        # Exploration
  kl_coef: 0.05          # Prevent divergence
  
  # Reward
  correct_reward: 1.0
  incorrect_reward: 0.0
  format_penalty: 0.1     # For missing \boxed{}
  
  # Training
  num_train_epochs: 1
  learning_rate: 1.0e-6
  
  # vLLM (fast rollout generation)
  use_vllm: true
  vllm_gpu_memory_utilization: 0.7
  vllm_tensor_parallel_size: 1
```

## vLLM Integration

GRPO requires generating **multiple rollouts per prompt** (e.g., 4-8). Using vLLM instead of HuggingFace `generate()` provides **10-20x speedup**.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  TRL GRPOTrainer with vLLM                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  vLLM Engine                      Actor Model               │
│  ────────────                     ───────────               │
│  • Fast batch generation          • Policy updates          │
│  • Continuous batching            • Gradient computation    │
│  • PagedAttention                 • KL penalty              │
│                                                             │
│  Rollouts ─────────────────────► Advantages + Rewards       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Disable vLLM (fallback to HF generate)

If you have issues with vLLM, disable it:

```bash
python -m tree.euclid.expv1.expv1_1.run_grpo \
    --config tree/euclid/expv1/expv1_1/config.yaml \
    --no-vllm
```

Or in config.yaml:
```yaml
grpo:
  use_vllm: false
```

## Reward Function

Binary reward based on answer correctness:

```python
def compute_reward(generated_text, ground_truth):
    predicted = extract_boxed_answer(generated_text)
    
    if predicted is None:
        return -0.1  # Format penalty
    
    if answers_match(predicted, ground_truth):
        return 1.0   # Correct
    else:
        return 0.0   # Incorrect
```

## Expected Results

### Comparison with ExpV1_0

| Model | Method | Expected Accuracy |
|-------|--------|-------------------|
| Qwen3-1.7B-Base | Baseline | 19.81% |
| Qwen3-1.7B-Base | SFT (ExpV1_0, all data) | 27.96% |
| Qwen3-1.7B-Base | GRPO (ExpV1_1) | TBD |

### Hypotheses

1. **GRPO may have similar performance to SFT** since both use solver-generated solutions
2. **GRPO might show better generalization** by learning from both correct and incorrect attempts
3. **GRPO requires more careful tuning** (temperature, KL coef, num_generations)

## Evaluation

Use the same evaluation script as ExpV1_0:

```bash
python -m tree.euclid.evaluate_math.evaluate_hf \
    --model ./output_grpo \
    --dataset-config number_theory \
    --split test \
    --output results_grpo.jsonl
```

## GPU Requirements

| Setting | VRAM Required |
|---------|---------------|
| 1.7B model, 4 generations, batch=2 | ~24GB |
| 1.7B model, 4 generations, batch=1 | ~16GB |
| With gradient checkpointing | -30% VRAM |

For limited VRAM, reduce `per_device_train_batch_size` or `num_generations`.

## Training Flow

```
For each batch of prompts:

1. GENERATION PHASE
   └─► Generate n=4 solutions per problem (temperature=0.8)
   
2. REWARD PHASE
   └─► Extract \boxed{} answer from each generation
   └─► Compare with ground truth → reward ∈ {0, 1}
   
3. ADVANTAGE COMPUTATION
   └─► For each problem group:
       advantage_i = (reward_i - mean) / (std + ε)
   
4. POLICY UPDATE
   └─► L = -Σ advantage_i × log π(response_i | prompt)
   └─► Add KL penalty to prevent divergence
   └─► Backprop and update weights
```

## Files

```
tree/euclid/expv1/expv1_1/
├── README.md               # This file
├── config.yaml             # Training configuration
├── requirements.txt        # Dependencies
├── run_grpo.py            # Main training script
├── data_utils.py          # Dataset loading utilities
├── reward_function.py     # Math reward computation
└── __init__.py
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce `num_generations` to 2
- Enable `gradient_checkpointing: true`

### Low Reward Signal
- Check that dataset has valid `answer` field
- Verify `extract_boxed_answer` works on your generations
- Consider reducing `format_penalty`

### Unstable Training
- Increase `kl_coef` to prevent divergence
- Reduce `learning_rate`
- Increase `num_generations` for better advantage estimates

## References

- GRPO: DeepSeek-Math paper
- TRL: https://huggingface.co/docs/trl/
- ExpV1_0: `tree/euclid/expv1/expv1_0/`
