# ExpV1_1: GRPO Training with Self-Consistency Rewards

This experiment compares **GRPO (Group Relative Policy Optimization)** vs **SFT (ExpV1_0)** on the same synthetic math dataset, using **self-consistency** to compute rewards (no ground truth needed).

## Overview

### Research Question
> Does GRPO with self-consistency rewards outperform SFT when training on synthetic math problems?

### Key Innovation: Self-Consistency Rewards
For **synthetic datasets** (generated via concept graph exploration), there is **no ground truth answer**. Instead of requiring a teacher or oracle, we compute rewards using **self-consistency across rollouts**:

```
For each prompt:
  1. Generate N rollouts (e.g., N=4)
  2. Extract \boxed{answer} from each rollout
  3. Find modal (majority) answer
  4. Reward rollouts that agree with modal answer
```

This is the **same mechanism** used in ExpV1_0's ZPD filtering, but now applied during GRPO training.

### Comparison with ExpV1_0

| Aspect | ExpV1_0 (SFT) | ExpV1_1 (GRPO) |
|--------|---------------|----------------|
| COT Source | Solver's best rollout | Solver's all rollouts |
| Ground Truth | Precomputed modal answer | Computed during training |
| Selection | Self-consistency (offline) | Self-consistency (online) |
| Signal | "Best" solution as target | Binary self-consistency reward |
| Learning | Imitation (from best) | Trial and error (from all) |

## What is GRPO?

**Group Relative Policy Optimization** (from DeepSeek-Math) is a simplified RL algorithm:

1. Generate **n rollouts** per problem (e.g., n=4)
2. Score each rollout with **self-consistency reward**
3. Compute **group-relative advantage**:
   ```
   advantage_i = (reward_i - mean) / (std + ε)
   ```
4. Update policy to **increase probability of high-advantage responses**

### Why GRPO over PPO?
- No value network needed (uses group statistics as baseline)
- Simpler and more stable
- Works well for outcome-supervised tasks

## Self-Consistency Reward Mechanism

```python
def compute_rewards(completions, num_generations):
    """
    For each group of rollouts (same prompt):
    1. Extract \boxed{} answer from each
    2. Find modal (majority) answer
    3. If enough agreement (>= min_agreement):
       - Matching rollouts → reward = 1.0
       - Non-matching → reward = 0.0
    4. If no consensus:
       - All rollouts → reward = 0.0
    """
    rewards = []
    for group in chunks(completions, num_generations):
        answers = [extract_boxed(c) for c in group]
        modal_answer = find_modal(answers)
        
        for ans in answers:
            if ans is None:
                rewards.append(-0.1)  # Format penalty
            elif ans matches modal_answer:
                rewards.append(1.0)   # Agrees with consensus
            else:
                rewards.append(0.0)   # Disagrees
    
    return rewards
```

### Why Self-Consistency Works
- **Synthetic problems have no oracle** - self-consistency is the only signal
- **Majority voting is a proxy for correctness** - if 3/4 rollouts agree, likely correct
- **Same principle as ExpV1_0** - but computed online during training
- **Forces model to be consistent** - rewards reproducible reasoning

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python -m tree.euclid.expv1.expv1_1.run_grpo \
    --config tree/euclid/expv1/expv1_1/config.yaml
```

### With Command-Line Overrides

```bash
python -m tree.euclid.expv1.expv1_1.run_grpo \
    --config tree/euclid/expv1/expv1_1/config.yaml \
    --model Qwen/Qwen3-1.7B-Base \
    --dataset tree/euclid/expv1/expv1_0/output_Qwen3-1.7B-Base_accepted/accepted_verified_retried.jsonl \
    --output ./output_grpo \
    --num-generations 4 \
    --min-agreement 2 \
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
| `--min-agreement` | Min rollouts for consensus | 2 |
| `--epochs` | Training epochs | 1 |
| `--lr` | Learning rate | 1e-6 |
| `--kl-coef` | KL penalty coefficient | 0.05 |
| `--batch-size` | Per-device batch size | 2 |
| `--temperature` | Sampling temperature | 0.8 |
| `--no-vllm` | Disable vLLM (use HF generate) | False |

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
grpo:
  # Model
  model_name_or_path: "Qwen/Qwen3-1.7B-Base"
  
  # GRPO specific
  num_generations: 4      # Group size for self-consistency
  temperature: 0.8        # Exploration
  kl_coef: 0.05          # Prevent divergence
  
  # Self-Consistency Reward
  correct_reward: 1.0     # Matches modal answer
  incorrect_reward: 0.0   # Doesn't match
  format_penalty: 0.1     # For missing \boxed{}
  min_agreement: 2        # Min rollouts for consensus
  
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
│  Rollouts ─────► Self-Consistency ─────► Advantages         │
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

## Expected Results

### Comparison with ExpV1_0

| Model | Method | Expected Accuracy |
|-------|--------|-------------------|
| Qwen3-1.7B-Base | Baseline | 19.81% |
| Qwen3-1.7B-Base | SFT (ExpV1_0, all data) | 27.96% |
| Qwen3-1.7B-Base | GRPO (ExpV1_1) | TBD |

### Hypotheses

1. **GRPO may have similar performance to SFT** since both use self-consistency
2. **GRPO might show better generalization** by learning from both correct and incorrect attempts
3. **Self-consistency rewards may be noisier** than ground truth, affecting convergence
4. **GRPO requires more careful tuning** (temperature, KL coef, num_generations, min_agreement)

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
   
2. SELF-CONSISTENCY REWARD PHASE
   └─► Extract \boxed{} answer from each generation
   └─► Find modal (majority) answer per prompt
   └─► Reward = 1.0 if matches modal, 0.0 if not
   └─► No ground truth needed!
   
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
├── reward_function.py     # Self-consistency reward computation
└── __init__.py
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce `num_generations` to 2
- Enable `gradient_checkpointing: true`

### Low Reward Signal / No Consensus
- Increase `num_generations` (more samples → better consensus)
- Decrease `min_agreement` (but may reward noise)
- Check that `extract_boxed_answer` works on your generations
- Lower `temperature` to reduce generation diversity

### Unstable Training
- Increase `kl_coef` to prevent divergence
- Reduce `learning_rate`
- Increase `num_generations` for better advantage estimates

### All Rewards are Zero
- Check if generations have `\boxed{}` format
- Check if `min_agreement` is too high for `num_generations`
- Review sample generations for formatting issues

## Key Differences from Standard GRPO

| Standard GRPO | ExpV1_1 GRPO |
|---------------|--------------|
| Requires ground truth answer | Uses self-consistency |
| Oracle determines correctness | Modal answer determines correctness |
| Works on benchmarks with labels | Works on ANY synthetic data |
| Reward = f(pred, gold) | Reward = f(pred, modal_from_rollouts) |

## References

- GRPO: DeepSeek-Math paper
- TRL: https://huggingface.co/docs/trl/
- ExpV1_0: `tree/euclid/expv1/expv1_0/`
- Self-Consistency: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"