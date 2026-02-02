# Training Metrics Documentation

This document explains all metrics reported during R-Zero questioner model training with Rentropy reward function.

## Table of Contents
- [Actor (Policy Model) Metrics](#actor-policy-model-metrics)
- [Critic (Value Estimation) Metrics](#critic-value-estimation-metrics)
- [Sequence Length Statistics](#sequence-length-statistics)
- [Performance Metrics](#performance-metrics)
- [Prompt/Response Length Metrics](#promptresponse-length-metrics)
- [Reward Breakdown](#reward-breakdown)
- [Timing Metrics](#timing-metrics)

---

## Actor (Policy Model) Metrics

These metrics track the policy model (the questioner) that generates questions.

### `entropy_loss`
- **What it is**: Measures the randomness/uncertainty in the model's action (token) selection
- **Interpretation**: 
  - Higher values (1.0-2.0) = More exploration, model is uncertain
  - Lower values (0.5-0.9) = More confident, model is converging to specific patterns
- **Healthy range**: 0.5-1.5 typically
- **Watch for**: Rapid drops may indicate mode collapse (model generating repetitive outputs)

### `grad_norm`
- **What it is**: Magnitude (L2 norm) of gradients during backpropagation
- **Interpretation**: 
  - Too high (>10) = Exploding gradients, unstable training
  - Too low (<0.1) = Vanishing gradients, model not learning
  - Healthy: 0.5-5.0 typically
- **Your values**: ~1.7-2.0 indicates stable gradient flow

### `kl_coef`
- **What it is**: Coefficient for KL divergence penalty term
- **Interpretation**: Controls how much to penalize divergence from the reference policy
- **Your config**: Fixed at 0.01 (very small penalty, allows significant learning)

### `kl_loss`
- **What it is**: Actual KL divergence loss being applied
- **Interpretation**: 
  - 0.0 = Model identical to reference policy (no learning)
  - Growing values = Model is learning to differ from initial state
- **Your trend**: 0.0 → 0.003 → 0.007 shows healthy policy evolution

### `lr` (Learning Rate)
- **What it is**: Step size for weight updates
- **Your value**: 1e-06 (0.000001) - very small for fine-tuning
- **Why small**: Prevents catastrophic forgetting of pre-trained knowledge

### `pg_clipfrac_higher`
- **What it is**: Fraction of samples where policy ratio was clipped because it was too HIGH
- **Interpretation**: 
  - PPO clips ratios outside [0.2, 3.0] range
  - Higher clipfrac = More aggressive updates being prevented
  - Your value (0.005 = 0.5%) = Very stable, minimal clipping needed

### `pg_clipfrac_lower`
- **What it is**: Fraction of samples clipped because ratio was too LOW
- **Interpretation**: 
  - Zero in your case = No samples went below threshold
  - Indicates model isn't over-correcting downward

### `pg_loss` (Policy Gradient Loss)
- **What it is**: Main RL training signal - encourages high-reward actions
- **Interpretation**: 
  - Growing values = Stronger learning signal
  - Your trend: 0.019 → 0.045 → 0.085 shows increasing policy updates
- **Note**: This is the loss being minimized (negative reward = positive loss)

### `ppo_kl`
- **What it is**: KL divergence computed specifically for PPO algorithm
- **Interpretation**: Separate from `kl_loss`, used for PPO-specific clipping
- **Your value**: 0.0 indicates PPO clipping isn't being triggered

---

## Critic (Value Estimation) Metrics

The critic estimates expected future rewards to compute advantages.

### `advantages.max/mean/min`
- **What it is**: Advantage = actual reward - expected reward (baseline)
- **Interpretation**: 
  - Positive = Better than expected
  - Negative = Worse than expected
  - Mean of -0.483 = On average, samples performed below baseline (model still learning)
- **Range**: Typically -1.5 to +1.5 (clipped)

### `returns.max/mean/min`
- **What it is**: Expected cumulative future rewards
- **Interpretation**: In GRPO (your algorithm), returns = advantages since γ=1.0, λ=1.0
- **Note**: Same values as advantages in your setup

### `rewards.max/mean/min`
- **What it is**: Raw reward values before advantage normalization
- **Interpretation**: 
  - Max 0.5 = Best reward (ZPD peaks at 0.5 for 50% solver accuracy)
  - Mean 0.121 = Average reward (positive is good!)
  - Min -1.0 = Worst reward (invalid format or unanswerable question)

### `score.max/mean/min`
- **What it is**: Same as rewards - raw reward values
- **Note**: Duplicate of rewards in your logging

---

## Sequence Length Statistics

Tracks token distribution across the batch and GPUs.

### `global_seqlen.balanced_max/min`
- **What it is**: After load balancing across GPUs, max/min tokens per GPU
- **Interpretation**: Shows how well work is distributed
- **Your values**: ~378,160 indicates good balancing

### `global_seqlen.max/mean/min`
- **What it is**: Longest/average/shortest total sequence in batch
- **Interpretation**: 
  - Max: Longest question+response pair
  - Mean: Average sequence length
  - Min: Shortest sequence (may be malformed)

### `minmax_diff`
- **What it is**: Difference between longest and shortest sequences
- **Interpretation**: 
  - Lower = More uniform batch (better for efficiency)
  - Higher = More variable lengths (may cause padding overhead)
- **Your trend**: Decreasing (64,950 → 61,515 → 64,950) shows stabilization

---

## Performance Metrics

System resource usage and throughput.

### `cpu_memory_used_gb`
- **What it is**: RAM usage on CPU
- **Your value**: ~42 GB indicates substantial memory usage

### `max_memory_allocated_gb`
- **What it is**: Peak GPU memory actually used for tensors
- **Your value**: ~26 GB on each GPU

### `max_memory_reserved_gb`
- **What it is**: Peak GPU memory reserved (includes cache, buffers)
- **Your value**: ~64 GB indicates significant memory overhead

### `mfu_actor` (Model FLOPS Utilization)
- **What it is**: Percentage of theoretical peak FLOPS achieved
- **Your value**: 0.0 = Not computed/not available

### `throughput`
- **What it is**: Tokens processed per second
- **Your value**: ~172-260 tokens/sec
- **Note**: Lower in Step 3 (172) due to longer sequences requiring more computation

### `time_per_step`
- **What it is**: Total wall-clock time for one training step
- **Your value**: ~1800-2200 seconds (30-37 minutes per step)
- **Breakdown**: Mostly spent in advantage computation (vLLM solver calls)

### `total_num_tokens`
- **What it is**: Total tokens processed this step
- **Your value**: ~1.5-2.0M tokens per step

---

## Prompt/Response Length Metrics

Tracks input/output sequence lengths.

### `prompt_length.clip_ratio`
- **What it is**: Fraction of prompts truncated (hit max_prompt_length limit)
- **Your value**: 0.0 = No prompts truncated (all fit in 2048 limit)

### `prompt_length.max/mean/min`
- **What it is**: Longest/average/shortest prompt length
- **Your value**: All 189 tokens = Same template used for all samples

### `response_length.clip_ratio`
- **What it is**: Fraction of responses truncated (hit max_response_length limit)
- **Your value**: 0.071-0.107 = 7-11% hit the 4096 token limit
- **Interpretation**: Some questions are generating very long responses

### `response_length.max/mean/min`
- **What it is**: Longest/average/shortest response length
- **Your trend**: 
  - Mean: 792 → 764 → 550 tokens (model learning to be more concise!)
  - Max: 4096 (hitting limit)
  - Min: 2 (likely malformed outputs)

---

## Reward Breakdown

Detailed reward components.

### `accuracy`
- **⚠️ MISLEADING NAME**: This is NOT mathematical accuracy!
- **What it actually is**: Diversity reward bonus (stored here for logging compatibility)
- **Your value**: 0.0 because you're in mode 1 (diversity logging only, no reward)
- **Note**: In modes 2-4, this would show cluster diversity bonuses

### `base_score`
- **What it is**: **Majority voting score** - how well vLLM solvers can answer the generated question
- **Interpretation**: 
  - Range: -1.0 to 1.0
  - 0.236 = ~24% solver agreement on average
  - **This is the key learning metric!**
- **Your trend**: -0.117 → -0.103 → 0.236 (dramatic improvement!)

### `diversity`
- **What it is**: Cluster diversity reward bonus
- **Your value**: 0.0 (disabled in mode 1)
- **Note**: In modes 2-4, this would show rarity/uniqueness bonuses

### `format`
- **What it is**: Binary score (1.0 or 0.0) for valid output format
- **Interpretation**: 
  - 1.0 = Has valid `<question>...</question>` tags and `\boxed{}` answer
  - 0.0 = Missing required format elements
- **Your trend**: 0.627 → 0.631 → 0.862 (86% format compliance!)

### `overall`
- **What it is**: Final reward = ZPD(base_score) + diversity
- **ZPD transform**: `min(score, 1-score)` peaks at 0.5 difficulty
- **Interpretation**: 
  - Positive = Good questions (in "zone of proximal development")
  - Negative = Questions too easy or too hard
- **Your trend**: -0.181 → -0.183 → 0.121 (turned positive!)

---

## Timing Metrics

### Per-Token Timing (milliseconds)

#### `adv` (Advantage Computation)
- **What it is**: Time to compute advantages per token
- **Your value**: ~0.6-1.1 ms/token
- **Note**: Includes vLLM solver calls (slowest part)

#### `gen` (Generation)
- **What it is**: Time to generate tokens during rollout
- **Your value**: ~0.14 ms/token
- **Interpretation**: Fast - model generates quickly

#### `old` (Old Policy)
- **What it is**: Time to compute old policy log probabilities
- **Your value**: ~0.024-0.025 ms/token

#### `ref` (Reference Policy)
- **What it is**: Time to compute reference policy log probabilities
- **Your value**: ~0.025-0.026 ms/token

#### `reward` (Reward Computation)
- **What it is**: Time to compute rewards
- **Your value**: ~0.001 ms/token
- **Interpretation**: Very fast - reward function is efficient

#### `update_actor` (Policy Update)
- **What it is**: Time to update policy weights per token
- **Your value**: ~0.16-0.18 ms/token

### Per-Step Timing (seconds)

#### `adv` (Advantage Computation)
- **What it is**: Total time computing advantages
- **Your value**: ~1200-1700 seconds (20-28 minutes!)
- **Why so long**: Includes vLLM solver calls to evaluate questions
- **Bottleneck**: This is the slowest part of training

#### `gen` (Generation)
- **What it is**: Total time generating responses
- **Your value**: ~160-230 seconds (2.5-4 minutes)

#### `old` (Old Policy)
- **What it is**: Total time computing old policy probs
- **Your value**: ~37-51 seconds

#### `ref` (Reference Policy)
- **What it is**: Total time computing reference policy probs
- **Your value**: ~39-49 seconds

#### `reward` (Reward Computation)
- **What it is**: Total time computing rewards
- **Your value**: ~1.2 seconds
- **Interpretation**: Very fast - reward function is efficient

#### `step` (Total Step Time)
- **What it is**: Total wall-clock time for one training step
- **Your value**: ~1800-2200 seconds (30-37 minutes)
- **Breakdown**: 
  - ~75% advantage computation (vLLM calls)
  - ~10% generation
  - ~15% other (policy updates, etc.)

#### `update_actor` (Policy Update)
- **What it is**: Total time updating model weights
- **Your value**: ~270-315 seconds (4.5-5 minutes)

---

## Key Takeaways

### What to Watch For:

1. **base_score**: Primary learning metric - should trend upward
2. **format**: Should increase toward 1.0 (model learning output format)
3. **entropy_loss**: Monitor for mode collapse (rapid drops)
4. **grad_norm**: Should stay in 0.5-5.0 range
5. **kl_loss**: Should grow slowly (0.0 → 0.01 range)
6. **overall reward**: Should turn positive as model learns

### Healthy Training Signs:

✅ Base score improving (negative → positive)  
✅ Format score increasing toward 1.0  
✅ Entropy stable (not collapsing to 0)  
✅ Grad norm stable (~1-2)  
✅ KL loss growing slowly  
✅ Overall reward turning positive  

### Warning Signs:

⚠️ Entropy dropping rapidly (<0.5) = Mode collapse  
⚠️ Grad norm >10 = Exploding gradients  
⚠️ Grad norm <0.1 = Vanishing gradients  
⚠️ Base score stuck negative = Model not learning  
⚠️ Format score stuck low = Model not learning format  

---

## References

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward Function**: Rentropy (R-Zero with cluster-entropy diversity)
- **Training Framework**: VERL (Versatile Efficient Reinforcement Learning)
- **Model**: Qwen3-4B-Base questioner model
