# VERL Training Pipeline - Complete Guide

**Document Purpose:** Comprehensive explanation of VERL (Versatile Reinforcement Learning) training pipeline for Rentropy question generation, with concrete examples and settings explained.

**Target Audience:** Understanding how questioner model training works in the Rentropy experiment.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Training Arguments Reference](#training-arguments-reference)
3. [The Complete Training Loop](#the-complete-training-loop)
4. [Key Concepts Explained](#key-concepts-explained)
5. [Common Confusions](#common-confusions)
6. [Production vs Smoke Test Settings](#production-vs-smoke-test-settings)

---

## System Architecture

### Two Separate vLLM Instances

The training system uses **two independent vLLM instances** with different purposes:

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌───────────────────┐              ┌──────────────────┐
│  1. ROLLOUT       │              │  2. REWARD       │
│  (Generate Qs)    │──questions──▶│  (Evaluate Qs)   │
└───────────────────┘              └──────────────────┘
        │                                   │
        ▼                                   ▼
┌──────────────────┐              ┌──────────────────┐
│ vLLM Instance A  │              │ vLLM Instance B  │
│ (Questioner)     │              │ (Solver)         │
│ Port: Internal   │              │ Port: 5000       │
│ Model: Training  │              │ Model: Fixed     │
│ Job: Generate Qs │              │ Job: Solve Qs    │
│ Managed: VERL    │              │ Managed: Manual  │
└──────────────────┘              └──────────────────┘
```

#### vLLM Instance A: Questioner Rollout Engine

**Purpose:** Generate questions during the rollout phase

**Who manages it:** VERL automatically spawns this internally

**Configuration:**
```bash
worker.rollout.n=2                          # 2 variations per prompt
worker.rollout.gpu_memory_utilization=0.25  # Use 25% GPU memory
worker.rollout.temperature=1.0              # Sampling temperature
worker.rollout.name=vllm                    # Use vLLM backend
```

**When it runs:** During the rollout phase of each training step

**Model:** The questioner model being trained (updates every step)

#### vLLM Instance B: Solver API

**Purpose:** Receive generated questions and solve them for reward computation

**Who manages it:** You start it manually before training:
```bash
python3 vllm_service_init/start_vllm_server.py \
    --port 5000 \
    --model_path $solver_model_path \
    --gpu_mem_util 0.2
```

**When it runs:** Throughout training, waiting for API requests from Rentropy

**Model:** Fixed solver model (never changes during training)

### GPU Memory Distribution

```
Total GPU Memory: 80 GB
├─ 20% (~16 GB): vLLM Instance B (Solver on port 5000)
├─ 25% (~20 GB): vLLM Instance A (Questioner rollout, VERL internal)
└─ 55% (~44 GB): Training (FSDP, gradients, optimizer states)
```

**Why two instances?** Separation of concerns:
- Generation (rollout) needs fast inference
- Evaluation (reward) needs stable, fixed model
- Different memory budgets and optimization profiles

---

## Training Arguments Reference

### Configuration File
```bash
config=examples/config_smoketest.yaml
```
**Purpose:** Base configuration file with default settings

**Analogy:** Recipe book with default instructions

**Usage:** Command-line arguments override these defaults

### Data Settings

#### `data.max_response_length=512`
**What:** Maximum length of generated questions (in tokens)

**Analogy:** Essay word limit of ~350-400 words

**Why it matters:** Longer = more GPU memory needed

**Smoke test:** 512 tokens (minimal)

**Production:** 1024-2048 tokens

#### `data.max_prompt_length=512`
**What:** Maximum length of input prompts

**Why it matters:** Combined with response length determines total context size

#### `data.rollout_batch_size=16`
**What:** How many DIFFERENT prompts to process in parallel

**Analogy:** Asking 16 students to work on different assignments simultaneously

**Purpose:** GPU efficiency - process multiple prompts at once

**Formula:** Total generations = `rollout_batch_size × rollout.n`

**Example:**
```
16 prompts × 2 variations = 32 questions per rollout
```

**Memory impact:** Larger = faster but more memory needed

**Smoke test:** 16 (minimal)

**Production:** 64-512

### Model Settings

#### `worker.actor.model.model_path=$questioner_model_path`
**What:** Which model to train (the "questioner")

**Example:** `Qwen/Qwen2.5-0.5B-Instruct`

**Note:** This is the model being trained, not the solver

### Saving & Checkpointing

#### `trainer.experiment_name=$save_path`
**What:** Name for this training run

**Example:** `rentropy_smoketest_20260131_mode1_questioner`

**Used for:** Logging, checkpoint directories, experiment tracking

#### `trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path`
**What:** Where to save trained model checkpoints

**Example:** `/root/euclid/rentropy/R-Zero/storage/models/your_experiment/`

**Structure:**
```
storage/models/your_experiment/
├── global_step_1/
│   └── actor/
│       ├── checkpoint/
│       └── huggingface/  ← Merged model ready to use
└── global_step_2/
    └── actor/
        └── huggingface/
```

#### `trainer.save_freq=1`
**What:** How often to save checkpoints (every N steps)

**Example:** Save after every step for smoke test

**Production:** 100-500 (save less frequently)

### Training Duration

#### `trainer.total_epochs=2`
**What:** Maximum number of times to iterate through entire dataset

**Calculation:**
```python
# Dataset: 12,000 prompts
# Batch size: 16
# Steps per epoch: 12,000 / 16 = 750

# total_epochs=2 means max 1,500 steps
```

**Note:** Often overridden by `max_steps`

#### `trainer.max_steps=2`
**What:** **HARD LIMIT** - stops after exactly N gradient updates

**Overrides:** Takes priority over `total_epochs`

**Smoke test:** 2 steps (just verify it works)

**Production:** 1,000-10,000+ steps

**Why both exist:** Can set either/or depending on preference

### Reward Function

#### `worker.reward.reward_function=./examples/reward_function/caller_rentropy.py:compute_score`
**What:** How to evaluate generated questions (the "grading rubric")

**Flow:**
1. Sends question to solver API (vLLM on port 5000)
2. Checks if answer is correct
3. Computes diversity rewards (rarity, batch uniqueness, cluster diversity)
4. Returns combined reward score

**Components:**
- ✅ Format reward: Is question properly formatted?
- ✅ Base score: Does solver solve it correctly?
- ✅ Diversity reward: Is it novel/diverse? (Rentropy modes 2-4)

### Hardware Settings

#### `trainer.n_gpus_per_node=1`
**What:** Number of GPUs to use for training

**Smoke test:** 1 GPU (cost-effective)

**Production:** 1-8 GPUs (with FSDP parallelism)

#### `CUDA_VISIBLE_DEVICES=0`
**What:** Which specific GPU to use

**Example:** GPU #0 out of available GPUs

### Sampling/Rollout Settings

#### `worker.rollout.n=2`
**What:** Number of question VARIATIONS to generate per prompt

**Analogy:** For each topic, generate 2 different question variations

**Why needed:** GRPO algorithm requires n≥2 for advantage estimation

**Formula:** Total questions = `rollout_batch_size × n`

**Example:**
```
16 prompts × 2 variations = 32 total questions
```

**Memory impact:** More variants = better exploration but slower

**Smoke test:** 2 (minimum for GRPO)

**Production:** 5-8

#### `worker.rollout.gpu_memory_utilization=0.25`
**What:** How much GPU memory the sampling engine (vLLM Instance A) can use

**Analogy:** Reserve 25% of GPU for question generation

**Why conservative:** Leave room for training (FSDP, gradients, optimizer)

**Smoke test:** 0.25 (25%)

**Production:** 0.6-0.8 (60-80%)

### Batch Sizes (Critical for Memory!)

#### `worker.actor.global_batch_size=1`
**What:** Total number of UNIQUE prompts to process per training step

**Analogy:** Give 1 unique assignment to class per lesson

**Formula:** Actual samples = `global_batch_size × rollout.n`

**Example:**
```
global_batch_size=1 × rollout.n=2 = 2 samples per training step
```

**Note:** Different from `rollout_batch_size`! This controls training, not rollout.

**Smoke test:** 1 (ultra-minimal)

**Production:** 64-128

#### `worker.actor.micro_batch_size_per_device_for_update=1`
**What:** How many samples to fit in GPU during **gradient computation** (backward pass)

**Analogy:** Grade 1 paper at a time when calculating corrections

**Purpose:** Memory efficiency during backpropagation

**Why separate:** Backward pass needs to store activations (more memory!)

**Smoke test:** 1 (safest)

**Production:** 2-4

#### `worker.actor.micro_batch_size_per_device_for_experience=1`
**What:** How many samples to fit in GPU during **forward pass** (collecting rollouts)

**Analogy:** Let 1 student answer at a time when collecting responses

**Purpose:** Memory efficiency during generation/rollout

**Smoke test:** 1 (safest)

**Production:** 4-8

### Prompt Template

#### `data.format_prompt=./examples/format_prompt/questioner.jinja`
**What:** Jinja template that formats system/user prompts

**Example template:**
```
You are an expert competition-math problem setter.
FIRST, think step-by-step to design a brand-new problem.
THEN, output exactly:

<question>
{The full problem statement}
</question>

\boxed{final_answer}
```

### Validation & Logging

#### `trainer.val_freq=-1`
**What:** How often to run validation (every N steps)

**Special:** -1 = disabled (no validation)

**Why disable:** Saves time during smoke test

**Production:** 100-500 (validate periodically)

#### `trainer.logger='["console"]'`
**What:** Where to log metrics

**Options:**
- `"console"`: Terminal output
- `"wandb"`: Weights & Biases cloud
- `"tensorboard"`: TensorBoard logging

**Smoke test:** Console only (no cloud logging)

**Production:** `["console", "wandb"]`

### Algorithm Settings

#### `algorithm.adv_estimator=grpo`
**What:** Which algorithm to use for advantage estimation

**Options:**
- `grpo`: Group Relative Policy Optimization
- `rloo`: Reward-weighted Leave-One-Out
- `gae`: Generalized Advantage Estimation (for PPO)

**Your choice:** GRPO (compares variations of same prompt)

#### `algorithm.kl_coef=0.01`
**What:** KL divergence penalty coefficient

**Purpose:** Prevent model from changing too drastically

**Formula:** `total_loss = policy_loss + kl_coef × kl_divergence`

**Higher value:** More conservative updates (slower learning, more stable)

**Lower value:** Aggressive updates (faster learning, less stable)

#### `algorithm.use_kl_loss=true`
**What:** Enable KL divergence penalty

**Why:** Keeps new policy close to old policy (stability)

### Optimizer Settings

#### `worker.actor.optim.lr=1e-5`
**What:** Learning rate (how big are weight updates)

**Example:**
```python
# Weight update:
new_weight = old_weight - lr × gradient
# 0.500001 = 0.5 - 0.00001 × (-0.1)
```

**Why small:** Large models need tiny, careful updates

#### `worker.actor.optim.strategy=adamw`
**What:** Which optimizer algorithm to use

**AdamW:** Adaptive learning rate with weight decay

**Alternative:** SGD, Adam, AdaFactor

#### `worker.actor.max_grad_norm=1.0`
**What:** Maximum gradient norm (gradient clipping)

**Purpose:** Prevent exploding gradients

**How it works:**
```python
if gradient_norm > 1.0:
    gradients = gradients × (1.0 / gradient_norm)
```

### FSDP Settings (Multi-GPU)

#### `worker.actor.fsdp.enable_full_shard=true`
**What:** Fully Sharded Data Parallel - split model across GPUs

**With 1 GPU:** Essentially a no-op, but enables proper initialization

**With multiple GPUs:** Each GPU holds only part of the model

---

## The Complete Training Loop

### High-Level Overview

```
for step in range(max_steps):  # e.g., 0, 1
    # Phase 1: ROLLOUT
    prompts = sample_prompts(dataset, rollout_batch_size)
    questions = generate_questions(prompts, rollout.n)
    
    # Phase 2: REWARD
    rewards = compute_rewards(questions)
    
    # Phase 3: TRAINING
    advantages = compute_advantages(rewards)
    batch = sample_training_batch(questions, advantages, global_batch_size)
    loss = compute_loss(batch)
    gradients = backward(loss)
    clip_gradients(gradients)
    update_model(gradients)
    
    # Phase 4: CHECKPOINT
    save_checkpoint(step)
```

### Detailed Step-by-Step Flow

#### PHASE 1: Rollout (Generation)

**Settings involved:**
- `data.rollout_batch_size=16`
- `worker.rollout.n=2`
- `worker.rollout.gpu_memory_utilization=0.25`

**What happens:**
```python
# Step 1.1: Sample prompts from dataset
prompts = random.sample(dataset, k=16)
# Example: ["Solve quadratic...", "Find triangle area...", ...]

# Step 1.2: Generate variations using vLLM Instance A (VERL internal)
all_questions = []
for prompt in prompts:
    # Generate n=2 variations per prompt
    variations = vllm_instance_a.generate(
        prompt,
        num_samples=2,
        temperature=1.0,
        max_tokens=512
    )
    all_questions.extend(variations)

# Result: 16 × 2 = 32 questions generated
```

**Example output:**
```
Prompt 1: "Generate geometry question"
  → Question 1a: "<question>Find area of triangle with base 10...</question> \boxed{50}"
  → Question 1b: "<question>Calculate perimeter of square...</question> \boxed{40}"

Prompt 2: "Generate algebra question"
  → Question 2a: "<question>Solve x² + 5x + 6 = 0</question> \boxed{-2, -3}"
  → Question 2b: "<question>Factor x² - 9</question> \boxed{(x+3)(x-3)}"

... (14 more prompts)
```

#### PHASE 2: Reward Computation

**Settings involved:**
- `worker.reward.reward_function=caller_rentropy.py:compute_score`

**What happens:**
```python
# Step 2.1: Send each question to Rentropy
rewards = []
for question in all_questions:
    # Rentropy calls vLLM Instance B (port 5000) to solve
    solution = requests.post(
        "http://localhost:5000/generate",
        json={"prompt": question}
    ).json()
    
    # Compute reward components
    format_score = check_format(question)      # e.g., 0.8
    correctness = verify_solution(solution)    # e.g., 1.0
    diversity = compute_diversity(question)    # e.g., 0.3
    
    # Combine rewards
    total_reward = combine_rewards(format_score, correctness, diversity)
    rewards.append(total_reward)

# Result: [0.9, 0.7, 0.85, ...] (32 reward scores)
```

**Reward breakdown:**
```
Question 1a: Format=0.9, Correctness=1.0, Diversity=0.2 → Total=0.8
Question 1b: Format=0.8, Correctness=0.5, Diversity=0.3 → Total=0.6
Question 2a: Format=1.0, Correctness=1.0, Diversity=0.4 → Total=0.9
...
```

#### PHASE 3: Training

##### Step 3.1: Compute Advantages (GRPO)

**Setting:** `algorithm.adv_estimator=grpo`

**What happens:**
```python
# For each set of variations (same prompt)
advantages = []
for i in range(rollout_batch_size):  # 16 prompts
    # Get the n=2 rewards for this prompt
    reward_1 = rewards[i*2 + 0]  # e.g., 0.8
    reward_2 = rewards[i*2 + 1]  # e.g., 0.6
    
    # Compute baseline (average of variations)
    baseline = (reward_1 + reward_2) / 2  # 0.7
    
    # Compute advantages (how much better than average)
    adv_1 = reward_1 - baseline  # +0.1 ✅ (good!)
    adv_2 = reward_2 - baseline  # -0.1 ❌ (bad!)
    
    advantages.extend([adv_1, adv_2])

# Result: [+0.1, -0.1, +0.15, -0.15, ...] (32 advantages)
```

**Intuition:** Advantages tell us which questions were better/worse relative to other attempts at the SAME prompt.

##### Step 3.2: Sample Training Batch

**Setting:** `worker.actor.global_batch_size=1`

**What happens:**
```python
# From 16 prompt groups (32 questions), select only 1 group for training
selected_group = random.choice(range(16))

# Extract that group's data
training_questions = all_questions[selected_group*2 : selected_group*2+2]
training_advantages = advantages[selected_group*2 : selected_group*2+2]

# Training batch: 2 questions (1 prompt × n=2)
```

**Why so small?** Memory efficiency for smoke test!

##### Step 3.3: Forward Pass (Micro-batched)

**Setting:** `worker.actor.micro_batch_size_per_device_for_experience=1`

**What happens:**
```python
# Even though batch=2, process 1 at a time to save memory
stored_logprobs = []

for question in training_questions:  # Process individually
    # Forward pass through model
    logprobs = model.forward([question])
    stored_logprobs.append(logprobs)
    
# Now have log probabilities for both questions
```

##### Step 3.4: Compute Loss

**Settings:**
- `algorithm.kl_coef=0.01`
- `algorithm.use_kl_loss=true`

**What happens:**
```python
total_loss = 0

for question, advantage, old_logprob in zip(
    training_questions, training_advantages, stored_logprobs
):
    # Get current model's probability
    new_logprob = model(question)
    
    # Compute probability ratio
    ratio = exp(new_logprob - old_logprob)
    
    # Policy loss: encourage good, discourage bad
    policy_loss = -advantage * ratio
    
    # KL penalty: don't change too much
    kl = kl_divergence(new_policy, old_policy)
    kl_loss = kl_coef * kl
    
    # Combine
    total_loss += policy_loss + kl_loss

# Example numbers:
# Question 1 (advantage=+0.1):
#   policy_loss = -0.1 × 1.05 = -0.105
#   kl_loss = 0.01 × 0.02 = 0.0002
#   loss = -0.1048 (negative → good for optimizer)

# Question 2 (advantage=-0.1):
#   policy_loss = -(-0.1) × 0.95 = +0.095
#   kl_loss = 0.01 × 0.02 = 0.0002
#   loss = +0.0952 (positive → bad for optimizer)
```

##### Step 3.5: Backward Pass (Micro-batched)

**Setting:** `worker.actor.micro_batch_size_per_device_for_update=1`

**What happens:**
```python
# Split into micro-batches for gradient computation
accumulated_gradients = []

for question, loss in zip(training_questions, losses):
    # Backward pass (one at a time)
    gradients = loss.backward()
    accumulated_gradients.append(gradients)

# Average the gradients
final_gradients = sum(accumulated_gradients) / len(accumulated_gradients)
```

**Why micro-batch:** Backward pass needs to store activations (memory intensive!)

##### Step 3.6: Clip Gradients

**Setting:** `worker.actor.max_grad_norm=1.0`

**What happens:**
```python
# Compute gradient norm
grad_norm = sqrt(sum(g² for g in gradients))

# If too large, scale down
if grad_norm > max_grad_norm:
    scale = max_grad_norm / grad_norm
    gradients = [g * scale for g in gradients]

# Example:
# Original norm: 2.5
# Scale: 1.0 / 2.5 = 0.4
# New gradients: [g * 0.4 for all gradients]
# New norm: 1.0 ✓
```

##### Step 3.7: Update Model Weights

**Settings:**
- `worker.actor.optim.lr=1e-5`
- `worker.actor.optim.strategy=adamw`

**What happens:**
```python
# AdamW optimizer update
for param in model.parameters():
    gradient = param.grad
    
    # Adaptive learning rate (simplified)
    adaptive_grad = adamw_compute(gradient, momentum, variance)
    
    # Update parameter
    param.data = param.data - lr * adaptive_grad
    
    # Example:
    # Old weight: 0.50000
    # Gradient: -0.1 (encourage this behavior)
    # Update: 0.50000 - (0.00001 × -0.1) = 0.50000 + 0.000001
    # New weight: 0.500001 (tiny change!)
```

**After 2 steps:** Weights have changed slightly to prefer high-reward questions.

#### PHASE 4: Checkpoint & Continue

**Settings:**
- `trainer.save_freq=1`
- `trainer.max_steps=2`

**What happens:**
```python
# Save checkpoint (every 1 step)
if step % save_freq == 0:
    save_checkpoint(model, optimizer, step)

# Check if done
if step >= max_steps - 1:  # step 1 >= 2-1
    print("Training complete!")
    break
else:
    continue  # Go back to Phase 1 with updated model
```

### Data Regeneration Per Step

**Critical:** Each step generates NEW data!

```
Step 1:
  ├─ Sample 16 NEW random prompts from dataset
  ├─ Generate 32 questions using Model v0
  ├─ Train on these 32 questions
  ├─ Update to Model v1
  └─ DISCARD all 32 questions (on-policy RL)

Step 2:
  ├─ Sample 16 DIFFERENT random prompts
  ├─ Generate 32 NEW questions using Model v1 (updated!)
  ├─ Train on these NEW 32 questions
  ├─ Update to Model v2
  └─ DISCARD all questions

Step 3+: STOP (max_steps=2 reached)
```

**Why regenerate?**
- GRPO is on-policy: must use fresh data from current model
- Old data becomes stale after model updates
- Ensures stable training (probability distributions match)

---

## Key Concepts Explained

### Difference: rollout_batch_size vs rollout.n

**The Confusion:**
```bash
data.rollout_batch_size=16  # ← What does this do?
worker.rollout.n=2          # ← What does this do?
```

**The Clarity:**

#### `rollout_batch_size=16`: BREADTH
- How many DIFFERENT prompts to process
- Like giving 16 students different assignments

#### `rollout.n=2`: DEPTH
- How many VARIATIONS per prompt
- Like each student tries 2 different approaches to their assignment

**Combined Effect:**
```
Total questions = rollout_batch_size × rollout.n
                = 16 × 2
                = 32 questions
```

**Visual:**
```
┌─────────────────────────────────────────────┐
│    rollout_batch_size = 16 prompts          │
└─────────────────────────────────────────────┘
     ↓           ↓           ↓           ↓
[Prompt 1]  [Prompt 2] ... [Prompt 16]
     ↓           ↓                       ↓
  ┌─────┐    ┌─────┐               ┌─────┐
  │Var A│    │Var A│               │Var A│
  │Var B│    │Var B│               │Var B│
  └─────┘    └─────┘               └─────┘
      ↑                                  ↑
      └────── rollout.n = 2 ─────────────┘
      (2 variations per prompt)
```

**Different Scenarios:**

| Scenario | rollout_batch_size | rollout.n | Total Qs | Purpose |
|----------|-------------------|-----------|----------|---------|
| Wide coverage | 64 | 2 | 128 | Fast, broad exploration |
| Deep exploration | 8 | 8 | 64 | Better advantages per prompt |
| Smoke test | 16 | 2 | 32 | Minimal memory |
| Production | 128 | 5 | 640 | Balanced |

### Micro-Batching Explained

**Why do we need TWO micro-batch settings?**

#### During Forward Pass: `micro_batch_size_per_device_for_experience=1`
```python
# Collect log probabilities (less memory intensive)
for micro_batch in split(questions, size=1):
    logprobs = model.forward(micro_batch)  # Only need output
    store(logprobs)
```
**Memory:** Model weights + activations (forward only)

#### During Backward Pass: `micro_batch_size_per_device_for_update=1`
```python
# Compute gradients (MORE memory intensive!)
for micro_batch in split(questions, size=1):
    loss = compute_loss(micro_batch)
    gradients = loss.backward()  # Stores ALL intermediate activations!
    accumulate(gradients)
```
**Memory:** Model weights + activations + gradients + optimizer states

**Backward needs more memory** because it must store all intermediate computations for the chain rule!

### On-Policy Learning

**VERL uses on-policy RL (GRPO, PPO, RLOO):**

**Rules:**
1. ✅ Must use fresh data from current policy
2. ❌ Cannot reuse old experiences
3. ✅ Stable but sample-inefficient

**Why?**
```python
# Step 1: Old model generates
old_prob = model_v0("question") = 0.01

# Train, model updates
# Step 2: Try to reuse old data
new_prob = model_v1("question") = 0.015

# PPO ratio:
ratio = new_prob / old_prob = 1.5  # TOO LARGE!
# Violates PPO assumptions → instability
```

**Solution:** Always regenerate with current model:
```python
# Step 2: Use fresh data
gen_prob = model_v1("new_question")
train_prob = model_v1("new_question")
ratio = train_prob / gen_prob ≈ 1.0  # Stable! ✓
```

### GRPO Advantage Estimation

**How it works:**

For each prompt, we have n=2 variations:
```python
prompt = "Generate geometry question"
question_1 = "<question>Triangle area...</question>"
question_2 = "<question>Rectangle perimeter...</question>"

reward_1 = 0.8
reward_2 = 0.6

# Compute baseline (average of this group)
baseline = (0.8 + 0.6) / 2 = 0.7

# Compute advantages
advantage_1 = 0.8 - 0.7 = +0.1  # Better than average!
advantage_2 = 0.6 - 0.7 = -0.1  # Worse than average!
```

**Intuition:**
- We're not comparing to absolute standards
- We're comparing variations of the SAME prompt
- "Which way of answering THIS prompt was better?"

**This is why we need `rollout.n ≥ 2`:** Need at least 2 samples to compute relative advantage!

---

## Common Confusions

### Q1: "Why do I manually start a vLLM server if VERL has its own?"

**A:** You need TWO jobs:
1. **Questioner** (VERL's internal vLLM) - Generates questions
2. **Solver** (Your manual vLLM) - Evaluates questions

Like having two students:
- Student A writes exam questions
- Student B takes the exam to verify questions are solvable

### Q2: "Does max_steps=2 reuse the same prompts?"

**A:** No! Each step:
1. Samples NEW random prompts
2. Generates NEW questions with UPDATED model
3. Trains on NEW data
4. Discards old data (on-policy RL)

### Q3: "Why are my batch sizes so complicated?"

**A:** Memory management!

```
global_batch_size = 1          # How many prompts for training
rollout_batch_size = 16        # How many prompts for generation
rollout.n = 2                  # Variations per prompt
micro_batch_update = 1         # Backward pass chunk size
micro_batch_experience = 1     # Forward pass chunk size
```

Different stages need different memory profiles!

### Q4: "Why is learning so slow with lr=1e-5?"

**A:** Safety for large language models!

```python
# Weight update per step:
weight_change = lr × gradient = 0.00001 × 0.1 = 0.000001

# After 1 step: weight barely changed!
# Need 1000s of steps for meaningful learning
```

Smoke test with max_steps=2 doesn't actually teach much!

### Q5: "What's the difference between global_batch_size and rollout_batch_size?"

**A:**

| Setting | Purpose | Phase | Value in Smoke Test |
|---------|---------|-------|---------------------|
| `rollout_batch_size` | How many prompts to generate from | Rollout (generation) | 16 |
| `global_batch_size` | How many prompts to train on | Training (optimization) | 1 |

**Why different?**
- Generate from many (16) to build experience buffer
- Train on few (1) to save memory during optimization

---

## Production vs Smoke Test Settings

### Smoke Test Settings (Current)

**Purpose:** Verify pipeline works, minimal compute

```bash
# Data
data.max_response_length=512
data.rollout_batch_size=16

# Training
trainer.max_steps=2
worker.actor.global_batch_size=1
worker.actor.micro_batch_size_per_device_for_update=1

# Rollout
worker.rollout.n=2
worker.rollout.gpu_memory_utilization=0.25

# Solver
solver_gpu_mem_util=0.2

# Results
- Runtime: ~5-10 minutes per mode
- Questions generated: 32 per step × 2 steps = 64 total
- Unique prompts seen: ~32 out of 12,000 (0.27%)
- Learning: Minimal (just 2 gradient updates)
```

### Production Settings (Recommended)

**Purpose:** Actually train a good questioner model

```bash
# Data
data.max_response_length=2048
data.rollout_batch_size=128

# Training
trainer.max_steps=10000
worker.actor.global_batch_size=64
worker.actor.micro_batch_size_per_device_for_update=4
trainer.save_freq=500

# Rollout
worker.rollout.n=5
worker.rollout.gpu_memory_utilization=0.7

# Solver
solver_gpu_mem_util=0.6

# Hardware
trainer.n_gpus_per_node=4  # Use 4 GPUs with FSDP

# Results
- Runtime: 2-5 days
- Questions generated: 128×5 = 640 per step × 10,000 steps = 6.4M total
- Unique prompts seen: 128 × 10,000 = 1.28M (with repeats)
- Learning: Substantial improvement expected
```

### Scaling Guidelines

**For different GPU memory:**

| GPU Memory | rollout_batch_size | global_batch_size | rollout.n | rollout_gpu_util |
|------------|-------------------|-------------------|-----------|------------------|
| 24 GB | 16 | 1 | 2 | 0.25 |
| 40 GB | 32 | 4 | 4 | 0.5 |
| 80 GB | 64 | 16 | 5 | 0.7 |
| 80 GB × 4 | 128 | 64 | 8 | 0.8 |

**Progressive scaling strategy:**
```bash
# Phase 1: Verify it works (smoke test)
max_steps=2, global_batch_size=1

# Phase 2: Short training run
max_steps=100, global_batch_size=4

# Phase 3: Medium training
max_steps=1000, global_batch_size=16

# Phase 4: Full production
max_steps=10000, global_batch_size=64
```

---

## Troubleshooting Guide

### Out of Memory (OOM) Errors

**Symptoms:** CUDA OOM, process killed

**Solutions:**
1. Reduce `rollout_batch_size`
2. Reduce `global_batch_size`
3. Reduce `rollout.n`
4. Lower `gpu_memory_utilization`
5. Reduce `max_response_length`
6. Set `micro_batch_size` to 1
7. Enable CPU offloading: `enable_cpu_offload=true`

### Training Not Improving

**Symptoms:** Rewards not increasing, loss not decreasing

**Solutions:**
1. Check if `max_steps` is too small (smoke test = no learning!)
2. Increase `global_batch_size` (more data per update)
3. Increase `lr` (faster learning, but less stable)
4. Check reward function is working (solver responding?)
5. Try more training steps (1000+ minimum for real learning)

### vLLM Port Conflicts

**Symptoms:** "Address already in use" on port 5000

**Solutions:**
1. Kill existing vLLM: `pkill -f "vllm_service_init.*port 5000"`
2. Use different port: `--port 5001`
3. Check running processes: `ps aux | grep vllm`

### Training Hangs

**Symptoms:** No progress, stuck on one step

**Causes:**
1. vLLM server not responding (solver timeout)
2. Rollout taking too long (increase timeout)
3. Deadlock in distributed training

**Solutions:**
1. Check vLLM logs for errors
2. Restart vLLM server
3. Check GPU utilization: `nvidia-smi`
4. Kill and restart training

---

## Quick Reference

### File Structure After Training

```
storage/
├── models/
│   └── experiment_name/
│       ├── global_step_1/
│       │   └── actor/
│       │       ├── checkpoint/          # FSDP checkpoint
│       │       └── huggingface/        # Merged HF format ← Use this!
│       └── global_step_2/
│           └── actor/
│               └── huggingface/
└── smoketest_results/
    ├── experiment_mode1_questioner.log
    └── results_summary.txt
```

### Essential Commands

```bash
# Start training
export STORAGE_PATH=/root/euclid/rentropy/R-Zero/storage
bash scripts/smoketest_questioner.sh \
    Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-0.5B-Instruct \
    my_experiment \
    1

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check vLLM processes
ps aux | grep vllm

# Kill vLLM
pkill -f "vllm_service_init"

# View logs
tail -f storage/smoketest_results/*.log

# Load trained model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "storage/models/my_experiment/global_step_2/actor/huggingface"
)
```

### Memory Budget Calculator

```python
def estimate_memory(
    rollout_batch_size,
    global_batch_size,
    rollout_n,
    model_size_gb,
    max_response_length
):
    """Estimate GPU memory needed"""
    
    # vLLM for rollout
    rollout_memory = model_size_gb * 1.2  # Model + KV cache
    
    # vLLM for solver
    solver_memory = model_size_gb * 1.2
    
    # Training: model + gradients + optimizer
    training_memory = model_size_gb * (1 + 1 + 2)  # weights + grads + adam states
    
    # Activations during forward/backward
    activation_memory = (
        global_batch_size * 
        max_response_length * 
        model_size_gb * 
        0.001
    )
    
    total = rollout_memory + solver_memory + training_memory + activation_memory
    
    return {
        "rollout": rollout_memory,
        "solver": solver_memory,
        "training": training_memory,
        "activations": activation_memory,
        "total": total
    }

# Example for smoke test:
# model = 0.5B params ≈ 1 GB
estimate_memory(
    rollout_batch_size=16,
    global_batch_size=1,
    rollout_n=2,
    model_size_gb=1,
    max_response_length=512
)
# Output: ~10 GB total (fits in 24 GB GPU with headroom)
```

---

## Glossary

**Actor:** The policy model being trained (questioner model)

**Advantage:** How much better a sample is compared to baseline

**FSDP:** Fully Sharded Data Parallel - distribute model across GPUs

**Global batch size:** Number of unique samples per training step

**GRPO:** Group Relative Policy Optimization - compare within groups

**KL divergence:** Measure of how much policy changed

**Micro-batch:** Subset of batch that fits in GPU memory

**On-policy:** Must use fresh data from current policy

**Rollout:** Generation phase (sampling from policy)

**vLLM:** Fast inference engine for LLMs

---

**Document version:** 1.0
**Last updated:** 2026-01-31
**For:** Rentropy questioner training with VERL
