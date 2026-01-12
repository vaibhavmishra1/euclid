# Knowledge-Set-Based Curriculum Learning for R-Zero

This module extends the R-Zero framework to implement curriculum learning based on mathematical knowledge sets extracted from the MATH dataset.

## Overview

### Key Concepts

1. **Knowledge Sets**: Each original MATH problem has a set of knowledge points (KPs). We keep these grouped together (not separated).
   - 868 knowledge sets = 868 original questions
   - Each set contains 2-5 knowledge points that work together

2. **Difficulty Levels 1-5**: Matches the MATH dataset levels
   - Level 1: Basic problem for students just learning
   - Level 2: Straightforward, requires solid understanding
   - Level 3: Moderate, requires reasoning
   - Level 4: Challenging, requires creative thinking
   - Level 5: Competition-level problem

3. **Initial Difficulty**: Each set starts at its original problem's level (not all at 1)

4. **Adaptive Curriculum**: Difficulty adjusts based on R-Zero reward:
   - If avg_reward > α (0.7): Increase difficulty
   - If avg_reward < β (0.3): Decrease difficulty
   - Otherwise: Maintain current difficulty

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├─────────────────────────────────────────────────────────────┤
│  1. Load 868 Knowledge Sets (difficulty from original level) │
│                         ↓                                    │
│  2. Generate 5 Questions/Set (using ALL KPs + difficulty)    │
│     → 868 × 5 = 4,340 questions per iteration               │
│                         ↓                                    │
│  3. Evaluate with Solver (R-Zero reward, unchanged)          │
│                         ↓                                    │
│  4. Compute Avg Reward per Set                               │
│     → If avg > 0.7: increase difficulty (max 5)             │
│     → If avg < 0.3: decrease difficulty (min 1)             │
│                         ↓                                    │
│  5. Train Solver on Questions                                │
│                         ↓                                    │
│  6. Train Challenger on Updated Prompts                      │
│                         ↓                                    │
│  7. Repeat                                                   │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Quick Start

```bash
# Set environment variables
export STORAGE_PATH="/path/to/storage"
export HUGGINGFACENAME="your_username"

# Run training
bash scripts/kp_main.sh \
    Qwen/Qwen3-4B \
    qwen3-4b-ks \
    /path/to/math_concepts_openai_all_number_theory.jsonl \
    3  # Number of iterations
```

### Input Data Format

Your JSONL file should have this format (from concept extraction):

```json
{
  "question": "If $AAA_4$ can be expressed as $33_b$...",
  "knowledge_points": [
    "Base conversion: For a number in base b...",
    "Properties of digits in base systems...",
    "Linear equation: If ax = b, then x = b/a"
  ],
  "level": "Level 4"
}
```

### Challenger Prompt Example

For a knowledge set at difficulty 4:

```
**Knowledge Points Required:**
1. Base conversion: For a number in base b, the value is calculated as Σ(d_i * b^i)
2. Properties of digits in base systems: In base b, digits must satisfy 0 ≤ d < b
3. Linear equation: If ax = b, then x = b/a

**Target Difficulty Level:** 4/5 (Level 4 - challenging problem requiring creative thinking)
```

## Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.7 | Increase difficulty if avg_reward > alpha |
| `beta` | 0.3 | Decrease difficulty if avg_reward < beta |
| `questions_per_set` | 5 | Questions generated per knowledge set |
| `num_iterations` | 3 | Number of training iterations |

### Difficulty Scale

| Level | Description |
|-------|-------------|
| 1 | Basic - students just learning the concepts |
| 2 | Straightforward - solid understanding required |
| 3 | Moderate - reasoning and concept connections |
| 4 | Challenging - creative thinking required |
| 5 | Competition-level - very challenging |

## File Structure

```
knowledge_curriculum/
├── __init__.py              # Module exports
├── knowledge_manager.py     # KnowledgeSetManager class
├── prompts.py              # Prompt templates (1-5 scale)
├── question_generate.py    # Question generation script
├── curriculum_trainer.py   # Main trainer
└── README.md               # This file

scripts/
├── kp_main.sh              # Main training script
├── kp_challenger_train.sh  # Challenger training
└── kp_solver_train.sh      # Solver training
```

## Key Design Decisions

1. **Knowledge Sets, Not Individual KPs**: We keep the KPs grouped as they appear in original problems. This preserves the multi-concept nature of real math problems.

2. **Initial Difficulty from Problem Level**: Instead of starting all sets at level 1, we use the original MATH dataset's level (1-5). This provides a better starting point.

3. **5-Level Scale**: Matches the MATH dataset exactly. Easier for the model to distinguish between levels.

4. **R-Zero Reward Unchanged**: We use R-Zero's reward function as-is. The reward captures both accuracy and diversity.

## Example Output

```
Iteration 2 Complete!
  
Knowledge Set Statistics:
  Total: 868 sets
  Avg Difficulty: 3.4
  Distribution: {1: 50, 2: 150, 3: 300, 4: 268, 5: 100}

Difficulty Adjustments:
  Increased: 120
  Decreased: 80
  Maintained: 600
  At max (5): 68
  At min (1): 0
```

## Citation

If you use this code, please cite the original R-Zero paper:

```bibtex
@article{huang2025rzeroselfevolvingreasoningllm,
    title={R-Zero: Self-Evolving Reasoning LLM from Zero Data},
    author={Chengsong Huang and others},
    year={2025},
    eprint={2508.05004},
    archivePrefix={arXiv},
}
```

## License

Apache License 2.0
