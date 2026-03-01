# Simple R-Zero Training Pipeline

A lightweight, simplified implementation of R-Zero using HuggingFace TRL (GRPO) and vLLM for inference.

## 🎯 Features

- **Simple**: Minimal hyperparameters, easy to understand
- **Lightweight**: Uses HuggingFace TRL's built-in GRPO trainer
- **Fast Inference**: vLLM for efficient generation
- **Concept-Based**: Train sequentially on 5 concepts
- **Self-Consistency**: Uses majority voting for rewards (no ground truth needed)

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Note**: TRL is only compatible with specific vLLM versions. This repo pins `vllm==0.12.0` in `requirements.txt`.

## 🚀 Quick Start

1. **Edit configuration** (optional):
   ```python
   # Edit config.py to change:
   # - base_model: Your model path
   # - questions_per_concept: Number of questions to generate
   # - num_iterations: Training iterations per concept
   ```

2. **Run training**:
   ```bash
   python train.py
   ```

That's it! The pipeline will:
1. Load 5 concepts from `examples/concept_sets.json`
2. For each concept:
   - Generate questions using vLLM
   - Solve questions with multiple rollouts
   - Compute self-consistency rewards
   - Train model using GRPO
   - Move to next iteration

## 📁 Project Structure

```
rconceptnew/
├── config.py              # Simple configuration
├── train.py               # Main training script
├── question_generator.py  # Question generation with vLLM
├── solver.py              # Solution generation with vLLM
├── reward.py              # Self-consistency reward computation
├── examples/
│   └── concept_sets.json  # 5 concept sets
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
@dataclass
class Config:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Your model
    questions_per_concept: int = 20                # Questions per concept
    num_iterations: int = 3                        # Iterations per concept
    num_rollouts: int = 4                          # Rollouts for self-consistency
    learning_rate: float = 1e-5                    # Learning rate
    batch_size: int = 4                            # Batch size
    max_steps: int = 50                            # Training steps per iteration
```

## 🔄 How It Works

1. **Question Generation**: Uses vLLM to generate questions conditioned on concepts
2. **Solution Generation**: Solves each question with multiple rollouts (e.g., 4 rollouts)
3. **Reward Computation**: Computes self-consistency reward = fraction of rollouts agreeing with majority answer
4. **ZPD Filtering**: Keeps questions with rewards in 0.3-0.8 range (Zone of Proximal Development)
5. **GRPO Training**: Uses TRL's GRPO trainer to improve the model
6. **Iteration**: Repeats for each concept, using updated model

## 📊 Output

Models are saved in `./output/concept_{iteration}_model/` after each iteration.

## 🎓 Key Simplifications

Compared to full R-Zero:
- ✅ Single script, no complex bash orchestration
- ✅ Uses TRL's built-in GRPO (no custom RL implementation)
- ✅ vLLM for fast inference (no complex worker setup)
- ✅ Simple self-consistency rewards (no complex scoring)
- ✅ Minimal hyperparameters (easy to tune)

## 🔧 Troubleshooting

**Out of memory?**
- Reduce `tensor_parallel_size` in config
- Reduce `batch_size` or `questions_per_concept`
- Use a smaller model

**Training too slow?**
- Reduce `max_steps` or `num_iterations`
- Reduce `num_rollouts`
- Use fewer GPUs

**Questions not good quality?**
- Increase `questions_per_concept` to get more candidates
- Adjust ZPD filtering range in `train.py`
- Try different temperature/top_p values

## 📝 Notes

- This is a simplified version for experimentation
- For production, you may want to add:
  - Better answer matching (semantic similarity)
  - More sophisticated reward functions
  - Evaluation on held-out sets
  - Checkpointing and resuming
