# MATH Dataset Evaluation with vLLM

Evaluate language models on the [MATH dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math) using vLLM for fast inference.

## Requirements

```bash
pip install vllm datasets tqdm
```

## Usage

### Basic Evaluation (Qwen 3B on Number Theory)

```bash
python -m euclid.evaluate_math.evaluate \
    --model vibhuiitj/qwen3-4b-base-variant1-feb5-solver-iter3 \
    --dataset-config number_theory \
    --split test 
    
```

### With Few-Shot Prompting

```bash
python -m tree.euclid.evaluate_math.evaluate \
    --model Qwen/Qwen2.5-3B \
    --dataset-config number_theory \
    --few-shot \
    --output results_qwen3b_number_theory_fewshot.jsonl
```

### Evaluate on Limited Examples

```bash
python -m tree.euclid.evaluate_math.evaluate \
    --model Qwen/Qwen2.5-3B \
    --dataset-config number_theory \
    --limit 100 \
    --output results_qwen3b_number_theory_100.jsonl
```

### Multi-GPU Evaluation

```bash
python -m tree.euclid.evaluate_math.evaluate \
    --model Qwen/Qwen2.5-3B \
    --dataset-config number_theory \
    --tensor-parallel-size 2 \
    --output results_qwen3b_number_theory.jsonl
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B` | Model name or path |
| `--dataset-config` | `number_theory` | MATH subset (algebra, geometry, etc.) |
| `--split` | `test` | Dataset split (train/test) |
| `--output` | `math_eval_results.jsonl` | Output file for results |
| `--limit` | `0` (all) | Max examples to evaluate |
| `--few-shot` | `False` | Use few-shot prompting |
| `--tensor-parallel-size` | `1` | Number of GPUs |
| `--max-tokens` | `2048` | Max generation tokens |
| `--temperature` | `0.0` | Sampling temperature |

## Dataset Configs

| Config | Test Size | Description |
|--------|-----------|-------------|
| `number_theory` | 540 | Number theory problems |
| `algebra` | 1,187 | Algebra problems |
| `geometry` | 479 | Geometry problems |
| `counting_and_probability` | 474 | Combinatorics & probability |
| `intermediate_algebra` | 903 | Advanced algebra |
| `prealgebra` | 871 | Pre-algebra |
| `precalculus` | 546 | Precalculus |

## Output Format

### Results JSONL (`results.jsonl`)

Each line contains:
```json
{
  "problem": "Find the GCD of 12 and 18",
  "level": "Level 1",
  "ground_truth": "6",
  "predicted": "6",
  "generated_text": "Using the Euclidean algorithm...",
  "correct": true
}
```

### Summary JSON (`results.summary.json`)

```json
{
  "model": "Qwen/Qwen2.5-3B",
  "dataset": "EleutherAI/hendrycks_math/number_theory",
  "split": "test",
  "total": 540,
  "correct": 180,
  "accuracy": 0.333,
  "level_stats": {
    "Level 1": {"correct": 50, "total": 100},
    "Level 2": {"correct": 40, "total": 110},
    ...
  }
}
```

## Available Qwen Models

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen2.5-0.5B | 0.5B | `Qwen/Qwen2.5-0.5B` |
| Qwen2.5-1.5B | 1.5B | `Qwen/Qwen2.5-1.5B` |
| Qwen2.5-3B | 3B | `Qwen/Qwen2.5-3B` |
| Qwen2.5-7B | 7B | `Qwen/Qwen2.5-7B` |
| Qwen2.5-Math-1.5B | 1.5B | `Qwen/Qwen2.5-Math-1.5B` |
| Qwen2.5-Math-7B | 7B | `Qwen/Qwen2.5-Math-7B` |
