# Few-Shot Question Generation

This script generates new math questions using few-shot prompting based on cluster membership from previously generated questions.

## Overview

The algorithm works as follows:

1. **Load clustered questions**: Load questions that have already been assigned to clusters (filters out very long questions)
2. **Sample cluster**: Pick a cluster randomly based on its occurrence probability (more frequent clusters are more likely to be sampled)
3. **Select examples**: Randomly select 2 questions from the chosen cluster (using shorter questions to avoid token limit issues)
4. **Generate new question**: Prompt the model to generate a new question on the same concept but with different details
5. **Repeat**: Continue until desired number of questions are generated
6. **Save**: Store generated questions with their source cluster information

## Requirements

- Python 3.8+
- vLLM (for question generation)
- numpy
- tqdm
- Access to GPU(s) for question generation

## Usage

### Basic Usage

```bash
python generate_few_shot_questions.py \
    --model /path/to/model \
    --questions_with_clusters storage/generated_question_with_clusters/variant1-feb5-questioner-iter3_with_clusters.json \
    --num_samples 10000
```

### Full Options

```bash
python generate_few_shot_questions.py \
    --model vibhuiitj/qwen3-4b-base-variant1-feb5-questioner-iter3 \
    --questions_with_clusters storage/generated_question_with_clusters/variant1-feb5-questioner-iter3_with_clusters.json \
    --num_samples 10000 \
    --num_examples 3 \
    --temperature 0.7 \
    --max_tokens 1024 \
    --batch_size 32 \
    --tensor_parallel_size 1 \
    --max_model_len 32768 \
    --seed 42 \
    --output_path storage/generated_question_few_shot/variant1_few_shot.json
```


python generate_few_shot_questions.py \
    --model vibhuiitj/qwen3-4b-base-variant2-feb5-solver-iter5 \
    --questions_with_clusters /workspace/euclid/questioner_model_evaluation/storage/generated_question_with_clusters/variant2-feb5-questioner-iter5_with_clusters.json \
    --num_samples 1000 \
    --num_examples 3 \
    --temperature 1.0 \
    --max_tokens 1024 \
    --batch_size 512 \
    --tensor_parallel_size 1 \
    --max_model_len 32768 \
    --seed 42 \
    --output_path storage/generated_question_few_shot/variant2_few_shot.json

## Arguments

- `--model`: Path to question generation model (required)
- `--questions_with_clusters`: Path to JSON file with questions and cluster IDs (required)
- `--num_samples`: Number of new questions to generate (default: 10000)
- `--output_path`: Output path for generated questions (default: auto-generated)
- `--num_examples`: Number of example questions per prompt (default: 2)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum tokens per question (default: 1024)
- `--batch_size`: Batch size for generation (default: 32)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--max_model_len`: Maximum model context length (default: 32768)

## Output

The script generates a JSON file with the following structure:

```json
[
  {
    "question": "Find the sum of all prime numbers less than 50.",
    "answer": "328",
    "score": 0,
    "cluster_id": 42,
    "source_cluster": 42,
    "example_questions": [
      "Find all prime numbers between 1 and 20.",
      "What is the largest prime number less than 100?"
    ],
    "generation_method": "few_shot_cluster_based"
  }
]
```

**Field Descriptions:**
- `question`: The generated problem statement (extracted from `<question>` tags)
- `answer`: The final answer (extracted from `\boxed{}`)
- `score`: 0 for successfully parsed questions, -1 for parse failures
- `cluster_id`: The cluster this question belongs to
- `source_cluster`: Same as cluster_id (for consistency)
- `example_questions`: The 2-3 example questions used to generate this
- `generation_method`: Always "few_shot_cluster_based"

### Output Directory

By default, questions are saved to:
```
storage/generated_question_few_shot/{input_filename}_few_shot_{model_name}.json
```

## How It Works

### 1. Question Filtering

Very long questions (>800 characters) are filtered out to prevent token limit issues:
- This ensures prompts stay within the model's context window
- Only reasonably-sized questions are used as examples

### 2. Cluster Probability Sampling

Clusters are sampled proportionally to their frequency in the original dataset:
- If Cluster A has 100 questions and Cluster B has 50 questions
- Cluster A has 2/3 probability, Cluster B has 1/3 probability

### 3. Few-Shot Prompt Construction

The script now uses the **same high-quality prompt format** as `question_generate.py`, with few-shot examples integrated:

**System Message:**
- Instructs the model to be an expert competition-math problem setter
- Requires thinking step-by-step, then outputting structured format
- Demands output in `<question>` tags with `\boxed{answer}` format
- Explicitly forbids explanations or extra markup

**User Message with Examples:**
```
Here are some example problems on related mathematical concepts:

Example 1:
Find all prime numbers between 1 and 20.

Example 2:
What is the largest prime number less than 100?

Based on the mathematical concepts demonstrated in the examples above, generate ONE NEW, more challenging
problem that:
1. Explores the same underlying mathematical concept or theory
2. Is slightly more difficult or requires deeper reasoning than the examples
3. Introduces a novel twist, additional constraint, or creative scenario
4. Is distinctly different from the examples (not just changing numbers)

Remember to format the output exactly as instructed: <question> tags and \boxed{} answer only.
```

**Key Improvements over Previous Version:**
- ✅ **Structured output**: Uses `<question>` tags and `\boxed{answer}` format, preventing solution contamination
- ✅ **Chat template**: Properly formatted chat messages for better model understanding
- ✅ **Explicit instructions**: Clear system-level instructions about being a problem setter
- ✅ **Parse validation**: Extracts and validates questions and answers, marking failures with score=-1
- ✅ **Consistent with baseline**: Same prompt structure as the original question generation script

**Design Philosophy:**
- **Progressive difficulty**: Generated questions should be slightly harder than examples
- **Novelty focus**: Encourages creative variations, not just number substitutions
- **Concept preservation**: Maintains the mathematical concept while adding complexity
- **Deep reasoning**: Promotes questions that require multi-step thinking
- **Clean output**: Structured format prevents embedding solutions in question text

### 4. Generation and Filtering

- Questions are generated in batches for efficiency
- Each generated question is tagged with its source cluster
- Example questions are saved for reference and analysis

## Benefits of This Approach

1. **Structured Output Format**: Using `<question>` tags and `\boxed{answer}` prevents the model from embedding solutions in questions (major issue in previous version)
2. **High-Quality Prompting**: Same expert problem-setter prompt as the baseline generation script
3. **Progressive Difficulty**: Generated questions are slightly harder than examples, creating a natural curriculum
4. **Concept Consistency**: Questions maintain thematic coherence within clusters
5. **Novel Variations**: Encourages creative twists and constraints, not just number substitutions
6. **Parse Validation**: Automatically validates output format and marks failures with score=-1
7. **Diversity**: Random sampling within clusters ensures variety
8. **Guided Generation**: Few-shot examples help the model understand the desired question type
9. **Traceability**: Each question is linked to its source cluster and examples
10. **Token Efficiency**: Filtering long questions prevents context window overflow
11. **Deep Reasoning**: Promotes multi-step problems requiring deeper mathematical thinking

### Improvements Over Previous Implementation

The updated script addresses the critical issues identified in the analysis:

**❌ Old Version Problems:**
- Generated solutions embedded in question text (10/11 questions)
- No structured output format
- Difficult to parse and validate
- Mathematical errors not caught

**✅ New Version Solutions:**
- Explicit `<question>` and `\boxed{}` format enforced
- Chat-based prompt matching the baseline script
- Automatic parsing and validation
- Parse failures marked with score=-1 for filtering
- Same high-quality instructions as original question generator

## Example Workflow

1. **Generate initial questions and cluster them**:
```bash
python evaluate_questioner_models.py \
    --model1 model1 --model2 model2 \
    --num_samples 10000
```

2. **Generate few-shot questions from clustered data**:
```bash
python generate_few_shot_questions.py \
    --model model1 \
    --questions_with_clusters storage/generated_question_with_clusters/model1_with_clusters.json \
    --num_samples 5000
```

3. **Analyze the new questions** (optional):
```bash
# You can re-cluster the few-shot questions to see if they maintain cluster coherence
python analyze_clusters.py \
    --save_name1 model1_few_shot \
    --save_name2 model1_original \
    --num_gpus 1
```

## Context Window Configuration

### Qwen3 Context Length Support

Qwen3 models natively support **32,768 tokens (32K)** context length, much larger than the previous 4K default:

```bash
# Use full 32K context (default, recommended)
python generate_few_shot_questions.py \
    --model your_model \
    --questions_with_clusters your_file.json \
    --max_model_len 32768

# Use smaller context if needed (faster, less memory)
python generate_few_shot_questions.py \
    --model your_model \
    --questions_with_clusters your_file.json \
    --max_model_len 8192
```

### Benefits of Longer Context:

1. **More examples**: Can use 3-4 examples instead of 2
2. **Longer questions**: Can include longer/more complex example questions
3. **Better guidance**: Model has more context to understand the pattern
4. **Fewer filters**: Don't need aggressive question length filtering

### Recommended Settings by Context Length:

| max_model_len | num_examples | max_question_chars | Use Case |
|---------------|--------------|-------------------|----------|
| 4096 | 2 | 800 | Memory constrained, fast generation |
| 8192 | 2-3 | 1200 | Balanced performance |
| 16384 | 3-4 | 1500 | Better quality examples |
| 32768 | 4-5 | 2000 | Maximum quality (default) |

## Notes

- **Context window**: Now defaults to 32K tokens (Qwen3's native support), much better than previous 4K limit
- **Question filtering**: Very long questions (>800 chars) are automatically filtered out to prevent token limit errors
- **Cluster filtering**: Clusters with fewer than `num_examples` questions are excluded from sampling
- The script automatically renormalizes probabilities for valid clusters
- Generated questions inherit the cluster ID from their source examples
- Temperature > 0 introduces randomness in generation (0.7 is a good balance)
- Higher batch sizes improve throughput but require more GPU memory
- With 32K context, you can now use 3+ examples and longer questions for better results!

## Filtering and Post-Processing

### Filter Out Parse Failures

Questions with `score=-1` failed to parse correctly (missing `<question>` tags or `\boxed{}`). Filter them out:

```python
import json

with open('generated_questions.json', 'r') as f:
    questions = json.load(f)

# Keep only successfully parsed questions
valid_questions = [q for q in questions if q['score'] != -1]

print(f"Kept {len(valid_questions)}/{len(questions)} valid questions")
```

### Extract Questions Only

If you only need the question text without metadata:

```python
questions_only = [q['question'] for q in valid_questions]
```

## Troubleshooting

**Issue**: "No clusters have at least N questions"
- **Solution**: Reduce `--num_examples` or use a dataset with more questions per cluster

**Issue**: Out of memory errors
- **Solution**: Reduce `--batch_size` or use more GPUs with `--tensor_parallel_size`

**Issue**: Generated questions are too similar to examples
- **Solution**: Increase `--temperature` for more diversity

**Issue**: Generated questions are off-topic
- **Solution**: Decrease `--temperature` or increase `--num_examples` for better guidance

**Issue**: High parse failure rate
- **Solution**: The model may not be following the format. Check the model's training and consider fine-tuning on formatted outputs
