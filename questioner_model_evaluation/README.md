# Questioner Model Evaluation

This directory contains scripts to evaluate two trained question generation models by analyzing the cluster distribution of generated questions.

## Overview

The evaluation process:
1. Generates 10,000 questions from each model using the same prompt
2. Assigns each question to a cluster using Qwen 0.6B embeddings and pre-computed cluster centroids
3. Computes and prints detailed statistics about cluster frequency distributions
4. Compares the diversity and coverage of both models

## Requirements

- Python 3.8+
- vLLM (for question generation)
- sentence-transformers (for embeddings)
- numpy
- Access to GPU(s) for question generation
- STORAGE_PATH environment variable set (or use --storage_path flag)

## Usage

### Basic Usage

```bash
python evaluate_questioner_models.py \
    --model1 /path/to/model1 \
    --model2 /path/to/model2 \
    --num_samples 10000
```

### Full Options

```bash
python evaluate_questioner_models.py \
    --model1 /path/to/model1 \
    --model2 /path/to/model2 \
    --num_samples 10000 \
    --num_gpus 1 \
    --centroids_path /workspace/euclid/rentropy/cluster_space/cluster_data/centroids.npy \
    --embedding_model Qwen/Qwen3-Embedding-0.6B \
    --storage_path /path/to/storage \
    --save_name1 model1_eval \
    --save_name2 model2_eval
```
python evaluate_questioner_models.py \
    --model1 vibhuiitj/qwen3-4b-base-variant1-feb5-questioner-iter3 \
    --model2 vibhuiitj/qwen3-4b-base-variant2-feb5-questioner-iter5 \
    --num_samples 10000 \
    --num_gpus 1 \
    --centroids_path /workspace/euclid/rentropy/cluster_space/cluster_data/centroids.npy \
    --embedding_model Qwen/Qwen3-Embedding-0.6B \
    --storage_path /workspace/euclid/questioner_model_evaluation/storage \
    --save_name1 variant1-feb5-questioner-iter3 \
    --save_name2 variant2-feb5-questioner-iter5 \
    --skip_generation 




### Skip Generation (Analyze Existing Results)

If you've already generated questions, you can skip generation and only analyze:

```bash
python evaluate_questioner_models.py \
    --model1 /path/to/model1 \
    --model2 /path/to/model2 \
    --skip_generation \
    --save_name1 model1_eval \
    --save_name2 model2_eval
```

## Arguments

- `--model1`: Path to first trained question generation model (required)
- `--model2`: Path to second trained question generation model (required)
- `--num_samples`: Number of questions to generate per model (default: 10000)
- `--num_gpus`: Number of GPUs to use for parallel generation (default: 8)
- `--centroids_path`: Path to centroids.npy file (default: `/workspace/euclid/rentropy/cluster_space/cluster_data/centroids.npy`)
- `--embedding_model`: Embedding model for cluster assignment (default: `Qwen/Qwen3-Embedding-0.6B`)
- `--storage_path`: Storage path for generated questions (default: uses STORAGE_PATH env var)
- `--skip_generation`: Skip question generation and only analyze existing results
- `--save_name1`: Save name prefix for model 1 results (default: `model1_eval`)
- `--save_name2`: Save name prefix for model 2 results (default: `model2_eval`)

## Output

The script prints detailed statistics to stdout including:
- Total questions generated
- Cluster coverage (unique clusters used / total clusters)
- Cluster count distribution (min, max, mean, median, std dev, percentiles)
- Diversity metrics (entropy, normalized entropy)
- Top 10 and bottom 10 clusters by frequency
- Comparison summary between the two models

Statistics are also saved to `results/cluster_statistics.json` for further analysis.

## How It Works

1. **Question Generation**: Uses the same prompt from `question_generate.py` to generate questions from both models. Questions are generated in parallel across multiple GPUs.

2. **Cluster Assignment**: 
   - Each question is embedded using Qwen 0.6B embedding model
   - Embeddings are compared to pre-computed cluster centroids (1024 clusters)
   - Each question is assigned to the nearest cluster (highest cosine similarity)

3. **Statistics Computation**:
   - Counts how many questions fall into each cluster
   - Computes distribution statistics (mean, median, std dev, percentiles)
   - Calculates entropy as a diversity measure
   - Identifies most/least frequent clusters

4. **Comparison**: Compares the two models on metrics like cluster coverage, diversity (entropy), and distribution characteristics.

## Example Output

```
================================================================================
CLUSTER DISTRIBUTION STATISTICS: Model 1
================================================================================

Basic Statistics:
  Total Questions: 10,000
  Unique Clusters Used: 856 / 1024
  Cluster Coverage: 83.59%

Cluster Count Distribution:
  Minimum: 1
  Maximum: 234
  Mean: 11.68
  Median: 7.00
  Std Dev: 15.23

Diversity Metrics:
  Entropy: 5.8234
  Max Entropy: 6.7520
  Normalized Entropy: 0.8625

Top 10 Most Frequent Clusters:
  Cluster   42:   234 questions (2.34%)
  Cluster  128:   189 questions (1.89%)
  ...
```

## Alternative: Using Bash Script for Generation

If you prefer to use the existing bash script for generation, you can:

1. Generate questions using the bash script:
```bash
cd /workspace/euclid/rentropy/R-Zero-main
bash question_generate/question_generate.bash /path/to/model1 1250 model1_eval
bash question_generate/question_generate.bash /path/to/model2 1250 model2_eval
```

2. Then analyze using the standalone analysis script:
```bash
python analyze_clusters.py \
    --save_name1 model1_eval \
    --save_name2 model2_eval \
    --num_gpus 8
```

## Notes

- The script assumes questions are saved in `{STORAGE_PATH}/generated_question/{save_name}_{gpu_id}.json`
- Each GPU generates `ceil(num_samples / num_gpus)` questions
- Cluster centroids are loaded from the specified centroids.npy file
- The embedding model is loaded once and reused for all questions
- Make sure `STORAGE_PATH` environment variable is set, or use `--storage_path` flag
- The script requires access to the `evaluation.datasets_loader` module (from R-Zero-main)
