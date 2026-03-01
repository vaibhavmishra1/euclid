### expv1_0 — Method 1 (offline) pipeline

This folder contains the **first runnable implementation** of the Method‑1 experiment described in:
- `tree/euclid/expv1/expv1_formal_planner/expv1_0.md`
- `tree/euclid/expv1/expv1_formal_planner/research_proposal_v0.md`

Goal (ExpV1_0):
- Build a synthetic dataset using **GRIP-style concept graph exploration**
- Keep **concept extraction + graph building** separate from **question generation + solver filtering**
- Add **dedup** and **solver-aware ZPD/self-consistency filtering**

This implementation is intentionally modular:
- Concept extraction, generator, teacher, solver inference, and dedup can be swapped between backends (local HF, vLLM, API).

---

## Pipeline Architecture

The pipeline is split into **5 independent stages** for maximum flexibility:

1. **Pipeline A**: Concept extraction + KCRG graph building
2. **Pipeline B1**: Question generation only (format validation + dedup)
3. **Pipeline B2**: ZPD filtering + COT generation (can run multiple times with different thresholds)
4. **Verification** (optional): Quality verification using OpenAI API
5. **Pipeline B3**: SFT training on accepted questions with full COT (supports verified data filtering)

---

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r tree/euclid/expv1/expv1_0/requirements.txt
   ```

2. **Configure models** in `config.yaml`:
   - Set `generation.generator_model` (for question generation)
   - Set `zpd.solver_model` (for ZPD filtering and COT generation)
   - Set `concept_extraction.concept_extractor_model` (for concept extraction)
   - Configure vLLM settings if using vLLM backend

3. **GPU requirements**:
   - Pipeline A: ~16GB+ VRAM (for 32B concept extractor)
   - Pipeline B1: ~8GB+ VRAM (for generator model)
   - Pipeline B2: ~8GB+ VRAM (for solver model)
   - Pipeline B3: ~16GB+ VRAM (for SFT training)

---

## Step-by-Step Instructions

### Step 1: Pipeline A — Concept Extraction + Graph Building

**Purpose**: Extract mathematical concepts from seed problems and build the concept relationship graph (KCRG).

**Command**:
```bash
python -m tree.euclid.expv1.expv1_0.run_build_concepts \
    --config tree/euclid/expv1/expv1_0/config.yaml
```

**What it does**:
1. Loads seed problems from HuggingFace dataset or JSONL file
2. Extracts concepts using LLM (default: Qwen2.5-32B-Instruct)
3. Cleans and canonicalizes concepts (3-pass pipeline)
4. Builds concept graph with explicit and implicit edges
5. Saves artifacts to `output/concepts/`

**Expected outputs**:
```
output/concepts/
├── seeds.jsonl                    # Original seed problems
├── seed_concepts.jsonl            # Extracted concepts per seed
├── concept_vocab.json             # Concept vocabulary
├── concept_graph.json             # KCRG graph
├── cleaned/
│   ├── canonical_vocab.json       # Cleaned concept vocabulary
│   ├── canonical_mapping.json     # Concept merging mappings
│   └── seed_concepts_canonical.jsonl
└── concept_graph_canonical.json   # Graph with cleaned concepts
```

**Time**: ~30-60 minutes for 200 seeds (depends on concept extractor model)

**Troubleshooting**:
- If concept extraction fails: Check GPU memory, reduce `concept_extraction.max_tokens`
- If graph is empty: Check `concept_graph.min_cooccurrence` threshold

---

### Step 2: Pipeline B1 — Question Generation

**Purpose**: Generate mathematical questions from concept bundles (format validation + dedup only).

**Command**:
```bash
python -m tree.euclid.expv1.expv1_0.run_generate_questions \
    --config tree/euclid/expv1/expv1_0/config.yaml
```

**What it does**:
1. Loads cleaned concepts and KCRG graph from Pipeline A
2. Samples concept bundles using explorator (GRIP-style)
3. Generates questions using generator model
4. Validates format (requires `<question>` tags, no answer leakage)
5. Deduplicates (lexical + embedding similarity)
6. Saves valid questions (no ZPD filtering yet)

**Expected outputs**:
```
output_{generator_model}_questions/
├── generated_questions.jsonl      # Valid questions (format: {idx, spec, problem, metadata})
├── candidates.jsonl               # All candidates with accept/reject reasons
├── metrics.json                    # Generation statistics
└── generator_io/                   # Raw generator outputs for debugging
    ├── 000000.txt
    ├── 000001.txt
    └── ...
```

**Configuration**:
- `generation.num_candidates`: Number of questions to generate (default: 20)
- `generation.generator_model`: Model for question generation
- `dedup.lexical`: Enable lexical dedup (default: true)
- `dedup.embedding`: Enable embedding-based dedup (default: false)

**Time**: ~5-10 minutes for 1000 candidates (depends on generator model)

**Troubleshooting**:
- Low acceptance rate: Check `generator_io/` logs for format issues
- Too many duplicates: Lower `dedup.cosine_threshold` or enable `dedup.embedding`

---

### Step 3: Pipeline B2 — ZPD Filtering + COT Generation

**Purpose**: Filter questions by difficulty (ZPD) and generate full chain-of-thought solutions.

**Command** (using config defaults):
```bash
python -m tree.euclid.expv1.expv1_0.run_filter_zpd \
    --config tree/euclid/expv1/expv1_0/config.yaml
```

**Command** (with custom thresholds):
```bash
python -m tree.euclid.expv1.expv1_0.run_filter_zpd \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --p-min 0.2 \
    --p-max 0.7
```

**What it does**:
1. Loads `generated_questions.jsonl` from Pipeline B1
2. For each question, runs solver multiple times (self-consistency)
3. Extracts full COT solution from best rollout
4. Calculates `p_succ` (solver success probability)
5. Filters by ZPD threshold: `p_min <= p_succ <= p_max`
6. Saves accepted questions with full COT solutions

**Expected outputs**:
```
output_{solver_model}_accepted_zpd{p_min}-{p_max}/
├── accepted.jsonl                 # Accepted questions with full COT
├── rejected.jsonl                  # Rejected questions with reasons
└── metrics.json                    # Filtering statistics
```

**Format of `accepted.jsonl`**:
```json
{
  "candidate": {
    "spec": {...},
    "problem": "Find the smallest...",
    "answer": "42",
    ...
  },
  "verification": {
    "modal_answer": "42",
    "p_succ": 0.625,
    "solution": "Let me solve this step by step...\n\\boxed{42}"
  },
  "zpd": {
    "p_succ": 0.625,
    "rollouts": 8,
    "details": {...}
  }
}
```

**Configuration**:
- `zpd.rollouts`: Number of solver rollouts (default: 8)
- `zpd.p_min`: Minimum p_succ threshold (default: 0.1)
- `zpd.p_max`: Maximum p_succ threshold (default: 1.0)
- `zpd.solver_model`: Model for solving and COT generation

**Time**: ~10-20 minutes for 1000 questions (depends on solver model and rollouts)

**Running multiple thresholds**:
```bash
# Easy-medium difficulty (p_succ 0.1-0.5)
python -m tree.euclid.expv1.expv1_0.run_filter_zpd \
    --config config.yaml --p-min 0.1 --p-max 0.5

# Medium-hard difficulty (p_succ 0.3-0.7)
python -m tree.euclid.expv1.expv1_0.run_filter_zpd \
    --config config.yaml --p-min 0.3 --p-max 0.7

# Hard difficulty (p_succ 0.5-0.9)
python -m tree.euclid.expv1.expv1_0.run_filter_zpd \
    --config config.yaml --p-min 0.5 --p-max 0.9
```

Each run creates a separate output directory with different thresholds.

**Troubleshooting**:
- Low acceptance rate: Adjust `p_min`/`p_max` thresholds
- No COT solutions: Check solver model output format
- Out of memory: Reduce `zpd.rollouts` or use smaller solver model

---

### Step 4 (Optional): Verification with OpenAI API

**Purpose**: Verify generated questions and answers using an external LLM (OpenAI GPT-4) to assess quality before SFT.

**Command** (sample and verify questions by difficulty):
```bash
python tree/euclid/expv1/expv1_0/verify_questions_and_answers.py \
    --accepted-file output_Qwen3-1.7B-Base_accepted/accepted.jsonl \
    --output output_Qwen3-1.7B-Base_accepted/verification_report.json \
    --verified-output output_Qwen3-1.7B-Base_accepted/accepted_verified.jsonl \
    --samples-per-bucket 100 \
    --p-min 0.3 --p-max 0.8
```

**What it does**:
1. Samples questions from each difficulty bucket (very_easy, easy, medium, hard, very_hard)
2. Sends each question + answer + COT to OpenAI for verification
3. Evaluates: question validity, answer correctness, COT quality, answer leakage
4. Saves verified questions with OpenAI assessments
5. Generates a comprehensive verification report

**Expected outputs**:
```
output_{model}_accepted/
├── accepted_verified.jsonl        # Questions with OpenAI verification results
├── verification_report.json       # Aggregated statistics by difficulty
└── ...
```

**Retry quota errors** (if verification failed due to API limits):
```bash
python tree/euclid/expv1/expv1_0/retry_quota_errors.py \
    --verified-file output_Qwen3-1.7B-Base_accepted/accepted_verified.jsonl \
    --max-retries 3 \
    --retry-delay 60
```

**Configuration**:
- `--samples-per-bucket`: Questions to sample per difficulty level (default: 100)
- `--p-min` / `--p-max`: Filter by ZPD p_succ before sampling
- `OPENAI_API_KEY`: Environment variable for OpenAI API key

**Verification metrics**:
- `question_validity`: valid / invalid
- `answer_correctness`: correct / incorrect / ambiguous
- `cot_quality`: high / medium / low
- `answer_leakage`: yes / no
- `overall_quality`: excellent / good / fair / poor

---

### Step 5: Pipeline B3 — SFT Training

**Purpose**: Fine-tune solver model on accepted questions with full COT solutions.

**Command** (basic, using all data):
```bash
python -m tree.euclid.expv1.expv1_0.run_sft \
    --config tree/euclid/expv1/expv1_0/config.yaml
```

**Command** (with explicit file path):
```bash
python -m tree.euclid.expv1.expv1_0.run_sft \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --accepted-path output_{model}_accepted_zpd0.2-0.7/accepted.jsonl
```

**Command** (using verified data with quality filters):
```bash
python -m tree.euclid.expv1.expv1_0.run_sft \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --accepted-path tree/euclid/expv1/expv1_0/output_Qwen3-1.7B-Base_accepted/accepted_verified_retried.jsonl \
    --require-valid --require-correct \
    --batch-size 128 \
    --num-epochs 3 \
    --use-bf16 \
    --use-flash-attention \
    --use-torch-compile
```

**Command** (high quality only):
```bash
python -m tree.euclid.expv1.expv1_0.run_sft \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --accepted-path output_Qwen3-1.7B-Base_accepted/accepted_verified_retried.jsonl \
    --require-valid --require-correct --require-no-leakage --min-overall-quality good
```

**What it does**:
1. Auto-detects data file from Pipeline B2 (or uses `--accepted-path`)
2. Supports both formats:
   - `accepted.jsonl` (Pipeline B2 output)
   - `accepted_verified.jsonl` (with OpenAI verification results)
3. Applies quality filters if using verified data
4. Extracts full COT solutions from `verification.solution` field
5. Formats training data: `prompt + "\n" + full_COT_solution`
6. Fine-tunes solver model using HuggingFace Trainer
7. Saves fine-tuned model

**Expected outputs**:
```
output_{solver_model}_accepted/
├── accepted.jsonl                     # (from Pipeline B2)
├── accepted_verified.jsonl            # (from verification step)
├── sft_dataset.jsonl                  # Formatted training data (no filters)
├── sft_dataset_valid_correct.jsonl    # With --require-valid --require-correct
└── ...

output/sft_model/                      # Fine-tuned model
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── ...
```

**Filtering options** (for verified data):
| Flag | Description |
|------|-------------|
| `--require-valid` | Only questions marked as valid by OpenAI |
| `--require-correct` | Only questions with correct answers |
| `--min-cot-quality` | Minimum COT quality: `low`, `medium`, `high` |
| `--require-no-leakage` | Only questions with no answer leakage |
| `--min-overall-quality` | Minimum quality: `poor`, `fair`, `good`, `excellent` |

**Configuration**:
- `sft.enabled`: Enable actual training (default: false)
- `sft.base_model`: Base model for fine-tuning (defaults to solver_model)
- `sft.max_steps`: Training steps (default: 100)
- `sft.output_dir`: Output directory for fine-tuned model

**Training format**:
```
Prompt: "You are a careful mathematical problem solver.
Please reason step by step and put your final answer inside \boxed{}.

Problem:
Find the smallest positive integer n such that..."

Completion: "Let me solve this step by step.
First, I need to find...
Therefore, the answer is \boxed{42}"
```

**Time**: 
- Dataset prep: ~1 minute
- Training: ~30-60 minutes for 100 steps (depends on model size and dataset)

**Troubleshooting**:
- "Missing accepted.jsonl": Run Pipeline B2 first, or specify `--accepted-path`
- Out of memory: Reduce batch size, use gradient checkpointing, or use LoRA
- No COT in training: Check that Pipeline B2 saved `verification.solution` field
- Filtering removed all data: Check verification report for quality breakdown

---

### Step 6: Evaluation (Optional)

**Purpose**: Evaluate fine-tuned model on benchmarks.

**Command**:
```bash
python -m tree.euclid.expv1.expv1_0.run_eval \
    --config tree/euclid/expv1/expv1_0/config.yaml
```

**What it does**:
1. Loads fine-tuned model from `sft.output_dir`
2. Evaluates on MATH benchmark subset
3. Compares before/after SFT performance

**Configuration**:
- `eval.dataset_config`: Benchmark subset (default: "number_theory")
- `eval.limit`: Number of test examples (default: 50)

---

## Benefits of the Multi-Stage Pipeline

- **Flexibility**: Try different ZPD thresholds without re-generating questions
- **Efficiency**: Don't waste solver calls on obviously bad questions
- **Iteration**: Experiment with difficulty bands easily
- **Resource management**: Generate questions on one GPU, filter on another
- **Debugging**: Easier to debug each stage independently
- **Full COT**: SFT trains on complete reasoning traces, not just answers
- **Quality control**: Optional verification step with OpenAI to filter bad data before SFT
- **Configurable filtering**: Train on all data or only verified high-quality samples

---

## Additional Tools

### Concept Cleaning (Optional)

If you want to re-run concept cleaning with different parameters:

```bash
python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --run-pass3 \
    --rebuild-graph
```

This re-runs Pass 3 (LLM verification) and rebuilds the concept graph.

---

## Common Workflows

### Workflow 1: Quick Test (Small Scale)

```bash
# 1. Generate 50 questions
# Edit config.yaml: generation.num_candidates = 50

python -m tree.euclid.expv1.expv1_0.run_generate_questions --config config.yaml

# 2. Filter with default ZPD
python -m tree.euclid.expv1.expv1_0.run_filter_zpd --config config.yaml

# 3. Check results
cat output_*_accepted*/accepted.jsonl | jq '.candidate.problem' | head -5
```

### Workflow 2: Production Run (Large Scale)

```bash
# 1. Generate 5000 questions (overnight)
# Edit config.yaml: generation.num_candidates = 5000

python -m tree.euclid.expv1.expv1_0.run_generate_questions --config config.yaml

# 2. Filter with multiple thresholds (next day)
python -m tree.euclid.expv1.expv1_0.run_filter_zpd --config config.yaml --p-min 0.2 --p-max 0.7
python -m tree.euclid.expv1.expv1_0.run_filter_zpd --config config.yaml --p-min 0.3 --p-max 0.8

# 3. Train on best threshold
python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml \
    --accepted-path output_*_accepted_zpd0.3-0.8/accepted.jsonl
```

### Workflow 3: With Quality Verification

```bash
# 1. Generate and filter questions (Pipelines B1 + B2)
python -m tree.euclid.expv1.expv1_0.run_generate_questions --config config.yaml
python -m tree.euclid.expv1.expv1_0.run_filter_zpd --config config.yaml --p-min 0.3 --p-max 0.8

# 2. Verify quality with OpenAI (sample from medium difficulty)
export OPENAI_API_KEY="your-key"
python tree/euclid/expv1/expv1_0/verify_questions_and_answers.py \
    --accepted-file output_*_accepted/accepted.jsonl \
    --output output_*_accepted/verification_report.json \
    --verified-output output_*_accepted/accepted_verified.jsonl \
    --samples-per-bucket 100 --p-min 0.3 --p-max 0.8

# 3. Review verification report
cat output_*_accepted/verification_report.json | jq '.summary'

# 4. Train SFT on verified high-quality data only
python -m tree.euclid.expv1.expv1_0.run_sft --config config.yaml \
    --accepted-path output_*_accepted/accepted_verified.jsonl \
    --require-valid --require-correct --min-overall-quality good
```

### Workflow 4: Iterative Refinement

```bash
# 1. Generate questions
python -m tree.euclid.expv1.expv1_0.run_generate_questions --config config.yaml

# 2. Try different thresholds to find optimal band
for p_min in 0.1 0.2 0.3; do
  for p_max in 0.6 0.7 0.8; do
    python -m tree.euclid.expv1.expv1_0.run_filter_zpd \
        --config config.yaml --p-min $p_min --p-max $p_max
  done
done

# 3. Compare metrics
cat output_*_accepted*/metrics.json | jq '{threshold: .zpd_threshold, accept_rate: .accept_rate}'
```

---

## Output File Formats

### `generated_questions.jsonl` (Pipeline B1)
```json
{
  "idx": 0,
  "spec": {
    "required_concepts": [{"type": "canonical", "name": "gcd"}],
    "hop_mode": "explicit",
    "target_domain": "",
    "answer_type": "final_answer"
  },
  "problem": "Find the smallest positive integer...",
  "raw_output": "...",
  "metadata": {"format_invalid": false, "answer_leaked": false}
}
```

### `accepted.jsonl` (Pipeline B2)
```json
{
  "candidate": {
    "spec": {...},
    "problem": "Find the smallest...",
    "answer": "42",
    "raw_output": "...",
    "metadata": {...}
  },
  "verification": {
    "modal_answer": "42",
    "p_succ": 0.625,
    "solution": "Let me solve this step by step...\n\\boxed{42}"
  },
  "zpd": {
    "p_succ": 0.625,
    "rollouts": 8,
    "num_correct": 0,
    "details": {
      "preds": ["42", "42", "42", ...],
      "modal_answer": "42",
      "full_outputs": ["...", "...", ...]
    }
  }
}
```

### `accepted_verified.jsonl` (Verification Step)
```json
{
  "id": "uuid-string",
  "difficulty_bucket": "medium",
  "original_data": {
    "candidate": {
      "spec": {...},
      "problem": "Find the smallest...",
      "answer": "42"
    },
    "verification": {
      "modal_answer": "42",
      "p_succ": 0.625,
      "solution": "Let me solve this step by step...\n\\boxed{42}"
    },
    "zpd": {...}
  },
  "verification": {
    "status": "success",
    "verification_result": {
      "question_validity": "valid",
      "answer_correctness": "correct",
      "cot_quality": "high",
      "answer_leakage": "no",
      "overall_quality": "excellent",
      "reasoning": "..."
    },
    "error": null
  },
  "verification_summary": {
    "question_validity": "valid",
    "answer_correctness": "correct",
    "cot_quality": "high",
    "answer_leakage": "no",
    "overall_quality": "excellent"
  }
}
```

### `sft_dataset.jsonl` (Pipeline B3)
```json
{
  "prompt": "You are a careful mathematical problem solver...\n\nProblem:\nFind the smallest...",
  "completion": "Let me solve this step by step...\n\\boxed{42}",
  "problem": "Find the smallest...",
  "answer": "42",
  "has_cot": true,
  "p_succ": 0.625,
  "difficulty_bucket": "medium"
}
```

---

## Notes

- This repo already contains an `expv0/exp0_0` baseline and the `R-Zero/` codebase. ExpV1_0 does **not** run co-evolution RL; it focuses on offline dataset construction + SFT for identifiability.
- The pipeline automatically handles model name sanitization for directory names (e.g., `Qwen/Qwen2.5-Math-7B-Instruct` → `Qwen2.5-Math-7B-Instruct`).
- All paths in config.yaml are relative to the workspace root.
- Pipeline B2 can be run multiple times on the same `generated_questions.jsonl` with different thresholds - each creates a separate output directory.