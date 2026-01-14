# exp0_0: Zero-Shot Baseline Experiment

**Experiment ID**: exp0_0  
**Type**: Baseline / No Training  
**Model**: Qwen/Qwen2.5-7B-Instruct  
**Dataset**: MATH-500  
**Date**: January 14, 2026

---

## 1. Objective

Run the complete CAQG pipeline **without any training** to establish baseline metrics before any optimization. This experiment answers the critical question: **How well does an off-the-shelf instruction-tuned model perform at iterative question generation?**

### What We Are Measuring

- Question generation quality (well-posedness, validity)
- Solvability rates (majority voting scores)
- Novelty scores (semantic distance from seed)
- Iteration depth before termination
- Failure mode distribution

### Why This Matters

Without a baseline, we cannot:
1. Quantify improvement from training
2. Identify which components need the most improvement
3. Set realistic targets for the trained model
4. Debug the pipeline before expensive training runs

---

## 2. Experimental Design

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     exp0_0 BASELINE PIPELINE (No Training)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MATH-500 Dataset                                                          │
│        │                                                                    │
│        ▼ (sample N seeds)                                                   │
│   ┌─────────────────┐                                                       │
│   │  Seed Questions │ ──────────────────────────────────────────┐          │
│   │   {q_1, ..., q_N}│                                           │          │
│   └────────┬────────┘                                           │          │
│            │                                                     │          │
│            ▼ FOR each iteration i = 1..I                        │          │
│   ┌─────────────────────────────────────────────────────────┐   │          │
│   │                                                          │   │          │
│   │  ┌──────────────────┐                                   │   │          │
│   │  │ Question Generator│  Qwen2.5-7B-Instruct + P1        │   │          │
│   │  │      (Q)          │  Generate K questions from seed   │   │          │
│   │  └────────┬─────────┘                                   │   │          │
│   │           │ {q'_1, ..., q'_K}                            │   │          │
│   │           ▼                                              │   │          │
│   │  ┌──────────────────┐                                   │   │          │
│   │  │     Solver       │  Qwen2.5-7B-Instruct + P2         │   │          │
│   │  │      (S)         │  Generate M solutions per q'      │   │          │
│   │  └────────┬─────────┘                                   │   │          │
│   │           │ {(q'_k, a_1), ..., (q'_k, a_M)} for each k   │   │          │
│   │           ▼                                              │   │          │
│   │  ┌──────────────────┐                                   │   │          │
│   │  │  Answer Verifier │  Qwen2.5-7B-Instruct + P3         │   │          │
│   │  │      (V1)        │  Compute r1, r2, r_m              │   │          │
│   │  └────────┬─────────┘                                   │   │          │
│   │           │                                              │   │          │
│   │           ▼                                              │   │          │
│   │  ┌──────────────────┐                                   │   │          │
│   │  │ Novelty Verifier │  Qwen2.5-7B-Instruct + P4         │   │          │
│   │  │      (V2)        │  Compute r3_novelty               │   │          │
│   │  └────────┬─────────┘                                   │   │          │
│   │           │                                              │   │          │
│   │           ▼                                              │   │          │
│   │  ┌──────────────────┐                                   │   │          │
│   │  │ Termination Check│                                   │   │          │
│   │  │  (a) Invalid?    │──► STOP & LOG                     │   │          │
│   │  │  (b) Too Hard?   │──► STOP & LOG                     │   │          │
│   │  │  (c) Not Novel?  │──► STOP & LOG                     │   │          │
│   │  │  Otherwise       │──► CONTINUE (i++)                 │   │          │
│   │  └──────────────────┘                                   │   │          │
│   │                                                          │   │          │
│   └─────────────────────────────────────────────────────────┘   │          │
│                                                                  │          │
│   ┌─────────────────────────────────────────────────────────────┘          │
│   │                                                                         │
│   ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         METRICS COLLECTION                           │  │
│   │  • Per-question: r1, r2, r_m, r3, solution_lengths                  │  │
│   │  • Per-iteration: valid_count, termination_reason                    │  │
│   │  • Aggregate: distributions, means, failure modes                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Configuration

```yaml
# config.yaml
experiment:
  name: exp0_0_baseline
  type: inference_only
  seed: 42

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  gpu_memory_utilization: 0.85
  
dataset:
  name: "math"
  source: "MATH-500"
  num_seeds: 50  # Start with 50, can scale to 100

generation:
  K: 4                    # Questions per seed per iteration
  M: 8                    # Solutions per question
  max_iterations: 5       # Maximum iteration depth
  temperature: 1.0
  top_p: 0.95
  max_tokens: 4096

thresholds:
  solvability_t: 0.5      # r_m threshold for "solvable"
  validity_t: 0.5         # r2 threshold for "valid question"
  novelty_t: 0.4          # r3 threshold for "novel"
  valid_ratio_t: 0.5      # Fraction of K that must be valid

output:
  save_dir: "results/"
  save_raw: true
  save_traces: true
```

---

## 3. Implementation Details

### 3.1 Directory Structure

```
exp0_0/
├── README.md                 # This document
├── config.yaml               # Hyperparameters
├── prompts/
│   ├── p1_generator.txt      # Question generator prompt (from v2 plan)
│   ├── p2_solver.txt         # Solver prompt (from v2 plan)
│   ├── p3_verifier.txt       # Answer verifier prompt (from v2 plan)
│   └── p4_novelty.txt        # Novelty verifier prompt (from v2 plan)
├── baseline_pipeline.py      # Main inference script
├── analyze_results.py        # Metrics computation & visualization
├── run_baseline.sh           # Shell wrapper
└── results/                  # Output directory (created at runtime)
    ├── raw_outputs.json
    ├── metrics_summary.json
    ├── iteration_traces.json
    └── failure_analysis.json
```

### 3.2 Dependencies

Reuse from existing R-Zero infrastructure:

| Component | Source File | What We Use |
|-----------|-------------|-------------|
| vLLM setup | `R-Zero/question_generate/question_generate.py` | Model loading, tokenizer, sampling params |
| Majority voting | `R-Zero/question_evaluate/evaluate.py` | Answer clustering, vote counting |
| Dataset loading | `R-Zero/evaluation/datasets_loader.py` | `MathDatasetHandler` class |
| Answer grading | `R-Zero/examples/reward_function/math.py` | `grade_answer` from mathruler |

### 3.3 Core Algorithm

```python
# Pseudocode for baseline_pipeline.py

def run_baseline(config):
    # Initialize
    model = load_vllm_model(config.model.name)
    seeds = load_math_seeds(config.dataset.num_seeds)
    all_results = []
    
    for seed_idx, seed in enumerate(seeds):
        trace = {"seed": seed, "iterations": []}
        context = seed
        
        for iteration in range(config.generation.max_iterations):
            iter_data = {"iteration": iteration, "questions": []}
            
            # Step 1: Generate K questions
            questions = generate_questions(model, context, K=config.generation.K)
            
            for q in questions:
                q_data = {"question": q, "solutions": [], "scores": {}}
                
                # Step 2: Generate M solutions
                solutions = generate_solutions(model, q, M=config.generation.M)
                q_data["solutions"] = solutions
                
                # Step 3: Verify answers (V1)
                r1_scores = [verify_solution(model, q, s) for s in solutions]
                r_m = majority_vote(r1_scores)
                r2 = verify_question_validity(model, q, solutions)
                
                q_data["scores"]["r1"] = r1_scores
                q_data["scores"]["r_m"] = r_m
                q_data["scores"]["r2"] = r2
                
                # Step 4: Verify novelty (V2)
                r3 = verify_novelty(model, seed, q)
                q_data["scores"]["r3"] = r3
                
                iter_data["questions"].append(q_data)
            
            # Step 5: Check termination conditions
            valid_questions = [q for q in iter_data["questions"] 
                             if q["scores"]["r2"] > config.thresholds.validity_t
                             and q["scores"]["r_m"] > config.thresholds.solvability_t]
            
            # Condition (a): Too many invalid
            if len(valid_questions) < config.generation.K * config.thresholds.valid_ratio_t:
                iter_data["termination"] = "invalid_questions"
                trace["iterations"].append(iter_data)
                break
            
            # Condition (b): Too difficult
            avg_solvability = mean([q["scores"]["r_m"] for q in valid_questions])
            if avg_solvability < config.thresholds.solvability_t:
                iter_data["termination"] = "too_difficult"
                trace["iterations"].append(iter_data)
                break
            
            # Condition (c): Not novel
            avg_novelty = mean([q["scores"]["r3"] for q in valid_questions])
            if avg_novelty < config.thresholds.novelty_t:
                iter_data["termination"] = "not_novel"
                trace["iterations"].append(iter_data)
                break
            
            # Continue to next iteration
            iter_data["termination"] = "continue"
            trace["iterations"].append(iter_data)
            context = valid_questions  # Use as seeds for next iteration
        
        all_results.append(trace)
    
    return all_results
```

---

## 4. Metrics Specification

### 4.1 Per-Question Metrics

| Metric | Symbol | Range | Description |
|--------|--------|-------|-------------|
| Format Success | `valid_format` | {0, 1} | Did the output parse correctly? |
| Solution Scores | `r1[1..M]` | [0, 1] | Per-solution correctness |
| Majority Vote | `r_m` | [0, 1] | Fraction of solutions agreeing |
| Question Validity | `r2` | [0, 1] | Is the question well-posed? |
| Novelty Score | `r3` | [0, 1] | Semantic distance from seed |
| Solution Length | `len_tokens` | N+ | Average token count |

### 4.2 Per-Iteration Metrics

| Metric | Description |
|--------|-------------|
| `num_valid` | Questions passing validity threshold |
| `num_solvable` | Questions with r_m > t |
| `num_novel` | Questions with r3 > t_novelty |
| `termination_reason` | Why iteration stopped |

### 4.3 Aggregate Metrics

| Metric | Formula | Expected Baseline |
|--------|---------|-------------------|
| Format Success Rate | `mean(valid_format)` | 70-85% |
| Question Validity Rate | `mean(r2 > 0.5)` | 40-60% |
| Solvability Rate | `mean(r_m > 0.5)` | 30-50% |
| Mean Novelty | `mean(r3)` | 0.2-0.4 |
| Mean Iteration Depth | `mean(final_iteration)` | 1-2 |
| Termination Distribution | `count(reason)/total` | Varies |

---

## 5. Expected Results & Hypotheses

### 5.1 Baseline Hypotheses

**H_baseline_1**: Without training, the model will generate mostly trivial variations (low novelty).

**H_baseline_2**: Many generated questions will be ill-posed or have no solution.

**H_baseline_3**: Iteration depth will be shallow (1-2) due to early termination.

**H_baseline_4**: Failure mode will be dominated by "not_novel" (condition c).

### 5.2 Expected Distributions

```
Expected Termination Reasons:
┌─────────────────────┬────────────┐
│ Reason              │ Expected % │
├─────────────────────┼────────────┤
│ invalid_questions   │ 25-35%     │
│ too_difficult       │ 15-25%     │
│ not_novel           │ 35-45%     │
│ max_iterations      │ 5-15%      │
└─────────────────────┴────────────┘

Expected r3 (Novelty) Distribution:
┌──────────────┬────────────┐
│ Range        │ Expected % │
├──────────────┼────────────┤
│ 0.0 - 0.2    │ 40-50%     │  ← Trivial variations
│ 0.2 - 0.4    │ 30-40%     │  ← Minor changes
│ 0.4 - 0.6    │ 10-20%     │  ← Moderate novelty
│ 0.6 - 1.0    │ 5-10%      │  ← Genuine novelty
└──────────────┴────────────┘
```

---

## 6. How to Run

### 6.1 Prerequisites

```bash
# Environment setup
pip install vllm transformers torch stopit mathruler datasets pandas

# Set storage path
export STORAGE_PATH="/path/to/storage"
export HF_TOKEN="your_huggingface_token"  # For model download
```

### 6.2 Execution

```bash
# From exp0_0 directory
cd tree/euclid/expv0/exp0_formal_planner/exp0_0

# Run baseline (single GPU)
CUDA_VISIBLE_DEVICES=0 python baseline_pipeline.py --config config.yaml

# Or use the wrapper script
bash run_baseline.sh
```

### 6.3 Analysis

```bash
# After pipeline completes
python analyze_results.py --results_dir results/

# Outputs:
# - results/metrics_summary.json
# - results/plots/ (if matplotlib available)
```

---

## 7. Success Criteria

The baseline experiment is successful if:

1. **Pipeline Completeness**: All N seeds are processed without crashes
2. **Data Quality**: At least 80% of outputs are parseable
3. **Metric Coverage**: All specified metrics are computed
4. **Reproducibility**: Running twice with same seed gives same results

The baseline metrics become the **targets to improve** in subsequent training experiments (exp0_1, exp0_2, etc.).

---

## 8. Next Steps After Baseline

| If Baseline Shows... | Then Prioritize... |
|---------------------|-------------------|
| Low validity (r2 < 0.4) | Better P1 prompt / constraint extraction |
| Low solvability (r_m < 0.3) | Difficulty calibration in generator |
| Low novelty (r3 < 0.3) | Transformation diversity training |
| Early termination (depth < 2) | Relaxed thresholds / better filtering |

---

## 9. Relation to Other Documents

| Document | Relation |
|----------|----------|
| `CAQG_research_plan.md` (v1) | Theoretical framework |
| `CAQG_research_plan_v2.md` | Full pipeline with prompts |
| `exp0_0_baseline_plan.md` (this) | Inference-only baseline |
| `exp0_1_*.md` (future) | First training experiment |

---

## Appendix A: Prompt Summaries

### P1 (Generator)
- Input: Seed question + previous questions
- Output: New question with transformation, answer, difficulty estimate
- Key: Must specify which transformation was applied

### P2 (Solver)  
- Input: Question
- Output: Step-by-step solution with boxed answer
- Key: Show all work, handle edge cases

### P3 (Answer Verifier V1)
- Input: Question + proposed solution + answer
- Output: r1 (solution correct), r2 (question valid)
- Key: Check both question AND solution

### P4 (Novelty Verifier V2)
- Input: Seed question + generated question + solver stats
- Output: r3 (novelty), r3_difficulty, r3_transformation
- Key: Provide detailed explanation (for future meta-verification)

---

## Appendix B: Compute Requirements

| Component | Estimate |
|-----------|----------|
| Model Memory | ~16GB VRAM (7B model) |
| Time per seed | ~5-10 minutes |
| Total (50 seeds) | ~4-8 hours |
| Storage | ~500MB for results |

Single A100 or 2x A10G recommended.
