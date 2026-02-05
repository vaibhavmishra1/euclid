#!/usr/bin/env python3
"""
Calculate maximum voting scores for generated questions.

For each question, rollout n solutions using a solver model, extract answers,
and calculate the maximum voting fraction (most frequent answer count / n).
Then average this across all questions for each model.

Usage:
    python calculate_max_voting.py \
        --solver_model vibhuiitj/qwen3-4b-base-variant2-feb5-solver-iter4 \
        --question_file1 storage/generated_question/variant1-feb5-questioner-iter3_0.json \
        --question_file2 storage/generated_question/variant2-feb5-questioner-iter5_0.json \
        --num_rollouts 4 \
        --output results/max_voting_scores.json \
        --limit 1000 \
        --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")

# Lazy import vLLM to allow checking args first
LLM = None
SamplingParams = None


def load_vllm():
    """Lazy load vLLM modules."""
    global LLM, SamplingParams
    if LLM is None:
        from vllm import LLM as _LLM, SamplingParams as _SamplingParams
        LLM = _LLM
        SamplingParams = _SamplingParams


# --------------------------------------------------------------------------- #
# Answer extraction and normalization (from evaluate.py)
# --------------------------------------------------------------------------- #

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the *last* \\boxed{...} content from a string (handles nested braces).
    """
    prefix = r"\boxed{"
    i = 0
    last: Optional[str] = None

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break
        j = start + len(prefix)
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[start + len(prefix) : j - 1].strip()
        i = j

    return last


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    - Remove whitespace
    - Convert fractions to standard form
    - Handle common LaTeX
    """
    if answer is None:
        return ""
    
    # Remove whitespace
    answer = answer.strip()
    answer = re.sub(r'\s+', '', answer)
    
    # Remove \text{}, \mathrm{}, etc.
    answer = re.sub(r'\\(text|mathrm|mathbf|mathit)\{([^}]*)\}', r'\2', answer)
    
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)
    
    # Remove \left and \right
    answer = re.sub(r'\\(left|right)', '', answer)
    
    # Remove $ signs
    answer = answer.replace('$', '')
    
    return answer.lower()


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #

def build_prompt(problem: str) -> str:
    """
    Build the prompt for the solver model.
    """
    prompt = f"""Please reason step by step, and put your final answer within  \\boxed{{}}.

Problem:
{problem}
"""
    return prompt


# --------------------------------------------------------------------------- #
# Maximum voting calculation
# --------------------------------------------------------------------------- #

def calculate_max_voting_score(
    question: str,
    solver_model: Any,
    sampling_params: Any,
    num_rollouts: int,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Calculate maximum voting score for a single question.
    
    Args:
        question: The math problem text
        solver_model: vLLM model instance
        sampling_params: vLLM sampling parameters
        num_rollouts: Number of solutions to generate
        batch_size: Batch size for generation
    
    Returns:
        Dictionary with:
            - max_voting_score: fraction (most frequent answer count / num_rollouts)
            - answers: list of normalized answers
            - answer_counts: dict of answer -> count
            - most_frequent_answer: the most common answer
            - most_frequent_count: count of most common answer
    """
    # Build prompt
    prompt = build_prompt(question)
    
    # Generate solutions in batches if needed
    all_answers = []
    
    # Process in batches to avoid memory issues
    for batch_start in range(0, num_rollouts, batch_size):
        batch_end = min(batch_start + batch_size, num_rollouts)
        batch_size_actual = batch_end - batch_start
        
        # Create batch of prompts (same prompt repeated)
        prompts_batch = [prompt] * batch_size_actual
        
        # Generate solutions for this batch
        outputs = solver_model.generate(prompts_batch, sampling_params)
        
        # Extract answers from this batch
        for output in outputs:
            generated_text = output.outputs[0].text
            answer = extract_boxed_answer(generated_text)
            if answer:
                normalized = normalize_answer(answer)
                all_answers.append(normalized)
            else:
                # If no boxed answer found, use empty string
                all_answers.append("")
    
    answers = all_answers
    
    # Count answer frequencies
    answer_counts = Counter(answers)
    
    # Find most frequent answer
    if len(answer_counts) > 0:
        most_frequent_answer, most_frequent_count = answer_counts.most_common(1)[0]
    else:
        most_frequent_answer = ""
        most_frequent_count = 0
    
    # Calculate max voting score
    # If the most frequent answer is empty, None, or invalid, set max voting score to 0
    # After normalization, empty answers become "" (empty string)
    if not most_frequent_answer or (isinstance(most_frequent_answer, str) and most_frequent_answer.strip() == ""):
        max_voting_score = 0.0
    else:
        max_voting_score = most_frequent_count / num_rollouts if num_rollouts > 0 else 0.0
    
    return {
        "max_voting_score": max_voting_score,
        "answers": answers,
        "answer_counts": dict(answer_counts),
        "most_frequent_answer": most_frequent_answer,
        "most_frequent_count": most_frequent_count,
        "num_unique_answers": len(answer_counts),
    }


def process_questions_batch(
    questions_batch: List[tuple],
    solver_model: Any,
    sampling_params: Any,
    num_rollouts: int,
) -> List[Dict[str, Any]]:
    """
    Process a batch of questions efficiently by batching all rollouts together.
    
    Args:
        questions_batch: List of (index, question_text) tuples
        solver_model: vLLM model instance
        sampling_params: vLLM sampling parameters
        num_rollouts: Number of solutions per question
    
    Returns:
        List of result dictionaries
    """
    if not questions_batch:
        return []
    
    # Build all prompts: for each question, create num_rollouts prompts
    all_prompts = []
    question_indices = []  # Track which question each prompt belongs to
    
    for idx, question_text in questions_batch:
        prompt = build_prompt(question_text)
        # Add num_rollouts copies of this prompt
        all_prompts.extend([prompt] * num_rollouts)
        question_indices.extend([idx] * num_rollouts)
    
    # Generate all solutions in one batch
    outputs = solver_model.generate(all_prompts, sampling_params)
    
    # Extract answers and group by question
    results = []
    answers_by_question = {}
    
    for i, output in enumerate(outputs):
        question_idx = question_indices[i]
        generated_text = output.outputs[0].text
        answer = extract_boxed_answer(generated_text)
        
        if question_idx not in answers_by_question:
            answers_by_question[question_idx] = []
        
        if answer:
            normalized = normalize_answer(answer)
            answers_by_question[question_idx].append(normalized)
        else:
            answers_by_question[question_idx].append("")
    
    # Calculate max voting scores for each question
    for idx, question_text in questions_batch:
        if idx not in answers_by_question:
            # No answers found
            results.append({
                "question_index": idx,
                "question": question_text,
                "max_voting_score": 0.0,
                "answers": [],
                "answer_counts": {},
                "most_frequent_answer": "",
                "most_frequent_count": 0,
                "num_unique_answers": 0,
            })
            continue
        
        answers = answers_by_question[idx]
        answer_counts = Counter(answers)
        
        if len(answer_counts) > 0:
            most_frequent_answer, most_frequent_count = answer_counts.most_common(1)[0]
        else:
            most_frequent_answer = ""
            most_frequent_count = 0
        
        # If the most frequent answer is empty, None, or invalid, set max voting score to 0
        # After normalization, empty answers become "" (empty string)
        if not most_frequent_answer or (isinstance(most_frequent_answer, str) and most_frequent_answer.strip() == ""):
            max_voting_score = 0.0
        else:
            max_voting_score = most_frequent_count / num_rollouts if num_rollouts > 0 else 0.0
        
        results.append({
            "question_index": idx,
            "question": question_text,
            "max_voting_score": max_voting_score,
            "answers": answers,
            "answer_counts": dict(answer_counts),
            "most_frequent_answer": most_frequent_answer,
            "most_frequent_count": most_frequent_count,
            "num_unique_answers": len(answer_counts),
        })
    
    return results


def plot_comparison(
    scores1: List[float],
    scores2: List[float],
    stats1: Dict[str, Any],
    stats2: Dict[str, Any],
    model1_name: str,
    model2_name: str,
    plot_path: Path
) -> None:
    """
    Create a comparison plot showing both models' distributions side by side.
    
    Args:
        scores1: Max voting scores for model 1
        scores2: Max voting scores for model 2
        stats1: Statistics for model 1
        stats2: Statistics for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        plot_path: Path to save the plot
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Create histogram bins from 0.0 to 1.0
    bins = np.linspace(0.0, 1.0, 21)  # 20 bins, each 0.05 wide
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Model 1
    counts1, bin_edges1, patches1 = ax1.hist(
        scores1,
        bins=bins,
        edgecolor='black',
        alpha=0.7,
        color='steelblue'
    )
    avg1 = stats1.get('average_max_voting_score', 0.0)
    ax1.axvline(avg1, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg1:.3f}')
    ax1.set_xlabel('Max Voting Score', fontsize=12)
    ax1.set_ylabel('Number of Questions', fontsize=12)
    ax1.set_title(f'{model1_name}\n'
                 f'Total: {stats1.get("num_evaluated", 0)}, '
                 f'Rollouts: {stats1.get("num_rollouts_per_question", 0)}',
                 fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks(np.arange(0.0, 1.1, 0.1))
    
    # Plot Model 2
    counts2, bin_edges2, patches2 = ax2.hist(
        scores2,
        bins=bins,
        edgecolor='black',
        alpha=0.7,
        color='orange'
    )
    avg2 = stats2.get('average_max_voting_score', 0.0)
    ax2.axvline(avg2, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg2:.3f}')
    ax2.set_xlabel('Max Voting Score', fontsize=12)
    ax2.set_ylabel('Number of Questions', fontsize=12)
    ax2.set_title(f'{model2_name}\n'
                 f'Total: {stats2.get("num_evaluated", 0)}, '
                 f'Rollouts: {stats2.get("num_rollouts_per_question", 0)}',
                 fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticks(np.arange(0.0, 1.1, 0.1))
    
    # Set overall title
    fig.suptitle('Max Voting Score Distribution Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {plot_path}")


def plot_distribution(
    max_voting_scores: List[float],
    statistics: Dict[str, Any],
    plot_path: Path,
    model_name: str = "Model"
) -> None:
    """
    Plot and save the distribution of max voting scores.
    
    Args:
        max_voting_scores: List of max voting scores
        statistics: Statistics dictionary
        plot_path: Path to save the plot
        model_name: Name of the model for the title
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Create histogram bins from 0.0 to 1.0
    bins = np.linspace(0.0, 1.0, 21)  # 20 bins, each 0.05 wide
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        max_voting_scores,
        bins=bins,
        edgecolor='black',
        alpha=0.7,
        color='steelblue'
    )
    
    # Add vertical line for average
    avg_score = statistics.get('average_max_voting_score', 0.0)
    ax.axvline(avg_score, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_score:.3f}')
    
    # Customize plot
    ax.set_xlabel('Max Voting Score', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_title(f'Max Voting Score Distribution - {model_name}\n'
                f'Total Questions: {statistics.get("num_evaluated", 0)}, '
                f'Rollouts per Question: {statistics.get("num_rollouts_per_question", 0)}',
                fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set x-axis limits and ticks
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    
    # Add text box with statistics
    stats_text = f'Mean: {avg_score:.3f}\n'
    stats_text += f'Min: {statistics.get("min_max_voting_score", 0.0):.3f}\n'
    stats_text += f'Max: {statistics.get("max_max_voting_score", 0.0):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to {plot_path}")


def evaluate_model_questions(
    questions: List[Dict[str, Any]],
    solver_model_name: str,
    num_rollouts: int,
    output_path: Path,
    tensor_parallel_size: int = 1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    batch_size: int = 32,
    model_name: str = None,
) -> Dict[str, Any]:
    """
    Evaluate all questions for a model and calculate average max voting score.
    
    Args:
        questions: List of question dicts with 'question' field
        solver_model_name: Name/path of solver model
        num_rollouts: Number of solutions to generate per question
        output_path: Path to save results
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for processing questions (not rollouts)
    
    Returns:
        Dictionary with statistics
    """
    load_vllm()
    
    print(f"Loading solver model: {solver_model_name}")
    llm = LLM(
        model=solver_model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["Problem:", "---", "\n\nProblem"],
    )
    
    print(f"Evaluating {len(questions)} questions with {num_rollouts} rollouts each...")
    print(f"Using batch size: {batch_size} questions per batch")
    
    # Filter and prepare questions
    valid_questions = []
    for i, q_data in enumerate(questions):
        question_text = q_data.get("question", "")
        if question_text and len(question_text.strip()) >= 10:
            valid_questions.append((i, question_text))
        else:
            print(f"Warning: Skipping question {i} (empty or too short)")
    
    # Process questions in batches
    results = []
    max_voting_scores = []
    
    # Calculate how many questions to process per batch
    # We want to maximize GPU utilization while staying within memory limits
    # Total prompts per batch = questions_per_batch * num_rollouts
    # Aim for batch_size total prompts, so questions_per_batch = batch_size / num_rollouts
    if num_rollouts > 0:
        questions_per_batch = max(1, batch_size // num_rollouts)
    else:
        questions_per_batch = batch_size
    
    # Ensure we don't create batches that are too large
    # Limit to reasonable number of questions per batch (e.g., 50)
    questions_per_batch = min(questions_per_batch, 50)
    
    print(f"Processing {questions_per_batch} questions per batch ({questions_per_batch * num_rollouts} total prompts per batch)")
    
    for batch_start in tqdm(range(0, len(valid_questions), questions_per_batch), desc="Processing batches"):
        batch_end = min(batch_start + questions_per_batch, len(valid_questions))
        questions_batch = valid_questions[batch_start:batch_end]
        
        try:
            batch_results = process_questions_batch(
                questions_batch=questions_batch,
                solver_model=llm,
                sampling_params=sampling_params,
                num_rollouts=num_rollouts,
            )
            results.extend(batch_results)
            max_voting_scores.extend([r["max_voting_score"] for r in batch_results])
            
        except Exception as e:
            print(f"Error processing batch {batch_start}-{batch_end}: {e}")
            # Add failed results for this batch
            for idx, question_text in questions_batch:
                results.append({
                    "question_index": idx,
                    "question": question_text,
                    "max_voting_score": 0.0,
                    "error": str(e)
                })
                max_voting_scores.append(0.0)
    
    # Calculate statistics
    if len(max_voting_scores) > 0:
        avg_max_voting = sum(max_voting_scores) / len(max_voting_scores)
        min_max_voting = min(max_voting_scores)
        max_max_voting = max(max_voting_scores)
    else:
        avg_max_voting = 0.0
        min_max_voting = 0.0
        max_max_voting = 0.0
    
    # Count distribution of max voting scores
    score_distribution = Counter()
    for score in max_voting_scores:
        # Bin scores into ranges
        if score == 1.0:
            score_distribution["1.0"] += 1
        elif score >= 0.8:
            score_distribution["0.8-1.0"] += 1
        elif score >= 0.6:
            score_distribution["0.6-0.8"] += 1
        elif score >= 0.4:
            score_distribution["0.4-0.6"] += 1
        elif score >= 0.2:
            score_distribution["0.2-0.4"] += 1
        else:
            score_distribution["0.0-0.2"] += 1
    
    statistics = {
        "model": solver_model_name,
        "num_questions": len(questions),
        "num_evaluated": len(results),
        "num_rollouts_per_question": num_rollouts,
        "average_max_voting_score": avg_max_voting,
        "min_max_voting_score": min_max_voting,
        "max_max_voting_score": max_max_voting,
        "score_distribution": dict(score_distribution),
        "max_voting_scores": max_voting_scores,  # Include raw scores for comparison plotting
    }
    
    # Save detailed results (without raw scores list to keep file size manageable)
    print(f"\nSaving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create statistics copy without raw scores for saving
    stats_for_save = {k: v for k, v in statistics.items() if k != 'max_voting_scores'}
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "statistics": stats_for_save,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Statistics:")
    print(f"  Total Questions: {statistics['num_questions']}")
    print(f"  Evaluated: {statistics['num_evaluated']}")
    print(f"  Rollouts per Question: {statistics['num_rollouts_per_question']}")
    print(f"  Average Max Voting Score: {statistics['average_max_voting_score']:.4f}")
    print(f"  Min Max Voting Score: {statistics['min_max_voting_score']:.4f}")
    print(f"  Max Max Voting Score: {statistics['max_max_voting_score']:.4f}")
    print(f"{'='*60}")
    
    # Plot and save distribution
    if HAS_MATPLOTLIB:
        plot_path = output_path.with_suffix('.png')
        display_name = model_name if model_name else Path(output_path.stem).stem.replace('max_voting_scores_', '')
        plot_distribution(max_voting_scores, statistics, plot_path, model_name=display_name)
    
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate maximum voting scores for generated questions"
    )
    parser.add_argument(
        "--solver_model",
        type=str,
        required=True,
        help="Solver model name or path (e.g., Qwen/Qwen2.5-3B)"
    )
    parser.add_argument(
        "--question_file1",
        type=str,
        required=True,
        help="Path to first model's generated questions JSON file"
    )
    parser.add_argument(
        "--question_file2",
        type=str,
        required=True,
        help="Path to second model's generated questions JSON file"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of solutions to generate per question (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/max_voting_scores.json",
        help="Output JSON path for results (default: results/max_voting_scores.json)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, default: 0.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for total prompts per batch (default: 128). With num_rollouts=4, this processes ~32 questions per batch"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions to evaluate (0 = all, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Load questions from both files
    print(f"Loading questions from {args.question_file1}...")
    with open(args.question_file1, 'r', encoding='utf-8') as f:
        questions1 = json.load(f)
    
    print(f"Loading questions from {args.question_file2}...")
    with open(args.question_file2, 'r', encoding='utf-8') as f:
        questions2 = json.load(f)
    
    # Limit if specified
    if args.limit > 0:
        questions1 = questions1[:args.limit]
        questions2 = questions2[:args.limit]
        print(f"Limited to {args.limit} questions per model")
    
    print(f"Model 1: {len(questions1)} questions")
    print(f"Model 2: {len(questions2)} questions")
    
    # Evaluate both models
    output_path = Path(args.output)
    
    # Evaluate Model 1
    print(f"\n{'='*60}")
    print("EVALUATING MODEL 1")
    print(f"{'='*60}")
    model1_name = Path(args.question_file1).stem.replace('_0', '').replace('variant', 'Variant')
    stats1 = evaluate_model_questions(
        questions=questions1,
        solver_model_name=args.solver_model,
        num_rollouts=args.num_rollouts,
        output_path=output_path.with_name(f"{output_path.stem}_model1{output_path.suffix}"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        model_name=model1_name,
    )
    
    # Evaluate Model 2
    print(f"\n{'='*60}")
    print("EVALUATING MODEL 2")
    print(f"{'='*60}")
    model2_name = Path(args.question_file2).stem.replace('_0', '').replace('variant', 'Variant')
    stats2 = evaluate_model_questions(
        questions=questions2,
        solver_model_name=args.solver_model,
        num_rollouts=args.num_rollouts,
        output_path=output_path.with_name(f"{output_path.stem}_model2{output_path.suffix}"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        model_name=model2_name,
    )
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nModel 1 Average Max Voting Score: {stats1['average_max_voting_score']:.4f}")
    print(f"Model 2 Average Max Voting Score: {stats2['average_max_voting_score']:.4f}")
    print(f"\nDifference: {abs(stats1['average_max_voting_score'] - stats2['average_max_voting_score']):.4f}")
    
    if stats1['average_max_voting_score'] > stats2['average_max_voting_score']:
        print(f"Model 1 has higher consistency (better max voting score)")
    elif stats2['average_max_voting_score'] > stats1['average_max_voting_score']:
        print(f"Model 2 has higher consistency (better max voting score)")
    else:
        print(f"Both models have equal consistency")
    
    # Save combined summary
    combined_summary = {
        "solver_model": args.solver_model,
        "num_rollouts": args.num_rollouts,
        "model1": {
            "question_file": args.question_file1,
            "statistics": stats1
        },
        "model2": {
            "question_file": args.question_file2,
            "statistics": stats2
        },
        "comparison": {
            "model1_avg_score": stats1['average_max_voting_score'],
            "model2_avg_score": stats2['average_max_voting_score'],
            "difference": abs(stats1['average_max_voting_score'] - stats2['average_max_voting_score']),
            "better_model": "model1" if stats1['average_max_voting_score'] > stats2['average_max_voting_score'] else "model2"
        }
    }
    
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined summary saved to {summary_path}")
    
    # Create comparison plot
    if HAS_MATPLOTLIB and 'max_voting_scores' in stats1 and 'max_voting_scores' in stats2:
        comparison_plot_path = output_path.with_name(f"{output_path.stem}_comparison.png")
        plot_comparison(
            scores1=stats1['max_voting_scores'],
            scores2=stats2['max_voting_scores'],
            stats1=stats1,
            stats2=stats2,
            model1_name=model1_name,
            model2_name=model2_name,
            plot_path=comparison_plot_path
        )
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
